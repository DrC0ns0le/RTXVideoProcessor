#include "rtx_processor.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include "cuda_kernels.h"

// Compatibility with CUDA headers for 10:10:10:2 array format like in SDK sample
#define CUDA_VERSION_INT_101010_2_DEFINED 12080
#if (CUDA_VERSION < CUDA_VERSION_INT_101010_2_DEFINED)
#ifndef CU_AD_FORMAT_UNORM_INT_101010_2
#define CU_AD_FORMAT_UNORM_INT_101010_2 ((CUarray_format)0x50)
#endif
#endif

#define CUDADRV_CHECK(x)                                                                            \
{                                                                                                   \
    CUresult rval;                                                                                  \
    if ((rval = (x)) != CUDA_SUCCESS)                                                               \
    {                                                                                               \
        const char *error_str;                                                                      \
        cuGetErrorString(rval, &error_str);                                                         \
        fprintf(stderr, "%s():%i: CUDA driver API error: %s\n", __FUNCTION__, __LINE__, error_str);   \
        throw std::runtime_error("CUDA error");                                                    \
    }                                                                                               \
}

RtxProcessor::RtxProcessor() {}
RtxProcessor::~RtxProcessor() { shutdown(); }

bool RtxProcessor::initialize(int gpuIndex, const RtxProcessConfig& cfg, uint32_t srcW, uint32_t srcH)
{
    if (m_initialized) return true;
    m_cfg = cfg;
    m_srcW = srcW;
    m_srcH = srcH;
    m_dstW = srcW * (cfg.enableVSR ? cfg.scaleFactor : 1);
    m_dstH = srcH * (cfg.enableVSR ? cfg.scaleFactor : 1);

    if (!initCuda(gpuIndex)) {
        setError("initCuda failed (check NVIDIA driver and CUDA installation)");
        return false;
    }
    if (!createRtx(cfg.enableTHDR, cfg.enableVSR)) {
        setError(std::string("RTX API create failed (THDR=") + (cfg.enableTHDR?"1":"0") + ", VSR=" + (cfg.enableVSR?"1":"0") + ")");
        return false;
    }
    if (!allocSurfaces(cfg.enableTHDR)) {
        if (cfg.enableTHDR) {
            setError("allocSurfaces failed with THDR enabled (HDR 10:10:10:2 surface may be unsupported by driver/GPU)");
        } else {
            setError("allocSurfaces failed while creating BGRA surfaces");
        }
        return false;
    }

    m_inPitch = m_srcW * 4; // BGRA8
    // Output pitch: FP16 not used. If THDR on, we use 10bit A2R10G10B10 (4 bytes per pixel)
    m_outPitch = m_dstW * 4;

    m_hostOut.resize(m_outPitch * m_dstH);

    m_initialized = true;
    return true;
}

bool RtxProcessor::processGpuNV12ToP010(const uint8_t* d_y, int pitchY,
                                        const uint8_t* d_uv, int pitchUV,
                                        AVFrame* encP010Frame,
                                        bool bt2020)
{
    if (!m_initialized || !d_y || !d_uv || !encP010Frame) {
        setError("processGpuNV12ToP010: invalid state or null args");
        return false;
    }

    CUDADRV_CHECK(cuCtxSetCurrent(m_ctx));

    // 1) NV12 (device) -> BGRA8 (device pitched)
    launch_nv12_to_bgra(d_y, pitchY, d_uv, pitchUV,
                        m_devBGRA, (int)m_devBGRAPitch,
                        (int)m_srcW, (int)m_srcH,
                        bt2020,
                        m_stream);

    // 2) Copy BGRA8 (device pitched) -> m_srcArray (CUDA array) for RTX input
    CUDA_MEMCPY2D copyIn{};
    copyIn.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyIn.srcDevice     = (CUdeviceptr)m_devBGRA;
    copyIn.srcPitch      = (unsigned int)m_devBGRAPitch;
    copyIn.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyIn.dstArray      = m_srcArray;
    copyIn.WidthInBytes  = (unsigned int)(m_srcW * 4);
    copyIn.Height        = m_srcH;
    CUDADRV_CHECK(cuMemcpy2D(&copyIn));

    // 3) RTX evaluate: m_srcTex -> m_dstSurf (ABGR10 when THDR enabled)
    API_RECT srcRect{0, 0, (int)m_srcW, (int)m_srcH};
    API_RECT dstRect{0, 0, (int)m_dstW, (int)m_dstH};
    API_VSR_Setting vsr{}; vsr.QualityLevel = m_cfg.vsrQuality;
    API_THDR_Setting thdr{};
    thdr.Contrast = m_cfg.thdrContrast;
    thdr.Saturation = m_cfg.thdrSaturation;
    thdr.MiddleGray = m_cfg.thdrMiddleGray;
    thdr.MaxLuminance = m_cfg.thdrMaxLuminance;

    bool ok = rtx_video_api_cuda_evaluate(m_srcTex, m_dstSurf, srcRect, dstRect,
                                          m_cfg.enableVSR ? &vsr : nullptr,
                                          m_cfg.enableTHDR ? &thdr : nullptr);
    if (!ok) {
        setError("RTX evaluate failed");
        return false;
    }

    // 4) Copy m_dstArray (ABGR10) -> device pitched staging
    CUDA_MEMCPY2D copyOut{};
    copyOut.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyOut.srcArray      = m_dstArray;
    copyOut.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyOut.dstDevice     = (CUdeviceptr)m_devABGR10;
    copyOut.dstPitch      = (unsigned int)m_devABGR10Pitch;
    copyOut.WidthInBytes  = (unsigned int)(m_dstW * 4);
    copyOut.Height        = m_dstH;
    CUDADRV_CHECK(cuMemcpy2D(&copyOut));

    // 5) ABGR10 -> P010 directly into FFmpeg CUDA frame planes
    uint8_t* d_outY  = encP010Frame->data[0];
    uint8_t* d_outUV = encP010Frame->data[1];
    int pitchOutY    = encP010Frame->linesize[0];
    int pitchOutUV   = encP010Frame->linesize[1];
    if (!d_outY || !d_outUV || pitchOutY <= 0 || pitchOutUV <= 0) {
        setError("processGpuNV12ToP010: invalid encoder CUDA frame planes");
        return false;
    }

    launch_abgr10_to_p010(m_devABGR10, (int)m_devABGR10Pitch,
                          d_outY, pitchOutY,
                          d_outUV, pitchOutUV,
                          (int)m_dstW, (int)m_dstH,
                          bt2020,
                          m_stream);

    cudaStreamSynchronize(m_stream);
    return true;
}

bool RtxProcessor::initializeWithContext(CUcontext externalCtx, const RtxProcessConfig& cfg, uint32_t srcW, uint32_t srcH)
{
    if (m_initialized) return true;
    m_cfg = cfg;
    m_srcW = srcW;
    m_srcH = srcH;
    m_dstW = srcW * (cfg.enableVSR ? cfg.scaleFactor : 1);
    m_dstH = srcH * (cfg.enableVSR ? cfg.scaleFactor : 1);

    m_ctx = externalCtx;
    m_externalCtx = true;
    CUDADRV_CHECK(cuCtxSetCurrent(m_ctx));
    if (!createRtx(cfg.enableTHDR, cfg.enableVSR)) {
        setError(std::string("RTX API create failed (THDR=") + (cfg.enableTHDR?"1":"0") + ", VSR=" + (cfg.enableVSR?"1":"0") + ")");
        return false;
    }
    if (!allocSurfaces(cfg.enableTHDR)) {
        setError("allocSurfaces failed in initializeWithContext");
        return false;
    }

    // Create stream
    if (!m_stream) cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);

    // Allocate device staging buffers
    size_t pitch = 0;
    if (cudaMallocPitch(&m_devBGRA, &pitch, m_srcW * 4, m_srcH) != cudaSuccess) {
        setError("cudaMallocPitch m_devBGRA failed");
        return false;
    }
    m_devBGRAPitch = pitch;
    if (cudaMallocPitch(&m_devABGR10, &pitch, m_dstW * 4, m_dstH) != cudaSuccess) {
        setError("cudaMallocPitch m_devABGR10 failed");
        return false;
    }
    m_devABGR10Pitch = pitch;

    m_initialized = true;
    return true;
}

bool RtxProcessor::process(const uint8_t* inBGRA, size_t inPitchBytes,
                           const uint8_t*& outData, uint32_t& outWidth, uint32_t& outHeight, size_t& outPitchBytes)
{
    if (!m_initialized) return false;

    // Copy input BGRA into CUDA array as source
    CUDA_MEMCPY2D copyIn{};
    copyIn.srcMemoryType = CU_MEMORYTYPE_HOST;
    copyIn.srcHost = inBGRA;
    copyIn.srcPitch = inPitchBytes;
    copyIn.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyIn.dstArray = m_srcArray;
    copyIn.dstPitch = 0;
    copyIn.WidthInBytes = m_inPitch;
    copyIn.Height = m_srcH;
    CUDADRV_CHECK(cuMemcpy2D(&copyIn));

    API_RECT srcRect{0, 0, (int)m_srcW, (int)m_srcH};
    API_RECT dstRect{0, 0, (int)m_dstW, (int)m_dstH};

    API_VSR_Setting vsr{};
    vsr.QualityLevel = m_cfg.vsrQuality; // 1..4

    API_THDR_Setting thdr{};
    thdr.Contrast = m_cfg.thdrContrast;
    thdr.Saturation = m_cfg.thdrSaturation;
    thdr.MiddleGray = m_cfg.thdrMiddleGray;
    thdr.MaxLuminance = m_cfg.thdrMaxLuminance;

    bool ok = rtx_video_api_cuda_evaluate(m_srcTex, m_dstSurf, srcRect, dstRect,
                                          m_cfg.enableVSR ? &vsr : nullptr,
                                          m_cfg.enableTHDR ? &thdr : nullptr);
    if (!ok) return false;

    // Copy CUDA dst array back to host
    CUDA_MEMCPY2D copyOut{};
    copyOut.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyOut.srcArray = m_dstArray;
    copyOut.dstMemoryType = CU_MEMORYTYPE_HOST;
    copyOut.dstHost = m_hostOut.data();
    copyOut.dstPitch = (unsigned int)m_outPitch;
    copyOut.WidthInBytes = (unsigned int)m_outPitch;
    copyOut.Height = m_dstH;
    CUDADRV_CHECK(cuMemcpy2D(&copyOut));

    outData = m_hostOut.data();
    outWidth = m_dstW;
    outHeight = m_dstH;
    outPitchBytes = m_outPitch;
    return true;
}

void RtxProcessor::shutdown()
{
    destroyRtx();
    freeSurfaces();
    deinitCuda();
    m_initialized = false;
}

bool RtxProcessor::initCuda(int gpuIndex)
{
    CUDADRV_CHECK(cuInit(0));
    int count = 0; CUDADRV_CHECK(cuDeviceGetCount(&count));
    if (gpuIndex < 0 || gpuIndex >= count) gpuIndex = 0;
    CUDADRV_CHECK(cuDeviceGet(&m_device, gpuIndex));
    CUDADRV_CHECK(cuCtxCreate(&m_ctx, 0, m_device));
    // stream optional
    return true;
}

void RtxProcessor::deinitCuda()
{
    if (m_stream) { cudaStreamDestroy(m_stream); m_stream = nullptr; }
    if (!m_externalCtx && m_ctx) { cuCtxDestroy(m_ctx); m_ctx = nullptr; }
}

bool RtxProcessor::createRtx(bool thdr, bool vsr)
{
    return rtx_video_api_cuda_create(m_ctx, m_stream, 0, thdr, vsr);
}

void RtxProcessor::destroyRtx()
{
    rtx_video_api_cuda_shutdown();
}

bool RtxProcessor::allocSurfaces(bool thdr)
{
    // Create source array BGRA8
    CUDA_ARRAY_DESCRIPTOR srcDesc{};
    srcDesc.Width = m_srcW;
    srcDesc.Height = m_srcH;
    srcDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    srcDesc.NumChannels = 4;

    CUDADRV_CHECK(cuArrayCreate(&m_srcArray, &srcDesc));

    CUDA_RESOURCE_DESC srcRes{};
    srcRes.resType = CU_RESOURCE_TYPE_ARRAY;
    srcRes.res.array.hArray = m_srcArray;

    CUDA_TEXTURE_DESC texDesc{};
    texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
    texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;

    CUDADRV_CHECK(cuTexObjectCreate(&m_srcTex, &srcRes, &texDesc, nullptr));

    // Destination: if THDR, we need 10-bit A2R10G10B10 (UNORM 10:10:10:2)
    CUDA_ARRAY_DESCRIPTOR dstDesc{};
    dstDesc.Width = m_dstW;
    dstDesc.Height = m_dstH;
    dstDesc.Format = thdr ? CU_AD_FORMAT_UNORM_INT_101010_2 : CU_AD_FORMAT_UNSIGNED_INT8;
    dstDesc.NumChannels = 4;

    CUresult cres = cuArrayCreate(&m_dstArray, &dstDesc);
    if (cres != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA does not support the HDR format needed for TrueHDR. Update NVIDIA driver.\n");
        return false;
    }

    CUDA_RESOURCE_DESC dstRes{};
    dstRes.resType = CU_RESOURCE_TYPE_ARRAY;
    dstRes.res.array.hArray = m_dstArray;
    CUDADRV_CHECK(cuSurfObjectCreate(&m_dstSurf, &dstRes));

    return true;
}

void RtxProcessor::freeSurfaces()
{
    if (m_srcTex) { cuTexObjectDestroy(m_srcTex); m_srcTex = 0; }
    if (m_dstSurf) { cuSurfObjectDestroy(m_dstSurf); m_dstSurf = 0; }
    if (m_srcArray) { cuArrayDestroy(m_srcArray); m_srcArray = nullptr; }
    if (m_dstArray) { cuArrayDestroy(m_dstArray); m_dstArray = nullptr; }
    if (m_devBGRA) { cudaFree(m_devBGRA); m_devBGRA = nullptr; m_devBGRAPitch = 0; }
    if (m_devABGR10) { cudaFree(m_devABGR10); m_devABGR10 = nullptr; m_devABGR10Pitch = 0; }
}
