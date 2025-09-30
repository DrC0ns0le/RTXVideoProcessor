#include "rtx_processor.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C"
{
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include "cuda_kernels.h"
#include "logger.h"

// Compatibility with CUDA headers for 10:10:10:2 array format like in SDK sample
#define CUDA_VERSION_INT_101010_2_DEFINED 12080
#if (CUDA_VERSION < CUDA_VERSION_INT_101010_2_DEFINED)
#ifndef CU_AD_FORMAT_UNORM_INT_101010_2
#define CU_AD_FORMAT_UNORM_INT_101010_2 ((CUarray_format)0x50)
#endif
#endif

#define CUDADRV_CHECK(x)                                                                                \
    {                                                                                                   \
        CUresult rval;                                                                                  \
        if ((rval = (x)) != CUDA_SUCCESS)                                                               \
        {                                                                                               \
            const char *error_str;                                                                      \
            cuGetErrorString(rval, &error_str);                                                         \
            fprintf(stderr, "%s():%i: CUDA driver API error: %s\n", __FUNCTION__, __LINE__, error_str); \
            throw std::runtime_error("CUDA error");                                                     \
        }                                                                                               \
    }

RTXProcessor::RTXProcessor() {}
RTXProcessor::~RTXProcessor() { shutdown(); }

bool RTXProcessor::initialize(int gpuIndex, const RTXProcessConfig &cfg, uint32_t srcW, uint32_t srcH)
{
    if (m_initialized)
        return true;
    m_cfg = cfg;
    m_srcW = srcW;
    m_srcH = srcH;
    m_dstW = srcW * (cfg.enableVSR ? cfg.scaleFactor : 1);
    m_dstH = srcH * (cfg.enableVSR ? cfg.scaleFactor : 1);

    if (!initCuda(gpuIndex))
    {
        setError("initCuda failed (check NVIDIA driver and CUDA installation)");
        return false;
    }
    if (!createRTX(cfg.enableTHDR, cfg.enableVSR))
    {
        setError(std::string("RTX API create failed (THDR=") + (cfg.enableTHDR ? "1" : "0") + ", VSR=" + (cfg.enableVSR ? "1" : "0") + ")");
        return false;
    }
    if (!allocSurfaces(cfg.enableTHDR))
    {
        if (cfg.enableTHDR)
        {
            setError("allocSurfaces failed with THDR enabled (HDR 10:10:10:2 surface may be unsupported by driver/GPU)");
        }
        else
        {
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

bool RTXProcessor::processGpuNV12ToP010(const uint8_t *d_y, int pitchY,
                                        const uint8_t *d_uv, int pitchUV,
                                        AVFrame *encP010Frame,
                                        bool bt2020)
{
    if (!m_initialized || !d_y || !d_uv || !encP010Frame)
    {
        setError("processGpuNV12ToP010: invalid state or null args");
        return false;
    }

    CUDADRV_CHECK(cuCtxSetCurrent(m_ctx));

    // Interpret the input using its native colorspace (bt2020 indicates input colorspace)
    // 1) NV12 (device) -> BGRA8 (device pitched)
    launch_nv12_to_bgra(d_y, pitchY, d_uv, pitchUV,
                        m_devBGRA, (int)m_devBGRAPitch,
                        (int)m_srcW, (int)m_srcH,
                        bt2020,
                        m_stream);

    // If both VSR and THDR are disabled, bypass RTX evaluate and convert directly to P010
    if (!m_cfg.enableVSR && !m_cfg.enableTHDR)
    {
        uint8_t *d_outY = encP010Frame->data[0];
        uint8_t *d_outUV = encP010Frame->data[1];
        int pitchOutY = encP010Frame->linesize[0];
        int pitchOutUV = encP010Frame->linesize[1];
        if (!d_outY || !d_outUV || pitchOutY <= 0 || pitchOutUV <= 0)
        {
            setError("processGpuNV12ToP010: invalid encoder CUDA frame planes");
            return false;
        }
        // Direct BGRA8 -> P010 using input colorspace
        launch_bgra8_to_p010(m_devBGRA, (int)m_devBGRAPitch,
                             d_outY, pitchOutY,
                             d_outUV, pitchOutUV,
                             (int)m_srcW, (int)m_srcH,
                             /*bt2020=*/bt2020,
                             m_stream);
        cudaStreamSynchronize(m_stream);
        return true;
    }

    // 2) Copy BGRA8 (device pitched) -> m_srcArray (CUDA array) for RTX input
    CUDA_MEMCPY2D copyIn{};
    copyIn.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyIn.srcDevice = (CUdeviceptr)m_devBGRA;
    copyIn.srcPitch = (unsigned int)m_devBGRAPitch;
    copyIn.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyIn.dstArray = m_srcArray;
    copyIn.WidthInBytes = (unsigned int)(m_srcW * 4);
    copyIn.Height = m_srcH;
    CUDADRV_CHECK(cuMemcpy2D(&copyIn));

    // 3) RTX evaluate: m_srcTex -> m_dstSurf (ABGR10 when THDR enabled)
    API_RECT srcRect{0, 0, (int)m_srcW, (int)m_srcH};
    API_RECT dstRect{0, 0, (int)m_dstW, (int)m_dstH};
    API_VSR_Setting vsr{};
    vsr.QualityLevel = m_cfg.vsrQuality;
    API_THDR_Setting thdr{};
    thdr.Contrast = m_cfg.thdrContrast;
    thdr.Saturation = m_cfg.thdrSaturation;
    thdr.MiddleGray = m_cfg.thdrMiddleGray;
    thdr.MaxLuminance = m_cfg.thdrMaxLuminance;

    bool ok = rtx_video_api_cuda_evaluate(m_srcTex, m_dstSurf, srcRect, dstRect,
                                          m_cfg.enableVSR ? &vsr : nullptr,
                                          m_cfg.enableTHDR ? &thdr : nullptr);
    if (!ok)
    {
        setError("RTX evaluate failed");
        return false;
    }

    // 4) Copy m_dstArray (ABGR10 if THDR, BGRA8 if SDR) -> device pitched staging
    CUDA_MEMCPY2D copyOut{};
    copyOut.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyOut.srcArray = m_dstArray;
    copyOut.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyOut.dstDevice = (CUdeviceptr)m_devABGR10;
    copyOut.dstPitch = (unsigned int)m_devABGR10Pitch;
    copyOut.WidthInBytes = (unsigned int)(m_dstW * 4);
    copyOut.Height = m_dstH;
    CUDADRV_CHECK(cuMemcpy2D(&copyOut));

    // 5) Convert RTX output to P010 directly into FFmpeg CUDA frame planes
    uint8_t *d_outY = encP010Frame->data[0];
    uint8_t *d_outUV = encP010Frame->data[1];
    int pitchOutY = encP010Frame->linesize[0];
    int pitchOutUV = encP010Frame->linesize[1];
    if (!d_outY || !d_outUV || pitchOutY <= 0 || pitchOutUV <= 0)
    {
        setError("processGpuNV12ToP010: invalid encoder CUDA frame planes");
        return false;
    }

    // Use BT.2020 for HDR output; BT.709 for SDR.
    if (m_cfg.enableTHDR)
    {
        // RTX produced ABGR10
        launch_abgr10_to_p010(m_devABGR10, (int)m_devABGR10Pitch,
                              d_outY, pitchOutY,
                              d_outUV, pitchOutUV,
                              (int)m_dstW, (int)m_dstH,
                              /*bt2020=*/true,
                              m_stream);
    }
    else
    {
        // RTX produced BGRA8
        launch_bgra8_to_p010(m_devABGR10, (int)m_devABGR10Pitch,
                             d_outY, pitchOutY,
                             d_outUV, pitchOutUV,
                             (int)m_dstW, (int)m_dstH,
                             /*bt2020=*/bt2020,
                             m_stream);
    }

    // Ensure all GPU operations complete before returning
    cudaStreamSynchronize(m_stream);
    return true;
}

bool RTXProcessor::processGpuP010ToP010(const uint8_t *d_y, int pitchY,
                                        const uint8_t *d_uv, int pitchUV,
                                        AVFrame *encP010Frame,
                                        bool bt2020)
{
    if (!m_initialized || !d_y || !d_uv || !encP010Frame)
    {
        setError("processGpuP010ToP010: invalid state or null args");
        return false;
    }

    CUDADRV_CHECK(cuCtxSetCurrent(m_ctx));

    // 1) P010 (device) -> X2BGR10LE (10-bit RGB, device pitched) - preserving full 10-bit precision
    launch_p010_to_x2bgr10(d_y, pitchY, d_uv, pitchUV,
                           m_devABGR10, (int)m_devABGR10Pitch,  // Reuse ABGR10 buffer for X2BGR10LE
                           (int)m_srcW, (int)m_srcH,
                           bt2020,
                           m_stream);

    // If both VSR and THDR are disabled, bypass RTX evaluate and convert directly to P010
    if (!m_cfg.enableVSR && !m_cfg.enableTHDR)
    {
        uint8_t *d_outY = encP010Frame->data[0];
        uint8_t *d_outUV = encP010Frame->data[1];
        int pitchOutY = encP010Frame->linesize[0];
        int pitchOutUV = encP010Frame->linesize[1];
        if (!d_outY || !d_outUV || pitchOutY <= 0 || pitchOutUV <= 0)
        {
            setError("processGpuP010ToP010: invalid encoder CUDA frame planes");
            return false;
        }
        // Direct X2BGR10LE -> P010 preserving 10-bit precision with BT.2020 colorspace
        launch_abgr10_to_p010(m_devABGR10, (int)m_devABGR10Pitch,
                              d_outY, pitchOutY,
                              d_outUV, pitchOutUV,
                              (int)m_srcW, (int)m_srcH,
                              /*bt2020=*/true,  // Always use BT.2020 for HDR content
                              m_stream);
        cudaStreamSynchronize(m_stream);
        return true;
    }

    // 2) Copy X2BGR10LE (device pitched) -> m_srcArray (CUDA array) for RTX input
    CUDA_MEMCPY2D copyIn{};
    copyIn.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyIn.srcDevice = (CUdeviceptr)m_devABGR10;  // Source is now X2BGR10LE
    copyIn.srcPitch = (unsigned int)m_devABGR10Pitch;
    copyIn.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyIn.dstArray = m_srcArray;
    copyIn.WidthInBytes = (unsigned int)(m_srcW * 4);  // 4 bytes per pixel for X2BGR10LE
    copyIn.Height = m_srcH;
    CUDADRV_CHECK(cuMemcpy2D(&copyIn));

    // 3) RTX evaluate: m_srcTex -> m_dstSurf (ABGR10 when processing HDR content)
    API_RECT srcRect{0, 0, (int)m_srcW, (int)m_srcH};
    API_RECT dstRect{0, 0, (int)m_dstW, (int)m_dstH};
    API_VSR_Setting vsr{};
    vsr.QualityLevel = m_cfg.vsrQuality;
    API_THDR_Setting thdr{};
    thdr.Contrast = m_cfg.thdrContrast;
    thdr.Saturation = m_cfg.thdrSaturation;
    thdr.MiddleGray = m_cfg.thdrMiddleGray;
    thdr.MaxLuminance = m_cfg.thdrMaxLuminance;

    bool rtx_ok = rtx_video_api_cuda_evaluate(m_srcTex, m_dstSurf, srcRect, dstRect,
                                              m_cfg.enableVSR ? &vsr : nullptr,
                                              m_cfg.enableTHDR ? &thdr : nullptr);
    if (!rtx_ok)
    {
        setError("processGpuP010ToP010: RTX evaluate failed");
        return false;
    }

    // 4) Copy m_dstArray (ABGR10) -> device pitched staging
    CUDA_MEMCPY2D copyOut{};
    copyOut.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyOut.srcArray = m_dstArray;
    copyOut.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyOut.dstDevice = (CUdeviceptr)m_devABGR10;
    copyOut.dstPitch = (unsigned int)m_devABGR10Pitch;
    copyOut.WidthInBytes = (unsigned int)(m_dstW * 4);
    copyOut.Height = m_dstH;
    CUDADRV_CHECK(cuMemcpy2D(&copyOut));

    // 5) ABGR10 (device pitched) -> P010 (encoder frame planes)
    uint8_t *d_outY = encP010Frame->data[0];
    uint8_t *d_outUV = encP010Frame->data[1];
    int pitchOutY = encP010Frame->linesize[0];
    int pitchOutUV = encP010Frame->linesize[1];
    if (!d_outY || !d_outUV || pitchOutY <= 0 || pitchOutUV <= 0)
    {
        setError("processGpuP010ToP010: invalid encoder CUDA frame planes");
        return false;
    }

    // RTX processed data is always in ABGR10 format, convert to P010 with BT.2020 for HDR
    launch_abgr10_to_p010(m_devABGR10, (int)m_devABGR10Pitch,
                          d_outY, pitchOutY,
                          d_outUV, pitchOutUV,
                          (int)m_dstW, (int)m_dstH,
                          /*bt2020=*/true,  // Always use BT.2020 for HDR content
                          m_stream);

    // Ensure all GPU operations complete before returning
    cudaStreamSynchronize(m_stream);
    return true;
}

bool RTXProcessor::processGpuP010ToNV12(const uint8_t *d_y, int pitchY,
                                        const uint8_t *d_uv, int pitchUV,
                                        AVFrame *encNV12Frame)
{
    if (!m_initialized || !d_y || !d_uv || !encNV12Frame)
    {
        setError("processGpuP010ToNV12: invalid state or null args");
        return false;
    }

    CUDADRV_CHECK(cuCtxSetCurrent(m_ctx));

    uint8_t *d_outY = encNV12Frame->data[0];
    uint8_t *d_outUV = encNV12Frame->data[1];
    int pitchOutY = encNV12Frame->linesize[0];
    int pitchOutUV = encNV12Frame->linesize[1];

    if (!d_outY || !d_outUV || pitchOutY <= 0 || pitchOutUV <= 0)
    {
        setError("processGpuP010ToNV12: invalid encoder CUDA frame planes");
        return false;
    }

    // If both VSR and THDR are disabled, bypass RTX and convert P010→NV12 directly
    if (!m_cfg.enableVSR && !m_cfg.enableTHDR)
    {
        // Simple downsample: P010 (10-bit) -> NV12 (8-bit) by taking upper 8 bits
        // This is appropriate when decoder outputs P010 for format negotiation reasons
        // but content is actually 8-bit SDR (lower 2 bits are zero)
        launch_p010_to_nv12(d_y, pitchY, d_uv, pitchUV,
                            d_outY, pitchOutY, d_outUV, pitchOutUV,
                            (int)m_srcW, (int)m_srcH,
                            m_stream);
        cudaStreamSynchronize(m_stream);
        return true;
    }

    // VSR and/or THDR enabled: need to go through RTX processing
    // Step 1: P010 (10-bit SDR padded) -> NV12 (8-bit) to extract actual 8-bit data
    // We need temporary NV12 buffers on device at source resolution
    uint8_t *d_tempY = nullptr;
    uint8_t *d_tempUV = nullptr;
    size_t tempPitchY = 0, tempPitchUV = 0;

    // Allocate temporary NV12 buffers (source resolution)
    cudaError_t err1 = cudaMallocPitch(&d_tempY, &tempPitchY, m_srcW, m_srcH);
    cudaError_t err2 = cudaMallocPitch(&d_tempUV, &tempPitchUV, m_srcW, m_srcH / 2);
    if (err1 != cudaSuccess || err2 != cudaSuccess || !d_tempY || !d_tempUV)
    {
        if (d_tempY) cudaFree(d_tempY);
        if (d_tempUV) cudaFree(d_tempUV);
        setError("processGpuP010ToNV12: failed to allocate temp NV12 buffers");
        return false;
    }

    // Downsample P010 → NV12 (extract 8-bit SDR data)
    launch_p010_to_nv12(d_y, pitchY, d_uv, pitchUV,
                        d_tempY, (int)tempPitchY, d_tempUV, (int)tempPitchUV,
                        (int)m_srcW, (int)m_srcH,
                        m_stream);

    // Step 2: NV12 (8-bit SDR) -> BGRA8 (device pitched)
    launch_nv12_to_bgra(d_tempY, (int)tempPitchY, d_tempUV, (int)tempPitchUV,
                        m_devBGRA, (int)m_devBGRAPitch,
                        (int)m_srcW, (int)m_srcH,
                        false, // bt2020 = false for SDR
                        m_stream);

    // Free temporary buffers
    cudaFree(d_tempY);
    cudaFree(d_tempUV);

    // Step 3: Copy BGRA8 (device pitched) -> m_srcArray (CUDA array) for RTX input
    CUDA_MEMCPY2D copyIn{};
    copyIn.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyIn.srcDevice = (CUdeviceptr)m_devBGRA;
    copyIn.srcPitch = (unsigned int)m_devBGRAPitch;
    copyIn.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyIn.dstArray = m_srcArray;
    copyIn.WidthInBytes = (unsigned int)(m_srcW * 4);
    copyIn.Height = m_srcH;
    CUDADRV_CHECK(cuMemcpy2D(&copyIn));

    // Step 4: RTX evaluate: m_srcTex -> m_dstSurf (BGRA8 for SDR output)
    API_RECT srcRect{0, 0, (int)m_srcW, (int)m_srcH};
    API_RECT dstRect{0, 0, (int)m_dstW, (int)m_dstH};
    API_VSR_Setting vsr{};
    vsr.QualityLevel = m_cfg.vsrQuality;
    // THDR should be disabled for SDR content
    bool ok = rtx_video_api_cuda_evaluate(m_srcTex, m_dstSurf, srcRect, dstRect,
                                          m_cfg.enableVSR ? &vsr : nullptr,
                                          nullptr); // No THDR for SDR
    if (!ok)
    {
        setError("RTX evaluate failed in processGpuP010ToNV12");
        return false;
    }

    // Step 5: Copy m_dstArray (BGRA8) -> device pitched staging
    CUDA_MEMCPY2D copyOut{};
    copyOut.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyOut.srcArray = m_dstArray;
    copyOut.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyOut.dstDevice = (CUdeviceptr)m_devABGR10;
    copyOut.dstPitch = (unsigned int)m_devABGR10Pitch;
    copyOut.WidthInBytes = (unsigned int)(m_dstW * 4);
    copyOut.Height = m_dstH;
    CUDADRV_CHECK(cuMemcpy2D(&copyOut));

    // Step 6: BGRA8 -> NV12 (output)
    launch_bgra8_to_nv12(m_devABGR10, (int)m_devABGR10Pitch,
                         d_outY, pitchOutY,
                         d_outUV, pitchOutUV,
                         (int)m_dstW, (int)m_dstH,
                         false, // bt2020 = false for SDR
                         m_stream);

    cudaStreamSynchronize(m_stream);
    return true;
}

bool RTXProcessor::processGpuP010SDRToP010(const uint8_t *d_y, int pitchY,
                                           const uint8_t *d_uv, int pitchUV,
                                           AVFrame *encP010Frame)
{
    if (!m_initialized || !d_y || !d_uv || !encP010Frame)
    {
        setError("processGpuP010SDRToP010: invalid state or null args");
        return false;
    }

    CUDADRV_CHECK(cuCtxSetCurrent(m_ctx));

    // Step 1: P010 → NV12 (extract 8-bit SDR)
    // Allocate temporary NV12 buffers at source resolution
    uint8_t *d_tempY = nullptr;
    uint8_t *d_tempUV = nullptr;
    size_t tempPitchY = 0, tempPitchUV = 0;

    cudaError_t err1 = cudaMallocPitch(&d_tempY, &tempPitchY, m_srcW, m_srcH);
    cudaError_t err2 = cudaMallocPitch(&d_tempUV, &tempPitchUV, m_srcW, m_srcH / 2);
    if (err1 != cudaSuccess || err2 != cudaSuccess || !d_tempY || !d_tempUV)
    {
        if (d_tempY) cudaFree(d_tempY);
        if (d_tempUV) cudaFree(d_tempUV);
        setError("processGpuP010SDRToP010: failed to allocate temp NV12 buffers");
        return false;
    }

    launch_p010_to_nv12(d_y, pitchY, d_uv, pitchUV,
                        d_tempY, (int)tempPitchY, d_tempUV, (int)tempPitchUV,
                        (int)m_srcW, (int)m_srcH,
                        m_stream);

    // Step 2: NV12 → BGRA8 → RTX (VSR+THDR) → ABGR10 → P010
    // Reuse the NV12ToP010 logic by calling it with the temp NV12 buffers
    bool result = processGpuNV12ToP010(d_tempY, (int)tempPitchY,
                                       d_tempUV, (int)tempPitchUV,
                                       encP010Frame,
                                       false); // bt2020=false for SDR input

    // Free temporary buffers
    cudaFree(d_tempY);
    cudaFree(d_tempUV);

    return result;
}

bool RTXProcessor::processGpuNV12ToNV12(const uint8_t *d_y, int pitchY,
                                        const uint8_t *d_uv, int pitchUV,
                                        AVFrame *encNV12Frame,
                                        bool bt2020)
{
    if (!m_initialized || !d_y || !d_uv || !encNV12Frame)
    {
        setError("processGpuNV12ToNV12: invalid state or null args");
        return false;
    }

    CUDADRV_CHECK(cuCtxSetCurrent(m_ctx));

    // NV12 -> BGRA8 (device)
    launch_nv12_to_bgra(d_y, pitchY, d_uv, pitchUV,
                        m_devBGRA, (int)m_devBGRAPitch,
                        (int)m_srcW, (int)m_srcH,
                        bt2020,
                        m_stream);

    // If both VSR and THDR are disabled, bypass RTX evaluate and convert directly to NV12
    if (!m_cfg.enableVSR && !m_cfg.enableTHDR)
    {
        uint8_t *d_outY = encNV12Frame->data[0];
        uint8_t *d_outUV = encNV12Frame->data[1];
        int pitchOutY = encNV12Frame->linesize[0];
        int pitchOutUV = encNV12Frame->linesize[1];
        if (!d_outY || !d_outUV || pitchOutY <= 0 || pitchOutUV <= 0)
        {
            setError("processGpuNV12ToNV12: invalid encoder CUDA frame planes");
            return false;
        }
        // Direct BGRA8 -> NV12 using input colorspace
        launch_bgra8_to_nv12(m_devBGRA, (int)m_devBGRAPitch,
                             d_outY, pitchOutY,
                             d_outUV, pitchOutUV,
                             (int)m_srcW, (int)m_srcH,
                             /*bt2020=*/bt2020,
                             m_stream);
        cudaStreamSynchronize(m_stream);
        return true;
    }

    // Copy BGRA8 -> m_srcArray for RTX input
    CUDA_MEMCPY2D copyIn{};
    copyIn.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyIn.srcDevice = (CUdeviceptr)m_devBGRA;
    copyIn.srcPitch = (unsigned int)m_devBGRAPitch;
    copyIn.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyIn.dstArray = m_srcArray;
    copyIn.WidthInBytes = (unsigned int)(m_srcW * 4);
    copyIn.Height = m_srcH;
    CUDADRV_CHECK(cuMemcpy2D(&copyIn));

    // RTX evaluate (BGRA8 in, BGRA8 or ABGR10 out depending on THDR)
    API_RECT srcRect{0, 0, (int)m_srcW, (int)m_srcH};
    API_RECT dstRect{0, 0, (int)m_dstW, (int)m_dstH};
    API_VSR_Setting vsr{}; vsr.QualityLevel = m_cfg.vsrQuality;
    API_THDR_Setting thdr{}; thdr.Contrast = m_cfg.thdrContrast; thdr.Saturation = m_cfg.thdrSaturation; thdr.MiddleGray = m_cfg.thdrMiddleGray; thdr.MaxLuminance = m_cfg.thdrMaxLuminance;
    bool ok = rtx_video_api_cuda_evaluate(m_srcTex, m_dstSurf, srcRect, dstRect,
                                          m_cfg.enableVSR ? &vsr : nullptr,
                                          m_cfg.enableTHDR ? &thdr : nullptr);
    if (!ok)
    {
        setError("RTX evaluate failed (NV12 path)");
        return false;
    }

    // Copy m_dstArray -> device pitched staging (BGRA8 if SDR)
    CUDA_MEMCPY2D copyOut{};
    copyOut.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyOut.srcArray = m_dstArray;
    copyOut.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyOut.dstDevice = (CUdeviceptr)m_devABGR10; // reused buffer (32bpp)
    copyOut.dstPitch = (unsigned int)m_devABGR10Pitch;
    copyOut.WidthInBytes = (unsigned int)(m_dstW * 4);
    copyOut.Height = m_dstH;
    CUDADRV_CHECK(cuMemcpy2D(&copyOut));

    // Convert to NV12 into FFmpeg CUDA frame planes
    uint8_t *d_outY = encNV12Frame->data[0];
    uint8_t *d_outUV = encNV12Frame->data[1];
    int pitchOutY = encNV12Frame->linesize[0];
    int pitchOutUV = encNV12Frame->linesize[1];
    if (!d_outY || !d_outUV || pitchOutY <= 0 || pitchOutUV <= 0)
    {
        setError("processGpuNV12ToNV12: invalid encoder CUDA frame planes");
        return false;
    }
    // For SDR output use BT.709; if input was BT.2020 but THDR disabled, we still use the input's matrix for colorimetry conversion to SDR YUV.
    launch_bgra8_to_nv12(m_devABGR10, (int)m_devABGR10Pitch,
                         d_outY, pitchOutY,
                         d_outUV, pitchOutUV,
                         (int)m_dstW, (int)m_dstH,
                         /*bt2020=*/bt2020,
                         m_stream);

    cudaStreamSynchronize(m_stream);
    return true;
}

bool RTXProcessor::initializeWithContext(CUcontext externalCtx, const RTXProcessConfig &cfg, uint32_t srcW, uint32_t srcH)
{
    if (m_initialized)
        return true;
    m_cfg = cfg;
    m_srcW = srcW;
    m_srcH = srcH;
    m_dstW = srcW * (cfg.enableVSR ? cfg.scaleFactor : 1);
    m_dstH = srcH * (cfg.enableVSR ? cfg.scaleFactor : 1);

    m_ctx = externalCtx;
    m_externalCtx = true;
    CUDADRV_CHECK(cuCtxSetCurrent(m_ctx));
    if (!createRTX(cfg.enableTHDR, cfg.enableVSR))
    {
        setError(std::string("RTX API create failed (THDR=") + (cfg.enableTHDR ? "1" : "0") + ", VSR=" + (cfg.enableVSR ? "1" : "0") + ")");
        return false;
    }
    if (!allocSurfaces(cfg.enableTHDR))
    {
        setError("allocSurfaces failed in initializeWithContext");
        return false;
    }

    // Create stream
    if (!m_stream)
        cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);

    // Allocate device staging buffers
    size_t pitch = 0;
    if (cudaMallocPitch(&m_devBGRA, &pitch, m_srcW * 4, m_srcH) != cudaSuccess)
    {
        setError("cudaMallocPitch m_devBGRA failed");
        return false;
    }
    m_devBGRAPitch = pitch;
    if (cudaMallocPitch(&m_devABGR10, &pitch, m_dstW * 4, m_dstH) != cudaSuccess)
    {
        setError("cudaMallocPitch m_devABGR10 failed");
        return false;
    }
    m_devABGR10Pitch = pitch;

    m_initialized = true;
    return true;
}

bool RTXProcessor::process(const uint8_t *inBGRA, size_t inPitchBytes,
                           const uint8_t *&outData, uint32_t &outWidth, uint32_t &outHeight, size_t &outPitchBytes)
{
    if (!m_initialized)
        return false;

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
    if (!ok)
        return false;

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

    // Synchronize to ensure copy is complete before returning host data
    if (m_stream)
        cudaStreamSynchronize(m_stream);

    outData = m_hostOut.data();
    outWidth = m_dstW;
    outHeight = m_dstH;
    outPitchBytes = m_outPitch;
    return true;
}

void RTXProcessor::shutdown()
{
    destroyRTX();
    freeSurfaces();
    deinitCuda();
    m_initialized = false;
}

bool RTXProcessor::initCuda(int gpuIndex)
{
    CUDADRV_CHECK(cuInit(0));
    int count = 0;
    CUDADRV_CHECK(cuDeviceGetCount(&count));
    if (gpuIndex < 0 || gpuIndex >= count)
        gpuIndex = 0;
    CUDADRV_CHECK(cuDeviceGet(&m_device, gpuIndex));
    CUDADRV_CHECK(cuCtxCreate(&m_ctx, 0, m_device));
    // stream optional
    return true;
}

void RTXProcessor::deinitCuda()
{
    if (m_stream)
    {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
    if (!m_externalCtx && m_ctx)
    {
        cuCtxDestroy(m_ctx);
        m_ctx = nullptr;
    }
}

bool RTXProcessor::createRTX(bool thdr, bool vsr)
{
    return rtx_video_api_cuda_create(m_ctx, m_stream, 0, thdr, vsr);
}

void RTXProcessor::destroyRTX()
{
    // Ensure all CUDA operations complete before RTX shutdown
    if (m_ctx)
    {
        cuCtxPushCurrent(m_ctx);
        cudaDeviceSynchronize();
        cuCtxPopCurrent(nullptr);
    }
    
    rtx_video_api_cuda_shutdown();
}

bool RTXProcessor::allocSurfaces(bool thdr)
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
    if (cres != CUDA_SUCCESS)
    {
        fprintf(stderr, "CUDA does not support the HDR format needed for TrueHDR. Update NVIDIA driver.\n");
        return false;
    }

    CUDA_RESOURCE_DESC dstRes{};
    dstRes.resType = CU_RESOURCE_TYPE_ARRAY;
    dstRes.res.array.hArray = m_dstArray;
    CUDADRV_CHECK(cuSurfObjectCreate(&m_dstSurf, &dstRes));

    return true;
}

void RTXProcessor::freeSurfaces()
{
    // Ensure all CUDA operations complete before freeing surfaces
    if (m_ctx)
    {
        cuCtxPushCurrent(m_ctx);
        cudaDeviceSynchronize();
    }
    
    if (m_srcTex)
    {
        cuTexObjectDestroy(m_srcTex);
        m_srcTex = 0;
    }
    if (m_dstSurf)
    {
        cuSurfObjectDestroy(m_dstSurf);
        m_dstSurf = 0;
    }
    if (m_srcSurf)
    {
        cuSurfObjectDestroy(m_srcSurf);
        m_srcSurf = 0;
    }
    if (m_srcArray)
    {
        cuArrayDestroy(m_srcArray);
        m_srcArray = nullptr;
    }
    if (m_dstArray)
    {
        cuArrayDestroy(m_dstArray);
        m_dstArray = nullptr;
    }
    if (m_devBGRA)
    {
        cudaFree(m_devBGRA);
        m_devBGRA = nullptr;
        m_devBGRAPitch = 0;
    }
    if (m_devABGR10)
    {
        cudaFree(m_devABGR10);
        m_devABGR10 = nullptr;
        m_devABGR10Pitch = 0;
    }
    
    if (m_ctx)
    {
        cuCtxPopCurrent(nullptr);
    }
}
