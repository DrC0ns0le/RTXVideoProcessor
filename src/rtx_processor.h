#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "rtx_video_api.h"

extern "C"
{
    struct AVFrame; // forward declaration to avoid pulling FFmpeg headers into this header
}

struct RTXProcessConfig
{
    bool enableVSR = true;
    bool enableTHDR = true;
    int vsrQuality = 4;  // 1..4 (4=highest)
    int scaleFactor = 2; // 2x
    // TrueHDR defaults per sample
    int thdrContrast = 100;
    int thdrSaturation = 100;
    int thdrMiddleGray = 50;
    int thdrMaxLuminance = 1000; // nits
};

struct FrameDesc
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pitch = 0; // bytes per row
    // BGRA8 or A2R10G10B10 depending on config/output
};

// Thin wrapper around RTX Video SDK CUDA API following the cuda_sample in the SDK
class RTXProcessor
{
public:
    RTXProcessor();
    ~RTXProcessor();

    // Initializes CUDA and RTX API. Output dimensions will be src*scaleFactor
    bool initialize(int gpuIndex, const RTXProcessConfig &cfg, uint32_t srcW, uint32_t srcH);

    // Initializes using an existing CUDA context (e.g., FFmpeg's AVHWDeviceContext CUDA context).
    // This ensures interop without cross-context copies.
    bool initializeWithContext(CUcontext externalCtx, const RTXProcessConfig &cfg, uint32_t srcW, uint32_t srcH);

    // Process one frame from host BGRA8 input into host 10-bit A2R10G10B10 or BGRA8 depending on THDR.
    // Provides a pointer to the internal output buffer to avoid copies. Returns false on failure.
    bool process(const uint8_t *inBGRA, size_t inPitchBytes,
                 const uint8_t *&outData, uint32_t &outWidth, uint32_t &outHeight, size_t &outPitchBytes);

    // Fully GPU path: consume an NV12 frame resident on the device (NVDEC output) and fill a P010 CUDA frame
    // (allocated by FFmpeg via hw_frames_ctx) for NVENC, avoiding any host copies.
    // d_y: device pointer to luma plane, pitchY: bytes per row of Y
    // d_uv: device pointer to interleaved UV plane, pitchUV: bytes per row of UV
    // encP010Frame: AVFrame with format AV_PIX_FMT_CUDA and sw_format P010; planes will be written on device.
    // bt2020: if true, use BT.2020 coefficients; otherwise use BT.709.
    bool processGpuNV12ToP010(const uint8_t *d_y, int pitchY,
                              const uint8_t *d_uv, int pitchUV,
                              AVFrame *encP010Frame,
                              bool bt2020);

    // Fully GPU path: consume an NV12 frame resident on the device (NVDEC output) and fill a BGRA8 CUDA array
    // (m_srcArray) for RTX processing, avoiding any host copies.
    // d_y: device pointer to luma plane, pitchY: bytes per row of Y
    // d_uv: device pointer to interleaved UV plane, pitchUV: bytes per row of UV
    // bt2020: if true, use BT.2020 coefficients; otherwise use BT.709.
    bool processGpuNV12ToBGRA(const uint8_t *d_y, int pitchY,
                              const uint8_t *d_uv, int pitchUV,
                              bool bt2020);

    void shutdown();

    // Returns a human-readable description of the last error (if any initialize/process failed)
    const std::string &lastError() const { return m_lastError; }

private:
    bool initCuda(int gpuIndex);
    void deinitCuda();

    bool createRTX(bool thdr, bool vsr);
    void destroyRTX();

    bool allocSurfaces(bool thdr);
    void freeSurfaces();

    void setError(const std::string &msg) { m_lastError = msg; }

    // Ensure we have a surface object bound to m_srcArray for CUDA kernels to write into it.
    bool ensureSrcSurface();

private:
    RTXProcessConfig m_cfg{};

    CUdevice m_device = 0;
    CUcontext m_ctx = nullptr;
    cudaStream_t m_stream = nullptr;
    bool m_externalCtx = false; // whether m_ctx is owned externally

    // Surfaces/Textures
    CUarray m_srcArray = nullptr;
    CUarray m_dstArray = nullptr;
    CUtexObject m_srcTex = 0;
    CUsurfObject m_dstSurf = 0;
    CUsurfObject m_srcSurf = 0; // for writing NV12->BGRA into m_srcArray via kernels

    // Device staging buffers to avoid host copies
    uint8_t *m_devBGRA = nullptr; // srcW x srcH x 4, pitched
    size_t m_devBGRAPitch = 0;
    uint8_t *m_devABGR10 = nullptr; // dstW x dstH x 4, pitched
    size_t m_devABGR10Pitch = 0;

    // Host staging
    std::vector<uint8_t> m_hostOut; // ABGR10 output when THDR enabled, BGRA8 otherwise

    // Sizes
    uint32_t m_srcW = 0, m_srcH = 0;
    uint32_t m_dstW = 0, m_dstH = 0;
    size_t m_inPitch = 0;  // bytes
    size_t m_outPitch = 0; // bytes

    bool m_initialized = false;

    std::string m_lastError;
};
