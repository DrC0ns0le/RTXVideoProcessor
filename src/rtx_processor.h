#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "rtx_video_api.h"

struct RtxProcessConfig {
    bool enableVSR = true;
    bool enableTHDR = true;
    int vsrQuality = 4;          // 1..4 (4=highest)
    int scaleFactor = 2;         // 2x
    // TrueHDR defaults per sample
    int thdrContrast = 100;
    int thdrSaturation = 100;
    int thdrMiddleGray = 50;
    int thdrMaxLuminance = 650;  // nits
};

struct FrameDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pitch = 0; // bytes per row
    // BGRA8 or A2R10G10B10 depending on config/output
};

// Thin wrapper around RTX Video SDK CUDA API following the cuda_sample in the SDK
class RtxProcessor {
public:
    RtxProcessor();
    ~RtxProcessor();

    // Initializes CUDA and RTX API. Output dimensions will be src*scaleFactor
    bool initialize(int gpuIndex, const RtxProcessConfig& cfg, uint32_t srcW, uint32_t srcH);

    // Process one frame from host BGRA8 input into host 10-bit A2R10G10B10 or BGRA8 depending on THDR.
    // Provides a pointer to the internal output buffer to avoid copies. Returns false on failure.
    bool process(const uint8_t* inBGRA, size_t inPitchBytes,
                 const uint8_t*& outData, uint32_t& outWidth, uint32_t& outHeight, size_t& outPitchBytes);

    void shutdown();

    // Returns a human-readable description of the last error (if any initialize/process failed)
    const std::string& lastError() const { return m_lastError; }

private:
    bool initCuda(int gpuIndex);
    void deinitCuda();

    bool createRtx(bool thdr, bool vsr);
    void destroyRtx();

    bool allocSurfaces(bool thdr);
    void freeSurfaces();

    void setError(const std::string& msg) { m_lastError = msg; }

private:
    RtxProcessConfig m_cfg{};

    CUdevice m_device = 0;
    CUcontext m_ctx = nullptr;
    cudaStream_t m_stream = nullptr;

    // Surfaces/Textures
    CUarray m_srcArray = nullptr;
    CUarray m_dstArray = nullptr;
    CUtexObject m_srcTex = 0;
    CUsurfObject m_dstSurf = 0;

    // Host staging
    std::vector<uint8_t> m_hostOut; // ABGR10 output when THDR enabled, BGRA8 otherwise

    // Sizes
    uint32_t m_srcW = 0, m_srcH = 0;
    uint32_t m_dstW = 0, m_dstH = 0;
    size_t   m_inPitch = 0;  // bytes
    size_t   m_outPitch = 0; // bytes

    bool m_initialized = false;

    std::string m_lastError;
};
