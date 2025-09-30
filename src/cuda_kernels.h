#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

// Launch NV12 -> BGRA8 conversion on device.
// d_y, d_uv are device pointers with pitches pitchY and pitchUV.
// outBGRA is device pointer with pitch outPitch (bytes per row), size w x h.
// If bt2020 is true, use BT.2020 YUV->RGB matrix; otherwise BT.709.
void launch_nv12_to_bgra(const uint8_t *d_y, int pitchY,
                         const uint8_t *d_uv, int pitchUV,
                         uint8_t *outBGRA, int outPitch,
                         int w, int h,
                         bool bt2020,
                         cudaStream_t stream);

// Launch ABGR10 (packed 10:10:10:2, 32bpp) -> P010 conversion on device.
// inABGR: device pointer with pitch inPitch, size w x h.
// outY/outUV: device pointers for P010 planes with pitches pitchY/pitchUV.
// If bt2020 is true, use BT.2020 RGB->YUV matrix; otherwise BT.709. Range mapping: full RGB -> limited YUV.
void launch_abgr10_to_p010(const uint8_t *inABGR, int inPitch,
                           uint8_t *outY, int pitchY,
                           uint8_t *outUV, int pitchUV,
                           int w, int h,
                           bool bt2020,
                           cudaStream_t stream);

// Launch BGRA8 (8-bit per channel) -> P010 conversion on device.
// inBGRA: device pointer with pitch inPitch, size w x h.
// outY/outUV: device pointers for P010 planes with pitches pitchY/pitchUV.
// If bt2020 is true, use BT.2020 RGB->YUV matrix; otherwise BT.709. Range mapping: full RGB -> limited YUV.
void launch_bgra8_to_p010(const uint8_t *inBGRA, int inPitch,
                          uint8_t *outY, int pitchY,
                          uint8_t *outUV, int pitchUV,
                          int w, int h,
                          bool bt2020,
                          cudaStream_t stream);

// Launch BGRA8 (8-bit per channel) -> NV12 (8-bit 4:2:0) conversion on device.
// inBGRA: device pointer with pitch inPitch, size w x h.
// outY/outUV: device pointers for NV12 planes with pitches pitchY/pitchUV.
// If bt2020 is true, use BT.2020 RGB->YUV matrix; otherwise BT.709. Range mapping: full RGB -> limited YUV.
void launch_bgra8_to_nv12(const uint8_t *inBGRA, int inPitch,
                          uint8_t *outY, int pitchY,
                          uint8_t *outUV, int pitchUV,
                          int w, int h,
                          bool bt2020,
                          cudaStream_t stream);

// Launch P010 (10-bit YUV420) -> NV12 (8-bit YUV420) conversion on device.
// Downsamples 10-bit to 8-bit by taking upper 8 bits.
// d_yIn, d_uvIn: device pointers to P010 input planes with pitches pitchYIn and pitchUVIn.
// d_yOut, d_uvOut: device pointers to NV12 output planes with pitches pitchYOut and pitchUVOut.
void launch_p010_to_nv12(const uint8_t *d_yIn, int pitchYIn,
                         const uint8_t *d_uvIn, int pitchUVIn,
                         uint8_t *d_yOut, int pitchYOut,
                         uint8_t *d_uvOut, int pitchUVOut,
                         int w, int h,
                         cudaStream_t stream);

// Launch P010 (10-bit YUV420) -> X2BGR10LE (10-bit RGB) conversion on device.
// d_y, d_uv are device pointers to P010 planes with pitches pitchY and pitchUV.
// outX2BGR10 is device pointer with pitch outPitch (bytes per row), size w x h.
// If bt2020 is true, use BT.2020 YUV->RGB matrix; otherwise BT.709.
void launch_p010_to_x2bgr10(const uint8_t *d_y, int pitchY,
                            const uint8_t *d_uv, int pitchUV,
                            uint8_t *outX2BGR10, int outPitch,
                            int w, int h,
                            bool bt2020,
                            cudaStream_t stream);
