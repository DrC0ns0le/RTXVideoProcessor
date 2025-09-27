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
