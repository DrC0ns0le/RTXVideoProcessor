#include "cuda_kernels.h"
#include <math.h>

// Utility: clamp to [a,b]
static __device__ inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static __device__ inline int clampi(int x, int a, int b) { return x < a ? a : (x > b ? b : x); }

// Convert ABGR10 (X2BGR10LE) 32bpp to P010 (Y:16bpp, UV:16bpp interleaved), BT.709 or BT.2020 limited range
// Assumptions:
// - inABGR points to pitched rows of 32-bit pixels, with bpp=4. Layout LE: bits 0..9 R, 10..19 G, 20..29 B, 30..31 A/unused.
// - outY pitch in bytes; outUV pitch in bytes. outUV height = h/2 and width = w (16-bit pairs per pixel).

__global__ void k_abgr10_to_p010(const uint8_t *__restrict__ inABGR, int inPitch,
                                 uint8_t *__restrict__ outY, int pitchY,
                                 uint8_t *__restrict__ outUV, int pitchUV,
                                 int w, int h, bool bt2020)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // process 2x2 block's left pixel x even
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2; // top row of 2x2 block
    if (x + 1 >= w || y + 1 >= h)
        return;

    // Accumulate Cb/Cr over 2x2
    float sumCb = 0.f, sumCr = 0.f;

    // Iterate 2x2
    float Ys[2][2];
    for (int dy = 0; dy < 2; ++dy)
    {
        const uint32_t *row = (const uint32_t *)(inABGR + (y + dy) * inPitch);
        for (int dx = 0; dx < 2; ++dx)
        {
            uint32_t p = row[x + dx];
            // X2BGR10LE (FFmpeg AV_PIX_FMT_X2BGR10LE): bits 0..9 B, 10..19 G, 20..29 R, 30..31 unused
            int r10 = (int)((p >> 0) & 0x3FF);  // Red in bits 0-9
            int g10 = (int)((p >> 10) & 0x3FF); // Green in bits 10-19
            int b10 = (int)((p >> 20) & 0x3FF); // Blue in bits 20-29
            float R = r10 / 1023.0f;
            float G = g10 / 1023.0f;
            float B = b10 / 1023.0f;

            // Compute Y' and chroma per BT.709 or BT.2020
            float Kr = bt2020 ? 0.2627f : 0.2126f;
            float Kb = bt2020 ? 0.0593f : 0.0722f;
            float Kg = 1.0f - Kr - Kb;
            float Yp = Kr * R + Kg * G + Kb * B;
            Ys[dy][dx] = Yp;
            float Cb = 0.5f * (B - Yp) / (1.0f - Kb);
            float Cr = 0.5f * (R - Yp) / (1.0f - Kr);
            sumCb += Cb;
            sumCr += Cr;
        }
    }

    float avgCb = sumCb * 0.25f;
    float avgCr = sumCr * 0.25f;

    // Map to limited range 10-bit
    auto mapY10 = [](float Y)
    {
        float v = 64.0f + Y * 876.0f; // [64..940]
        int vi = (int)lrintf(v);
        return (unsigned short)clampi(vi, 64, 940);
    };
    auto mapC10 = [](float C)
    {
        float v = 512.0f + C * 896.0f; // [64..960]
        int vi = (int)lrintf(v);
        return (unsigned short)clampi(vi, 64, 960);
    };

    // Write Y plane (two rows, two cols)
    unsigned short *yRow0 = (unsigned short *)(outY + y * pitchY);
    unsigned short *yRow1 = (unsigned short *)(outY + (y + 1) * pitchY);
    unsigned short y00 = (unsigned short)(mapY10(Ys[0][0]) << 6);
    unsigned short y01 = (unsigned short)(mapY10(Ys[0][1]) << 6);
    unsigned short y10 = (unsigned short)(mapY10(Ys[1][0]) << 6);
    unsigned short y11 = (unsigned short)(mapY10(Ys[1][1]) << 6);
    yRow0[x + 0] = y00;
    yRow0[x + 1] = y01;
    yRow1[x + 0] = y10;
    yRow1[x + 1] = y11;

    // Write interleaved UV (at (y/2, x))
    int uvy = y / 2; // chroma row
    int ux = x / 2;  // chroma column (subsampled)
    unsigned short *uvRow = (unsigned short *)(outUV + uvy * pitchUV);
    unsigned short U = (unsigned short)(mapC10(avgCb) << 6);
    unsigned short V = (unsigned short)(mapC10(avgCr) << 6);
    // Interleaved 16-bit U,V per chroma sample
    uvRow[ux * 2 + 0] = U;
    uvRow[ux * 2 + 1] = V;
}

void launch_abgr10_to_p010(const uint8_t *inABGR, int inPitch,
                           uint8_t *outY, int pitchY,
                           uint8_t *outUV, int pitchUV,
                           int w, int h,
                           bool bt2020,
                           cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((w + block.x * 2 - 1) / (block.x * 2), (h + block.y * 2 - 1) / (block.y * 2));
    k_abgr10_to_p010<<<grid, block, 0, stream>>>(inABGR, inPitch, outY, pitchY, outUV, pitchUV, w, h, bt2020);
}

// Convert BGRA8 (8-bit per channel) -> P010 limited range, using BT.709 or BT.2020 coefficients.
__global__ void k_bgra8_to_p010(const uint8_t *__restrict__ inBGRA, int inPitch,
                                uint8_t *__restrict__ outY, int pitchY,
                                uint8_t *__restrict__ outUV, int pitchUV,
                                int w, int h, bool bt2020)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // process 2x2 block
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (x + 1 >= w || y + 1 >= h)
        return;

    float sumCb = 0.f, sumCr = 0.f;
    float Ys[2][2];
    for (int dy = 0; dy < 2; ++dy)
    {
        const uint8_t *row = inBGRA + (y + dy) * inPitch;
        for (int dx = 0; dx < 2; ++dx)
        {
            const uint8_t *p = row + (x + dx) * 4;
            // Our pipeline stores R,G,B,A order in the 4 bytes per pixel
            float R = p[0] / 255.0f;
            float G = p[1] / 255.0f;
            float B = p[2] / 255.0f;

            float Kr = bt2020 ? 0.2627f : 0.2126f;
            float Kb = bt2020 ? 0.0593f : 0.0722f;
            float Kg = 1.0f - Kr - Kb;
            float Yp = Kr * R + Kg * G + Kb * B;
            Ys[dy][dx] = Yp;
            float Cb = 0.5f * (B - Yp) / (1.0f - Kb);
            float Cr = 0.5f * (R - Yp) / (1.0f - Kr);
            sumCb += Cb;
            sumCr += Cr;
        }
    }

    float avgCb = sumCb * 0.25f;
    float avgCr = sumCr * 0.25f;

    auto mapY10 = [](float Y)
    {
        float v = 64.0f + Y * 876.0f; // [64..940]
        int vi = (int)lrintf(v);
        return (unsigned short)clampi(vi, 64, 940);
    };
    auto mapC10 = [](float C)
    {
        float v = 512.0f + C * 896.0f; // [64..960]
        int vi = (int)lrintf(v);
        return (unsigned short)clampi(vi, 64, 960);
    };

    unsigned short *yRow0 = (unsigned short *)(outY + y * pitchY);
    unsigned short *yRow1 = (unsigned short *)(outY + (y + 1) * pitchY);
    unsigned short y00 = (unsigned short)(mapY10(Ys[0][0]) << 6);
    unsigned short y01 = (unsigned short)(mapY10(Ys[0][1]) << 6);
    unsigned short y10 = (unsigned short)(mapY10(Ys[1][0]) << 6);
    unsigned short y11 = (unsigned short)(mapY10(Ys[1][1]) << 6);
    yRow0[x + 0] = y00;
    yRow0[x + 1] = y01;
    yRow1[x + 0] = y10;
    yRow1[x + 1] = y11;

    int uvy = y / 2;
    int ux = x / 2;
    unsigned short *uvRow = (unsigned short *)(outUV + uvy * pitchUV);
    unsigned short U = (unsigned short)(mapC10(avgCb) << 6);
    unsigned short V = (unsigned short)(mapC10(avgCr) << 6);
    uvRow[ux * 2 + 0] = U;
    uvRow[ux * 2 + 1] = V;
}

void launch_bgra8_to_p010(const uint8_t *inBGRA, int inPitch,
                          uint8_t *outY, int pitchY,
                          uint8_t *outUV, int pitchUV,
                          int w, int h,
                          bool bt2020,
                          cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((w + block.x * 2 - 1) / (block.x * 2), (h + block.y * 2 - 1) / (block.y * 2));
    k_bgra8_to_p010<<<grid, block, 0, stream>>>(inBGRA, inPitch, outY, pitchY, outUV, pitchUV, w, h, bt2020);
}

// Convert BGRA8 -> NV12 (8-bit limited range)
__global__ void k_bgra8_to_nv12(const uint8_t *__restrict__ inBGRA, int inPitch,
                                uint8_t *__restrict__ outY, int pitchY,
                                uint8_t *__restrict__ outUV, int pitchUV,
                                int w, int h, bool bt2020)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // process 2x2 block
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (x + 1 >= w || y + 1 >= h)
        return;

    float sumCb = 0.f, sumCr = 0.f;
    float Ys[2][2];
    for (int dy = 0; dy < 2; ++dy)
    {
        const uint8_t *row = inBGRA + (y + dy) * inPitch;
        for (int dx = 0; dx < 2; ++dx)
        {
            const uint8_t *p = row + (x + dx) * 4;
            float R = p[0] / 255.0f;
            float G = p[1] / 255.0f;
            float B = p[2] / 255.0f;

            float Kr = bt2020 ? 0.2627f : 0.2126f;
            float Kb = bt2020 ? 0.0593f : 0.0722f;
            float Kg = 1.0f - Kr - Kb;
            float Yp = Kr * R + Kg * G + Kb * B;
            Ys[dy][dx] = Yp;
            float Cb = 0.5f * (B - Yp) / (1.0f - Kb);
            float Cr = 0.5f * (R - Yp) / (1.0f - Kr);
            sumCb += Cb;
            sumCr += Cr;
        }
    }

    float avgCb = sumCb * 0.25f;
    float avgCr = sumCr * 0.25f;

    auto mapY8 = [](float Y)
    {
        float v = 16.0f + Y * 219.0f; // [16..235]
        int vi = (int)lrintf(v);
        return (uint8_t)clampi(vi, 16, 235);
    };
    auto mapC8 = [](float C)
    {
        float v = 128.0f + C * 224.0f; // [16..240]
        int vi = (int)lrintf(v);
        return (uint8_t)clampi(vi, 16, 240);
    };

    uint8_t *yRow0 = outY + y * pitchY;
    uint8_t *yRow1 = outY + (y + 1) * pitchY;
    yRow0[x + 0] = mapY8(Ys[0][0]);
    yRow0[x + 1] = mapY8(Ys[0][1]);
    yRow1[x + 0] = mapY8(Ys[1][0]);
    yRow1[x + 1] = mapY8(Ys[1][1]);

    int uvy = y / 2;
    int ux = x / 2;
    uint8_t *uvRow = outUV + uvy * pitchUV;
    uvRow[ux * 2 + 0] = mapC8(avgCb);
    uvRow[ux * 2 + 1] = mapC8(avgCr);
}

void launch_bgra8_to_nv12(const uint8_t *inBGRA, int inPitch,
                          uint8_t *outY, int pitchY,
                          uint8_t *outUV, int pitchUV,
                          int w, int h,
                          bool bt2020,
                          cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((w + block.x * 2 - 1) / (block.x * 2), (h + block.y * 2 - 1) / (block.y * 2));
    k_bgra8_to_nv12<<<grid, block, 0, stream>>>(inBGRA, inPitch, outY, pitchY, outUV, pitchUV, w, h, bt2020);
}

static __device__ inline void yuv_to_rgb(float Yp, float Uc, float Vc, bool bt2020, float &R, float &G, float &B)
{
    if (bt2020)
    {
        // BT.2020
        R = Yp + 1.4746f * Vc;
        G = Yp - 0.16455f * Uc - 0.57135f * Vc;
        B = Yp + 1.8814f * Uc;
    }
    else
    {
        // BT.709
        R = Yp + 1.5748f * Vc;
        G = Yp - 0.1873f * Uc - 0.4681f * Vc;
        B = Yp + 1.8556f * Uc;
    }
}

__global__ void k_nv12_to_rgba(const uint8_t *__restrict__ d_y, int pitchY,
                               const uint8_t *__restrict__ d_uv, int pitchUV,
                               uint8_t *__restrict__ outBGRA, int outPitch,
                               int w, int h, bool bt2020)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    // NV12: Y plane full res, UV interleaved at half res
    int uvx = (x / 2) * 2; // U,V pair starting index
    int uvy = y / 2;

    float Y = (float)d_y[y * pitchY + x];
    float U = (float)d_uv[uvy * pitchUV + uvx + 0];
    float V = (float)d_uv[uvy * pitchUV + uvx + 1];

    // Limited range mapping
    float Yp = (Y - 16.0f) / 219.0f;
    float Uc = (U - 128.0f) / 224.0f;
    float Vc = (V - 128.0f) / 224.0f;

    float R, G, B;
    yuv_to_rgb(Yp, Uc, Vc, bt2020, R, G, B);
    R = clampf(R, 0.0f, 1.0f);
    G = clampf(G, 0.0f, 1.0f);
    B = clampf(B, 0.0f, 1.0f);

    uint8_t r8 = (uint8_t)(R * 255.0f + 0.5f);
    uint8_t g8 = (uint8_t)(G * 255.0f + 0.5f);
    uint8_t b8 = (uint8_t)(B * 255.0f + 0.5f);

    uint8_t *dst = outBGRA + y * outPitch + x * 4;
    dst[0] = r8;  // R
    dst[1] = g8;  // G
    dst[2] = b8;  // B
    dst[3] = 255; // A
}

__global__ void k_p010_to_x2bgr10(const uint8_t *__restrict__ d_y, int pitchY,
                                  const uint8_t *__restrict__ d_uv, int pitchUV,
                                  uint8_t *__restrict__ outX2BGR10, int outPitch,
                                  int w, int h, bool bt2020)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    // P010: Y plane full res (16-bit), UV interleaved at half res (16-bit pairs)
    // 10-bit data is in upper 10 bits of each 16-bit value
    int uvy = y / 2;

    const uint16_t *y_plane = (const uint16_t *)d_y;
    const uint16_t *uv_plane = (const uint16_t *)d_uv;

    uint16_t Y16 = y_plane[y * (pitchY / 2) + x];
    uint16_t U16 = uv_plane[uvy * (pitchUV / 2) + (x / 2) * 2 + 0];
    uint16_t V16 = uv_plane[uvy * (pitchUV / 2) + (x / 2) * 2 + 1];

    // Extract 10-bit values from upper 10 bits of 16-bit data
    float Y = (float)(Y16 >> 6); // P010: 10 bits in upper bits, so shift right 6
    float U = (float)(U16 >> 6);
    float V = (float)(V16 >> 6);

    // Limited range mapping for 10-bit (0-1023 range)
    float Yp = (Y - 64.0f) / 876.0f;   // 10-bit: Y range 64-940 (876 levels)
    float Uc = (U - 512.0f) / 896.0f;  // 10-bit: UV range 64-960 (896 levels), centered at 512
    float Vc = (V - 512.0f) / 896.0f;

    float R, G, B;
    yuv_to_rgb(Yp, Uc, Vc, bt2020, R, G, B);
    R = clampf(R, 0.0f, 1.0f);
    G = clampf(G, 0.0f, 1.0f);
    B = clampf(B, 0.0f, 1.0f);

    // Convert to 10-bit values (0-1023 range)
    uint32_t r10 = (uint32_t)(R * 1023.0f + 0.5f);
    uint32_t g10 = (uint32_t)(G * 1023.0f + 0.5f);
    uint32_t b10 = (uint32_t)(B * 1023.0f + 0.5f);

    // Pack into X2BGR10LE format: bits 0-9 R, 10-19 G, 20-29 B, 30-31 unused
    uint32_t packed = (r10 & 0x3FF) | ((g10 & 0x3FF) << 10) | ((b10 & 0x3FF) << 20);

    uint32_t *dst = (uint32_t *)(outX2BGR10 + y * outPitch + x * 4);
    *dst = packed;
}

// Kernel: P010 (10-bit YUV420) -> NV12 (8-bit YUV420)
// Simple downsampling: take upper 8 bits of 10-bit data
__global__ void k_p010_to_nv12(const uint8_t *__restrict__ d_yIn, int pitchYIn,
                                const uint8_t *__restrict__ d_uvIn, int pitchUVIn,
                                uint8_t *__restrict__ d_yOut, int pitchYOut,
                                uint8_t *__restrict__ d_uvOut, int pitchUVOut,
                                int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Process Y plane
    if (x < w && y < h) {
        const uint16_t *yIn = (const uint16_t *)d_yIn;
        uint16_t y16 = yIn[y * (pitchYIn / 2) + x];
        // P010: 10 bits in upper bits, shift right 2 to get upper 8 bits
        uint8_t y8 = (uint8_t)(y16 >> 8);
        d_yOut[y * pitchYOut + x] = y8;
    }

    // Process UV plane (half resolution)
    int uvY = y / 2;
    int uvX = (x / 2) * 2; // Align to UV pair
    if (uvX < w && uvY < h / 2 && (x % 2 == 0)) {
        const uint16_t *uvIn = (const uint16_t *)d_uvIn;
        uint16_t u16 = uvIn[uvY * (pitchUVIn / 2) + uvX + 0];
        uint16_t v16 = uvIn[uvY * (pitchUVIn / 2) + uvX + 1];
        // Downsample to 8-bit
        uint8_t u8 = (uint8_t)(u16 >> 8);
        uint8_t v8 = (uint8_t)(v16 >> 8);
        d_uvOut[uvY * pitchUVOut + uvX + 0] = u8;
        d_uvOut[uvY * pitchUVOut + uvX + 1] = v8;
    }
}

void launch_nv12_to_bgra(const uint8_t *d_y, int pitchY,
                         const uint8_t *d_uv, int pitchUV,
                         uint8_t *outBGRA, int outPitch,
                         int w, int h,
                         bool bt2020,
                         cudaStream_t stream)
{
    dim3 block(32, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    k_nv12_to_rgba<<<grid, block, 0, stream>>>(d_y, pitchY, d_uv, pitchUV, outBGRA, outPitch, w, h, bt2020);
}

void launch_p010_to_nv12(const uint8_t *d_yIn, int pitchYIn,
                         const uint8_t *d_uvIn, int pitchUVIn,
                         uint8_t *d_yOut, int pitchYOut,
                         uint8_t *d_uvOut, int pitchUVOut,
                         int w, int h,
                         cudaStream_t stream)
{
    dim3 block(32, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    k_p010_to_nv12<<<grid, block, 0, stream>>>(d_yIn, pitchYIn, d_uvIn, pitchUVIn,
                                                d_yOut, pitchYOut, d_uvOut, pitchUVOut,
                                                w, h);
}

void launch_p010_to_x2bgr10(const uint8_t *d_y, int pitchY,
                            const uint8_t *d_uv, int pitchUV,
                            uint8_t *outX2BGR10, int outPitch,
                            int w, int h,
                            bool bt2020,
                            cudaStream_t stream)
{
    dim3 block(32, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    k_p010_to_x2bgr10<<<grid, block, 0, stream>>>(d_y, pitchY, d_uv, pitchUV, outX2BGR10, outPitch, w, h, bt2020);
}
