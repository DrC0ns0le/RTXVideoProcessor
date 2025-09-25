#include "cuda_kernels.h"
#include <math.h>

// Utility: clamp to [a,b]
static __device__ inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static __device__ inline int clampi(int x, int a, int b) { return x < a ? a : (x > b ? b : x); }

// Convert ABGR10 (X2BGR10LE) 32bpp to P010 (Y:16bpp, UV:16bpp interleaved), BT.709 or BT.2020 limited range
// Assumptions:
// - inABGR points to pitched rows of 32-bit pixels, with bpp=4. Layout LE: bits 0..9 R, 10..19 G, 20..29 B, 30..31 A/unused.
// - outY pitch in bytes; outUV pitch in bytes. outUV height = h/2 and width = w (16-bit pairs per pixel).

__global__ void k_abgr10_to_p010(const uint8_t* __restrict__ inABGR, int inPitch,
                                 uint8_t* __restrict__ outY, int pitchY,
                                 uint8_t* __restrict__ outUV, int pitchUV,
                                 int w, int h, bool bt2020)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // process 2x2 block's left pixel x even
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2; // top row of 2x2 block
    if (x + 1 >= w || y + 1 >= h) return;

    // Accumulate Cb/Cr over 2x2
    float sumCb = 0.f, sumCr = 0.f;

    // Iterate 2x2
    float Ys[2][2];
    for (int dy = 0; dy < 2; ++dy) {
        const uint32_t* row = (const uint32_t*)(inABGR + (y + dy) * inPitch);
        for (int dx = 0; dx < 2; ++dx) {
            uint32_t p = row[x + dx];
            // X2BGR10LE (FFmpeg AV_PIX_FMT_X2BGR10LE): bits 0..9 B, 10..19 G, 20..29 R, 30..31 unused
            int r10 = (int)((p >> 0)  & 0x3FF);   // Red in bits 0-9
            int g10 = (int)((p >> 10) & 0x3FF);   // Green in bits 10-19
            int b10 = (int)((p >> 20) & 0x3FF);   // Blue in bits 20-29
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
    auto mapY10 = [](float Y) {
        float v = 64.0f + Y * 876.0f; // [64..940]
        int vi = (int)lrintf(v);
        return (unsigned short)clampi(vi, 64, 940);
    };
    auto mapC10 = [](float C) {
        float v = 512.0f + C * 896.0f; // [64..960]
        int vi = (int)lrintf(v);
        return (unsigned short)clampi(vi, 64, 960);
    };

    // Write Y plane (two rows, two cols)
    unsigned short* yRow0 = (unsigned short*)(outY + y * pitchY);
    unsigned short* yRow1 = (unsigned short*)(outY + (y + 1) * pitchY);
    unsigned short y00 = (unsigned short)(mapY10(Ys[0][0]) << 6);
    unsigned short y01 = (unsigned short)(mapY10(Ys[0][1]) << 6);
    unsigned short y10 = (unsigned short)(mapY10(Ys[1][0]) << 6);
    unsigned short y11 = (unsigned short)(mapY10(Ys[1][1]) << 6);
    yRow0[x + 0] = y00;
    yRow0[x + 1] = y01;
    yRow1[x + 0] = y10;
    yRow1[x + 1] = y11;

    // Write interleaved UV (at (y/2, x))
    int uvy = y / 2;           // chroma row
    int ux  = x / 2;           // chroma column (subsampled)
    unsigned short* uvRow = (unsigned short*)(outUV + uvy * pitchUV);
    unsigned short U = (unsigned short)(mapC10(avgCb) << 6);
    unsigned short V = (unsigned short)(mapC10(avgCr) << 6);
    // Interleaved 16-bit U,V per chroma sample
    uvRow[ux * 2 + 0] = U;
    uvRow[ux * 2 + 1] = V;
}

void launch_abgr10_to_p010(const uint8_t* inABGR, int inPitch,
                           uint8_t* outY, int pitchY,
                           uint8_t* outUV, int pitchUV,
                           int w, int h,
                           bool bt2020,
                           cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((w + block.x*2 - 1) / (block.x*2), (h + block.y*2 - 1) / (block.y*2));
    k_abgr10_to_p010<<<grid, block, 0, stream>>>(inABGR, inPitch, outY, pitchY, outUV, pitchUV, w, h, bt2020);
}

static __device__ inline void yuv_to_rgb(float Yp, float Uc, float Vc, bool bt2020, float& R, float& G, float& B)
{
    if (bt2020) {
        // BT.2020
        R = Yp + 1.4746f * Vc;
        G = Yp - 0.16455f * Uc - 0.57135f * Vc;
        B = Yp + 1.8814f * Uc;
    } else {
        // BT.709
        R = Yp + 1.5748f * Vc;
        G = Yp - 0.1873f * Uc - 0.4681f * Vc;
        B = Yp + 1.8556f * Uc;
    }
}

__global__ void k_nv12_to_rgba(const uint8_t* __restrict__ d_y, int pitchY,
                               const uint8_t* __restrict__ d_uv, int pitchUV,
                               uint8_t* __restrict__ outBGRA, int outPitch,
                               int w, int h, bool bt2020)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

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

    uint8_t* dst = outBGRA + y * outPitch + x * 4;
    dst[0] = r8;  // R
    dst[1] = g8;  // G
    dst[2] = b8;  // B
    dst[3] = 255; // A
}

void launch_nv12_to_bgra(const uint8_t* d_y, int pitchY,
                         const uint8_t* d_uv, int pitchUV,
                         uint8_t* outBGRA, int outPitch,
                         int w, int h,
                         bool bt2020,
                         cudaStream_t stream)
{
    dim3 block(32, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    k_nv12_to_rgba<<<grid, block, 0, stream>>>(d_y, pitchY, d_uv, pitchUV, outBGRA, outPitch, w, h, bt2020);
}
