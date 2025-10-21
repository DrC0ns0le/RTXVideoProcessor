#pragma once

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/pixfmt.h>
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

#include <memory>

#include "rtx_processor.h"
#include "frame_pool.h"

class IProcessor
{
public:
    virtual ~IProcessor() = default;
    // Produces an encoder-ready frame (GPU: AV_PIX_FMT_CUDA P010; CPU: AV_PIX_FMT_P010LE)
    // Returns false on failure.
    virtual bool process(const AVFrame *decframe, AVFrame *&outFrame) = 0;
    virtual void shutdown() = 0;
};

class GpuProcessor : public IProcessor
{
public:
    GpuProcessor(RTXProcessor &rtx, CudaFramePool &pool, AVColorSpace colorSpace, bool thdrEnabled)
        : m_rtx(rtx), m_pool(pool), m_bt2020(colorSpace == AVCOL_SPC_BT2020_NCL), m_thdrEnabled(thdrEnabled) {}

    bool process(const AVFrame *decframe, AVFrame *&outFrame) override
    {
        if (!decframe || decframe->format != AV_PIX_FMT_CUDA)
            return false;

        // Determine the actual pixel format of the CUDA frame
        AVPixelFormat sw_format = AV_PIX_FMT_NONE;
        if (decframe->hw_frames_ctx)
        {
            AVHWFramesContext *hw_frames_ctx = (AVHWFramesContext *)decframe->hw_frames_ctx->data;
            sw_format = hw_frames_ctx->sw_format;
        }

        AVFrame *enc_hw = m_pool.acquire();
        bool ok = false;

        if (sw_format == AV_PIX_FMT_P010LE)
        {
            // P010 input: could be true HDR (BT.2020) or SDR (BT.709) in 10-bit container (Main10)
            // Key distinction: m_bt2020 tells us if input is true HDR or just SDR in P010 format

            if (m_bt2020)
            {
                // True HDR input (BT.2020): preserve 10-bit pipeline
                // P010 -> X2BGR10LE (10-bit) -> RTX -> ABGR10 -> P010
                ok = m_rtx.processGpuP010ToP010(decframe->data[0], decframe->linesize[0],
                                                decframe->data[1], decframe->linesize[1],
                                                enc_hw, m_bt2020);
            }
            else
            {
                // SDR input in P010 format (BT.709): treat as 8-bit SDR
                if (m_thdrEnabled)
                {
                    // THDR enabled for SDR: P010 -> NV12 (8-bit) -> BGRA8 -> RTX (VSR+THDR) -> ABGR10 -> P010
                    ok = m_rtx.processGpuP010SDRToP010(decframe->data[0], decframe->linesize[0],
                                                       decframe->data[1], decframe->linesize[1],
                                                       enc_hw);
                }
                else
                {
                    // No THDR: P010 -> NV12 (8-bit) -> BGRA8 -> RTX (VSR only) -> BGRA8 -> NV12
                    ok = m_rtx.processGpuP010ToNV12(decframe->data[0], decframe->linesize[0],
                                                    decframe->data[1], decframe->linesize[1],
                                                    enc_hw);
                }
            }
        }
        else
        {
            // NV12 input: SDR content
            if (m_thdrEnabled)
            {
                ok = m_rtx.processGpuNV12ToP010(decframe->data[0], decframe->linesize[0],
                                                decframe->data[1], decframe->linesize[1],
                                                enc_hw, m_bt2020);
            }
            else
            {
                ok = m_rtx.processGpuNV12ToNV12(decframe->data[0], decframe->linesize[0],
                                                decframe->data[1], decframe->linesize[1],
                                                enc_hw, m_bt2020);
            }
        }
        if (!ok)
            return false;
        outFrame = enc_hw;
        return true;
    }

    void shutdown() override { /* RTX owned by caller */ }

private:
    RTXProcessor &m_rtx;
    CudaFramePool &m_pool;
    bool m_bt2020 = false;
    bool m_thdrEnabled = false;
};

class CpuProcessor : public IProcessor
{
public:
    CpuProcessor(RTXProcessor &rtx,
                 int srcW, int srcH,
                 int dstW, int dstH)
        : m_rtx(rtx), m_srcW(srcW), m_srcH(srcH), m_dstW(dstW), m_dstH(dstH)
    {
        // Allocate BGRA buffer
        m_bgra.reset(av_frame_alloc());
        m_bgra->format = AV_PIX_FMT_RGBA;
        m_bgra->width = m_srcW;
        m_bgra->height = m_srcH;
        if (av_frame_get_buffer(m_bgra.get(), 32) < 0)
            throw std::runtime_error("CpuProcessor: alloc BGRA failed");

        // Defer building output and sws until setConfig() (needs THDR on/off)
        m_sws_to_yuv = nullptr;
    }

    bool process(const AVFrame *decframe, AVFrame *&outFrame) override
    {
        if (!decframe)
            return false;
        // Build/update sws_to_argb if needed
        if (!m_sws_to_argb || m_last_src_format != decframe->format || m_last_src_w != decframe->width || m_last_src_h != decframe->height)
        {
            if (m_sws_to_argb)
                sws_freeContext(m_sws_to_argb);
            m_sws_to_argb = sws_getContext(
                decframe->width, decframe->height, (AVPixelFormat)decframe->format,
                m_srcW, m_srcH, AV_PIX_FMT_RGBA,
                SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!m_sws_to_argb)
                return false;
            // Select input colorspace based on decoded frame colorspace
            const int *coeffs = (decframe->colorspace == AVCOL_SPC_BT2020_NCL)
                                    ? sws_getCoefficients(SWS_CS_BT2020)
                                    : sws_getCoefficients(SWS_CS_ITU709);
            sws_setColorspaceDetails(m_sws_to_argb, coeffs, 0, coeffs, 1, 0, 1 << 16, 1 << 16);
            m_last_src_format = decframe->format;
            m_last_src_w = decframe->width;
            m_last_src_h = decframe->height;
        }

        const uint8_t *srcData[AV_NUM_DATA_POINTERS] = {decframe->data[0], decframe->data[1], decframe->data[2], decframe->data[3]};
        int srcLines[AV_NUM_DATA_POINTERS] = {decframe->linesize[0], decframe->linesize[1], decframe->linesize[2], decframe->linesize[3]};
        if (av_frame_make_writable(m_bgra.get()) < 0)
            return false;
        sws_scale(m_sws_to_argb, srcData, srcLines, 0, decframe->height, m_bgra->data, m_bgra->linesize);

        // RTX CPU process -> ABGR10
        const uint8_t *rtx_data = nullptr;
        uint32_t rtxW = 0, rtxH = 0;
        size_t rtxPitch = 0;
        if (!m_rtx.process(m_bgra->data[0], (size_t)m_bgra->linesize[0], rtx_data, rtxW, rtxH, rtxPitch))
            return false;
        if (rtxW != (uint32_t)m_dstW || rtxH != (uint32_t)m_dstH)
            return false;

        // Build m_sws_to_yuv if needed based on THDR config and output format
        if (!m_sws_to_yuv)
        {
            AVPixelFormat srcPix = m_rtx_cfg.enableTHDR ? AV_PIX_FMT_X2BGR10LE : AV_PIX_FMT_BGRA;
            AVPixelFormat dstPix = m_rtx_cfg.enableTHDR ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
            m_sws_to_yuv = sws_getContext(
                m_dstW, m_dstH, srcPix,
                m_dstW, m_dstH, dstPix,
                SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!m_sws_to_yuv)
                return false;
            const int *coeffs = m_rtx_cfg.enableTHDR ? sws_getCoefficients(SWS_CS_BT2020)
                                                     : sws_getCoefficients(SWS_CS_ITU709);
            sws_setColorspaceDetails(m_sws_to_yuv,
                                     coeffs, m_rtx_cfg.enableTHDR ? 1 : 0,
                                     coeffs, m_rtx_cfg.enableTHDR ? 0 : 0,
                                     0, 1 << 16, 1 << 16);
        }

        // Ensure output buffer is allocated and writable
        if (!m_out)
            return false;
        if (av_frame_make_writable(m_out.get()) < 0)
            return false;
        const uint8_t *in_planes[1] = {rtx_data};
        int in_lines[1] = {static_cast<int>(rtxPitch)};
        sws_scale(m_sws_to_yuv, in_planes, in_lines, 0, m_dstH, m_out->data, m_out->linesize);

        outFrame = m_out.get();
        return true;
    }

    void setConfig(const RTXProcessConfig &cfg)
    {
        m_rtx_cfg = cfg;
        // Rebuild output frame and sws to match THDR on/off
        if (m_sws_to_yuv)
        {
            sws_freeContext(m_sws_to_yuv);
            m_sws_to_yuv = nullptr;
        }
        // Allocate output frame in requested format
        m_out.reset(av_frame_alloc());
        if (!m_out)
            throw std::runtime_error("CpuProcessor: alloc out frame failed");
        m_out->format = m_rtx_cfg.enableTHDR ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
        m_out->width = m_dstW;
        m_out->height = m_dstH;
        if (av_frame_get_buffer(m_out.get(), 32) < 0)
            throw std::runtime_error("CpuProcessor: alloc out buffer failed");

        AVPixelFormat srcPix = m_rtx_cfg.enableTHDR ? AV_PIX_FMT_X2BGR10LE : AV_PIX_FMT_BGRA;
        AVPixelFormat dstPix = m_rtx_cfg.enableTHDR ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
        m_sws_to_yuv = sws_getContext(
            m_dstW, m_dstH, srcPix,
            m_dstW, m_dstH, dstPix,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!m_sws_to_yuv)
            throw std::runtime_error("CpuProcessor: sws_to_yuv alloc failed in setConfig");
        const int *coeffs = m_rtx_cfg.enableTHDR ? sws_getCoefficients(SWS_CS_BT2020)
                                                 : sws_getCoefficients(SWS_CS_ITU709);
        sws_setColorspaceDetails(m_sws_to_yuv,
                                 coeffs, m_rtx_cfg.enableTHDR ? 1 : 0,
                                 coeffs, m_rtx_cfg.enableTHDR ? 0 : 0,
                                 0, 1 << 16, 1 << 16);
    }

    void shutdown() override
    {
        if (m_sws_to_argb)
        {
            sws_freeContext(m_sws_to_argb);
            m_sws_to_argb = nullptr;
        }
        if (m_sws_to_yuv)
        {
            sws_freeContext(m_sws_to_yuv);
            m_sws_to_yuv = nullptr;
        }
    }

private:
    RTXProcessor &m_rtx;
    RTXProcessConfig m_rtx_cfg{};
    int m_srcW = 0, m_srcH = 0;
    int m_dstW = 0, m_dstH = 0;

    SwsContext *m_sws_to_argb = nullptr;
    SwsContext *m_sws_to_yuv = nullptr; // to P010 (HDR) or NV12 (SDR)
    int m_last_src_format = AV_PIX_FMT_NONE;
    int m_last_src_w = 0, m_last_src_h = 0;

    FramePtr m_bgra{nullptr, &av_frame_free_single_fp};
    FramePtr m_out{nullptr, &av_frame_free_single_fp};

    bool m_rtx_initialized = false;
};
