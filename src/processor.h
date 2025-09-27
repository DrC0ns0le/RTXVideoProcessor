#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/pixfmt.h>
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

#include <memory>

#include "rtx_processor.h"
#include "frame_pool.h"

class IProcessor {
public:
    virtual ~IProcessor() = default;
    // Produces an encoder-ready frame (GPU: AV_PIX_FMT_CUDA P010; CPU: AV_PIX_FMT_P010LE)
    // Returns false on failure.
    virtual bool process(const AVFrame *decframe, AVFrame *&outFrame) = 0;
    virtual void shutdown() = 0;
};

class GpuProcessor : public IProcessor {
public:
    GpuProcessor(RTXProcessor &rtx, CudaFramePool &pool, AVColorSpace colorSpace)
        : m_rtx(rtx), m_pool(pool), m_bt2020(colorSpace == AVCOL_SPC_BT2020_NCL) {}

    bool process(const AVFrame *decframe, AVFrame *&outFrame) override {
        if (!decframe || decframe->format != AV_PIX_FMT_CUDA) return false;
        AVFrame *enc_hw = m_pool.acquire();
        if (!m_rtx.processGpuNV12ToP010(decframe->data[0], decframe->linesize[0],
                                        decframe->data[1], decframe->linesize[1],
                                        enc_hw, m_bt2020))
            return false;
        outFrame = enc_hw;
        return true;
    }

    void shutdown() override { /* RTX owned by caller */ }

private:
    RTXProcessor &m_rtx;
    CudaFramePool &m_pool;
    bool m_bt2020 = false;
};

class CpuProcessor : public IProcessor {
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

        // Allocate P010 buffer
        m_p010.reset(av_frame_alloc());
        m_p010->format = AV_PIX_FMT_P010LE;
        m_p010->width = m_dstW;
        m_p010->height = m_dstH;
        if (av_frame_get_buffer(m_p010.get(), 32) < 0)
            throw std::runtime_error("CpuProcessor: alloc P010 failed");

        // CPU path colorspace for X2BGR10LE->P010 (ABGR10 to P010)
        m_sws_to_p010 = sws_getContext(
            m_dstW, m_dstH, AV_PIX_FMT_X2BGR10LE,
            m_dstW, m_dstH, AV_PIX_FMT_P010LE,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!m_sws_to_p010)
            throw std::runtime_error("CpuProcessor: sws_to_p010 alloc failed");
        const int *coeffs_bt2020 = sws_getCoefficients(SWS_CS_BT2020);
        sws_setColorspaceDetails(m_sws_to_p010,
                                 coeffs_bt2020, 1,
                                 coeffs_bt2020, 0,
                                 0, 1 << 16, 1 << 16);
    }

    bool process(const AVFrame *decframe, AVFrame *&outFrame) override {
        if (!decframe) return false;
        // Build/update sws_to_argb if needed
        if (!m_sws_to_argb || m_last_src_format != decframe->format || m_last_src_w != decframe->width || m_last_src_h != decframe->height) {
            if (m_sws_to_argb) sws_freeContext(m_sws_to_argb);
            m_sws_to_argb = sws_getContext(
                decframe->width, decframe->height, (AVPixelFormat)decframe->format,
                m_srcW, m_srcH, AV_PIX_FMT_RGBA,
                SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!m_sws_to_argb) return false;
            const int *coeffs = sws_getCoefficients(SWS_CS_ITU709);
            sws_setColorspaceDetails(m_sws_to_argb, coeffs, 0, coeffs, 1, 0, 1 << 16, 1 << 16);
            m_last_src_format = decframe->format;
            m_last_src_w = decframe->width; m_last_src_h = decframe->height;
        }

        const uint8_t *srcData[AV_NUM_DATA_POINTERS] = {decframe->data[0], decframe->data[1], decframe->data[2], decframe->data[3]};
        int srcLines[AV_NUM_DATA_POINTERS] = {decframe->linesize[0], decframe->linesize[1], decframe->linesize[2], decframe->linesize[3]};
        if (av_frame_make_writable(m_bgra.get()) < 0) return false;
        sws_scale(m_sws_to_argb, srcData, srcLines, 0, decframe->height, m_bgra->data, m_bgra->linesize);

        // RTX CPU process -> ABGR10
        const uint8_t *rtx_data = nullptr; uint32_t rtxW = 0, rtxH = 0; size_t rtxPitch = 0;
        if (!m_rtx.process(m_bgra->data[0], (size_t)m_bgra->linesize[0], rtx_data, rtxW, rtxH, rtxPitch)) return false;
        if (rtxW != (uint32_t)m_dstW || rtxH != (uint32_t)m_dstH) return false;

        // ABGR10 -> P010
        if (av_frame_make_writable(m_p010.get()) < 0) return false;
        const uint8_t *abgr_planes[1] = {rtx_data};
        int abgr_lines[1] = {static_cast<int>(rtxPitch)};
        sws_scale(m_sws_to_p010, abgr_planes, abgr_lines, 0, m_dstH, m_p010->data, m_p010->linesize);

        outFrame = m_p010.get();
        return true;
    }

    void setConfig(const RTXProcessConfig &cfg) { m_rtx_cfg = cfg; }

    void shutdown() override {
        if (m_sws_to_argb) { sws_freeContext(m_sws_to_argb); m_sws_to_argb = nullptr; }
        if (m_sws_to_p010) { sws_freeContext(m_sws_to_p010); m_sws_to_p010 = nullptr; }
    }

private:
    RTXProcessor &m_rtx;
    RTXProcessConfig m_rtx_cfg{};
    int m_srcW = 0, m_srcH = 0;
    int m_dstW = 0, m_dstH = 0;

    SwsContext *m_sws_to_argb = nullptr;
    SwsContext *m_sws_to_p010 = nullptr;
    int m_last_src_format = AV_PIX_FMT_NONE;
    int m_last_src_w = 0, m_last_src_h = 0;

    FramePtr m_bgra{nullptr, &av_frame_free_single_fp};
    FramePtr m_p010{nullptr, &av_frame_free_single_fp};

    bool m_rtx_initialized = false;
};
