#pragma once

extern "C"
{
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
}

#include <vector>
#include <memory>
#include <stdexcept>

// RAII deleter helpers
static inline void av_frame_free_single_fp(AVFrame *f)
{
    if (f)
        av_frame_free(&f);
}

using FramePtr = std::unique_ptr<AVFrame, void (*)(AVFrame *)>;

// Simple round-robin CUDA encoder frame pool backed by an existing hw_frames_ctx.
class CudaFramePool
{
public:
    CudaFramePool() = default;

    // hw_frames_ctx: encoder's CUDA frames context (AV_PIX_FMT_CUDA with sw_format P010)
    // width/height are for clarity; frames will inherit from hw_frames_ctx
    void initialize(AVBufferRef *hw_frames_ctx, int width, int height, int pool_size)
    {
        if (!hw_frames_ctx)
            throw std::runtime_error("CudaFramePool: hw_frames_ctx is null");
        m_hw_frames_ctx = av_buffer_ref(hw_frames_ctx);
        if (!m_hw_frames_ctx)
            throw std::runtime_error("CudaFramePool: av_buffer_ref failed");
        m_frames.clear();
        m_frames.reserve(pool_size);
        for (int i = 0; i < pool_size; ++i)
        {
            FramePtr f(av_frame_alloc(), &av_frame_free_single_fp);
            if (!f)
                throw std::runtime_error("CudaFramePool: av_frame_alloc failed");
            // Set format/size for clarity; buffer will be fetched per use
            f->format = AV_PIX_FMT_CUDA;
            f->width = width;
            f->height = height;
            // Pre-bind a buffer once to back the frame; later uses will refresh with av_hwframe_get_buffer
            int err = av_hwframe_get_buffer(m_hw_frames_ctx, f.get(), 0);
            if (err < 0)
                throw std::runtime_error("CudaFramePool: av_hwframe_get_buffer failed during init");
            m_frames.push_back(std::move(f));
        }
        m_index = 0;
    }

    ~CudaFramePool()
    {
        if (m_hw_frames_ctx)
            av_buffer_unref(&m_hw_frames_ctx);
        // frames free automatically
    }

    // Returns a frame ready for writing. Calls av_frame_unref() and fetches a fresh buffer.
    AVFrame *acquire()
    {
        if (m_frames.empty())
            throw std::runtime_error("CudaFramePool not initialized");

        FramePtr &slot = m_frames[m_index];
        m_index = (m_index + 1) % m_frames.size();
        av_frame_unref(slot.get());
        slot->format = AV_PIX_FMT_CUDA;
        // width/height kept from initialization
        int err = av_hwframe_get_buffer(m_hw_frames_ctx, slot.get(), 0);
        if (err < 0)
            throw std::runtime_error("CudaFramePool: av_hwframe_get_buffer failed in acquire");
        return slot.get();
    }

private:
    AVBufferRef *m_hw_frames_ctx = nullptr;
    std::vector<FramePtr> m_frames;
    size_t m_index = 0;
};
