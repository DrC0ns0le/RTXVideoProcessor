#pragma once

extern "C"
{
#include <libavutil/avutil.h>
#include <libavutil/rational.h>
#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
}

#include <cstdint>
#include "logger.h"

/**
 * Simplified timestamp manager for FFmpeg 8 compatibility
 *
 * Philosophy: Set AVFrame->pts correctly, let FFmpeg handle the rest
 * - Encoder generates DTS automatically
 * - Muxer applies output_ts_offset via AVFormatContext->output_ts_offset
 * - Muxer validates monotonicity and DTS <= PTS
 * - No complex baseline tracking or manual A/V sync needed
 */
class TimestampManager
{
public:
    enum class Mode
    {
        NORMAL, // Zero-based timestamps
        COPYTS  // Preserve original timestamps
    };

    struct Config
    {
        Mode mode = Mode::NORMAL;
        int64_t input_seek_us = 0;         // Input seeking offset (-ss)
        int64_t output_seek_target_us = 0; // Output seeking target (frame dropping)
        bool start_at_zero = false;        // -start_at_zero flag
    };

    explicit TimestampManager(const Config &config) : cfg_(config)
    {
        LOG_DEBUG("TimestampManager initialized: mode=%s, input_seek=%lld us, start_at_zero=%s",
                  config.mode == Mode::COPYTS ? "COPYTS" : "NORMAL",
                  config.input_seek_us,
                  config.start_at_zero ? "enabled" : "disabled");
    }

    // ========== Video Frame PTS Derivation ==========

    /**
     * Derive PTS for encoded video frame (simplified API)
     * Returns single PTS value - encoder will generate DTS automatically
     */
    int64_t deriveVideoPTS(const AVFrame *frame,
                           AVRational in_time_base,
                           AVRational out_time_base)
    {
        // Get input PTS
        int64_t in_pts = (frame->pts != AV_NOPTS_VALUE)
                             ? frame->pts
                             : frame->best_effort_timestamp;

        // Handle missing timestamps with synthetic counter
        if (in_pts == AV_NOPTS_VALUE)
        {
            LOG_WARN("Frame has no PTS, using synthetic timestamp %lld", frame_counter_);
            return frame_counter_++;
        }

        // Increment frame counter for statistics
        frame_counter_++;

        // Derive PTS based on mode
        int64_t out_pts;
        if (cfg_.mode == Mode::COPYTS)
        {
            out_pts = deriveCopytsPTS(in_pts, in_time_base, out_time_base);
        }
        else
        {
            out_pts = deriveNormalPTS(in_pts, in_time_base, out_time_base);
        }

        return out_pts;
    }

    /**
     * Check if frame should be dropped during output seeking
     * Used for -ss output seeking (frame-level seeking after decode)
     */
    bool shouldDropFrameForOutputSeek(const AVFrame *frame, AVRational time_base)
    {
        // No output seeking requested
        if (output_seek_complete_ || cfg_.output_seek_target_us == 0)
        {
            return false;
        }

        // Get frame PTS
        int64_t frame_pts = (frame->pts != AV_NOPTS_VALUE)
                                ? frame->pts
                                : frame->best_effort_timestamp;

        if (frame_pts == AV_NOPTS_VALUE)
        {
            output_seek_complete_ = true;
            LOG_WARN("Output seek aborted: no frame timestamps");
            return false;
        }

        // Convert to microseconds
        int64_t frame_pts_us = av_rescale_q(frame_pts, time_base, {1, AV_TIME_BASE});

        // Drop frames before target
        if (frame_pts_us < cfg_.output_seek_target_us)
        {
            dropped_frame_count_++;
            return true;
        }

        // Reached target - stop dropping
        output_seek_complete_ = true;
        LOG_DEBUG("Output seek complete at %.3fs (dropped %d frames)",
                  frame_pts_us / 1000000.0, dropped_frame_count_);
        return false;
    }

    // ========== Packet Timestamp Rescaling ==========

    /**
     * Simple timestamp rescaling for copied streams
     * Just rescales from input to output timebase - no complex adjustments
     */
    void rescalePacketTimestamps(AVPacket *pkt,
                                 AVRational in_time_base,
                                 AVRational out_time_base)
    {
        av_packet_rescale_ts(pkt, in_time_base, out_time_base);
    }

    // ========== Statistics ==========

    int64_t getFrameCount() const { return frame_counter_; }
    int getDroppedFrames() const { return dropped_frame_count_; }

private:
    Config cfg_;

    // Baselines for timestamp derivation
    int64_t video_baseline_pts_ = AV_NOPTS_VALUE;  // NORMAL mode: baseline from first frame
    int64_t copyts_baseline_pts_ = AV_NOPTS_VALUE; // COPYTS mode: for -start_at_zero

    // State tracking
    bool output_seek_complete_ = false;
    int64_t frame_counter_ = 0;
    int dropped_frame_count_ = 0;

    // ========== Internal Implementation ==========

    /**
     * COPYTS mode: Preserve original timestamps
     * Optionally apply -start_at_zero offset
     */
    int64_t deriveCopytsPTS(int64_t in_pts, AVRational in_tb, AVRational out_tb)
    {
        // Rescale to output timebase
        int64_t out_pts = av_rescale_q_rnd(in_pts, in_tb, out_tb, AV_ROUND_NEAR_INF);

        // Handle overflow
        if (out_pts == AV_NOPTS_VALUE)
        {
            LOG_ERROR("Timestamp overflow in COPYTS mode, using 0");
            return 0;
        }

        // Apply -start_at_zero offset (subtract first frame PTS)
        if (cfg_.start_at_zero)
        {
            if (copyts_baseline_pts_ == AV_NOPTS_VALUE)
            {
                copyts_baseline_pts_ = out_pts;
                LOG_DEBUG("COPYTS baseline established at %lld for -start_at_zero", copyts_baseline_pts_);
            }
            out_pts -= copyts_baseline_pts_;
        }

        // Clamp negative timestamps to zero
        if (out_pts < 0)
        {
            LOG_DEBUG("Negative PTS %lld in COPYTS mode, clamping to 0", out_pts);
            out_pts = 0;
        }

        return out_pts;
    }

    /**
     * NORMAL mode: Zero-based timestamps
     * Adjust for input seeking to maintain continuous timeline
     */
    int64_t deriveNormalPTS(int64_t in_pts, AVRational in_tb, AVRational out_tb)
    {
        // Establish baseline from first frame
        // FFmpeg behavior: output timestamps always start at 0 (without -copyts)
        if (video_baseline_pts_ == AV_NOPTS_VALUE)
        {
            video_baseline_pts_ = in_pts;
            LOG_DEBUG("Video baseline established from first frame: %lld (input timebase)", video_baseline_pts_);
        }

        // Calculate relative PTS (zero-based output)
        // Example: if first frame is at 60s, baseline=60s, so output PTS starts at 0
        int64_t relative_pts = in_pts - video_baseline_pts_;

        // Rescale to output timebase
        int64_t out_pts = av_rescale_q_rnd(relative_pts, in_tb, out_tb, AV_ROUND_NEAR_INF);

        // FFmpeg behavior: clamp negative timestamps to zero
        // This can happen if frames arrive out of order or decoder issues
        if (out_pts < 0)
        {
            LOG_DEBUG("Negative PTS %lld in NORMAL mode, clamping to 0", out_pts);
            out_pts = 0;
        }

        return out_pts;
    }
};
