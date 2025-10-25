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
        bool vsync_cfr = false;            // CFR mode: generate timestamps at constant frame rate
        AVRational cfr_frame_rate = {0, 1}; // Frame rate for CFR mode
        bool clamp_negative_copyts = true; // If false, allow negative PTS in COPYTS (avoid_negative_ts=disabled)
        bool hls_mode = false;             // HLS fMP4 mode: preserve tfdt baseMediaDecodeTime for A/V sync
    };

    explicit TimestampManager(const Config &config) : cfg_(config)
    {
        LOG_DEBUG("TimestampManager initialized: mode=%s, input_seek=%lld us, start_at_zero=%s",
                  config.mode == Mode::COPYTS ? "COPYTS" : "NORMAL",
                  config.input_seek_us,
                  config.start_at_zero ? "enabled" : "disabled");
    }

    // CFR helper utilities (no side effects on last index)
    bool cfrActive() const { return cfg_.vsync_cfr; }
    void ensureCfrBaseline(int64_t in_pts)
    {
        if (cfr_baseline_in_pts_ == AV_NOPTS_VALUE)
            cfr_baseline_in_pts_ = in_pts;
    }
    int64_t getLastCfrIndex() const { return cfr_last_index_; }
    void setLastCfrIndex(int64_t idx) { cfr_last_index_ = idx; }
    int64_t cfrTargetIndexFromInput(int64_t in_pts, AVRational in_tb) const
    {
        if (cfr_baseline_in_pts_ == AV_NOPTS_VALUE) return 0;
        int64_t delta_in = in_pts - cfr_baseline_in_pts_;
        return av_rescale_q_rnd(delta_in, in_tb, cfg_.cfr_frame_rate, AV_ROUND_NEAR_INF);
    }
    int64_t cfrIndexToPts(int64_t index, AVRational out_tb) const
    {
        return av_rescale_q_rnd(index, av_inv_q(cfg_.cfr_frame_rate), out_tb, AV_ROUND_NEAR_INF);
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
        // CFR mode: quantize input PTS to CFR ticks and drop frames mapping to the same CFR index
        if (cfg_.vsync_cfr)
        {
            int64_t in_pts = (frame->pts != AV_NOPTS_VALUE)
                                 ? frame->pts
                                 : frame->best_effort_timestamp;
            if (in_pts == AV_NOPTS_VALUE)
            {
                // If no input PTS, fall back to frame counter-based CFR
                int64_t cfr_pts_fc = av_rescale_q_rnd(frame_counter_, av_inv_q(cfg_.cfr_frame_rate), out_time_base, AV_ROUND_NEAR_INF);
                frame_counter_++;
                return cfr_pts_fc;
            }

            // Establish CFR baseline on first frame
            if (cfr_baseline_in_pts_ == AV_NOPTS_VALUE)
                cfr_baseline_in_pts_ = in_pts;

            // Compute target CFR frame index from input PTS relative to baseline
            int64_t delta_in = in_pts - cfr_baseline_in_pts_;
            int64_t target_index = av_rescale_q_rnd(delta_in, in_time_base, cfg_.cfr_frame_rate, AV_ROUND_NEAR_INF);

            // Drop if this frame maps to the same CFR index as last emitted
            if (cfr_last_index_ != AV_NOPTS_VALUE && target_index <= cfr_last_index_)
            {
                return AV_NOPTS_VALUE; // signal drop
            }

            cfr_last_index_ = target_index;

            // Quantized CFR PTS in output timebase
            int64_t cfr_pts = av_rescale_q_rnd(target_index, av_inv_q(cfg_.cfr_frame_rate), out_time_base, AV_ROUND_NEAR_INF);
            return cfr_pts;
        }

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
     *
     * FFmpeg's behavior: With double -ss (e.g., -ss 60 -i input -ss 60), frames
     * before the output seek target are decoded but NOT muxed. This ensures:
     * - Content starts at the output seek position (60s), not the keyframe (50s)
     * - Duration is measured from output seek target (60+12=72s), not keyframe (50+12=62s)
     * - HLS tfdt values reflect the correct playback timeline position
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

        // Drop frames before target (FFmpeg behavior for output seeking)
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

    // Get the copyts baseline for audio synchronization
    // Returns the baseline PTS that was subtracted from timestamps (in input timebase ticks)
    // Audio can use this to apply the same normalization for A/V sync
    int64_t getCopytsBaseline() const { return copyts_baseline_pts_; }
    bool hasCopytsBaseline() const { return copyts_baseline_pts_ != AV_NOPTS_VALUE; }

private:
    Config cfg_;

    // Baselines for timestamp derivation
    int64_t video_baseline_pts_ = AV_NOPTS_VALUE;  // NORMAL mode: baseline from first frame
    int64_t copyts_baseline_pts_ = AV_NOPTS_VALUE; // COPYTS mode: for -start_at_zero

    // State tracking
    bool output_seek_complete_ = false;
    int64_t frame_counter_ = 0;
    int dropped_frame_count_ = 0;

    // CFR quantization state
    int64_t cfr_baseline_in_pts_ = AV_NOPTS_VALUE; // input-pts baseline for CFR quantization
    int64_t cfr_last_index_ = AV_NOPTS_VALUE;      // last emitted CFR frame index

    // ========== Internal Implementation ==========

    /**
     * COPYTS mode: Preserve original timestamps
     * Optionally apply -start_at_zero offset or output seeking normalization
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

        // FFmpeg behavior with output seeking and COPYTS:
        // - Non-HLS: Normalize to zero (avoid_negative_ts=auto)
        // - HLS: Normalize to output seek target for correct tfdt (baseMediaDecodeTime)
        //
        // HLS players like Stremio rely on tfdt to sync A/V tracks. The tfdt must reflect
        // the playback timeline position (e.g., 60s seek = tfdt starts at ~60s equivalent),
        // not the decode timeline (which may start earlier due to keyframe seeking).
        bool should_normalize = cfg_.start_at_zero || (cfg_.output_seek_target_us > 0);

        if (should_normalize)
        {
            if (copyts_baseline_pts_ == AV_NOPTS_VALUE)
            {
                // For HLS mode with output seeking, baseline relative to output seek target
                // This ensures tfdt reflects the playback position, not decode position
                if (cfg_.hls_mode && cfg_.output_seek_target_us > 0)
                {
                    // Set baseline to (first_frame_pts - output_seek_target)
                    // This makes timestamps start near output_seek_target in the timeline
                    int64_t seek_target_pts = av_rescale_q(cfg_.output_seek_target_us, {1, AV_TIME_BASE}, out_tb);
                    copyts_baseline_pts_ = out_pts - seek_target_pts;
                    LOG_DEBUG("COPYTS baseline for HLS: first_pts=%lld, seek_target=%lld, baseline=%lld",
                              out_pts, seek_target_pts, copyts_baseline_pts_);
                }
                else
                {
                    // Non-HLS or start_at_zero: normalize to zero
                    copyts_baseline_pts_ = out_pts;
                    const char* reason = cfg_.start_at_zero ? "-start_at_zero" : "output seeking";
                    LOG_DEBUG("COPYTS baseline established at %lld for %s", copyts_baseline_pts_, reason);
                }
            }
            out_pts -= copyts_baseline_pts_;
        }

        // Clamp negative timestamps only if requested
        if (out_pts < 0 && cfg_.clamp_negative_copyts)
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
        // Establish baseline
        // FFmpeg behavior: output timestamps always start at 0 (without -copyts)
        if (video_baseline_pts_ == AV_NOPTS_VALUE)
        {
            // When seeking is used, use the seek target as baseline to ensure A/V sync
            // This is critical because:
            // - With accurate_seek: first frame IS the target (baseline = first frame = target) ✓
            // - With noaccurate_seek: first frame is BEFORE target (baseline must be target for A/V sync) ✓
            if (cfg_.input_seek_us > 0)
            {
                video_baseline_pts_ = av_rescale_q(cfg_.input_seek_us, {1, AV_TIME_BASE}, in_tb);
                LOG_DEBUG("Video baseline set from seek target: %lld us -> %lld (input timebase)",
                         cfg_.input_seek_us, video_baseline_pts_);
            }
            else
            {
                video_baseline_pts_ = in_pts;
                LOG_DEBUG("Video baseline established from first frame: %lld (input timebase)", video_baseline_pts_);
            }
        }

        // Calculate relative PTS (zero-based output)
        int64_t relative_pts = in_pts - video_baseline_pts_;

        // Rescale to output timebase
        int64_t out_pts = av_rescale_q_rnd(relative_pts, in_tb, out_tb, AV_ROUND_NEAR_INF);

        // FFmpeg behavior: clamp negative timestamps to zero
        // This can happen with noaccurate_seek when first frame is before seek target
        if (out_pts < 0)
        {
            LOG_DEBUG("Negative PTS %lld in NORMAL mode (frame before seek target), clamping to 0", out_pts);
            out_pts = 0;
        }

        return out_pts;
    }
};
