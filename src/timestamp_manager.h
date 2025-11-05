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
        int64_t input_seek_us = 0;          // Input seeking offset (-ss before -i)
        int64_t output_seek_target_us = 0;  // Output seeking target (-ss after -i, frame dropping)
        int64_t output_ts_offset_us = 0;    // Output timestamp offset (-output_ts_offset, Stremio mode)
        bool start_at_zero = false;         // -start_at_zero flag
        bool vsync_cfr = false;             // CFR mode: generate timestamps at constant frame rate
        AVRational cfr_frame_rate = {0, 1}; // Frame rate for CFR mode
        bool clamp_negative_copyts = true;  // If false, allow negative PTS in COPYTS (avoid_negative_ts=disabled)
        bool hls_mode = false;              // HLS fMP4 mode: preserve tfdt baseMediaDecodeTime for A/V sync
        double frame_drop_threshold = 0.0;  // CFR frame drop threshold (default 0 = use -1.1 hardcoded threshold)
    };

    explicit TimestampManager(const Config &config) : cfg_(config)
    {
        LOG_DEBUG("TimestampManager initialized: mode=%s, input_seek=%lld us, start_at_zero=%s",
                  config.mode == Mode::COPYTS ? "COPYTS" : "NORMAL",
                  config.input_seek_us,
                  config.start_at_zero ? "enabled" : "disabled");
    }

    // CFR helper utilities (FFmpeg-compliant delta-based approach)
    bool cfrActive() const { return cfg_.vsync_cfr; }

    /**
     * CFR frame synchronization (matches FFmpeg's video_sync_process)
     * Returns number of frames to output: 0=drop, 1=normal, >1=duplicate
     * Assigns PTS via out_pts parameter
     */
    int64_t cfrSync(const AVFrame *frame, AVRational in_tb, AVRational out_tb, int64_t *out_pts, double *out_duration)
    {
        int64_t in_pts = (frame->pts != AV_NOPTS_VALUE) ? frame->pts : frame->best_effort_timestamp;

        // CFR working timebase: av_inv_q(framerate) where 1 tick = 1 frame duration
        // For framerate 500/21, this gives timebase 21/500
        AVRational cfr_tb = av_inv_q(cfg_.cfr_frame_rate);

        // Convert input PTS to CFR timebase (FFmpeg: adjust_frame_pts_to_encoder_tb)
        int64_t sync_ipts = av_rescale_q_rnd(in_pts, in_tb, cfr_tb, AV_ROUND_NEAR_INF);

        // Initialize next_pts on first frame (in CFR timebase)
        if (cfr_next_pts_ == AV_NOPTS_VALUE)
        {
            cfr_next_pts_ = sync_ipts;
        }

        // Calculate delta (drift between input and expected position)
        // Since both sync_ipts and cfr_next_pts_ are in CFR timebase where 1 tick = 1 frame,
        // delta is naturally in frame units!
        // FFmpeg: delta0 = sync_ipts - ofp->next_pts; delta = delta0 + duration (where duration=1)
        double delta0 = (double)(sync_ipts - cfr_next_pts_);
        double delta = delta0 + 1.0;  // +1.0 for current frame duration (1 frame)

        // Calculate frame duration for encoder (in encoder timebase ticks)
        double frame_duration_ticks = av_q2d(av_inv_q(cfg_.cfr_frame_rate)) / av_q2d(out_tb);
        if (out_duration)
            *out_duration = frame_duration_ticks;

        int64_t nb_frames = 1; // Default: output one frame

        // FFmpeg's VSYNC_CFR logic
        if (cfg_.frame_drop_threshold > 0.0 && delta < cfg_.frame_drop_threshold && frame_counter_ > 0)
        {
            nb_frames = 0; // Drop frame
        }
        else if (delta < -1.1)
        {
            nb_frames = 0; // Drop frame
        }
        else if (delta > 1.1)
        {
            nb_frames = (int64_t)llrint(delta); // Duplicate frame
        }

        // Assign output PTS in CFR timebase (FFmpeg: frame_out->pts = ofp->next_pts)
        // Caller must convert this to encoder timebase using av_rescale_q(pts, cfr_tb, enc_tb)
        *out_pts = cfr_next_pts_;

        // Increment next_pts for each output frame (FFmpeg: ofp->next_pts++)
        // This stays in CFR timebase where incrementing by 1 = advancing by 1 frame
        cfr_next_pts_ += nb_frames;

        return nb_frames;
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
        // NOTE: CFR mode is now handled by cfrSync() in main.cpp
        // This method is only used for non-CFR paths

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

    // Get the copyts baseline for this stream
    // Returns the baseline PTS that was subtracted from timestamps (in timebase ticks)
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

    // CFR state (FFmpeg-compliant)
    int64_t cfr_next_pts_ = AV_NOPTS_VALUE;  // Next expected output PTS (FFmpeg: ofp->next_pts)

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
        // 1. HLS mode: Normalize timestamps for HLS segment compatibility
        //    - output_ts_offset is handled HERE (not by muxer) for HLS compatibility
        // 2. Non-HLS with output seeking or start_at_zero: Normalize to zero
        //    - Muxer will apply output_ts_offset (if set) to normalized timestamps
        //
        // Priority: hls_mode (with offset) > start_at_zero/output_seek_target (normalize to zero)
        bool should_normalize = cfg_.start_at_zero || (cfg_.output_seek_target_us > 0) || (cfg_.hls_mode && cfg_.output_ts_offset_us > 0);

        if (should_normalize)
        {
            if (copyts_baseline_pts_ == AV_NOPTS_VALUE)
            {
                // HLS mode with output_ts_offset: Apply offset in TimestampManager
                // (Muxer offset is disabled for HLS mode - see main.cpp line 282)
                if (cfg_.hls_mode && cfg_.output_ts_offset_us > 0)
                {
                    int64_t offset_pts = av_rescale_q(cfg_.output_ts_offset_us, {1, AV_TIME_BASE}, out_tb);
                    copyts_baseline_pts_ = out_pts - offset_pts;
                    LOG_DEBUG("COPYTS baseline for HLS (-output_ts_offset): first_pts=%lld, offset=%lld us (%lld ticks), baseline=%lld",
                              out_pts, cfg_.output_ts_offset_us, offset_pts, copyts_baseline_pts_);
                }
                // HLS mode with output seeking (no explicit offset): baseline relative to output seek target
                else if (cfg_.hls_mode && cfg_.output_seek_target_us > 0)
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
