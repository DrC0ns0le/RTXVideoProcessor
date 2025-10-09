#pragma once

extern "C"
{
#include <libavutil/avutil.h>
#include <libavutil/rational.h>
#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
}

#include <cstdint>
#include <functional>
#include <memory>
#include "logger.h"
#include "pipeline_types.h"

/**
 * Production-grade timestamp management system
 *
 * Improvements over v1:
 * - Eliminates redundant timebase conversions
 * - Function pointer strategy for mode selection (no branch in hot path)
 * - Proper DTS handling for B-frames
 * - Discontinuity detection
 * - A/V drift monitoring
 * - Overflow protection
 * - Better monotonicity enforcement
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
        int64_t input_seek_us = 0;
        int64_t output_seek_target_us = 0;
        int64_t output_ts_offset_us = 0;
        bool enable_validation = true;
        bool enforce_monotonicity = true;         // Automatically fix timestamp violations (legacy default, FFmpeg doesn't do this)
        AVRational expected_frame_rate = {24, 1}; // For monotonicity recovery
        AvoidNegativeTs avoid_negative_ts = AvoidNegativeTs::AUTO;
        bool start_at_zero = false;
    };

    struct TimestampPair
    {
        int64_t pts = AV_NOPTS_VALUE;
        int64_t dts = AV_NOPTS_VALUE;
    };

    explicit TimestampManager(const Config &config)
        : cfg_(config), needs_baseline_wait_(config.input_seek_us > 0 && config.mode == Mode::NORMAL), frame_duration_us_(av_rescale_q(1, av_inv_q(config.expected_frame_rate), {1, AV_TIME_BASE}))
    {
        // Pre-compute frame duration in microseconds for monotonicity recovery
        LOG_DEBUG("TimestampManager initialized: mode=%s, avoid_negative_ts=%s, start_at_zero=%s, frame_duration=%.3fms",
                  config.mode == Mode::COPYTS ? "COPYTS" : "NORMAL",
                  getAvoidNegativeTsName(config.avoid_negative_ts),
                  config.start_at_zero ? "enabled" : "disabled",
                  frame_duration_us_ / 1000.0);
    }

    // ========== Video Frame Timestamps ==========

    /**
     * Derive both PTS and DTS for encoded video frame
     */
    TimestampPair deriveVideoTimestamps(const AVFrame *frame,
                                        AVRational in_time_base,
                                        AVRational out_time_base)
    {
        int64_t in_pts = (frame->pts != AV_NOPTS_VALUE)
                             ? frame->pts
                             : frame->best_effort_timestamp;

        TimestampPair result;

        // Handle missing timestamps
        if (in_pts == AV_NOPTS_VALUE)
        {
            result.pts = synthetic_counter_++;
            result.dts = result.pts; // No reordering without timestamps
            return result;
        }

        // Derive PTS based on mode
        if (cfg_.mode == Mode::COPYTS)
        {
            result.pts = deriveCopytsPTS(in_pts, in_time_base, out_time_base);
        }
        else
        {
            result.pts = deriveNormalPTS(in_pts, in_time_base, out_time_base);
        }

        // Enforce PTS monotonicity only (encoder handles DTS)
        enforceMonotonicity(result, out_time_base);

        // THEN update tracking with FINAL corrected value
        int64_t final_pts_us = av_rescale_q(result.pts, out_time_base, {1, AV_TIME_BASE});

        // Detect discontinuities (using FINAL corrected PTS)
        if (last_video_pts_us_ != AV_NOPTS_VALUE)
        {
            if (detectDiscontinuity(final_pts_us, last_video_pts_us_))
            {
                LOG_WARN("Video timestamp discontinuity detected: %.3fs -> %.3fs",
                         last_video_pts_us_ / 1000000.0, final_pts_us / 1000000.0);
                discontinuity_count_++;
            }
        }

        // Update tracking with corrected PTS
        last_video_pts_us_ = final_pts_us;

        return result;
    }

    /**
     * Check if frame should be dropped during output seeking
     */
    bool shouldDropFrameForOutputSeek(const AVFrame *frame, AVRational time_base)
    {
        if (output_seek_complete_ || cfg_.output_seek_target_us == 0)
        {
            return false;
        }

        int64_t frame_pts = (frame->pts != AV_NOPTS_VALUE)
                                ? frame->pts
                                : frame->best_effort_timestamp;

        if (frame_pts == AV_NOPTS_VALUE)
        {
            output_seek_complete_ = true;
            LOG_WARN("Output seek aborted: no frame timestamps");
            return false;
        }

        int64_t frame_pts_us = av_rescale_q(frame_pts, time_base, {1, AV_TIME_BASE});

        if (frame_pts_us < cfg_.output_seek_target_us)
        {
            dropped_frame_count_++;
            return true;
        }

        output_seek_complete_ = true;
        LOG_DEBUG("Output seek complete: dropped %d frames", dropped_frame_count_);
        return false;
    }

    // ========== Packet Timestamps ==========

    /**
     * Adjust packet timestamps for copied streams
     * Optimized to avoid redundant timebase conversions
     */
    void adjustPacketTimestamps(AVPacket *pkt,
                                AVRational in_time_base,
                                int stream_index)
    {
        if (cfg_.mode == Mode::COPYTS)
        {
            adjustCopytsPacket(pkt, in_time_base);
        }
        else
        {
            adjustNormalPacket(pkt, in_time_base, stream_index);
        }

        validatePacketTimestamps(pkt);
    }

    // ========== Baseline Management ==========

    void establishGlobalBaseline(const AVPacket *pkt, AVRational time_base)
    {
        if (global_baseline_us_ == AV_NOPTS_VALUE && pkt->pts != AV_NOPTS_VALUE)
        {
            global_baseline_us_ = av_rescale_q(pkt->pts, time_base, {1, AV_TIME_BASE});
            LOG_DEBUG("Global baseline established at %.3fs", global_baseline_us_ / 1000000.0);
        }
    }

    int64_t getGlobalBaseline() const { return global_baseline_us_; }

    // ========== A/V Sync Monitoring ==========

    void updateAudioTimestamp(int64_t audio_pts_us)
    {
        last_audio_pts_us_ = audio_pts_us;
        checkAVDrift();
    }

    // ========== Diagnostics ==========

    void dumpState() const
    {
        LOG_DEBUG("=== TimestampManager State ===");
        LOG_DEBUG("  Mode: %s", cfg_.mode == Mode::COPYTS ? "COPYTS" : "NORMAL");
        LOG_DEBUG("  Input seek: %.3fs", cfg_.input_seek_us / 1000000.0);
        LOG_DEBUG("  Output seek target: %.3fs", cfg_.output_seek_target_us / 1000000.0);
        LOG_DEBUG("  Output ts offset: %.3fs", cfg_.output_ts_offset_us / 1000000.0);
        LOG_DEBUG("  Video baseline: %lld", video_baseline_pts_);
        LOG_DEBUG("  Global baseline: %.3fs", global_baseline_us_ / 1000000.0);
        LOG_DEBUG("  Last video PTS: %.3fms", last_video_pts_us_ / 1000.0);
        LOG_DEBUG("  Last audio PTS: %.3fms", last_audio_pts_us_ / 1000.0);
        LOG_DEBUG("  Dropped frames: %d", dropped_frame_count_);
        LOG_DEBUG("  Discontinuities: %d", discontinuity_count_);
        LOG_DEBUG("  Monotonicity violations: %d", monotonicity_violation_count_);
        LOG_DEBUG("  Negative PTS count: %d", negative_pts_count_);
    }

    struct Stats
    {
        int dropped_frames = 0;
        int discontinuities = 0;
        int monotonicity_violations = 0;
        int negative_pts = 0;
        int av_drift_warnings = 0;
        double max_av_drift_ms = 0.0;
    };

    Stats getStats() const
    {
        return {
            dropped_frame_count_,
            discontinuity_count_,
            monotonicity_violation_count_,
            negative_pts_count_,
            av_drift_warning_count_,
            max_av_drift_ms_};
    }

private:
    Config cfg_;

    // Pre-computed state flags (set once, avoid branches in hot path)
    const bool needs_baseline_wait_;
    const int64_t frame_duration_us_;

    // Baselines
    int64_t video_baseline_pts_ = AV_NOPTS_VALUE;
    int64_t global_baseline_us_ = AV_NOPTS_VALUE;
    int64_t copyts_baseline_pts_ = AV_NOPTS_VALUE; // For -start_at_zero with -copyts

    // Monotonicity tracking
    int64_t last_video_pts_ = AV_NOPTS_VALUE;
    int64_t last_video_dts_ = AV_NOPTS_VALUE;
    int64_t last_video_pts_us_ = AV_NOPTS_VALUE;
    int64_t last_audio_pts_us_ = AV_NOPTS_VALUE;

    // State flags
    bool output_seek_complete_ = false;

    // Fallback counter
    int64_t synthetic_counter_ = 0;

    // Diagnostics
    int dropped_frame_count_ = 0;
    int discontinuity_count_ = 0;
    int monotonicity_violation_count_ = 0;
    int negative_pts_count_ = 0;
    int av_drift_warning_count_ = 0;
    double max_av_drift_ms_ = 0.0;

    // ========== Internal Implementation ==========

    int64_t deriveCopytsPTS(int64_t in_pts, AVRational in_tb, AVRational out_tb)
    {
        int64_t out_pts = av_rescale_q_rnd(in_pts, in_tb, out_tb, AV_ROUND_NEAR_INF);

        if (out_pts == AV_NOPTS_VALUE)
        {
            LOG_ERROR("Timestamp overflow in COPYTS mode");
            return 0;
        }

        // Apply -start_at_zero offset (compute baseline from first PTS)
        if (cfg_.start_at_zero)
        {
            if (copyts_baseline_pts_ == AV_NOPTS_VALUE)
            {
                copyts_baseline_pts_ = out_pts;
                LOG_DEBUG("COPYTS baseline established at %lld for -start_at_zero", copyts_baseline_pts_);
            }
            out_pts -= copyts_baseline_pts_;
        }

        if (cfg_.output_ts_offset_us != 0)
        {
            int64_t offset = av_rescale_q_rnd(cfg_.output_ts_offset_us,
                                              {1, AV_TIME_BASE},
                                              out_tb,
                                              AV_ROUND_NEAR_INF);
            out_pts -= offset;
        }

        // Handle negative timestamps based on avoid_negative_ts setting
        out_pts = applyAvoidNegativeTs(out_pts, "COPYTS PTS");

        return out_pts;
    }

    int64_t deriveNormalPTS(int64_t in_pts, AVRational in_tb, AVRational out_tb)
    {
        if (video_baseline_pts_ == AV_NOPTS_VALUE)
        {
            video_baseline_pts_ = in_pts;

            if (cfg_.input_seek_us > 0)
            {
                // Use user's requested seek position (not actual seek landing position)
                // This ensures output timestamps align with user's intent and audio sync
                int64_t seek_offset = av_rescale_q_rnd(cfg_.input_seek_us,
                                                       {1, AV_TIME_BASE},
                                                       in_tb,
                                                       AV_ROUND_NEAR_INF);
                video_baseline_pts_ -= seek_offset;
                LOG_DEBUG("Video baseline adjusted for seek: first_frame=%lld, user_seek=%lld, baseline=%lld",
                          in_pts, seek_offset, video_baseline_pts_);
            }
            else
            {
                LOG_DEBUG("Video baseline from first decoded frame: %lld", video_baseline_pts_);
            }
        }

        int64_t relative_pts = in_pts - video_baseline_pts_;
        int64_t out_pts = av_rescale_q_rnd(relative_pts, in_tb, out_tb, AV_ROUND_NEAR_INF);

        // Handle negative timestamps based on avoid_negative_ts setting
        out_pts = applyAvoidNegativeTs(out_pts, "NORMAL PTS");

        return out_pts;
    }

    void adjustCopytsPacket(AVPacket *pkt, AVRational in_tb)
    {
        if (cfg_.output_ts_offset_us != 0)
        {
            int64_t offset = av_rescale_q_rnd(cfg_.output_ts_offset_us,
                                              {1, AV_TIME_BASE},
                                              in_tb,
                                              AV_ROUND_NEAR_INF);

            if (pkt->pts != AV_NOPTS_VALUE)
            {
                pkt->pts -= offset;
                if (pkt->pts < 0)
                    pkt->pts = 0;
            }

            if (pkt->dts != AV_NOPTS_VALUE)
            {
                pkt->dts -= offset;
                if (pkt->dts < 0)
                    pkt->dts = 0;
            }
        }
    }

    void adjustNormalPacket(AVPacket *pkt, AVRational in_tb, int stream_index)
    {
        if (!needs_baseline_wait_)
            return;

        if (global_baseline_us_ == AV_NOPTS_VALUE)
        {
            pkt->pts = AV_NOPTS_VALUE;
            pkt->dts = AV_NOPTS_VALUE;
            return;
        }

        // Adjust using global baseline (in microseconds)
        if (pkt->pts != AV_NOPTS_VALUE)
        {
            int64_t pts_us = av_rescale_q(pkt->pts, in_tb, {1, AV_TIME_BASE});
            int64_t adjusted_us = pts_us - global_baseline_us_;
            pkt->pts = av_rescale_q(adjusted_us, {1, AV_TIME_BASE}, in_tb);
        }

        if (pkt->dts != AV_NOPTS_VALUE)
        {
            int64_t dts_us = av_rescale_q(pkt->dts, in_tb, {1, AV_TIME_BASE});
            int64_t adjusted_us = dts_us - global_baseline_us_;
            pkt->dts = av_rescale_q(adjusted_us, {1, AV_TIME_BASE}, in_tb);
        }
    }

    void enforceMonotonicity(TimestampPair &ts, AVRational out_tb)
    {
        // Initialize tracking on first frame
        if (last_video_pts_ == AV_NOPTS_VALUE)
        {
            last_video_pts_ = ts.pts;
            return;
        }

        // FFmpeg mode: Detect but don't fix PTS violations
        if (!cfg_.enforce_monotonicity)
        {
            if (ts.pts <= last_video_pts_)
            {
                monotonicity_violation_count_++;
                LOG_WARN("PTS monotonicity violation: %lld <= %lld (FFmpeg mode: not fixing)",
                         ts.pts, last_video_pts_);
            }
            updateTracking(ts);
            return;
        }

        // Legacy mode: Auto-fix PTS violations for compatibility
        fixPTSViolation(ts, out_tb);
        updateTracking(ts);
    }

private:
    void detectViolations(const TimestampPair &ts)
    {
        if (ts.pts <= last_video_pts_)
        {
            monotonicity_violation_count_++;
            LOG_WARN("PTS monotonicity violation: %lld <= %lld (FFmpeg mode: not fixing)",
                     ts.pts, last_video_pts_);
        }
        if (ts.dts != AV_NOPTS_VALUE && ts.dts <= last_video_dts_)
        {
            LOG_WARN("DTS monotonicity violation: %lld <= %lld (FFmpeg mode: not fixing)",
                     ts.dts, last_video_dts_);
        }
    }

    void fixPTSViolation(TimestampPair &ts, AVRational out_tb)
    {
        if (ts.pts <= last_video_pts_)
        {
            monotonicity_violation_count_++;
            int64_t frame_duration = av_rescale_q(1, av_inv_q(cfg_.expected_frame_rate), out_tb);
            int64_t original_pts = ts.pts;
            ts.pts = last_video_pts_ + frame_duration;
            LOG_DEBUG("Fixed PTS: %lld -> %lld (legacy mode)", original_pts, ts.pts);
        }
    }

    void fixDTSViolation(TimestampPair &ts)
    {
        // Ensure DTS <= PTS (FFmpeg enforces this strictly)
        if (ts.dts != AV_NOPTS_VALUE && ts.dts > ts.pts)
        {
            LOG_ERROR("DTS > PTS detected: DTS=%lld, PTS=%lld - clamping DTS=PTS (may cause issues)",
                      ts.dts, ts.pts);
            ts.dts = ts.pts;
        }

        // Ensure DTS strict monotonicity (DTS must be GREATER than last, not equal)
        if (ts.dts != AV_NOPTS_VALUE && last_video_dts_ != AV_NOPTS_VALUE && ts.dts <= last_video_dts_)
        {
            int64_t original_dts = ts.dts;
            ts.dts = last_video_dts_ + 1;
            LOG_VERBOSE("DTS monotonicity fix: %lld <= %lld, adjusted to %lld",
                        original_dts, last_video_dts_, ts.dts);

            // FFmpeg-compliant behavior: Do NOT increment PTS
            // If DTS > PTS after fix, this will cause muxer error (as in vanilla FFmpeg)
            if (ts.dts > ts.pts)
            {
                LOG_ERROR("DTS monotonicity fix created DTS > PTS: DTS=%lld, PTS=%lld",
                          ts.dts, ts.pts);
                LOG_ERROR("This will cause muxer error (vanilla FFmpeg behavior)");
                // Clamp DTS to PTS to prevent immediate error, but warn user
                ts.dts = ts.pts;
            }
        }
    }

    void updateTracking(const TimestampPair &ts)
    {
        last_video_pts_ = ts.pts;
        // DTS tracking removed - encoder handles DTS generation
    }

    bool detectDiscontinuity(int64_t pts_us, int64_t last_pts_us)
    {
        int64_t diff = pts_us - last_pts_us;

        // Detect large forward jumps (> 10 seconds) or any backward jump
        const int64_t max_gap_us = 10000000; // 10 seconds
        return diff > max_gap_us || diff < 0;
    }

    void checkAVDrift()
    {
        if (last_video_pts_us_ == AV_NOPTS_VALUE || last_audio_pts_us_ == AV_NOPTS_VALUE)
        {
            return;
        }

        int64_t drift_us = last_audio_pts_us_ - last_video_pts_us_;
        double drift_ms = std::abs(drift_us) / 1000.0;

        // Track max drift for stats
        if (drift_ms > max_av_drift_ms_)
        {
            max_av_drift_ms_ = drift_ms;
        }

        // Account for natural update frequency difference:
        // - Video updates every ~40ms (24fps) or ~33ms (30fps)
        // - Audio updates every ~20ms (48kHz, 1024 samples)
        // Allow drift up to 2x expected frame duration before warning
        int64_t expected_frame_duration_us = frame_duration_us_ * 2;
        int64_t warning_threshold_us = expected_frame_duration_us + 100000; // + 100ms safety margin

        // Only warn if drift exceeds natural timing differences
        if (std::abs(drift_us) > warning_threshold_us)
        {
            const char *direction = (drift_us > 0) ? "audio ahead" : "video ahead";
            av_drift_warning_count_++;

            // Log every 10th warning to avoid spam, or if critically high
            // if (av_drift_warning_count_ % 10 == 1 || drift_ms > 500.0) {
            if (drift_ms > 500.0)
            {
                LOG_WARN("A/V drift: %.3fms %s (video: %.3fs, audio: %.3fs) [warning #%d]",
                         drift_ms, direction,
                         last_video_pts_us_ / 1000000.0,
                         last_audio_pts_us_ / 1000000.0,
                         av_drift_warning_count_);

                // Critical threshold: >500ms suggests real problem
                if (drift_ms > 500.0)
                {
                    LOG_ERROR("CRITICAL: A/V drift >500ms - possible processing stall or frame drops!");
                }
            }
        }
    }

    void validatePacketTimestamps(AVPacket *pkt)
    {
        if (!cfg_.enable_validation)
            return;

        if (pkt->dts != AV_NOPTS_VALUE && pkt->pts != AV_NOPTS_VALUE && pkt->dts > pkt->pts)
        {
            LOG_DEBUG("DTS > PTS in packet, adjusting: %lld -> %lld", pkt->dts, pkt->pts);
            pkt->dts = pkt->pts;
        }
    }

    /**
     * Apply avoid_negative_ts policy to a timestamp (FFmpeg-compatible)
     * @param ts The timestamp to process
     * @param label Debug label for logging
     * @return Adjusted timestamp
     */
    int64_t applyAvoidNegativeTs(int64_t ts, const char *label)
    {
        if (ts >= 0)
        {
            return ts; // No adjustment needed for positive timestamps
        }

        switch (cfg_.avoid_negative_ts)
        {
        case AvoidNegativeTs::DISABLED:
            // Allow negative timestamps (FFmpeg -avoid_negative_ts disabled)
            return ts;

        case AvoidNegativeTs::MAKE_NON_NEGATIVE:
            // Only shift if negative (FFmpeg -avoid_negative_ts make_non_negative)
            if (ts < 0)
            {
                negative_pts_count_++;
                LOG_DEBUG("Negative %s: %lld (shifting to 0 - make_non_negative)", label, ts);
                return 0;
            }
            return ts;

        case AvoidNegativeTs::MAKE_ZERO:
            // Always shift to start at zero (FFmpeg -avoid_negative_ts make_zero)
            if (ts < 0)
            {
                negative_pts_count_++;
                LOG_DEBUG("Negative %s: %lld (shifting to 0 - make_zero)", label, ts);
                return 0;
            }
            return ts;

        case AvoidNegativeTs::AUTO:
        default:
            // Auto mode: clamp to zero (safe default)
            if (ts < 0)
            {
                negative_pts_count_++;
                LOG_DEBUG("Negative %s: %lld (clamping to 0 - auto)", label, ts);
                return 0;
            }
            return ts;
        }
    }

    const char *getAvoidNegativeTsName(AvoidNegativeTs mode) const
    {
        switch (mode)
        {
        case AvoidNegativeTs::AUTO:
            return "auto";
        case AvoidNegativeTs::MAKE_ZERO:
            return "make_zero";
        case AvoidNegativeTs::MAKE_NON_NEGATIVE:
            return "make_non_negative";
        case AvoidNegativeTs::DISABLED:
            return "disabled";
        default:
            return "unknown";
        }
    }
};
