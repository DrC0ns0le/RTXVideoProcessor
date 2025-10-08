#pragma once

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/rational.h>
#include <libavformat/avformat.h>
}

#include <cstdint>

// Derive zero-based output PTS in encoder time_base from an input decoded frame.
// v_start_pts is updated on first valid input PTS to establish a zero baseline.
// Falls back to a synthetic counter if no timestamps are available.
// seek_offset_us: seek offset in microseconds to account for seeking
static inline int64_t derive_output_pts(int64_t &v_start_pts,
                                        const AVFrame *decframe,
                                        AVRational in_time_base,
                                        AVRational out_time_base,
                                        int64_t seek_offset_us = 0)
{
    int64_t in_pts = (decframe->pts != AV_NOPTS_VALUE)
                         ? decframe->pts
                         : decframe->best_effort_timestamp;
    // Establish a zero-based timeline from the first valid input PTS
    if (in_pts != AV_NOPTS_VALUE)
    {
        if (v_start_pts == AV_NOPTS_VALUE)
        {
            v_start_pts = in_pts;
            // When seeking, adjust the baseline to account for the seek offset
            if (seek_offset_us > 0)
            {
                int64_t seek_offset_in_timebase = av_rescale_q(seek_offset_us, {1, AV_TIME_BASE}, in_time_base);
                v_start_pts -= seek_offset_in_timebase;
            }
        }
        in_pts -= v_start_pts;
    }
    static int64_t synthetic_counter = 0;
    int64_t out_pts = (in_pts != AV_NOPTS_VALUE)
                          ? av_rescale_q(in_pts, in_time_base, out_time_base)
                          : synthetic_counter++;
    return out_pts;
}

// Helper function to derive zero-based PTS for copied streams
static inline int64_t derive_copied_stream_pts(int64_t &stream_start_pts,
                                               int64_t packet_pts,
                                               int64_t seek_offset_us,
                                               AVRational stream_time_base)
{
    if (packet_pts == AV_NOPTS_VALUE) {
        return AV_NOPTS_VALUE;
    }

    // Establish baseline from first packet
    if (stream_start_pts == AV_NOPTS_VALUE) {
        stream_start_pts = packet_pts;
        // When seeking, adjust the baseline to account for the seek offset
        if (seek_offset_us > 0) {
            int64_t seek_offset_in_timebase = av_rescale_q(seek_offset_us, {1, AV_TIME_BASE}, stream_time_base);
            stream_start_pts -= seek_offset_in_timebase;
        }
    }

    // Return timestamp relative to baseline
    return packet_pts - stream_start_pts;
}

// Helper function to derive PTS using global minimum baseline
static inline int64_t derive_global_baseline_pts(int64_t packet_pts,
                                                 int64_t global_min_pts_us,
                                                 AVRational stream_time_base)
{
    if (packet_pts == AV_NOPTS_VALUE || global_min_pts_us == AV_NOPTS_VALUE) {
        return packet_pts; // No adjustment if no baseline established
    }

    // Convert packet PTS to microseconds
    int64_t packet_pts_us = av_rescale_q(packet_pts, stream_time_base, {1, AV_TIME_BASE});

    // Subtract global minimum to make all streams start from 0
    int64_t adjusted_pts_us = packet_pts_us - global_min_pts_us;

    // Convert back to stream timebase
    return av_rescale_q(adjusted_pts_us, {1, AV_TIME_BASE}, stream_time_base);
}

// Derive output PTS for -copyts mode (preserves original timestamps)
// This function does NOT subtract v_start_pts, preserving the original timeline
// output_ts_offset_us: optional offset to SUBTRACT from timestamps (-output_ts_offset)
static inline int64_t derive_copyts_pts(const AVFrame *decframe,
                                        AVRational in_time_base,
                                        AVRational out_time_base,
                                        int64_t output_ts_offset_us = 0)
{
    int64_t in_pts = (decframe->pts != AV_NOPTS_VALUE)
                         ? decframe->pts
                         : decframe->best_effort_timestamp;

    if (in_pts == AV_NOPTS_VALUE) {
        return AV_NOPTS_VALUE;
    }

    // Convert to output timebase without baseline adjustment (copyts behavior)
    int64_t out_pts = av_rescale_q(in_pts, in_time_base, out_time_base);

    // Apply output timestamp offset if specified
    // NOTE: output_ts_offset SUBTRACTS from timestamps to shift timeline backward
    // Example: frame at 424s with offset 424 → output at 0s (424 - 424 = 0)
    if (output_ts_offset_us != 0) {
        int64_t offset_in_out_tb = av_rescale_q(output_ts_offset_us, {1, AV_TIME_BASE}, out_time_base);
        out_pts -= offset_in_out_tb;  // Subtract to shift timeline backward

        // Edge case: Prevent negative timestamps (can happen if offset > actual PTS)
        // FFmpeg clamps to 0 in this case
        if (out_pts < 0) {
            out_pts = 0;
        }
    }

    return out_pts;
}

// Derive output PTS for -copyts mode for copied packets (no re-encoding)
static inline int64_t derive_copyts_packet_pts(int64_t packet_pts,
                                                AVRational in_time_base,
                                                AVRational out_time_base,
                                                int64_t output_ts_offset_us = 0)
{
    if (packet_pts == AV_NOPTS_VALUE) {
        return AV_NOPTS_VALUE;
    }

    // Convert to output timebase without baseline adjustment (copyts behavior)
    int64_t out_pts = av_rescale_q(packet_pts, in_time_base, out_time_base);

    // Apply output timestamp offset if specified (subtracts to shift timeline backward)
    if (output_ts_offset_us != 0) {
        int64_t offset_in_out_tb = av_rescale_q(output_ts_offset_us, {1, AV_TIME_BASE}, out_time_base);
        out_pts -= offset_in_out_tb;  // Subtract to shift timeline backward

        // Edge case: Prevent negative timestamps (can happen if offset > actual PTS)
        // FFmpeg clamps to 0 in this case
        if (out_pts < 0) {
            out_pts = 0;
        }
    }

    return out_pts;
}
