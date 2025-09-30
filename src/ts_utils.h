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
