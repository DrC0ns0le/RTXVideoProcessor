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
static inline int64_t derive_output_pts(int64_t &v_start_pts,
                                        const AVFrame *decframe,
                                        AVRational in_time_base,
                                        AVRational out_time_base)
{
    int64_t in_pts = (decframe->pts != AV_NOPTS_VALUE)
                         ? decframe->pts
                         : decframe->best_effort_timestamp;
    // Establish a zero-based timeline from the first valid input PTS
    if (in_pts != AV_NOPTS_VALUE)
    {
        if (v_start_pts == AV_NOPTS_VALUE)
            v_start_pts = in_pts;
        in_pts -= v_start_pts;
    }
    static int64_t synthetic_counter = 0;
    int64_t out_pts = (in_pts != AV_NOPTS_VALUE)
                          ? av_rescale_q(in_pts, in_time_base, out_time_base)
                          : synthetic_counter++;
    return out_pts;
}
