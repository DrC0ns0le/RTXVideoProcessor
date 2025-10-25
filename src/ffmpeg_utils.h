#pragma once

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

#include <stdexcept>
#include <string>

#include "pipeline_types.h"
#include "audio_config.h"

inline void ff_check(int err, const char *what)
{
    if (err < 0)
    {
        char buf[256];
        av_strerror(err, buf, sizeof(buf));
        throw std::runtime_error(std::string(what) + ": " + buf);
    }
}

inline std::string ff_ts(double seconds)
{
    char b[64];
    snprintf(b, sizeof(b), "%.3fs", seconds);
    return b;
}

// Attach HDR mastering metadata and content light level side data to a video stream
void add_mastering_and_cll(AVStream *st, int max_luminance_nits);

// Open input and locate video stream, prepare decoder. Tries to enable CUDA device.
bool open_input(const char *inPath, InputContext &in, const InputOpenOptions *options = nullptr);
void close_input(InputContext &in);

// Open output, create video encoder stream and map non-video streams.
bool open_output(const char *outPath, const InputContext &in, OutputContext &out, const std::vector<std::string> &streamMaps = {}, const std::string &outputFormatName = "");
void close_output(OutputContext &out);

// Audio configuration functions
void configure_audio_from_params(const AudioParameters &params, OutputContext &out);
bool apply_stream_mappings(const std::vector<std::string> &mappings, const InputContext &in, OutputContext &out);
bool setup_audio_encoder(const InputContext &in, OutputContext &out);
bool setup_audio_filter(const InputContext &in, OutputContext &out);
bool process_audio_frame(AVFrame *input_frame, OutputContext &out);

// Initialize audio PTS tracking after seeking
void init_audio_pts_after_seek(const InputContext &in, OutputContext &out, int64_t global_baseline_pts_us = AV_NOPTS_VALUE);
