#pragma once

#include "config_parser.h"
#include "ffmpeg_utils.h"
#include <string>

// Output detection helpers
bool is_pipe_output(const char *path);
bool will_use_hls_output(const PipelineConfig &cfg);

// HLS configuration
void finalize_hls_options(PipelineConfig *cfg, OutputContext *out);

// Video encoder configuration
AVBufferRef *configure_video_encoder(PipelineConfig &cfg, InputContext &in, OutputContext &out,
                                     bool inputIsHDR, bool use_cuda_path, int dstW, int dstH,
                                     const AVRational &fr, bool hls_enabled, bool mux_is_isobmff);

// Stream metadata configuration
void configure_stream_metadata(InputContext &in, OutputContext &out, PipelineConfig &cfg,
                               bool inputIsHDR, bool mux_is_isobmff, bool hls_enabled);
