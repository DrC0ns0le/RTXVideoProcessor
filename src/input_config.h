#pragma once

#include "config_parser.h"
#include "ffmpeg_utils.h"

// Input detection helpers
bool is_network_input(const char *input);

// Input processing configuration
bool configure_input_hdr_detection(PipelineConfig &cfg, InputContext &in);
void configure_vsr_auto_disable(PipelineConfig &cfg, const InputContext &in);
void configure_audio_processing(PipelineConfig &cfg, InputContext &in, OutputContext &out);

// Progress tracking setup
int64_t setup_progress_tracking(const InputContext &in, const AVRational &fr);
