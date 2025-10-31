#pragma once

#include "pipeline_types.h"
#include "config_parser.h"
#include <vector>
#include <string>

extern "C"
{
#include <libavformat/avformat.h>
}

// Parse a single -map argument string into a StreamMapSpec
// Returns true on success, false on parse error
bool parse_map_spec(const std::string &map_arg, StreamMapSpec &spec);

// Resolve stream mapping from parsed specs and input contexts
// This implements FFmpeg's stream selection algorithm
std::vector<MappedStream> resolve_stream_mapping(
    const std::vector<InputContext> &inputs,
    const PipelineConfig &cfg);

// Helper: Check if a stream matches a stream specifier
bool stream_matches_spec(
    const AVStream *stream,
    int stream_index,
    const StreamMapSpec &spec);

// Helper: Apply default FFmpeg stream selection (when no -map is specified)
void apply_default_mapping(
    const std::vector<InputContext> &inputs,
    const PipelineConfig &cfg,
    std::vector<MappedStream> &mapped_streams);
