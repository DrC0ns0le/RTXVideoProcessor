#pragma once

#include "rtx_processor.h"
#include <string>
#include <vector>
#include <cstdint>

// Configuration structure for the entire processing pipeline
struct PipelineConfig
{
    bool verbose = false;
    bool debug = false;
    bool cpuOnly = false;
    bool ffCompatible = false;

    char *inputPath = nullptr;
    char *outputPath = nullptr;

    // NVENC settings
    std::string tune;
    std::string preset;
    std::string rc; // cbr, vbr, constqp

    int gop; // keyframe interval, multiple of seconds
    int bframes;
    int qp;

    int targetBitrateMultiplier;

    RTXProcessConfig rtxCfg;

    std::string inputFormatName;
    std::string outputFormatName;
    std::string fflags;

    bool overwrite = true;

    // HLS options
    int maxDelay = -1;
    int hlsTime = -1;
    std::string hlsSegmentType;
    std::string hlsInitFilename;
    int64_t hlsStartNumber = -1;
    std::string hlsSegmentFilename;
    std::string hlsPlaylistType;
    int hlsListSize = -1;

    // Stream mapping options
    std::vector<std::string> streamMaps;

    // Audio codec options
    std::string audioCodec;
    int audioChannels = -1;
    int audioBitrate = -1;
    int audioSampleRate = -1;
    std::string audioFilter;

    // Seek options
    std::string seekTime;          // Input seeking (-ss before -i)
    std::string outputSeekTime;    // Output seeking (-ss after -i)
    std::string outputTsOffset;    // Output timestamp offset (-output_ts_offset)
    bool copyts = false;           // Preserve original timestamps (-copyts)
    bool noAccurateSeek = false;   // Fast seek to nearest keyframe (-noaccurate_seek)
    bool seek2any = false;         // Allow seeking to non-keyframes (-seek2any)
    bool seekTimestamp = false;    // Use timestamp-based seeking (-seek_timestamp)

    // Timestamp handling options (FFmpeg compatibility)
    std::string avoidNegativeTs = "auto";  // FFmpeg -avoid_negative_ts (auto/make_zero/make_non_negative/disabled)
    bool startAtZero = false;              // FFmpeg -start_at_zero

    // Muxer options (essential - affect output structure/playback)
    std::string movflags;          // User-specified movflags (-movflags)
    int64_t fragDuration = 0;      // Fragment duration in microseconds (-frag_duration)
    int fragmentIndex = -1;        // Fragment index number (-fragment_index)
    int useEditlist = -1;          // Use edit list in MP4 (-use_editlist)
    int maxMuxingQueueSize = -1;   // Max muxing queue size (-max_muxing_queue_size)
};

// Print usage/help information
void print_help(const char *argv0);

// Parse command-line arguments and initialize configuration
// Returns true if parsing succeeded, false if help was shown or error occurred
void parse_arguments(int argc, char **argv, PipelineConfig *cfg);
