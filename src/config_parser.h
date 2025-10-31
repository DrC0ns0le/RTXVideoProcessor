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

    char *inputPath = nullptr; // DEPRECATED: Use inputPaths for multi-input support
    char *outputPath = nullptr;

    // Multi-input support
    std::vector<std::string> inputPaths;  // Multiple input files (-i flag can be repeated)

    // NVENC settings
    std::string tune;
    std::string preset;
    std::string rc; // cbr, vbr, constqp

    int gop; // keyframe interval, multiple of seconds (--nvenc-gop)
    int gopFrames = -1; // GOP size in frames (-g), takes precedence over gop if set
    int bframes;
    int qp;

    // Advanced GOP/keyframe control
    int scThreshold = -1;     // Scene change threshold (-sc_threshold)
    int keyintMin = -1;       // Minimum GOP length (-keyint_min)
    bool noScenecut = false;  // Disable scene detection (-no-scenecut)
    bool forcedIdr = false;   // Force IDR frames (-forced-idr)

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
    std::string hlsFlags;          // HLS muxer flags (e.g., "independent_segments", "delete_segments")
    std::string hlsSegmentOptions; // Options to pass to segment muxer (e.g., "movflags=+frag_discont")

    // Stream mapping options
    std::vector<std::string> streamMaps;  // Raw -map arguments (e.g., "0:0", "1:a", "-0:v")

    // Stream disable flags (FFmpeg -vn, -an, -sn, -dn)
    bool disableVideo = false;    // -vn: Disable video streams
    bool disableAudio = false;    // -an: Disable audio streams
    bool disableSubtitle = false; // -sn: Disable subtitle streams
    bool disableData = false;     // -dn: Disable data streams

    // Metadata and chapter mapping (Jellyfin compatibility)
    int mapMetadata = 0;          // -map_metadata: -1 = disable, 0+ = input index
    bool hasMapMetadata = false;  // Track if explicitly set (vs default)
    int mapChapters = 0;          // -map_chapters: -1 = disable, 0+ = input index
    bool hasMapChapters = false;  // Track if explicitly set (vs default)

    // Audio codec options
    std::string audioCodec;
    bool audioCodecApplyToAll = false; // true for -codec:a, false for -codec:a:0
    int audioChannels = -1;
    int audioBitrate = -1;
    int audioSampleRate = -1;
    std::string audioFilter;

    // Seek and duration options
    std::string seekTime;        // Input seeking (-ss before -i)
    std::string duration;        // Duration limit (-t)
    std::string outputSeekTime;  // Output seeking (-ss after -i)
    std::string outputTsOffset;  // Output timestamp offset (-output_ts_offset)
    bool copyts = false;         // Preserve original timestamps (-copyts)
    bool noAccurateSeek = false; // Fast seek to nearest keyframe (-noaccurate_seek)
    bool seek2any = false;       // Allow seeking to non-keyframes (-seek2any)
    bool seekTimestamp = false;  // Use timestamp-based seeking (-seek_timestamp)

    // Timestamp handling options (FFmpeg compatibility)
    std::string avoidNegativeTs = "auto"; // FFmpeg -avoid_negative_ts (auto/make_zero/make_non_negative/disabled)
    bool startAtZero = false;             // FFmpeg -start_at_zero
    std::string vsync;                    // FFmpeg -vsync (passthrough/cfr/vfr/drop/auto) - default: auto
    std::string outputFrameRate;          // Output framerate override (-r, -r:v) - overrides input framerate

    // Muxer options (essential - affect output structure/playback)
    std::string movflags;        // User-specified movflags (-movflags)
    int64_t fragDuration = 0;    // Fragment duration in microseconds (-frag_duration)
    int fragmentIndex = -1;      // Fragment index number (-fragment_index)
    int useEditlist = -1;        // Use edit list in MP4 (-use_editlist)
    int maxMuxingQueueSize = -1; // Max muxing queue size (-max_muxing_queue_size)
};

// Print usage/help information
void print_help(const char *argv0);

// Parse command-line arguments and initialize configuration
// Returns true if parsing succeeded, false if help was shown or error occurred
void parse_arguments(int argc, char **argv, PipelineConfig *cfg);
