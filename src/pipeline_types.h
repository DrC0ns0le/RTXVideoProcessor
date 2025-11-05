#pragma once

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavfilter/avfilter.h>
#include <libswresample/swresample.h>
#include <libavutil/audio_fifo.h>
}

#include <cstdint>
#include <string>
#include <vector>
#include <map>

struct InputOpenOptions
{
    std::string fflags;
    bool preferP010ForHDR = false; // Request P010 output for HDR content
    std::string seekTime;          // Seek time in FFmpeg format (e.g., "00:09:06.671")

    // Seek behavior flags (FFmpeg compatibility)
    bool noAccurateSeek = false; // Use AVSEEK_FLAG_ANY for fast seeking (-noaccurate_seek)
    bool seek2any = false;       // Allow seeking to non-keyframes (-seek2any)
    bool seekTimestamp = false;  // Enable/disable seeking by timestamp with -ss (when disabled, adds stream start_time to seek position)

    // Decoder error handling (FFmpeg compatibility)
    bool enableErrorConcealment = true; // Enable error concealment for incomplete frames (legacy default)
    bool flushOnSeek = false;           // Flush decoder buffers after seeking (FFmpeg does NOT do this)
};

struct HlsMuxOptions
{
    bool enabled = false;
    bool overwrite = true; // by default, overwrite the output file
    int maxDelay = -1;
    int hlsTime = -1;
    std::string segmentType;
    std::string initFilename;
    int64_t startNumber = -1;
    std::string segmentFilename;
    std::string playlistType;
    int listSize = -1;
    bool autoDiscontinuity = true; // Automatically mark discontinuity on seek (legacy default, FFmpeg doesn't do this)
    std::string customFlags;       // User-specified HLS flags via -hls_flags (e.g., "independent_segments")
    std::string segmentOptions;    // Options to pass to segment muxer via hls_segment_options (e.g., "movflags=+frag_discont")
};

// FFmpeg-compatible timestamp handling modes
enum class AvoidNegativeTs
{
    AUTO,              // FFmpeg default: auto (make_non_negative for MOV, make_zero for others)
    MAKE_ZERO,         // Shift timestamps to start at zero
    MAKE_NON_NEGATIVE, // Only shift if negative
    DISABLED           // Allow negative timestamps (may break some muxers)
};

struct InputContext
{
    AVFormatContext *fmt = nullptr;
    int vstream = -1;
    AVStream *vst = nullptr;
    AVCodecContext *vdec = nullptr;

    // Audio input - multi-stream support
    std::map<int, AVCodecContext*> audio_decoders;  // stream_index -> decoder context
    int primary_audio_stream = -1;  // "Best" audio stream for info/logging

    AVBufferRef *hw_device_ctx = nullptr; // CUDA device

    // Seek offset tracking for A/V sync
    int64_t seek_offset_us = 0; // Seek offset in microseconds
};

struct AudioConfig
{
    std::string codec;
    int channels = -1;
    int bitrate = -1;
    int sampleRate = -1;
    std::string filter;
    bool enabled = false;
    bool copyts = false; // Preserve original timestamps
    bool applyToAllAudioStreams = false; // true for -codec:a, false for -codec:a:0
};

// Per-stream audio encoder context
struct AudioEncoderContext
{
    int input_stream_index = -1;           // Which input stream this encodes
    AVStream *output_stream = nullptr;     // Output stream
    AVCodecContext *encoder = nullptr;     // Encoder context
    SwrContext *resampler = nullptr;       // Resampler for format conversion
    AVAudioFifo *fifo = nullptr;           // Sample buffering for fixed frame sizes
    AVFilterGraph *filter_graph = nullptr; // Audio filter graph
    AVFilterContext *buffersrc = nullptr;  // Filter input
    AVFilterContext *buffersink = nullptr; // Filter output
    int64_t accumulated_samples = 0;       // Sample counter for PTS calculation
    int64_t start_pts = AV_NOPTS_VALUE;    // First audio PTS for baseline
};

// Stream mapping decision for each input stream
enum class StreamMapDecision
{
    EXCLUDE,       // Stream should not be included in output
    COPY,          // Stream should be copied to output
    PROCESS_VIDEO, // Video stream to be processed (encoded)
    PROCESS_AUDIO  // Audio stream to be processed (re-encoded)
};

// Stream mapping specification (parsed from -map arguments)
struct StreamMapSpec
{
    int input_file_index = -1;     // Input file index (e.g., 0 from "-map 0:1")
    int stream_index = -1;         // Stream index within file (e.g., 1 from "-map 0:1", -1 for "any")
    AVMediaType stream_type = AVMEDIA_TYPE_UNKNOWN; // Stream type filter (e.g., VIDEO from "-map 0:v")
    int stream_type_index = -1;    // Index within stream type (e.g., 0 from "-map 0:v:0", -1 for "all")
    bool is_negative = false;      // true for "-map -0:1" (exclude stream)
    bool is_optional = false;      // true for "-map 0:v?" (don't error if missing)

    // Metadata filtering (e.g., "0:m:language:eng")
    bool has_metadata_filter = false;
    std::string metadata_key;      // e.g., "language" from "0:m:language:eng"
    std::string metadata_value;    // e.g., "eng" from "0:m:language:eng"

    std::string raw_specifier;     // Original stream specifier string for debugging
};

// Resolved mapping for a specific output stream
struct MappedStream
{
    int input_index = -1;          // Which input file (index into input contexts vector)
    int input_stream_index = -1;   // Which stream in that input file
    AVMediaType type = AVMEDIA_TYPE_UNKNOWN; // Stream media type
    bool requires_processing = false; // true if stream needs encoding/processing
    bool copy_stream = false;      // true for pass-through copy (subtitles, data, etc.)
};

struct OutputContext
{
    AVFormatContext *fmt = nullptr;
    AVStream *vstream = nullptr;
    AVCodecContext *venc = nullptr;

    // Audio processing - multi-stream only
    AudioConfig audioConfig;
    std::map<int, AudioEncoderContext> audio_encoders;  // input_stream_index -> encoder context
    
    int64_t output_seek_target_us = 0; // Output seeking target for reference
    bool hls_mode = false; // HLS fMP4 output mode
    bool audio_output_seek_complete = false; // Track when audio has passed output seek point

    // Stream mapping: final decision for each input stream
    std::vector<StreamMapDecision> stream_decisions;
    // Mapping from input stream index to output stream index (-1 if not mapped)
    std::vector<int> input_to_output_map;

    // Multi-input stream mapping
    std::vector<MappedStream> mapped_streams;        // All output streams (video/audio/subtitle/data)
    std::vector<AVStream*> all_output_streams;       // Corresponding AVStream* for each mapped stream
    std::vector<AVCodecContext*> passthrough_codecs; // Decoder contexts for copy streams (subtitles, etc.)

    HlsMuxOptions hlsOptions;
    AVDictionary *muxOptions = nullptr;

    // DTS monotonicity tracking for video packets
    int64_t last_video_dts = AV_NOPTS_VALUE;
};
