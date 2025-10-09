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

struct InputOpenOptions
{
    std::string fflags;
    bool preferP010ForHDR = false; // Request P010 output for HDR content
    std::string seekTime;          // Seek time in FFmpeg format (e.g., "00:09:06.671")

    // Seek behavior flags (FFmpeg compatibility)
    bool noAccurateSeek = false; // Use AVSEEK_FLAG_ANY for fast seeking (-noaccurate_seek)
    bool seek2any = false;       // Allow seeking to non-keyframes (-seek2any)
    bool seekTimestamp = false;  // Use timestamp-based seeking (-seek_timestamp)

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

    // Audio input
    int astream = -1;
    AVStream *ast = nullptr;
    AVCodecContext *adec = nullptr;

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
};

// Stream mapping decision for each input stream
enum class StreamMapDecision
{
    EXCLUDE,       // Stream should not be included in output
    COPY,          // Stream should be copied to output
    PROCESS_VIDEO, // Video stream to be processed (encoded)
    PROCESS_AUDIO  // Audio stream to be processed (re-encoded)
};

struct OutputContext
{
    AVFormatContext *fmt = nullptr;
    AVStream *vstream = nullptr;
    AVCodecContext *venc = nullptr;

    // Audio processing
    AVStream *astream = nullptr;
    AVCodecContext *aenc = nullptr;
    AudioConfig audioConfig;

    // Audio filtering
    AVFilterGraph *filter_graph = nullptr;
    AVFilterContext *buffersrc_ctx = nullptr;
    AVFilterContext *buffersink_ctx = nullptr;

    // Audio resampling
    SwrContext *swr_ctx = nullptr;

    // Audio buffering for fixed frame sizes
    AVAudioFifo *audio_fifo = nullptr;
    int64_t next_audio_pts = 0;
    int64_t last_audio_dts = AV_NOPTS_VALUE;
    int64_t a_start_pts = AV_NOPTS_VALUE; // First audio PTS for baseline

    // Video DTS tracking for monotonicity enforcement
    int64_t last_video_dts = AV_NOPTS_VALUE;

    // Stream mapping: final decision for each input stream
    std::vector<StreamMapDecision> stream_decisions;
    // Mapping from input stream index to output stream index (-1 if not mapped)
    std::vector<int> input_to_output_map;
    HlsMuxOptions hlsOptions;
    AVDictionary *muxOptions = nullptr;

    // FFmpeg-compatible timestamp options
    AvoidNegativeTs avoidNegativeTs = AvoidNegativeTs::AUTO;
    bool startAtZero = false;
};
