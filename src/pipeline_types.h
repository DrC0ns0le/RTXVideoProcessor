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
    bool copyts = false; // Preserve original timestamps
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
    int64_t a_start_pts = AV_NOPTS_VALUE; // First audio PTS for baseline
    // Track last emitted audio DTS (in output stream time_base) to ensure strict monotonicity
    int64_t last_audio_dts = AV_NOPTS_VALUE;
    // Track actual accumulated samples for precise PTS calculation (prevents resampling drift)
    int64_t accumulated_audio_samples = 0;

    // Shared timestamp baseline for HLS A/V sync
    // When HLS+COPYTS+output seeking is used, video and audio must use the same baseline
    // to ensure tfdt values are aligned. Set by video TimestampManager, used by audio.
    int64_t copyts_baseline_pts = AV_NOPTS_VALUE; // In microseconds
    int64_t output_seek_target_us = 0; // Output seeking target for reference
    bool hls_mode = false; // HLS fMP4 output mode
    bool audio_output_seek_complete = false; // Track when audio has passed output seek point

    // Stream mapping: final decision for each input stream
    std::vector<StreamMapDecision> stream_decisions;
    // Mapping from input stream index to output stream index (-1 if not mapped)
    std::vector<int> input_to_output_map;
    HlsMuxOptions hlsOptions;
    AVDictionary *muxOptions = nullptr;
};
