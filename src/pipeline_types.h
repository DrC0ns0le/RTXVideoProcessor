#pragma once

extern "C" {
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
    std::string seekTime; // Seek time in FFmpeg format (e.g., "00:09:06.671")
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
    std::string filter;
    bool enabled = false;
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

    std::vector<int> map_streams; // input->output map, -1 for unmapped
    std::vector<std::string> streamMappings; // parsed -map arguments
    HlsMuxOptions hlsOptions;
    AVDictionary *muxOptions = nullptr;
};
