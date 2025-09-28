#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
}

#include <cstdint>
#include <string>
#include <vector>

struct InputOpenOptions
{
    std::string fflags;
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
    AVBufferRef *hw_device_ctx = nullptr; // CUDA device
};

struct OutputContext
{
    AVFormatContext *fmt = nullptr;
    AVStream *vstream = nullptr;
    AVCodecContext *venc = nullptr;
    std::vector<int> map_streams; // input->output map, -1 for unmapped
    HlsMuxOptions hlsOptions;
    AVDictionary *muxOptions = nullptr;
};
