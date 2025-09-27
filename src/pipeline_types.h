#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
}

#include <vector>

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
};
