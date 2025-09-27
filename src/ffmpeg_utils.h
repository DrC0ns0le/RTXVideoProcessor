#pragma once

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

#include <stdexcept>
#include <string>

inline void ff_check(int err, const char *what)
{
    if (err < 0)
    {
        char buf[256];
        av_strerror(err, buf, sizeof(buf));
        throw std::runtime_error(std::string(what) + ": " + buf);
    }
}

inline std::string ff_ts(double seconds)
{
    char b[64];
    snprintf(b, sizeof(b), "%.3fs", seconds);
    return b;
}
