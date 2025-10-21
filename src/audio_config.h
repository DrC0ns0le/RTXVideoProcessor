#pragma once

#include <string>
#include <vector>

struct AudioParameters
{
    std::string codec;
    int channels = -1;
    int bitrate = -1;
    int sampleRate = -1;
    std::string filter;
    std::vector<std::string> streamMaps;
};