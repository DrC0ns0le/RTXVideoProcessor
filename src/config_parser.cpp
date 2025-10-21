#include "config_parser.h"
#include "logger.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <system_error>

// Helper function: strip file: prefix and quotes from paths
static char *extract_ffmpeg_file_path(char *value)
{
    if (!value)
        return value;

    // Preserve pipe: prefix for stdout/stdin handling
    if (std::strncmp(value, "pipe:", 5) == 0)
        return value;

    // Strip file: prefix if present
    if (std::strncmp(value, "file:", 5) == 0)
        value += 5;

    size_t len = std::strlen(value);
    if (len >= 2)
    {
        char first = value[0];
        char last = value[len - 1];
        if ((first == '"' && last == '"') || (first == '\'' && last == '\''))
        {
            value[len - 1] = '\0';
            ++value;
        }
    }

    return value;
}

// Helper function: check if string ends with suffix
static bool endsWith(const std::string &str, const std::string &suffix)
{
    if (suffix.size() > str.size())
        return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Helper function: convert string to lowercase
static std::string lowercase_copy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Helper function: get environment variable as string
static const char *get_env_var(const char *name)
{
    return std::getenv(name);
}

// Helper function: get environment variable as integer with default
static int get_env_int(const char *name, int default_value)
{
    const char *value = std::getenv(name);
    if (!value)
        return default_value;
    try
    {
        return std::stoi(value);
    }
    catch (...)
    {
        fprintf(stderr, "Warning: Invalid integer value for %s: %s (using default: %d)\n", name, value, default_value);
        return default_value;
    }
}

// Helper function: get environment variable as boolean (1/true/yes = true, 0/false/no = false)
static bool get_env_bool(const char *name, bool default_value)
{
    const char *value = std::getenv(name);
    if (!value)
        return default_value;
    std::string lower = lowercase_copy(value);
    if (lower == "1" || lower == "true" || lower == "yes")
        return true;
    if (lower == "0" || lower == "false" || lower == "no")
        return false;
    fprintf(stderr, "Warning: Invalid boolean value for %s: %s (using default: %s)\n",
            name, value, default_value ? "true" : "false");
    return default_value;
}

// Helper function: check if output is a pipe/stdout
static bool is_pipe_output(const char *path)
{
    if (!path)
        return false;
    return (std::strcmp(path, "-") == 0) ||
           (std::strcmp(path, "pipe:") == 0) ||
           (std::strcmp(path, "pipe:1") == 0) ||
           (std::strncmp(path, "pipe:", 5) == 0);
}

void print_help(const char *argv0)
{
    fprintf(stderr, "RTXVideoProcessor build %s\n", BUILD_VERSION);
    fprintf(stderr, "Usage: %s input output.{mp4|mkv|m3u8|-} [options]\n", argv0);
    fprintf(stderr, "\nInput can be:\n");
    fprintf(stderr, "  - Local file: input.mp4, input.mkv\n");
    fprintf(stderr, "  - HTTP/HTTPS URL: http://example.com/video.mp4\n");
    fprintf(stderr, "  - RTMP/RTSP stream: rtmp://server/stream\n");
    fprintf(stderr, "\nOutput can be:\n");
    fprintf(stderr, "  - Local file: output.mp4, output.mkv, output.m3u8\n");
    fprintf(stderr, "  - Stdout pipe: - or pipe:1\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -v, --verbose Enable verbose logging\n");
    fprintf(stderr, "  -d, --debug Enable debug logging\n");
    fprintf(stderr, "  --cpu         Bypass GPU for video processing pipeline other than RTX processing\n");
    fprintf(stderr, "\nVSR options:\n");
    fprintf(stderr, "  --no-vsr      Disable VSR (env: RTX_NO_VSR=1)\n");
    fprintf(stderr, "  --vsr-quality     Set VSR quality, default 4 (env: RTX_VSR_QUALITY)\n");
    fprintf(stderr, "\nTHDR options:\n");
    fprintf(stderr, "  --no-thdr     Disable THDR (env: RTX_NO_THDR=1)\n");
    fprintf(stderr, "  --thdr-contrast   Set THDR contrast, default 115 (env: RTX_THDR_CONTRAST)\n");
    fprintf(stderr, "  --thdr-saturation Set THDR saturation, default 75 (env: RTX_THDR_SATURATION)\n");
    fprintf(stderr, "  --thdr-middle-gray Set THDR middle gray, default 30 (env: RTX_THDR_MIDDLE_GRAY)\n");
    fprintf(stderr, "  --thdr-max-luminance Set THDR max luminance, default 1000 (env: RTX_THDR_MAX_LUMINANCE)\n");
    fprintf(stderr, "\nNVENC options:\n");
    fprintf(stderr, "  --nvenc-tune        Set NVENC tune, default hq (env: RTX_NVENC_TUNE)\n");
    fprintf(stderr, "  --nvenc-preset      Set NVENC preset, default p7 (env: RTX_NVENC_PRESET)\n");
    fprintf(stderr, "  --nvenc-rc          Set NVENC rate control, default constqp (env: RTX_NVENC_RC)\n");
    fprintf(stderr, "  --nvenc-gop         Set NVENC GOP (seconds), default 3 (env: RTX_NVENC_GOP)\n");
    fprintf(stderr, "  --nvenc-bframes     Set NVENC bframes, default 2 (env: RTX_NVENC_BFRAMES)\n");
    fprintf(stderr, "  --nvenc-qp          Set NVENC QP, default 21 (env: RTX_NVENC_QP)\n");
    fprintf(stderr, "  --nvenc-bitrate-multiplier Set NVENC bitrate multiplier, default 2 (env: RTX_NVENC_BITRATE_MULTIPLIER)\n");
    fprintf(stderr, "\nEnvironment variables can be used to set defaults. Command-line flags override environment variables.\n");
    fprintf(stderr, "\nHLS options (detected automatically for .m3u8 outputs):\n");
    fprintf(stderr, "  -hls_time <seconds>             Set target segment duration (default 4)\n");
    fprintf(stderr, "  -hls_segment_type <mpegts|fmp4> Select segment container (default fmp4)\n");
    fprintf(stderr, "  -hls_segment_filename <pattern> Segment naming pattern (auto-generated)\n");
    fprintf(stderr, "  -hls_fmp4_init_filename <file>  Initialization segment path for fMP4\n");
    fprintf(stderr, "  -start_number <n>               Starting segment number (default 0)\n");
    fprintf(stderr, "  -hls_playlist_type <type>       Playlist type (event, vod, live)\n");
    fprintf(stderr, "  -hls_list_size <count>          Playlist size (0 = keep all segments)\n");
    fprintf(stderr, "  -hls_flags <flags>              HLS muxer flags (e.g., independent_segments, delete_segments)\n");
    fprintf(stderr, "  -hls_segment_options <opts>     Options to pass to segment muxer (e.g., movflags=+frag_discont)\n");
}

// Parse arguments in FFmpeg-compatible mode (-i input -f format output)
static void parse_compatibility_mode(int argc, char **argv, PipelineConfig *cfg)
{
    // Enable verbose logging
    cfg->verbose = true;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-fflags")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-fflags requires an argument\n");
                exit(1);
            }
            cfg->fflags = argv[++i];
        }
        else if (arg == "-y")
        {
            cfg->overwrite = true;
        }
        else if (arg == "-f")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-f requires an argument\n");
                exit(1);
            }
            if (cfg->inputFormatName.empty())
            {
                cfg->inputFormatName = argv[++i];
            }
            else
            {
                cfg->outputFormatName = argv[++i];
            }
        }
        else if (arg == "-i")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-i requires an input path\n");
                exit(1);
            }
            cfg->inputPath = extract_ffmpeg_file_path(argv[++i]);
        }
        else if (arg == "-max_delay")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-max_delay requires a value\n");
                exit(1);
            }
            const char *value = argv[++i];
            try
            {
                cfg->maxDelay = std::stoi(value);
            }
            catch (...)
            {
                fprintf(stderr, "Invalid value for -max_delay: %s\n", value);
                exit(1);
            }
        }
        else if (arg == "-hls_time")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-hls_time requires a value\n");
                exit(1);
            }
            const char *value = argv[++i];
            try
            {
                cfg->hlsTime = std::stoi(value);
            }
            catch (...)
            {
                fprintf(stderr, "Invalid value for -hls_time: %s\n", value);
                exit(1);
            }
        }
        else if (arg == "-hls_segment_type")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-hls_segment_type requires a value\n");
                exit(1);
            }
            cfg->hlsSegmentType = extract_ffmpeg_file_path(argv[++i]);
        }
        else if (arg == "-hls_fmp4_init_filename")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-hls_fmp4_init_filename requires a value\n");
                exit(1);
            }
            cfg->hlsInitFilename = extract_ffmpeg_file_path(argv[++i]);
        }
        else if (arg == "-start_number")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-start_number requires a value\n");
                exit(1);
            }
            const char *value = argv[++i];
            try
            {
                cfg->hlsStartNumber = std::stoll(value);
            }
            catch (...)
            {
                fprintf(stderr, "Invalid value for -start_number: %s\n", value);
                exit(1);
            }
        }
        else if (arg == "-hls_segment_filename")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-hls_segment_filename requires a value\n");
                exit(1);
            }
            cfg->hlsSegmentFilename = extract_ffmpeg_file_path(argv[++i]);
        }
        else if (arg == "-hls_playlist_type")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-hls_playlist_type requires a value\n");
                exit(1);
            }
            cfg->hlsPlaylistType = argv[++i];
        }
        else if (arg == "-hls_list_size")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-hls_list_size requires a value\n");
                exit(1);
            }
            const char *value = argv[++i];
            try
            {
                cfg->hlsListSize = std::stoi(value);
            }
            catch (...)
            {
                fprintf(stderr, "Invalid value for -hls_list_size: %s\n", value);
                exit(1);
            }
        }
        else if (arg == "-hls_flags")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-hls_flags requires a value\n");
                exit(1);
            }
            cfg->hlsFlags = argv[++i];
        }
        else if (arg == "-hls_segment_options")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-hls_segment_options requires a value\n");
                exit(1);
            }
            cfg->hlsSegmentOptions = argv[++i];
        }
        else if (arg == "-map")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-map requires an argument\n");
                exit(1);
            }
            cfg->streamMaps.push_back(argv[++i]);
        }
        else if (arg.substr(0, 7) == "-codec:" || arg.substr(0, 3) == "-c:")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "%s requires an argument\n", arg.c_str());
                exit(1);
            }
            if (arg == "-codec:a:0" || arg == "-c:a:0" || arg == "-codec:a" || arg == "-c:a")
            {
                cfg->audioCodec = argv[++i];
            }
        }
        else if (arg == "-ac")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-ac requires an argument\n");
                exit(1);
            }
            const char *value = argv[++i];
            try
            {
                cfg->audioChannels = std::stoi(value);
            }
            catch (...)
            {
                fprintf(stderr, "Invalid value for -ac: %s\n", value);
                exit(1);
            }
        }
        else if (arg == "-ab")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-ab requires an argument\n");
                exit(1);
            }
            const char *value = argv[++i];
            try
            {
                cfg->audioBitrate = std::stoi(value);
            }
            catch (...)
            {
                fprintf(stderr, "Invalid value for -ab: %s\n", value);
                exit(1);
            }
        }
        else if (arg == "-ar" || arg == "-ar:a")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-ar requires an argument\n");
                exit(1);
            }
            const char *value = argv[++i];
            try
            {
                cfg->audioSampleRate = std::stoi(value);
            }
            catch (...)
            {
                fprintf(stderr, "Invalid value for -ar: %s\n", value);
                exit(1);
            }
        }
        else if (arg == "-af")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-af requires an argument\n");
                exit(1);
            }
            cfg->audioFilter = argv[++i];
        }
        else if (arg == "-ss")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-ss requires a time value\n");
                exit(1);
            }
            // Determine if this is input seeking or output seeking based on context
            // If we haven't seen -i yet, it's input seeking
            // If we have seen -i, it's output seeking
            if (cfg->inputPath == nullptr)
            {
                cfg->seekTime = argv[++i];
            }
            else
            {
                cfg->outputSeekTime = argv[++i];
            }
        }
        else if (arg == "-t")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-t requires a time value\n");
                exit(1);
            }
            cfg->duration = argv[++i];
        }
        else if (arg == "-copyts")
        {
            cfg->copyts = true;
        }
        else if (arg == "-start_at_zero")
        {
            cfg->startAtZero = true;
        }
        else if (arg == "-avoid_negative_ts")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-avoid_negative_ts requires a value (auto/make_zero/make_non_negative/disabled)\n");
                exit(1);
            }
            cfg->avoidNegativeTs = argv[++i];
        }
        else if (arg == "-output_ts_offset")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-output_ts_offset requires a time value\n");
                exit(1);
            }
            cfg->outputTsOffset = argv[++i];
        }
        else if (arg == "-noaccurate_seek")
        {
            cfg->noAccurateSeek = true;
        }
        else if (arg == "-seek2any")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-seek2any requires a value\n");
                exit(1);
            }
            cfg->seek2any = (std::stoi(argv[++i]) != 0);
        }
        else if (arg == "-seek_timestamp")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-seek_timestamp requires a value\n");
                exit(1);
            }
            cfg->seekTimestamp = (std::stoi(argv[++i]) != 0);
        }
        else if (arg == "-movflags")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-movflags requires flags\n");
                exit(1);
            }
            cfg->movflags = argv[++i];
        }
        else if (arg == "-frag_duration")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-frag_duration requires a value\n");
                exit(1);
            }
            cfg->fragDuration = std::stoll(argv[++i]);
        }
        else if (arg == "-fragment_index")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-fragment_index requires a value\n");
                exit(1);
            }
            cfg->fragmentIndex = std::stoi(argv[++i]);
        }
        else if (arg == "-use_editlist")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-use_editlist requires a value\n");
                exit(1);
            }
            cfg->useEditlist = std::stoi(argv[++i]);
        }
        else if (arg == "-max_muxing_queue_size")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-max_muxing_queue_size requires a value\n");
                exit(1);
            }
            cfg->maxMuxingQueueSize = std::stoi(argv[++i]);
        }
        // RTX VSR flags
        else if (arg == "--no-vsr")
        {
            cfg->rtxCfg.enableVSR = false;
        }
        else if (arg == "--vsr-quality")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "--vsr-quality requires an argument\n");
                exit(1);
            }
            cfg->rtxCfg.vsrQuality = std::stoi(argv[++i]);
        }
        // RTX THDR flags
        else if (arg == "--no-thdr")
        {
            if (!cfg->rtxCfg.enableVSR)
                LOG_WARN("Both VSR & THDR are disabled, bypassing RTX evaluate");
            cfg->rtxCfg.enableTHDR = false;
        }
        else if (arg == "--thdr-contrast")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "--thdr-contrast requires an argument\n");
                exit(1);
            }
            cfg->rtxCfg.thdrContrast = std::stoi(argv[++i]);
        }
        else if (arg == "--thdr-saturation")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "--thdr-saturation requires an argument\n");
                exit(1);
            }
            cfg->rtxCfg.thdrSaturation = std::stoi(argv[++i]);
        }
        else if (arg == "--thdr-middle-gray")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "--thdr-middle-gray requires an argument\n");
                exit(1);
            }
            cfg->rtxCfg.thdrMiddleGray = std::stoi(argv[++i]);
        }
        else if (arg == "--thdr-max-luminance")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "--thdr-max-luminance requires an argument\n");
                exit(1);
            }
            cfg->rtxCfg.thdrMaxLuminance = std::stoi(argv[++i]);
        }
        // Detect output path: file extensions or pipe/stdout
        if (endsWith(arg, ".m3u8") || endsWith(arg, ".mp4") || endsWith(arg, ".mkv") ||
            arg == "-" || arg == "pipe:" || arg == "pipe:1")
        {
            cfg->outputPath = argv[i];
            LOG_DEBUG("Set outputPath = '%s'\n", cfg->outputPath);
        }
    }
}

// Parse arguments in simple mode (input output [options])
static void parse_simple_mode(int argc, char **argv, PipelineConfig *cfg)
{
    int i = 1;
    cfg->inputPath = argv[i++];
    cfg->outputPath = argv[i++];

    for (; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v")
            cfg->verbose = true;
        else if (arg == "--debug" || arg == "-d")
            cfg->debug = true;
        else if (arg == "--cpu" || arg == "-cpu")
            cfg->cpuOnly = true;
        else if (arg == "--help" || arg == "-h")
        {
            print_help(argv[0]);
            exit(0);
        }

        // HLS / output format options
        else if (arg == "-hls_time")
        {
            if (i + 1 < argc)
            {
                const char *value = argv[++i];
                try
                {
                    cfg->hlsTime = std::stoi(value);
                }
                catch (...)
                {
                    fprintf(stderr, "Invalid value for -hls_time: %s\n", value);
                    exit(1);
                }
            }
            else
            {
                fprintf(stderr, "Missing argument for -hls_time\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "-hls_segment_type")
        {
            if (i + 1 < argc)
            {
                cfg->hlsSegmentType = extract_ffmpeg_file_path(argv[++i]);
            }
            else
            {
                fprintf(stderr, "Missing argument for -hls_segment_type\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "-hls_fmp4_init_filename")
        {
            if (i + 1 < argc)
            {
                cfg->hlsInitFilename = extract_ffmpeg_file_path(argv[++i]);
            }
            else
            {
                fprintf(stderr, "Missing argument for -hls_fmp4_init_filename\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "-start_number")
        {
            if (i + 1 < argc)
            {
                const char *value = argv[++i];
                try
                {
                    cfg->hlsStartNumber = std::stoll(value);
                }
                catch (...)
                {
                    fprintf(stderr, "Invalid value for -start_number: %s\n", value);
                    exit(1);
                }
            }
            else
            {
                fprintf(stderr, "Missing argument for -start_number\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "-hls_segment_filename")
        {
            if (i + 1 < argc)
            {
                cfg->hlsSegmentFilename = extract_ffmpeg_file_path(argv[++i]);
            }
            else
            {
                fprintf(stderr, "Missing argument for -hls_segment_filename\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "-hls_playlist_type")
        {
            if (i + 1 < argc)
            {
                cfg->hlsPlaylistType = argv[++i];
            }
            else
            {
                fprintf(stderr, "Missing argument for -hls_playlist_type\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "-hls_list_size")
        {
            if (i + 1 < argc)
            {
                const char *value = argv[++i];
                try
                {
                    cfg->hlsListSize = std::stoi(value);
                }
                catch (...)
                {
                    fprintf(stderr, "Invalid value for -hls_list_size: %s\n", value);
                    exit(1);
                }
            }
            else
            {
                fprintf(stderr, "Missing argument for -hls_list_size\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "-hls_flags")
        {
            if (i + 1 < argc)
            {
                cfg->hlsFlags = argv[++i];
            }
            else
            {
                fprintf(stderr, "Missing argument for -hls_flags\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "-hls_segment_options")
        {
            if (i + 1 < argc)
            {
                cfg->hlsSegmentOptions = argv[++i];
            }
            else
            {
                fprintf(stderr, "Missing argument for -hls_segment_options\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "-max_delay")
        {
            if (i + 1 < argc)
            {
                const char *value = argv[++i];
                try
                {
                    cfg->maxDelay = std::stoi(value);
                }
                catch (...)
                {
                    fprintf(stderr, "Invalid value for -max_delay: %s\n", value);
                    exit(1);
                }
            }
            else
            {
                fprintf(stderr, "Missing argument for -max_delay\n");
                print_help(argv[0]);
                exit(1);
            }
        }

        // VSR
        else if (arg == "--no-vsr")
            cfg->rtxCfg.enableVSR = false;
        else if (arg == "--vsr-quality")
        {
            if (i + 1 < argc)
            {
                cfg->rtxCfg.vsrQuality = std::stoi(argv[++i]);
            }
            else
            {
                fprintf(stderr, "Missing argument for --vsr-quality\n");
                print_help(argv[0]);
                exit(1);
            }
        }

        // THDR
        else if (arg == "--no-thdr")
        {
            if (!cfg->rtxCfg.enableVSR)
                LOG_WARN("Both VSR & THDR are disabled, bypassing RTX evaluate");
            cfg->rtxCfg.enableTHDR = false;
        }
        else if (arg == "--thdr-contrast")
        {
            if (i + 1 < argc)
            {
                cfg->rtxCfg.thdrContrast = std::stoi(argv[++i]);
            }
            else
            {
                fprintf(stderr, "Missing argument for --thdr-contrast\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "--thdr-saturation")
        {
            if (i + 1 < argc)
            {
                cfg->rtxCfg.thdrSaturation = std::stoi(argv[++i]);
            }
            else
            {
                fprintf(stderr, "Missing argument for --thdr-saturation\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "--thdr-middle-gray")
        {
            if (i + 1 < argc)
            {
                cfg->rtxCfg.thdrMiddleGray = std::stoi(argv[++i]);
            }
            else
            {
                fprintf(stderr, "Missing argument for --thdr-middle-gray\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "--thdr-max-luminance")
        {
            if (i + 1 < argc)
            {
                cfg->rtxCfg.thdrMaxLuminance = std::stoi(argv[++i]);
            }
            else
            {
                fprintf(stderr, "Missing argument for --thdr-max-luminance\n");
                print_help(argv[0]);
                exit(1);
            }
        }

        // NVENC
        else if (arg == "--nvenc-tune")
        {
            if (i + 1 < argc)
            {
                cfg->tune = argv[++i];
            }
            else
            {
                fprintf(stderr, "Missing argument for --nvenc-tune\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "--nvenc-preset")
        {
            if (i + 1 < argc)
            {
                cfg->preset = argv[++i];
            }
            else
            {
                fprintf(stderr, "Missing argument for --nvenc-preset\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "--nvenc-rc")
        {
            if (i + 1 < argc)
            {
                cfg->rc = argv[++i];
            }
            else
            {
                fprintf(stderr, "Missing argument for --nvenc-rc\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "--nvenc-gop")
        {
            if (i + 1 < argc)
            {
                cfg->gop = std::stoi(argv[++i]);
            }
            else
            {
                fprintf(stderr, "Missing argument for --nvenc-gop\n");
                print_help(argv[0]);
                exit(1);
            }
        }
        else if (arg == "--nvenc-bframes")
        {
            if (i + 1 < argc)
            {
                cfg->bframes = std::stoi(argv[++i]);
            }
            else
            {
                fprintf(stderr, "Missing argument for --nvenc-bframes\n");
                print_help(argv[0]);
                exit(1);
            }
        }

        else
        {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_help(argv[0]);
            exit(1);
        }
    }
}

void parse_arguments(int argc, char **argv, PipelineConfig *cfg)
{
    if (argc < 3)
    {
        print_help(argv[0]);
        exit(1);
    }

    // Set default values (with environment variable overrides)
    // Command-line flags will override these
    cfg->rtxCfg.enableVSR = !get_env_bool("RTX_NO_VSR", false);
    cfg->rtxCfg.scaleFactor = 2;
    cfg->rtxCfg.vsrQuality = get_env_int("RTX_VSR_QUALITY", 4);

    cfg->rtxCfg.enableTHDR = !get_env_bool("RTX_NO_THDR", false);
    cfg->rtxCfg.thdrContrast = get_env_int("RTX_THDR_CONTRAST", 115);
    cfg->rtxCfg.thdrSaturation = get_env_int("RTX_THDR_SATURATION", 75);
    cfg->rtxCfg.thdrMiddleGray = get_env_int("RTX_THDR_MIDDLE_GRAY", 30);
    cfg->rtxCfg.thdrMaxLuminance = get_env_int("RTX_THDR_MAX_LUMINANCE", 1000);

    const char *env_tune = get_env_var("RTX_NVENC_TUNE");
    cfg->tune = env_tune ? env_tune : "hq";
    const char *env_preset = get_env_var("RTX_NVENC_PRESET");
    cfg->preset = env_preset ? env_preset : "p7";
    const char *env_rc = get_env_var("RTX_NVENC_RC");
    cfg->rc = env_rc ? env_rc : "constqp";
    cfg->gop = get_env_int("RTX_NVENC_GOP", 3);
    cfg->bframes = get_env_int("RTX_NVENC_BFRAMES", 2);
    cfg->qp = get_env_int("RTX_NVENC_QP", 21);
    cfg->targetBitrateMultiplier = get_env_int("RTX_NVENC_BITRATE_MULTIPLIER", 2);

    // Determine parsing mode: simple (input output [opts]) vs FFmpeg-compatible (-i input -f format output)
    // Simple mode: first arg is input file (positional)
    // FFmpeg mode: uses -i flag for input
    bool uses_input_flag = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-i") == 0)
        {
            uses_input_flag = true;
            break;
        }
    }

    if (uses_input_flag)
    {
        // FFmpeg-compatible mode (uses -i flag)
        cfg->ffCompatible = true;
        parse_compatibility_mode(argc, argv, cfg);
    }
    else
    {
        // Simple mode (positional input/output)
        parse_simple_mode(argc, argv, cfg);
    }
}
