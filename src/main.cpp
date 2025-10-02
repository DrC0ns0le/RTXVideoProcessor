#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <queue>
#include <filesystem>
#include <cctype>
#include <cstring>
#include <system_error>
#include <cerrno>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

extern "C"
{
#include <libavfilter/buffersrc.h>
#include <libavfilter/buffersink.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavcodec/packet.h>
#include <libavutil/common.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/display.h>
#include <libavutil/rational.h>
#include <libavutil/parseutils.h>
#include <libswscale/swscale.h>
}

#include <cuda_runtime.h>

#include "ffmpeg_utils.h"
#include "rtx_processor.h"
#include "frame_pool.h"
#include "ts_utils.h"
#include "timestamp_manager.h"
#include "processor.h"
#include "logger.h"
#include "audio_config.h"
#include "async_demuxer.h"

// Compatibility for older FFmpeg versions
#ifndef AV_FRAME_FLAG_KEY
#define AV_FRAME_FLAG_KEY (1 << 0)
#endif

// RAII deleters for FFmpeg types that require ** double-pointer frees
static inline void av_frame_free_single(AVFrame *f)
{
    if (f)
        av_frame_free(&f);
}
static inline void av_packet_free_single(AVPacket *p)
{
    if (p)
        av_packet_free(&p);
}

bool endsWith(const std::string &str, const std::string &suffix)
{
    if (suffix.size() > str.size())
        return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

#ifdef _WIN32
static std::string find_ffmpeg_on_path()
{
    const char *path_env = std::getenv("PATH");
    if (!path_env)
        return {};

    const std::string exe_name = "ffmpeg.exe";
    std::string path_str(path_env);
    size_t start = 0;
    while (start <= path_str.size())
    {
        size_t end = path_str.find(';', start);
        std::string dir = path_str.substr(start, end == std::string::npos ? std::string::npos : end - start);

        dir.erase(std::remove(dir.begin(), dir.end(), '"'), dir.end());
        auto first = dir.find_first_not_of(" \t");
        if (first == std::string::npos)
        {
            dir.clear();
        }
        else
        {
            auto last = dir.find_last_not_of(" \t");
            dir = dir.substr(first, last - first + 1);
        }

        if (!dir.empty() && dir != "." && dir != ".\\" && dir != "./")
        {
            std::filesystem::path candidate = std::filesystem::path(dir) / exe_name;
            std::error_code ec;
            if (std::filesystem::exists(candidate, ec) && !std::filesystem::is_directory(candidate, ec))
            {
                return candidate.string();
            }
        }

        if (end == std::string::npos)
            break;
        start = end + 1;
    }

    return {};
}

static std::string quote_windows_arg(const std::string &arg)
{
    if (arg.empty())
        return "\"\"";

    bool needs_quotes = arg.find_first_of(" \t\"") != std::string::npos;
    if (!needs_quotes)
        return arg;

    std::string result;
    result.reserve(arg.size() + 2);
    result.push_back('"');

    size_t backslash_count = 0;
    for (char ch : arg)
    {
        if (ch == '\\')
        {
            ++backslash_count;
        }
        else if (ch == '"')
        {
            result.append(backslash_count * 2 + 1, '\\');
            result.push_back('"');
            backslash_count = 0;
        }
        else
        {
            if (backslash_count > 0)
            {
                result.append(backslash_count, '\\');
                backslash_count = 0;
            }
            result.push_back(ch);
        }
    }

    if (backslash_count > 0)
    {
        result.append(backslash_count * 2, '\\');
    }

    result.push_back('"');
    return result;
}

static std::string format_windows_error(DWORD error_code)
{
    if (error_code == 0)
        return "";

    LPSTR buffer = nullptr;
    DWORD size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        error_code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buffer),
        0,
        nullptr);

    std::string message;
    if (size != 0 && buffer)
    {
        message.assign(buffer, size);
        while (!message.empty() && (message.back() == '\r' || message.back() == '\n'))
            message.pop_back();
    }
    else
    {
        message = "Unknown error";
    }

    if (buffer)
        LocalFree(buffer);

    return message;
}
#endif

// Helper to check if input is a network URL
static bool is_network_input(const char* input)
{
    if (!input) return false;
    return (std::strncmp(input, "http://", 7) == 0 ||
            std::strncmp(input, "https://", 8) == 0 ||
            std::strncmp(input, "rtmp://", 7) == 0 ||
            std::strncmp(input, "rtsp://", 7) == 0 ||
            std::strncmp(input, "tcp://", 6) == 0 ||
            std::strncmp(input, "udp://", 6) == 0);
}

static bool passthrough_required(int argc, char **argv)
{
    if (argc <= 0)
        return false;

    bool binary_is_ffmpeg = endsWith(argv[0], "ffmpeg") || endsWith(argv[0], "ffmpeg.exe");

    if (!binary_is_ffmpeg)
        return false;

    // Check if input is a network URL (we support these now via FFmpeg's network protocols)
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc)
        {
            // Network inputs are supported via FFmpeg's built-in protocols
            // No passthrough needed just because of network input
            break;
        }
    }

    bool input_looks_supported = (argc > 1) && (endsWith(argv[1], ".mp4") || endsWith(argv[1], ".mkv"));
    bool output_looks_supported = (argc > 2) && (endsWith(argv[2], ".mp4") || endsWith(argv[2], ".mkv") || endsWith(argv[2], ".m3u8"));

    bool requests_hls_muxer = false;
    bool requests_mp4_to_pipe = false;
    bool requests_video_exclusion = false;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-f") == 0 && i + 1 < argc)
        {
            if (std::strcmp(argv[i + 1], "hls") == 0)
            {
                requests_hls_muxer = true;
            }
            else if (std::strcmp(argv[i + 1], "mp4") == 0)
            {
                // Check if output is pipe
                for (int j = i + 2; j < argc; ++j)
                {
                    if (std::strcmp(argv[j], "-") == 0 ||
                        std::strcmp(argv[j], "pipe:") == 0 ||
                        std::strcmp(argv[j], "pipe:1") == 0)
                    {
                        requests_mp4_to_pipe = true;
                        break;
                    }
                }
            }
        }
        else if (std::strcmp(argv[i], "-map") == 0 && i + 1 < argc && std::strcmp(argv[i + 1], "-0:v?") == 0)
        {
            requests_video_exclusion = true;
        }
    }

    // Passthrough is needed if either input and output are not supported and if hls or mp4-to-pipe is not requested
    // or if excluding video

    return (!input_looks_supported && !output_looks_supported && !requests_hls_muxer && !requests_mp4_to_pipe) || requests_video_exclusion;
}

static int run_ffmpeg_passthrough(int argc, char **argv)
{
    fprintf(stderr, "Running ffmpeg passthrough\n");
    std::string ffmpeg_binary;
#ifdef _WIN32
    bool use_absolute_ffmpeg = false;
    if (auto resolved = find_ffmpeg_on_path(); !resolved.empty())
    {
        ffmpeg_binary = std::move(resolved);
        use_absolute_ffmpeg = true;
    }
    else
    {
        ffmpeg_binary = "ffmpeg";
    }
#else
    ffmpeg_binary = "ffmpeg";
#endif

    std::vector<std::string> forwarded_args_storage;
    forwarded_args_storage.reserve(static_cast<size_t>(argc));
    forwarded_args_storage.emplace_back(ffmpeg_binary);
    for (int i = 1; i < argc; ++i)
    {
        forwarded_args_storage.emplace_back(argv[i] ? argv[i] : "");
    }

#ifdef _WIN32
    std::string command_line;
    command_line.reserve(256);
    for (size_t i = 0; i < forwarded_args_storage.size(); ++i)
    {
        if (i > 0)
            command_line.push_back(' ');
        command_line.append(quote_windows_arg(forwarded_args_storage[i]));
    }

    std::vector<char> command_line_buffer(command_line.begin(), command_line.end());
    command_line_buffer.push_back('\0');

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);

    // Inherit stdin/stdout/stderr for pipe support
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

    const char *application_name = use_absolute_ffmpeg ? forwarded_args_storage[0].c_str() : nullptr;

    BOOL success = CreateProcessA(
        application_name,
        command_line_buffer.data(),
        nullptr,
        nullptr,
        TRUE,  // bInheritHandles = TRUE to inherit stdout/stderr/stdin
        0,
        nullptr,
        nullptr,
        &si,
        &pi);

    if (!success)
    {
        DWORD err = GetLastError();
        std::string err_msg = format_windows_error(err);
        fprintf(stderr, "Failed to launch ffmpeg passthrough binary '%s': %s (0x%08lX)\n",
                forwarded_args_storage[0].c_str(), err_msg.c_str(), static_cast<unsigned long>(err));
        return err ? static_cast<int>(err) : 1;
    }

    DWORD wait_result = WaitForSingleObject(pi.hProcess, INFINITE);
    if (wait_result == WAIT_FAILED)
    {
        DWORD err = GetLastError();
        std::string err_msg = format_windows_error(err);
        fprintf(stderr, "Failed waiting for ffmpeg passthrough process: %s (0x%08lX)\n",
                err_msg.c_str(), static_cast<unsigned long>(err));
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return err ? static_cast<int>(err) : 1;
    }

    DWORD exit_code = 0;
    if (!GetExitCodeProcess(pi.hProcess, &exit_code))
    {
        DWORD err = GetLastError();
        std::string err_msg = format_windows_error(err);
        fprintf(stderr, "Failed to retrieve ffmpeg passthrough exit code: %s (0x%08lX)\n",
                err_msg.c_str(), static_cast<unsigned long>(err));
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return err ? static_cast<int>(err) : 1;
    }

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    return static_cast<int>(exit_code);
#else
    std::vector<char *> passthrough_argv;
    passthrough_argv.reserve(forwarded_args_storage.size() + 1);
    for (auto &stored : forwarded_args_storage)
    {
        passthrough_argv.push_back(const_cast<char *>(stored.c_str()));
    }
    passthrough_argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0)
    {
        fprintf(stderr, "Failed to fork for ffmpeg passthrough: %s\n", std::strerror(errno));
        return errno ? errno : 1;
    }
    if (pid == 0)
    {
        execvp(ffmpeg_binary.c_str(), passthrough_argv.data());
        fprintf(stderr, "Failed to exec ffmpeg passthrough binary '%s': %s\n", ffmpeg_binary.c_str(), std::strerror(errno));
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0)
    {
        fprintf(stderr, "Failed to wait for ffmpeg passthrough process: %s\n", std::strerror(errno));
        return errno ? errno : 1;
    }

    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    if (WIFSIGNALED(status))
        return 128 + WTERMSIG(status);
    return 1;
#endif
}

using FramePtr = std::unique_ptr<AVFrame, void (*)(AVFrame *)>;
using PacketPtr = std::unique_ptr<AVPacket, void (*)(AVPacket *)>;

// Pipeline types are provided by pipeline_types.h via ffmpeg_utils.h

// Unified helper to send a frame to the encoder and interleaved-write all produced packets
static inline void encode_and_write(AVCodecContext *enc,
                                    AVStream *vstream,
                                    AVFormatContext *ofmt,
                                    AVFrame *frame,
                                    PacketPtr &opkt,
                                    const char *ctx_label)
{
    ff_check(avcodec_send_frame(enc, frame), ctx_label);
    while (true)
    {
        int ret = avcodec_receive_packet(enc, opkt.get());
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;
        ff_check(ret, "receive encoded packet");
        opkt->stream_index = vstream->index;

        av_packet_rescale_ts(opkt.get(), enc->time_base, vstream->time_base);
        ff_check(av_interleaved_write_frame(ofmt, opkt.get()), "write video packet");
        av_packet_unref(opkt.get());
    }
}

// Ensure SWS context converts from source frame format to RGBA with proper colorspace
static inline void ensure_sws_to_argb(SwsContext *&sws_to_argb,
                                      int &last_src_format,
                                      int srcW, int srcH,
                                      AVPixelFormat srcFmt,
                                      AVColorSpace colorspace)
{
    if (!sws_to_argb || last_src_format != srcFmt)
    {
        if (sws_to_argb)
            sws_freeContext(sws_to_argb);
        sws_to_argb = sws_getContext(
            srcW, srcH, srcFmt,
            srcW, srcH, AV_PIX_FMT_RGBA,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_to_argb)
            throw std::runtime_error("sws_to_argb alloc failed");
        const int *coeffs = (colorspace == AVCOL_SPC_BT2020_NCL)
                                ? sws_getCoefficients(SWS_CS_BT2020)
                                : sws_getCoefficients(SWS_CS_ITU709);
        sws_setColorspaceDetails(sws_to_argb, coeffs, 0, coeffs, 1, 0, 1 << 16, 1 << 16);
        last_src_format = srcFmt;
    }
}

// Ensure CPU RTX is initialized and process the BGRA frame
static inline void ensure_rtx_cpu_and_process(RTXProcessor &rtx_cpu,
                                              bool &rtx_cpu_init,
                                              const RTXProcessConfig &rtxCfg,
                                              int srcW, int srcH,
                                              const uint8_t *inBGRA, size_t inPitch,
                                              const uint8_t *&outData, uint32_t &outW, uint32_t &outH, size_t &outPitch)
{
    if (!rtx_cpu_init)
    {
        if (!rtx_cpu.initialize(0, rtxCfg, srcW, srcH))
        {
            std::string detail = rtx_cpu.lastError();
            if (detail.empty())
                detail = "unknown error";
            throw std::runtime_error(std::string("Failed to initialize RTX CPU path: ") + detail);
        }
        rtx_cpu_init = true;
    }
    if (!rtx_cpu.process(inBGRA, inPitch, outData, outW, outH, outPitch))
        throw std::runtime_error("RTX CPU processing failed");
}

struct PipelineConfig
{
    bool verbose = false;
    bool debug = false;
    bool cpuOnly = false;
    bool ffCompatible = false;

    char *inputPath = nullptr;
    char *outputPath = nullptr;

    // NVENC settings
    std::string tune;
    std::string preset;
    std::string rc; // cbr, vbr, constqp

    int gop; // keyframe interval, multiple of seconds
    int bframes;
    int qp;

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

    // Stream mapping options
    std::vector<std::string> streamMaps;

    // Audio codec options
    std::string audioCodec;
    int audioChannels = -1;
    int audioBitrate = -1;
    int audioSampleRate = -1;
    std::string audioFilter;

    // Seek options
    std::string seekTime;          // Input seeking (-ss before -i)
    std::string outputSeekTime;    // Output seeking (-ss after -i)
    std::string outputTsOffset;    // Output timestamp offset (-output_ts_offset)
    bool copyts = false;           // Preserve original timestamps (-copyts)
    bool noAccurateSeek = false;   // Fast seek to nearest keyframe (-noaccurate_seek)
    bool seek2any = false;         // Allow seeking to non-keyframes (-seek2any)
    bool seekTimestamp = false;    // Use timestamp-based seeking (-seek_timestamp)

    // Timestamp handling options (FFmpeg compatibility)
    std::string avoidNegativeTs = "auto";  // FFmpeg -avoid_negative_ts (auto/make_zero/make_non_negative/disabled)
    bool startAtZero = false;              // FFmpeg -start_at_zero

    // Muxer options (essential - affect output structure/playback)
    std::string movflags;          // User-specified movflags (-movflags)
    int64_t fragDuration = 0;      // Fragment duration in microseconds (-frag_duration)
    int fragmentIndex = -1;        // Fragment index number (-fragment_index)
    int useEditlist = -1;          // Use edit list in MP4 (-use_editlist)
    int maxMuxingQueueSize = -1;   // Max muxing queue size (-max_muxing_queue_size)
};

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

static std::string lowercase_copy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Helper to check if output is a pipe/stdout
static bool is_pipe_output(const char* path)
{
    if (!path) return false;
    return (std::strcmp(path, "-") == 0) ||
           (std::strcmp(path, "pipe:") == 0) ||
           (std::strcmp(path, "pipe:1") == 0) ||
           (std::strncmp(path, "pipe:", 5) == 0);
}

// Helper to check if HLS output will be used
static bool will_use_hls_output(const PipelineConfig& cfg)
{
    const bool format_requests_hls = !cfg.outputFormatName.empty() && lowercase_copy(cfg.outputFormatName) == "hls";

    if (cfg.outputPath && *cfg.outputPath)
    {
        std::filesystem::path playlistPath(cfg.outputPath);
        const std::string extLower = lowercase_copy(playlistPath.extension().string());
        return format_requests_hls || (extLower == ".m3u8" || extLower == ".m3u");
    }

    return format_requests_hls;
}

// Configures the output context for HLS muxing
static void finalize_hls_options(PipelineConfig *cfg, OutputContext *out)
{

    HlsMuxOptions hlsOpts;

    const bool format_requests_hls = !cfg->outputFormatName.empty() && lowercase_copy(cfg->outputFormatName) == "hls";
    
    if (cfg->outputPath && *cfg->outputPath)
    {
        std::filesystem::path playlistPath(cfg->outputPath);
        const std::string extLower = lowercase_copy(playlistPath.extension().string());
        
        if (format_requests_hls && (extLower != ".m3u8" && extLower != ".m3u"))
        {
            // If the format requests HLS but the output path doesn't have a .m3u8 or .m3u extension, throw an error
            throw std::runtime_error("Output path must have .m3u8 or .m3u extension when HLS is requested");
        }
        else if (!format_requests_hls && (extLower != ".m3u8" && extLower != ".m3u"))
        {
            // If the format doesn't request HLS and the output path doesn't have a .m3u8 or .m3u extension, return
            return;
        }
    }

    // Output format is HLS
    if (cfg->outputFormatName.empty())
        cfg->outputFormatName = "hls";

    hlsOpts.enabled = true;
    hlsOpts.overwrite = cfg->overwrite;
    // FFmpeg compatibility: Don't automatically mark discontinuities on seek
    hlsOpts.autoDiscontinuity = !cfg->ffCompatible;

    // Parse playlist path (skip directory creation for pipe output)
    std::filesystem::path playlistPath(cfg->outputPath);
    std::filesystem::path playlistDir = playlistPath.parent_path();

    if (!is_pipe_output(cfg->outputPath))
    {
        if (!playlistDir.empty())
        {
            std::error_code ec;
            std::filesystem::create_directories(playlistDir, ec);
            if (ec)
            {
                throw std::runtime_error("Failed to create HLS output directory: " + playlistDir.string() + " (" + ec.message() + ")");
            }
        }
    }
    std::string playlistStem = playlistPath.stem().string();

    // Set the segment type to fmp4 if it's not set
    std::string segmentTypeLower = lowercase_copy(cfg->hlsSegmentType);
    if (segmentTypeLower.empty())
    {
        cfg->hlsSegmentType = "fmp4";
        segmentTypeLower = "fmp4";
    }
    bool useFmp4 = (segmentTypeLower == "fmp4");
    if (!useFmp4 && segmentTypeLower != "mpegts")
    {
        cfg->hlsSegmentType = "mpegts";
        segmentTypeLower = "mpegts";
        useFmp4 = false;
    }
    hlsOpts.segmentType = segmentTypeLower;

    if (!cfg->hlsSegmentFilename.empty())
    {
        hlsOpts.segmentFilename = cfg->hlsSegmentFilename;
    }
    else
    {
        // Use .mp4 extension for fMP4 segments for better browser compatibility
        const std::string segmentExt = useFmp4 ? ".mp4" : ".ts";
        const std::string pattern = playlistStem.empty() ? (std::string("segment_%05d") + segmentExt)
                                                         : (playlistStem + "_%05d" + segmentExt);
        std::filesystem::path segPath = playlistDir.empty() ? std::filesystem::path(pattern) : (playlistDir / pattern);
        hlsOpts.segmentFilename = segPath.string();
    }

    if (useFmp4 && !cfg->hlsInitFilename.empty())
    {
        hlsOpts.initFilename = cfg->hlsInitFilename;
    }
    else if (useFmp4)
    {
        const std::string initName = playlistStem.empty() ? "init.mp4" : (playlistStem + "_init.mp4");
        std::filesystem::path initPath = playlistDir.empty() ? std::filesystem::path(initName) : (playlistDir / initName);
        hlsOpts.initFilename = initPath.string();
    }

    if (cfg->hlsTime <= 0)
        hlsOpts.hlsTime = 4;
    else
        hlsOpts.hlsTime = cfg->hlsTime;

    if (cfg->hlsListSize < 0)
        hlsOpts.listSize = 0;
    else
        hlsOpts.listSize = cfg->hlsListSize;

    if (cfg->hlsStartNumber < 0)
        hlsOpts.startNumber = 0;
    else
        hlsOpts.startNumber = cfg->hlsStartNumber;

    if (cfg->maxDelay >= 0)
        hlsOpts.maxDelay = cfg->maxDelay;

    if (cfg->hlsPlaylistType.empty())
        hlsOpts.playlistType = "";
    else
        hlsOpts.playlistType = cfg->hlsPlaylistType;

    // Set the HLS options
    out->hlsOptions = hlsOpts;
}

static void compatibility_mode(int argc, char **argv, PipelineConfig *cfg)
{
    // Enable verbose logging
    cfg->verbose = true;

    // Compatibility mode for ffmpeg, no positional arguments
    if (!cfg)
        return;

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
            if (cfg->inputPath == nullptr) {
                cfg->seekTime = argv[++i];
            } else {
                cfg->outputSeekTime = argv[++i];
            }
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
        // Detect output path: file extensions or pipe/stdout
        if (endsWith(arg, ".m3u8") || endsWith(arg, ".mp4") || endsWith(arg, ".mkv") ||
            arg == "-" || arg == "pipe:" || arg == "pipe:1")
        {
            cfg->outputPath = argv[i];
            LOG_DEBUG("Set outputPath = '%s'\n", cfg->outputPath);
        }
    }
}

static void print_help(const char *argv0)
{
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
    fprintf(stderr, "  --no-vsr      Disable VSR\n");
    fprintf(stderr, "  --vsr-quality     Set VSR quality, default 4\n");
    fprintf(stderr, "\nTHDR options:\n");
    fprintf(stderr, "  --no-thdr     Disable THDR\n");
    fprintf(stderr, "  --thdr-contrast   Set THDR contrast, default 115\n");
    fprintf(stderr, "  --thdr-saturation Set THDR saturation, default 75\n");
    fprintf(stderr, "  --thdr-middle-gray Set THDR middle gray, default 30\n");
    fprintf(stderr, "  --thdr-max-luminance Set THDR max luminance, default 1000\n");
    fprintf(stderr, "\nNVENC options:\n");
    fprintf(stderr, "  --nvenc-tune        Set NVENC tune, default hq\n");
    fprintf(stderr, "  --nvenc-preset      Set NVENC preset, default p4\n");
    fprintf(stderr, "  --nvenc-rc          Set NVENC rate control, default constqp\n");
    fprintf(stderr, "  --nvenc-gop         Set NVENC GOP (seconds), default 3\n");
    fprintf(stderr, "  --nvenc-bframes     Set NVENC bframes, default 0\n");
    fprintf(stderr, "  --nvenc-qp          Set NVENC QP, default 21\n");
    fprintf(stderr, "  --nvenc-bitrate-multiplier Set NVENC bitrate multiplier, default 2\n");
    fprintf(stderr, "\nHLS options (detected automatically for .m3u8 outputs):\n");
    fprintf(stderr, "  -hls_time <seconds>            Set target segment duration (default 4)\n");
    fprintf(stderr, "  -hls_segment_type <mpegts|fmp4> Select segment container (default fmp4)\n");
    fprintf(stderr, "  -hls_segment_filename <pattern> Segment naming pattern (auto-generated)\n");
    fprintf(stderr, "  -hls_fmp4_init_filename <file>  Initialization segment path for fMP4\n");
    fprintf(stderr, "  -start_number <n>               Starting segment number (default 0)\n");
    fprintf(stderr, "  -hls_playlist_type <type>       Playlist type (event, vod, live)\n");
    fprintf(stderr, "  -hls_list_size <count>          Playlist size (0 = keep all segments)\n");
}

static void init_setup(int argc, char **argv, PipelineConfig *cfg)
{
    if (argc < 3)
    {
        print_help(argv[0]);
        exit(1);
    }

    // Default VSR settings
    cfg->rtxCfg.enableVSR = true;
    cfg->rtxCfg.scaleFactor = 2;
    cfg->rtxCfg.vsrQuality = 4;

    // Default THDR settings
    cfg->rtxCfg.enableTHDR = true;
    cfg->rtxCfg.thdrContrast = 115;
    cfg->rtxCfg.thdrSaturation = 75;
    cfg->rtxCfg.thdrMiddleGray = 30;
    cfg->rtxCfg.thdrMaxLuminance = 1000;

    // Default NVENC settings
    cfg->tune = "hq";
    cfg->preset = "p4";
    cfg->rc = "constqp";

    cfg->gop = 3;  // 3 seconds GOP for HLS compatibility (will be multiplied by framerate)
    cfg->bframes = 0;
    cfg->qp = 21;
    cfg->targetBitrateMultiplier = 2;

    int i = 1;
    // if input path does not end with mp4 or mkv, use alternate arg method
    if (std::string(argv[i]).find(".mp4") == std::string::npos && std::string(argv[i]).find(".mkv") == std::string::npos && argc > 5)
    {
        cfg->ffCompatible = true; // obfuscate as ffmpeg, with compatible args to vanilla ffmpeg
        compatibility_mode(argc, argv, cfg);
        return;
    }
    else
    {
        cfg->inputPath = argv[i++];
        cfg->outputPath = argv[i++];
    }

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
            if (cfg->rtxCfg.enableVSR)
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

// Helper function for input HDR detection and configuration
static bool configure_input_hdr_detection(PipelineConfig& cfg, InputContext& in) {
    // Detect HDR content and disable THDR if input is already HDR
    bool inputIsHDR = false;
    if (in.vst && in.vst->codecpar) {
        AVColorTransferCharacteristic trc = in.vst->codecpar->color_trc;
        inputIsHDR = (trc == AVCOL_TRC_SMPTE2084) ||  // PQ (HDR10)
                    (trc == AVCOL_TRC_ARIB_STD_B67);   // HLG (Hybrid Log-Gamma)
    }

    if (inputIsHDR) {
        if (cfg.rtxCfg.enableTHDR) {
            LOG_INFO("Input content is HDR (transfer characteristic: %s). Disabling THDR to preserve HDR metadata.",
                     in.vst->codecpar->color_trc == AVCOL_TRC_SMPTE2084 ? "PQ/HDR10" : "HLG");
            cfg.rtxCfg.enableTHDR = false;
            cfg.targetBitrateMultiplier = cfg.targetBitrateMultiplier * 0.8; // 10bits -> 8bits, 20% less bitrate
        }

        // Reopen input with P010 preference for HDR content to enable full 10-bit pipeline
        close_input(in);
        InputOpenOptions inputOpts;
        inputOpts.fflags = cfg.fflags;
        inputOpts.preferP010ForHDR = true;
        inputOpts.seekTime = cfg.seekTime;
        inputOpts.noAccurateSeek = cfg.noAccurateSeek;
        inputOpts.seek2any = cfg.seek2any;
        inputOpts.seekTimestamp = cfg.seekTimestamp;
        // FFmpeg compatibility: Disable non-standard behaviors
        inputOpts.enableErrorConcealment = !cfg.ffCompatible;
        inputOpts.flushOnSeek = false;
        open_input(cfg.inputPath, in, &inputOpts);
        LOG_INFO("Configured decoder for P010 output to preserve full 10-bit HDR pipeline");
    }

    return inputIsHDR;
}

// Helper function for VSR auto-disable logic
static void configure_vsr_auto_disable(PipelineConfig& cfg, const InputContext& in) {
    // Auto-disable VSR for inputs >= 1920x1080 in either orientation
    if (cfg.rtxCfg.enableVSR) {
        bool ge1080p = (in.vdec->width > 2560 && in.vdec->height > 1440) ||
                       (in.vdec->width > 1440 && in.vdec->height > 2560);
        if (ge1080p) {
            LOG_INFO("Input resolution is %dx%d (>=1080p). Disabling VSR.", in.vdec->width, in.vdec->height);
            cfg.rtxCfg.enableVSR = false;
            cfg.targetBitrateMultiplier = cfg.targetBitrateMultiplier / 2.0; // approx. 2x less bitrate
        }
    }
}

// Helper function for audio processing configuration
static void configure_audio_processing(PipelineConfig& cfg, InputContext& in, OutputContext& out) {
    // Configure audio processing if compatibility mode is enabled
    if (cfg.ffCompatible) {
        LOG_DEBUG("Compatibility mode enabled, configuring audio...\n");

        // Create AudioParameters from PipelineConfig to avoid struct duplication issues
        AudioParameters audioParams;
        audioParams.codec = cfg.audioCodec;
        audioParams.channels = cfg.audioChannels;
        audioParams.bitrate = cfg.audioBitrate;
        audioParams.sampleRate = cfg.audioSampleRate;
        audioParams.filter = cfg.audioFilter;
        audioParams.streamMaps = cfg.streamMaps;

        configure_audio_from_params(audioParams, out);
        LOG_DEBUG("Audio config completed, enabled=%s\n", out.audioConfig.enabled ? "true" : "false");

        if (out.audioConfig.enabled) {
            LOG_DEBUG("Applying stream mappings...\n");
            apply_stream_mappings(cfg.streamMaps, in, out);

            LOG_DEBUG("Setting up audio encoder...\n");
            if (!setup_audio_encoder(in, out)) {
                LOG_WARN("Failed to setup audio encoder, disabling audio processing");
                out.audioConfig.enabled = false;
            } else {
                LOG_DEBUG("Audio encoder setup complete\n");
                LOG_DEBUG("Setting up audio filter...\n");
                if (!setup_audio_filter(in, out)) {
                    LOG_WARN("Failed to setup audio filter, continuing without filtering");
                    // Don't disable audio processing, just filtering
                } else {
                    LOG_DEBUG("Audio filter setup complete\n");
                }
            }
        }
    }
    LOG_DEBUG("Audio configuration complete, proceeding...\n");
}

// Helper function for progress tracking setup
static int64_t setup_progress_tracking(const InputContext& in, const AVRational& fr) {
    // Get total duration and frames for progress tracking
    int64_t total_frames = 0;
    if (in.vst->nb_frames > 0) {
        total_frames = in.vst->nb_frames;
    } else {
        // Estimate total frames from duration if frame count is not available
        double duration_sec = 0.0;
        if (in.vst->duration > 0 && in.vst->duration != AV_NOPTS_VALUE) {
            duration_sec = in.vst->duration * av_q2d(in.vst->time_base);
        } else if (in.fmt->duration != AV_NOPTS_VALUE) {
            duration_sec = static_cast<double>(in.fmt->duration) / AV_TIME_BASE;
        }

        if (duration_sec > 0.0 && fr.num > 0 && fr.den > 0) {
            total_frames = static_cast<int64_t>(duration_sec * av_q2d(fr) + 0.5);
        }
    }
    return total_frames;
}

// Helper function for video encoder configuration
static AVBufferRef* configure_video_encoder(PipelineConfig& cfg, InputContext& in, OutputContext& out,
                                           bool inputIsHDR, bool use_cuda_path, int dstW, int dstH,
                                           const AVRational& fr, bool hls_enabled, bool mux_is_isobmff) {
    // Configure encoder now that sizes are known
    LOG_DEBUG("Configuring HEVC encoder...");
    out.venc->codec_id = AV_CODEC_ID_HEVC;
    out.venc->width = dstW;
    out.venc->height = dstH;
    out.venc->time_base = av_inv_q(fr);
    out.venc->framerate = fr;

    // Prefer CUDA frames if decoder is CUDA-capable to avoid copies
    bool outputHDR = cfg.rtxCfg.enableTHDR || inputIsHDR;
    if (use_cuda_path) {
        out.venc->pix_fmt = AV_PIX_FMT_CUDA; // NVENC consumes CUDA frames via hw_frames_ctx
    } else {
        // CPU fallback: choose NV12 for SDR, P010 for HDR
        out.venc->pix_fmt = outputHDR ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
    }

    // For HLS, align GOP with segment duration for clean boundaries
    // At high frame rates, use shorter segments to reduce IDR encoding overhead
    int gop_duration_sec = cfg.gop;
    if (hls_enabled && out.hlsOptions.hlsTime > 0) {
        gop_duration_sec = out.hlsOptions.hlsTime;

        // For high frame rates (>50fps), recommend shorter segments to reduce IDR cost
        double fps = (double)fr.num / fr.den;
        if (fps > 50 && gop_duration_sec > 2) {
            LOG_WARN("High frame rate (%.1f fps) with %d-sec segments may cause periodic slowdowns",
                     fps, gop_duration_sec);
            LOG_WARN("Consider using -hls_time 2 for better performance at high FPS");
        }

        LOG_INFO("Aligning GOP size (%d sec = %d frames) with HLS segment duration",
                 gop_duration_sec, gop_duration_sec * fr.num / std::max(1, fr.den));
    }
    out.venc->gop_size = gop_duration_sec * fr.num / std::max(1, fr.den);
    out.venc->max_b_frames = 2;
    out.venc->color_range = AVCOL_RANGE_MPEG;

    // Use HDR color settings if THDR is enabled OR if input is HDR content
    if (outputHDR) {
        if (inputIsHDR) {
            // Preserve input HDR characteristics when input is HDR
            out.venc->color_trc = in.vst->codecpar->color_trc;
            out.venc->color_primaries = in.vst->codecpar->color_primaries;
            out.venc->colorspace = in.vst->codecpar->color_space;
        } else {
            // THDR enabled: use default HDR settings
            out.venc->color_trc = AVCOL_TRC_SMPTE2084; // PQ
            out.venc->color_primaries = AVCOL_PRI_BT2020;
            out.venc->colorspace = AVCOL_SPC_BT2020_NCL;
        }
    } else {
        out.venc->color_trc = AVCOL_TRC_BT709;
        out.venc->color_primaries = AVCOL_PRI_BT709;
        out.venc->colorspace = AVCOL_SPC_BT709;
    }

    // Ensure muxers that require extradata (e.g., Matroska) receive global headers
    if ((out.fmt->oformat->flags & AVFMT_GLOBALHEADER) || !mux_is_isobmff || hls_enabled) {
        out.venc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    int64_t target_bitrate = (in.fmt->bit_rate > 0) ? (int64_t)(in.fmt->bit_rate * cfg.targetBitrateMultiplier) : (int64_t)25000000;
    LOG_VERBOSE("Input bitrate: %.2f (Mbps), Target bitrate: %.2f (Mbps)\n", in.fmt->bit_rate / 1000000.0, target_bitrate / 1000000.0);
    LOG_VERBOSE("Encoder settings - tune: %s, preset: %s, rc: %s, qp: %d, gop: %d, bframes: %d",
                cfg.tune.c_str(), cfg.preset.c_str(), cfg.rc.c_str(), cfg.qp, cfg.gop, cfg.bframes);
    av_opt_set(out.venc->priv_data, "tune", cfg.tune.c_str(), 0);
    av_opt_set(out.venc->priv_data, "preset", cfg.preset.c_str(), 0);
    av_opt_set(out.venc->priv_data, "rc", cfg.rc.c_str(), 0);
    av_opt_set_int(out.venc->priv_data, "qp", cfg.qp, 0);
    av_opt_set(out.venc->priv_data, "temporal-aq", "1", 0);

    // Note: async_depth can cause frame reordering issues with some muxers
    // Removed for now - can re-enable if needed for extreme high-fps scenarios
    // av_opt_set_int(out.venc->priv_data, "async_depth", 4, 0);
    // av_opt_set_int(out.venc->priv_data, "delay", 4, 0);

    // For HLS output, force IDR frames to ensure clean segment boundaries
    if (hls_enabled) {
        av_opt_set(out.venc->priv_data, "forced-idr", "1", 0);
        // Note: strict_gop disabled - it prevents adaptive scene-change IDRs which hurts encoding performance
        // forced-idr alone is sufficient to ensure IDRs at GOP boundaries for HLS
        // av_opt_set(out.venc->priv_data, "strict_gop", "1", 0);
        LOG_INFO("Enabled forced-idr for HLS segment alignment (strict_gop disabled for better performance)");
    }

    // Set HEVC profile based on HDR output: use Main10 for HDR, Main for SDR
    if (outputHDR) {
        av_opt_set(out.venc->priv_data, "profile", "main10", 0);
    } else {
        av_opt_set(out.venc->priv_data, "profile", "main", 0);
    }

    // If using CUDA path, create encoder hw_frames_ctx on the same device before opening encoder
    AVBufferRef *enc_hw_frames = nullptr;
    if (use_cuda_path) {
        enc_hw_frames = av_hwframe_ctx_alloc(in.hw_device_ctx);
        if (!enc_hw_frames)
            throw std::runtime_error("av_hwframe_ctx_alloc failed for encoder");
        AVHWFramesContext *fctx = (AVHWFramesContext *)enc_hw_frames->data;
        fctx->format = AV_PIX_FMT_CUDA;
        // Choose NV12 for SDR, P010 for HDR (THDR enabled or input is HDR)
        fctx->sw_format = outputHDR ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
        fctx->width = dstW;
        fctx->height = dstH;
        fctx->initial_pool_size = 64;
        ff_check(av_hwframe_ctx_init(enc_hw_frames), "init encoder hwframe ctx");
        out.venc->hw_frames_ctx = av_buffer_ref(enc_hw_frames);
    }

    ff_check(avcodec_open2(out.venc, out.venc->codec, nullptr), "open encoder");
    LOG_VERBOSE("Pipeline: decode=%s, colorspace+scale+pack=%s\n",
                (in.vdec->hw_device_ctx ? "GPU(NVDEC)" : "CPU"),
                (use_cuda_path ? "GPU(RTX/CUDA)" : "CPU(SWS)"));

    return enc_hw_frames;
}

// Helper function for stream metadata and codec parameters
static void configure_stream_metadata(InputContext& in, OutputContext& out, PipelineConfig& cfg,
                                    bool inputIsHDR, bool mux_is_isobmff, bool hls_enabled) {
    bool outputHDR = cfg.rtxCfg.enableTHDR || inputIsHDR;

    ff_check(avcodec_parameters_from_context(out.vstream->codecpar, out.venc), "enc params to stream");
    if (out.vstream->codecpar->extradata_size == 0 || out.vstream->codecpar->extradata == nullptr) {
        throw std::runtime_error("HEVC encoder did not provide extradata; required for Matroska outputs");
    }
    LOG_DEBUG("Encoder extradata size: %d bytes", out.vstream->codecpar->extradata_size);
    if (out.vstream->codecpar->extradata_size >= 4) {
        const uint8_t *ed = out.vstream->codecpar->extradata;
        LOG_DEBUG("Extradata head: %02X %02X %02X %02X", ed[0], ed[1], ed[2], ed[3]);
    }

    // Prefer 'hvc1' brand to carry parameter sets (VPS/SPS/PPS) in-band, which improves fMP4 compatibility, and required by macOS
    if (out.vstream->codecpar) {
        if (mux_is_isobmff) {
            out.vstream->codecpar->codec_tag = MKTAG('h', 'v', 'c', '1');
        } else {
            out.vstream->codecpar->codec_tag = 0;
        }
        out.vstream->codecpar->color_range = AVCOL_RANGE_MPEG;
        if (outputHDR) {
            if (inputIsHDR) {
                // Preserve input HDR characteristics when input is HDR
                out.vstream->codecpar->color_trc = in.vst->codecpar->color_trc;
                out.vstream->codecpar->color_primaries = in.vst->codecpar->color_primaries;
                out.vstream->codecpar->color_space = in.vst->codecpar->color_space;
            } else {
                // THDR enabled: use default HDR settings
                out.vstream->codecpar->color_trc = AVCOL_TRC_SMPTE2084;
                out.vstream->codecpar->color_primaries = AVCOL_PRI_BT2020;
                out.vstream->codecpar->color_space = AVCOL_SPC_BT2020_NCL;
            }
        } else {
            out.vstream->codecpar->color_trc = AVCOL_TRC_BT709;
            out.vstream->codecpar->color_primaries = AVCOL_PRI_BT709;
            out.vstream->codecpar->color_space = AVCOL_SPC_BT709;
        }
    }

    if (outputHDR && !hls_enabled) {
        if (inputIsHDR) {
            // For HDR input, try to preserve original mastering metadata
            LOG_INFO("Preserving HDR mastering metadata from input stream");
            // TODO: Copy mastering display color volume and content light level from input
            // For now, add default HDR metadata to ensure proper HDR signaling
            add_mastering_and_cll(out.vstream, 4000); // Conservative max luminance for HDR passthrough
        } else {
            // THDR enabled: add THDR mastering metadata
            add_mastering_and_cll(out.vstream, cfg.rtxCfg.thdrMaxLuminance);
        }
    }
}

// Helper function for frame buffer and context initialization
static void initialize_frame_buffers_and_contexts(bool use_cuda_path, int dstW, int dstH,
                                                 CudaFramePool& cuda_pool, SwsContext*& sws_to_p010,
                                                 FramePtr& bgra_frame, FramePtr& p010_frame,
                                                 const InputContext& in, const OutputContext& out) {
    // Initialize CUDA frame pool if using CUDA path
    if (use_cuda_path) {
        const int POOL_SIZE = 8; // Adjust based on your needs
        cuda_pool.initialize(out.venc->hw_frames_ctx, dstW, dstH, POOL_SIZE);
    }

    // Prepare CPU fallback buffers and sws_to_p010 even if CUDA path is enabled, to allow on-the-fly fallback
    p010_frame->format = AV_PIX_FMT_P010LE;
    p010_frame->width = dstW;
    p010_frame->height = dstH;
    ff_check(av_frame_get_buffer(p010_frame.get(), 32), "alloc p010");

    // CPU path colorspace for RGB(A)->P010
    sws_to_p010 = sws_getContext(
        dstW, dstH, AV_PIX_FMT_X2BGR10LE,
        dstW, dstH, AV_PIX_FMT_P010LE,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_to_p010)
        throw std::runtime_error("sws_to_p010 alloc failed");
    const int *coeffs_bt2020 = sws_getCoefficients(SWS_CS_BT2020);
    sws_setColorspaceDetails(sws_to_p010,
                             coeffs_bt2020, 1,
                             coeffs_bt2020, 0,
                             0, 1 << 16, 1 << 16);

    bgra_frame->format = AV_PIX_FMT_RGBA;
    bgra_frame->width = in.vdec->width;
    bgra_frame->height = in.vdec->height;
    ff_check(av_frame_get_buffer(bgra_frame.get(), 32), "alloc bgra");
}

// Helper function for RTX processor initialization
static void initialize_rtx_processor(RTXProcessor& rtx, bool& rtx_init, bool use_cuda_path,
                                   const PipelineConfig& cfg, const InputContext& in) {
    if (use_cuda_path) {
        AVHWDeviceContext *devctx = (AVHWDeviceContext *)in.hw_device_ctx->data;
        AVCUDADeviceContext *cudactx = (AVCUDADeviceContext *)devctx->hwctx;
        CUcontext cu = cudactx->cuda_ctx;
        if (!rtx.initializeWithContext(cu, cfg.rtxCfg, in.vdec->width, in.vdec->height)) {
            std::string detail = rtx.lastError();
            if (detail.empty())
                detail = "unknown error";
            throw std::runtime_error(std::string("Failed to init RTX GPU path: ") + detail);
        }
        rtx_init = true;
    } else {
        if (!rtx.initialize(0, cfg.rtxCfg, in.vdec->width, in.vdec->height)) {
            std::string detail = rtx.lastError();
            if (detail.empty())
                detail = "unknown error";
            throw std::runtime_error(std::string("Failed to init RTX CPU path: ") + detail);
        }
        rtx_init = true;
    }
}

static void apply_movflags(AVDictionary** muxopts, bool is_pipe, bool hls_enabled, const PipelineConfig& cfg) {
    // User-specified flags take priority (FFmpeg-compatible)
    if (!cfg.movflags.empty()) {
        av_dict_set(muxopts, "movflags", cfg.movflags.c_str(), 0);
        LOG_DEBUG("Applied user movflags: %s\n", cfg.movflags.c_str());
        return;
    }

    // Legacy mode: Auto-apply flags for compatibility
    if (cfg.ffCompatible) return;

    if (is_pipe) {
        av_dict_set(muxopts, "movflags", "+empty_moov+default_base_moof+delay_moov+dash+write_colr", 0);
    } else if (!hls_enabled) {
        av_dict_set(muxopts, "movflags", "+faststart+write_colr", 0);
    } else {
        av_dict_set(muxopts, "movflags", "+frag_keyframe+delay_moov+faststart+write_colr", 0);
    }
}

static void apply_fragment_options(AVDictionary** muxopts, const PipelineConfig& cfg) {
    if (cfg.fragDuration > 0) {
        av_dict_set(muxopts, "frag_duration", std::to_string(cfg.fragDuration).c_str(), 0);
    }
    if (cfg.fragmentIndex >= 0) {
        av_dict_set(muxopts, "fragment_index", std::to_string(cfg.fragmentIndex).c_str(), 0);
    }
    if (cfg.useEditlist >= 0) {
        av_dict_set(muxopts, "use_editlist", std::to_string(cfg.useEditlist).c_str(), 0);
    }
}

static void write_muxer_header(OutputContext& out, bool hls_enabled, bool mux_is_isobmff,
                               const AVRational& fr, bool is_pipe, const PipelineConfig& cfg) {
    out.vstream->time_base = out.venc->time_base;
    out.vstream->avg_frame_rate = fr;

    AVDictionary *muxopts = out.muxOptions;

    if (mux_is_isobmff) {
        apply_movflags(&muxopts, is_pipe, hls_enabled, cfg);
        apply_fragment_options(&muxopts, cfg);
    }

    // Apply FFmpeg-compatible timestamp handling to muxer (affects all formats)
    av_dict_set(&muxopts, "avoid_negative_ts", cfg.avoidNegativeTs.c_str(), 0);
    LOG_DEBUG("Muxer avoid_negative_ts: %s", cfg.avoidNegativeTs.c_str());

    if (cfg.maxMuxingQueueSize > 0) {
        out.fmt->max_interleave_delta = cfg.maxMuxingQueueSize;
    }

    ff_check(avformat_write_header(out.fmt, &muxopts), "write header");
    if (muxopts) av_dict_free(&muxopts);
    out.muxOptions = nullptr;
}

int run_pipeline(PipelineConfig cfg)
{
    Logger::instance().setVerbose(cfg.verbose || cfg.debug);
    Logger::instance().setDebug(cfg.debug);

    LOG_VERBOSE("Starting video processing pipeline");
    LOG_DEBUG("Input: %s", cfg.inputPath);
    LOG_DEBUG("Output: %s", cfg.outputPath);
    LOG_VERBOSE("CPU-only mode: %s", cfg.cpuOnly ? "enabled" : "disabled");

    // Start pipeline
    InputContext in{};
    OutputContext out{};

    try
    {
        // Stage 1: Open input and configure HDR detection
        InputOpenOptions inputOpts;
        inputOpts.fflags = cfg.fflags;
        inputOpts.seekTime = cfg.seekTime;
        inputOpts.noAccurateSeek = cfg.noAccurateSeek;
        inputOpts.seek2any = cfg.seek2any;
        inputOpts.seekTimestamp = cfg.seekTimestamp;
        // FFmpeg compatibility: Disable non-standard behaviors
        inputOpts.enableErrorConcealment = !cfg.ffCompatible;  // FFmpeg doesn't enable error concealment by default
        inputOpts.flushOnSeek = false;  // FFmpeg never flushes decoder on seek
        open_input(cfg.inputPath, in, &inputOpts);
        bool inputIsHDR = configure_input_hdr_detection(cfg, in);

        // Stage 2: Configure VSR auto-disable
        configure_vsr_auto_disable(cfg, in);

        // Stage 3: Setup output and HLS options
        LOG_DEBUG("Finalizing HLS options...");
        finalize_hls_options(&cfg, &out);

        // Stage 3b: Pre-configure audio intent to guide stream creation
        bool willReencodeAudio = cfg.ffCompatible &&
                                ((!cfg.audioCodec.empty() && cfg.audioCodec != "copy") ||
                                 cfg.audioChannels > 0 || cfg.audioBitrate > 0 || !cfg.audioFilter.empty());
        if (willReencodeAudio) {
            out.audioConfig.enabled = true;
            out.audioConfig.codec = cfg.audioCodec.empty() ? "aac" : cfg.audioCodec;
        }

        LOG_DEBUG("Opening output...");
        open_output(cfg.outputPath, in, out);
        LOG_DEBUG("Output opened successfully");

        // Stage 4: Configure audio processing (complete the audio setup)
        configure_audio_processing(cfg, in, out);

        // Stage 5: Calculate frame rate and setup progress tracking
        const bool hls_enabled = out.hlsOptions.enabled;
        const bool hls_segments_are_fmp4 = hls_enabled && lowercase_copy(out.hlsOptions.segmentType) == "fmp4";

        // Read input bitrate and fps
        AVRational fr = in.vst->avg_frame_rate.num ? in.vst->avg_frame_rate : in.vst->r_frame_rate;
        if (fr.num == 0 || fr.den == 0)
            fr = {in.vst->time_base.den, in.vst->time_base.num};

        int64_t total_frames = setup_progress_tracking(in, fr);

        // Progress tracking variables
        int64_t processed_frames = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto last_update = start_time;
        const int update_interval_ms = 500; // Update progress every 500ms
        std::string progress_bar(50, ' ');

        // Prepare sws contexts (created on first decoded frame when actual format is known)
        SwsContext *sws_to_argb = nullptr;
        int last_src_format = AV_PIX_FMT_NONE;

        // A2R10G10B10 -> P010 for NVENC
        // In this pipeline, the RTX output maps to A2R10G10B10; prefer doing P010 conversion on GPU when possible.
        int effScale = cfg.rtxCfg.enableVSR ? cfg.rtxCfg.scaleFactor : 1;
        int dstW = in.vdec->width * effScale;
        int dstH = in.vdec->height * effScale;
        SwsContext *sws_to_p010 = nullptr; // CPU fallback
        bool use_cuda_path = (in.vdec->hw_device_ctx != nullptr) && !cfg.cpuOnly;

        // Stage 6: Calculate output dimensions and setup muxer format
        LOG_VERBOSE("Processing path: %s", use_cuda_path ? "GPU (CUDA)" : "CPU");
        LOG_VERBOSE("Output resolution: %dx%d (scale factor: %d)", dstW, dstH, effScale);
        std::string muxer_name = (out.fmt && out.fmt->oformat && out.fmt->oformat->name)
                                     ? out.fmt->oformat->name
                                     : "";
        bool mux_is_isobmff = muxer_name.find("mp4") != std::string::npos ||
                              muxer_name.find("mov") != std::string::npos;
        if (hls_segments_are_fmp4) {
            mux_is_isobmff = true;
        }
        LOG_VERBOSE("Output container: %s",
                    muxer_name.empty() ? "unknown" : muxer_name.c_str());

        // Stage 7: Configure video encoder
        AVBufferRef *enc_hw_frames = configure_video_encoder(cfg, in, out, inputIsHDR, use_cuda_path,
                                                            dstW, dstH, fr, hls_enabled, mux_is_isobmff);

        // Stage 8: Configure stream metadata and codec parameters
        configure_stream_metadata(in, out, cfg, inputIsHDR, mux_is_isobmff, hls_enabled);
        if (enc_hw_frames)
            av_buffer_unref(&enc_hw_frames);

        // Stage 9: Initialize frame buffers and processing contexts
        CudaFramePool cuda_pool;
        FramePtr frame(av_frame_alloc(), &av_frame_free_single);
        FramePtr swframe(av_frame_alloc(), &av_frame_free_single);
        FramePtr bgra_frame(av_frame_alloc(), &av_frame_free_single);
        FramePtr p010_frame(av_frame_alloc(), &av_frame_free_single);

        initialize_frame_buffers_and_contexts(use_cuda_path, dstW, dstH, cuda_pool, sws_to_p010,
                                            bgra_frame, p010_frame, in, out);

        // Frame buffering for handling processing spikes
        std::queue<std::pair<AVFrame *, int64_t>> frame_buffer; // frame and output_pts pairs
        const int MAX_BUFFER_SIZE = 4;

        // Stage 10: Write muxer header
        bool isPipeOutput = is_pipe_output(cfg.outputPath);
        write_muxer_header(out, hls_enabled, mux_is_isobmff, fr, isPipeOutput, cfg);

        PacketPtr pkt(av_packet_alloc(), &av_packet_free_single);
        PacketPtr opkt(av_packet_alloc(), &av_packet_free_single);

        // Legacy variables removed - now using TimestampManager
        // All PTS/DTS handling centralized in ts_manager

        const uint8_t *rtx_data = nullptr;
        uint32_t rtxW = 0, rtxH = 0;
        size_t rtxPitch = 0;

        // Progress display function
        auto show_progress = [&]()
        {
            if (total_frames <= 0)
                return; // Skip if we can't determine total frames

            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            auto time_since_last_update = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();

            if (time_since_last_update < update_interval_ms && processed_frames < total_frames)
                return;

            last_update = now;

            double progress = static_cast<double>(processed_frames) / total_frames;
            int bar_width = 50;
            int pos = static_cast<int>(bar_width * progress);

            std::string bar;
            bar.reserve(bar_width + 10);
            bar = "[";
            for (int i = 0; i < bar_width; ++i)
            {
                if (i < pos)
                    bar += "=";
                else if (i == pos)
                    bar += ">";
                else
                    bar += " ";
            }
            bar += "] ";

            // Calculate FPS
            double fps = (elapsed_ms > 0) ? (processed_frames * 1000.0) / elapsed_ms : 0.0;

            // Calculate ETA
            double remaining_sec = (elapsed_ms > 0) ? (total_frames - processed_frames) / (processed_frames / (elapsed_ms / 1000.0)) : 0;
            int remaining_mins = static_cast<int>(remaining_sec) / 60;
            int remaining_secs = static_cast<int>(remaining_sec) % 60;

            // Format progress line
            std::ostringstream oss;
            oss << "\r" << bar;
            oss << std::setw(5) << std::fixed << std::setprecision(1) << (progress * 100.0) << "% ";
            oss << "[" << processed_frames << "/" << total_frames << "] ";
            oss << std::setw(5) << std::fixed << std::setprecision(1) << fps << " fps ";
            oss << "ETA: " << std::setw(2) << std::setfill('0') << remaining_mins << ":"
                << std::setw(2) << std::setfill('0') << remaining_secs;

            // Clear the line and print
            fprintf(stderr, "\r\033[2K"); // Clear the entire line and move cursor to start
            fprintf(stderr, "%s", oss.str().c_str());
            fflush(stderr);
        };

        // Stage 11: Initialize RTX processor
        RTXProcessor rtx;
        bool rtx_init = false;
        initialize_rtx_processor(rtx, rtx_init, use_cuda_path, cfg, in);

        // Note: Audio PTS will be aligned with video baseline after first video packet
        // This ensures proper A/V sync during seek operations

        // Calculate whether output should be HDR
        bool outputHDR = cfg.rtxCfg.enableTHDR || inputIsHDR;

        // Build a processor abstraction for the loop
        std::unique_ptr<IProcessor> processor;
        if (use_cuda_path)
        {
            processor = std::make_unique<GpuProcessor>(rtx, cuda_pool, in.vdec->colorspace, outputHDR);
        }
        else
        {
            auto cpuProc = std::make_unique<CpuProcessor>(rtx, in.vdec->width, in.vdec->height, dstW, dstH);
            // Create a modified config for CPU processor to ensure it uses HDR pixel formats when outputting HDR
            RTXProcessConfig cpuConfig = cfg.rtxCfg;
            cpuConfig.enableTHDR = outputHDR; // Use HDR pixel formats for both THDR and HDR input
            cpuProc->setConfig(cpuConfig);
            processor = std::move(cpuProc);
        }

        // Initialize centralized timestamp manager (V2)
        TimestampManager::Config ts_config;
        ts_config.mode = cfg.copyts ? TimestampManager::Mode::COPYTS : TimestampManager::Mode::NORMAL;
        ts_config.input_seek_us = in.seek_offset_us;
        // FFmpeg compatibility: Don't automatically fix timestamp violations
        ts_config.enforce_monotonicity = !cfg.ffCompatible;  // FFmpeg reports errors, doesn't auto-fix

        // Parse avoid_negative_ts setting (FFmpeg-compatible)
        std::string avoid_ts_lower = cfg.avoidNegativeTs;
        std::transform(avoid_ts_lower.begin(), avoid_ts_lower.end(), avoid_ts_lower.begin(), ::tolower);
        if (avoid_ts_lower == "disabled") {
            ts_config.avoid_negative_ts = AvoidNegativeTs::DISABLED;
        } else if (avoid_ts_lower == "make_zero") {
            ts_config.avoid_negative_ts = AvoidNegativeTs::MAKE_ZERO;
        } else if (avoid_ts_lower == "make_non_negative") {
            ts_config.avoid_negative_ts = AvoidNegativeTs::MAKE_NON_NEGATIVE;
        } else {
            ts_config.avoid_negative_ts = AvoidNegativeTs::AUTO;  // Default
        }

        // Set start_at_zero flag (FFmpeg-compatible)
        ts_config.start_at_zero = cfg.startAtZero;

        // Store settings in OutputContext for muxer reference
        out.avoidNegativeTs = ts_config.avoid_negative_ts;
        out.startAtZero = ts_config.start_at_zero;

        // Detect actual frame rate for better monotonicity recovery
        AVRational detected_fr = av_guess_frame_rate(in.fmt, in.vst, nullptr);
        if (detected_fr.num > 0 && detected_fr.den > 0) {
            ts_config.expected_frame_rate = detected_fr;
            LOG_DEBUG("Detected frame rate: %d/%d (%.3f fps)",
                     detected_fr.num, detected_fr.den, av_q2d(detected_fr));
        } else {
            ts_config.expected_frame_rate = {24, 1};  // Default fallback
            LOG_WARN("Could not detect frame rate, using default 24fps");
        }

        // Parse output seeking time
        if (!cfg.outputSeekTime.empty()) {
            int ret = av_parse_time(&ts_config.output_seek_target_us, cfg.outputSeekTime.c_str(), 1);
            if (ret < 0) {
                throw std::runtime_error("Invalid output seek time format: " + cfg.outputSeekTime);
            }
            LOG_DEBUG("Output seeking enabled: target = %.3fs", ts_config.output_seek_target_us / 1000000.0);

            // Edge case: Warn if output seek < input seek
            if (in.seek_offset_us > 0 && ts_config.output_seek_target_us < in.seek_offset_us) {
                LOG_WARN("Output seek target (%.3fs) < input seek (%.3fs) - may cause unexpected behavior",
                         ts_config.output_seek_target_us / 1000000.0, in.seek_offset_us / 1000000.0);
            }
        }

        // Parse output timestamp offset
        if (!cfg.outputTsOffset.empty()) {
            int ret = av_parse_time(&ts_config.output_ts_offset_us, cfg.outputTsOffset.c_str(), 1);
            if (ret < 0) {
                throw std::runtime_error("Invalid output timestamp offset format: " + cfg.outputTsOffset);
            }
            LOG_DEBUG("Output timestamp offset: %.3fs", ts_config.output_ts_offset_us / 1000000.0);
        }

        // Edge case: Validate copyts + output_ts_offset usage
        if (ts_config.output_ts_offset_us != 0 && !cfg.copyts) {
            LOG_WARN("-output_ts_offset specified without -copyts. This may produce unexpected timestamps.");
            LOG_WARN("Typically use: -copyts -output_ts_offset <value> together");
        }

        // Create timestamp manager (V2)
        TimestampManager ts_manager(ts_config);
        ts_manager.dumpState();

        // Initialize async demuxer for non-blocking I/O
        AsyncDemuxer::Config demux_config;
        // Scale buffer based on frame rate: target 2-3 seconds of buffering
        // At 30fps: 60-90 frames, at 60fps: 120-180 frames, at 120fps: 240-360 frames
        double fps = av_q2d(fr);
        size_t target_buffer_seconds = 3;
        demux_config.max_queue_size = static_cast<size_t>(fps * target_buffer_seconds);
        demux_config.max_queue_size = std::max<size_t>(60, std::min<size_t>(360, demux_config.max_queue_size));
        demux_config.enable_stats = true;
        AsyncDemuxer async_demuxer(in.fmt, demux_config);
        LOG_DEBUG("Async demuxer queue size: %zu frames (%.1f fps, %zu sec buffer)",
                  demux_config.max_queue_size, fps, target_buffer_seconds);

        // Start async demuxing thread
        if (!async_demuxer.start()) {
            throw std::runtime_error("Failed to start async demuxer");
        }

        // Read packets
        LOG_DEBUG("Starting frame processing loop with async demuxing...");
        LOG_DEBUG("Video stream index: %d, Audio stream index: %d", in.vstream, in.astream);
        LOG_DEBUG("Audio config enabled: %s", out.audioConfig.enabled ? "true" : "false");
        LOG_DEBUG("Copyts mode: %s", cfg.copyts ? "enabled" : "disabled");
        LOG_DEBUG("FFmpeg compatibility: avoid_negative_ts=%s, start_at_zero=%s",
                 cfg.avoidNegativeTs.c_str(), cfg.startAtZero ? "enabled" : "disabled");
        LOG_DEBUG("Starting processing with seek_offset_us = %lld (%.3fs)", in.seek_offset_us, in.seek_offset_us / 1000000.0);
        int packet_count = 0;
        bool audio_pts_aligned = false;  // Track if we've aligned audio PTS with video baseline
        while (true)
        {
            // Get packet from async demuxer (non-blocking I/O)
            AVPacket* raw_pkt = async_demuxer.getPacket();
            if (!raw_pkt) {
                // Check for EOF or error
                if (async_demuxer.isEOF()) {
                    break;
                }
                int err = async_demuxer.getError();
                if (err != 0) {
                    char errbuf[AV_ERROR_MAX_STRING_SIZE];
                    av_make_error_string(errbuf, sizeof(errbuf), err);
                    throw std::runtime_error(std::string("Async demuxer error: ") + errbuf);
                }
                break;
            }

            // Transfer ownership to smart pointer
            av_packet_unref(pkt.get());
            av_packet_move_ref(pkt.get(), raw_pkt);
            av_packet_free(&raw_pkt);
            packet_count++;

            // Establish global baseline from the first VIDEO packet when seeking to ensure A/V sync
            if (in.seek_offset_us > 0 && ts_manager.getGlobalBaseline() == AV_NOPTS_VALUE &&
                pkt->stream_index == in.vstream && pkt->pts != AV_NOPTS_VALUE) {
                AVStream *stream = in.fmt->streams[pkt->stream_index];
                ts_manager.establishGlobalBaseline(pkt.get(), stream->time_base);

                // Align audio PTS with the newly established video baseline
                if (out.audioConfig.enabled && !audio_pts_aligned) {
                    init_audio_pts_after_seek(in, out, ts_manager.getGlobalBaseline());
                    audio_pts_aligned = true;
                    LOG_DEBUG("Audio PTS aligned with video baseline at %.3fs", ts_manager.getGlobalBaseline() / 1000000.0);
                }
            }

            // Process packets by type
            if (pkt->stream_index == in.vstream)
            {
                ff_check(avcodec_send_packet(in.vdec, pkt.get()), "send packet");
                av_packet_unref(pkt.get());

                while (true)
                {
                    int ret = avcodec_receive_frame(in.vdec, frame.get());
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                        break;
                    ff_check(ret, "receive frame");

                    AVFrame *decframe = frame.get();
                    FramePtr tmp(nullptr, &av_frame_free_single);
                    bool frame_is_cuda = (decframe->format == AV_PIX_FMT_CUDA);
                    if (frame_is_cuda && !use_cuda_path)
                    {
                        // Decoder produced CUDA but encoder path is CPU: transfer to SW
                        if (!swframe)
                            swframe.reset(av_frame_alloc());
                        ff_check(av_hwframe_transfer_data(swframe.get(), decframe, 0), "hwframe transfer");
                        decframe = swframe.get();
                        frame_is_cuda = false;
                    }

                    // Output seeking: Drop frames until target reached (handled by TimestampManager)
                    if (ts_manager.shouldDropFrameForOutputSeek(decframe, in.vst->time_base)) {
                        av_frame_unref(frame.get());
                        if (swframe)
                            av_frame_unref(swframe.get());
                        continue;
                    }

                    // Derive output PTS/DTS using centralized timestamp manager V2
                    // Handles NORMAL/COPYTS modes, output_ts_offset, monotonicity, DTS, etc.
                    TimestampManager::TimestampPair timestamps = ts_manager.deriveVideoTimestamps(
                        decframe, in.vst->time_base, out.venc->time_base);

                    // Use unified processor
                    AVFrame *outFrame = nullptr;
                    if (!processor->process(decframe, outFrame))
                    {
                        // Processing failed; no runtime fallback by design
                        throw std::runtime_error("Processor failed to produce output frame");
                    }

                    // Update progress prior to encoding
                    processed_frames++;
                    show_progress();

                    // Set consistent PTS/DTS to prevent stuttering
                    outFrame->pts = timestamps.pts;
                    outFrame->pkt_dts = timestamps.dts;

                    // IMPORTANT: Must sync before encoder accesses CUDA frame data
                    // RTX processor syncs internally, but this ensures frame is ready for NVENC
                    if (use_cuda_path)
                        cudaStreamSynchronize(0);

                    // Encode frame
                    encode_and_write(out.venc, out.vstream, out.fmt, outFrame, opkt, "send frame to encoder");
                    av_frame_unref(frame.get());
                    if (swframe)
                        av_frame_unref(swframe.get());
                }
            }
            else if (cfg.ffCompatible && out.audioConfig.enabled && in.astream >= 0 && pkt->stream_index == in.astream)
            {
                // FFmpeg does NOT wait for video baseline before processing audio
                // Dropping audio packets while waiting for video can cause A/V desync
                // Only drop if explicitly required for custom sync logic (non-FFmpeg behavior)
                // NOTE: Disabled by default for FFmpeg compatibility
                // if (in.seek_offset_us > 0 && ts_manager.getGlobalBaseline() == AV_NOPTS_VALUE) {
                //     av_packet_unref(pkt.get());
                //     continue;
                // }

                // Process audio packets when audio encoding is enabled
                if (in.adec && out.aenc)
                {
                    ff_check(avcodec_send_packet(in.adec, pkt.get()), "send audio packet");
                    av_packet_unref(pkt.get());

                    FramePtr audio_frame(av_frame_alloc(), &av_frame_free_single);
                    while (true)
                    {
                        int ret = avcodec_receive_frame(in.adec, audio_frame.get());
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                            break;
                        ff_check(ret, "receive audio frame");

                        // Update A/V drift monitoring
                        if (audio_frame->pts != AV_NOPTS_VALUE) {
                            int64_t audio_pts_us = av_rescale_q(audio_frame->pts,
                                                                in.ast->time_base,
                                                                {1, AV_TIME_BASE});
                            ts_manager.updateAudioTimestamp(audio_pts_us);
                        }

                        // Use the new helper function to process audio
                        if (process_audio_frame(audio_frame.get(), out, opkt.get()))
                        {
                            // If we got a packet, write it
                            if (opkt->data)
                            {
                                ff_check(av_interleaved_write_frame(out.fmt, opkt.get()), "write audio packet");
                                av_packet_unref(opkt.get());
                            }
                        }

                        av_frame_unref(audio_frame.get());
                    }
                }
                else
                {
                    // Fallback: copy audio packet without re-encoding
                    int out_index = out.map_streams[pkt->stream_index];


                    if (out_index >= 0)
                    {
                        AVStream *ist = in.fmt->streams[pkt->stream_index];
                        AVStream *ost = out.fmt->streams[out_index];

                        // Use centralized timestamp manager for audio copy packets
                        ts_manager.adjustPacketTimestamps(pkt.get(), ist->time_base, pkt->stream_index);

                        // Check if packet was invalidated (waiting for baseline)
                        if (pkt->pts == AV_NOPTS_VALUE && pkt->dts == AV_NOPTS_VALUE) {
                            av_packet_unref(pkt.get());
                            continue;
                        }

                        av_packet_rescale_ts(pkt.get(), ist->time_base, ost->time_base);
                        pkt->stream_index = out_index;
                        ff_check(av_interleaved_write_frame(out.fmt, pkt.get()), "write copied audio packet");
                    }
                    av_packet_unref(pkt.get());
                }
            }
            else
            {
                // Copy other streams
                int out_index = out.map_streams[pkt->stream_index];

                // Skip subtitle streams (they should already be filtered but double-check)
                AVStream *input_stream = in.fmt->streams[pkt->stream_index];
                if (input_stream->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE)
                {
                    av_packet_unref(pkt.get());
                    continue;
                }

                if (out_index >= 0)
                {
                    AVStream *ist = in.fmt->streams[pkt->stream_index];
                    AVStream *ost = out.fmt->streams[out_index];

                    // Use centralized timestamp manager for all copied streams
                    ts_manager.adjustPacketTimestamps(pkt.get(), ist->time_base, pkt->stream_index);

                    av_packet_rescale_ts(pkt.get(), ist->time_base, ost->time_base);
                    pkt->stream_index = out_index;
                    ff_check(av_interleaved_write_frame(out.fmt, pkt.get()), "write copied packet");
                }
                av_packet_unref(pkt.get());
            }
        }

        // Flush encoder
        LOG_DEBUG("Finished processing all frames, flushing encoder...");
        ff_check(avcodec_send_frame(out.venc, nullptr), "send flush");
        while (true)
        {
            int ret = avcodec_receive_packet(out.venc, opkt.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            ff_check(ret, "receive packet flush");
            opkt->stream_index = out.vstream->index;
            av_packet_rescale_ts(opkt.get(), out.venc->time_base, out.vstream->time_base);
            ff_check(av_interleaved_write_frame(out.fmt, opkt.get()), "write packet flush");
            av_packet_unref(opkt.get());
        }

        // Flush audio encoder if enabled
        if (cfg.ffCompatible && out.audioConfig.enabled && out.aenc)
        {
            LOG_DEBUG("Flushing audio encoder...");
            ff_check(avcodec_send_frame(out.aenc, nullptr), "send audio flush");
            while (true)
            {
                int ret = avcodec_receive_packet(out.aenc, opkt.get());
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;
                ff_check(ret, "receive audio packet flush");
                opkt->stream_index = out.astream->index;
                av_packet_rescale_ts(opkt.get(), out.aenc->time_base, out.astream->time_base);
                ff_check(av_interleaved_write_frame(out.fmt, opkt.get()), "write audio packet flush");
                av_packet_unref(opkt.get());
            }
        }

        ff_check(av_write_trailer(out.fmt), "write trailer");

        // Stop async demuxer
        async_demuxer.stop();

        // Print final progress and statistics
        if (total_frames > 0)
        {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            double total_sec = total_ms / 1000.0;
            double avg_fps = (total_sec > 0) ? processed_frames / total_sec : 0.0;

            LOG_DEBUG("Processing completed in %.1fs @ %.1f fps", total_sec, avg_fps);

            // Async demuxer statistics
            AsyncDemuxer::Stats demux_stats = async_demuxer.getStats();
            LOG_DEBUG("Demuxer stats: packets=%zu, queue_waits=%zu, queue_depth=%zu/%zu%s",
                     demux_stats.total_reads, demux_stats.queue_full_waits,
                     demux_stats.min_queue_depth != SIZE_MAX ? demux_stats.min_queue_depth : 0,
                     demux_stats.max_queue_depth,
                     demux_stats.queue_full_waits > 0 ? " (backpressure)" : "");

            // Report timestamp statistics
            TimestampManager::Stats ts_stats = ts_manager.getStats();
            if (ts_stats.dropped_frames > 0 || ts_stats.discontinuities > 0 ||
                ts_stats.monotonicity_violations > 0 || ts_stats.negative_pts > 0 ||
                ts_stats.av_drift_warnings > 0) {
                LOG_DEBUG("Timestamp stats: dropped=%d, discontinuities=%d, monotonic_fix=%d, neg_pts=%d, av_drift=%d (max=%.1fms)%s",
                         ts_stats.dropped_frames, ts_stats.discontinuities,
                         ts_stats.monotonicity_violations, ts_stats.negative_pts,
                         ts_stats.av_drift_warnings, ts_stats.max_av_drift_ms,
                         ts_stats.max_av_drift_ms > 500.0 ? " WARNING:HIGH_DRIFT" : "");
            }

            // Overall health assessment
            bool healthy = true;
            std::string warnings;
            if (demux_stats.queue_full_waits > demux_stats.total_reads * 0.1) {
                warnings += "demux_backpressure ";
                healthy = false;
            }
            if (ts_stats.max_av_drift_ms > 500.0) {
                warnings += "high_av_drift ";
                healthy = false;
            }
            LOG_DEBUG("Pipeline health: %s%s", healthy ? "OK" : "WARN",
                     warnings.empty() ? "" : (" - " + warnings).c_str());
        }
        if (sws_to_argb)
            sws_freeContext(sws_to_argb);
        sws_freeContext(sws_to_p010);
        // Ensure all CUDA operations complete before cleanup
        if (use_cuda_path)
        {
            cudaDeviceSynchronize();
        }

        // Known issues: if running over SSH, unable to shutdown properly and will hang indefinitely
        if (rtx_init)
        {
            LOG_VERBOSE("Shutting down RTX...");
            rtx.shutdown();
            LOG_VERBOSE("RTX shutdown complete.");
        }

        close_output(out);
        close_input(in);

        return 0;
    }
    catch (const std::exception &ex)
    {
        fprintf(stderr, "Error: %s\n", ex.what());
        close_output(out);
        close_input(in);
        return 2;
    }
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    // Only attach to parent console if stdout is not already redirected/piped
    // When spawned from Node.js with stdio:["ignore","pipe","pipe"],
    // stdout is already connected to a pipe and should not be reopened
    DWORD mode;
    BOOL stdoutIsPipe = !GetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), &mode);

    if (!stdoutIsPipe && AttachConsole(ATTACH_PARENT_PROCESS))
    {
        // Reopen stdout/stderr to the parent console
        FILE* fp;
        freopen_s(&fp, "CONOUT$", "w", stdout);
        freopen_s(&fp, "CONOUT$", "w", stderr);
        freopen_s(&fp, "CONIN$", "r", stdin);
    }
#endif

    if (passthrough_required(argc, argv))
    {
        return run_ffmpeg_passthrough(argc, argv);
    }

    PipelineConfig cfg;

    init_setup(argc, argv, &cfg);

    // Set log level
    av_log_set_level(AV_LOG_WARNING);

    int ret = run_pipeline(cfg);
}