#include "output_config.h"
#include "logger.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <system_error>

extern "C"
{
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/mastering_display_metadata.h>
}

// String utility: check if string ends with suffix
bool endsWith(const std::string &str, const std::string &suffix)
{
    if (suffix.size() > str.size())
        return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// String utility: convert to lowercase
std::string lowercase_copy(const std::string &s)
{
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });
    return result;
}

// Check if output is a pipe/stdout
bool is_pipe_output(const char *path)
{
    if (!path)
        return false;
    return (std::strcmp(path, "-") == 0) ||
           (std::strcmp(path, "pipe:") == 0) ||
           (std::strcmp(path, "pipe:1") == 0) ||
           (std::strncmp(path, "pipe:", 5) == 0);
}

// Check if HLS output will be used
bool will_use_hls_output(const PipelineConfig &cfg)
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

// Configure HLS muxing options
void finalize_hls_options(PipelineConfig *cfg, OutputContext *out)
{
    HlsMuxOptions hlsOpts;

    const bool format_requests_hls = !cfg->outputFormatName.empty() && lowercase_copy(cfg->outputFormatName) == "hls";

    if (cfg->outputPath && *cfg->outputPath)
    {
        std::filesystem::path playlistPath(cfg->outputPath);
        const std::string extLower = lowercase_copy(playlistPath.extension().string());

        if (format_requests_hls && (extLower != ".m3u8" && extLower != ".m3u"))
        {
            throw std::runtime_error("Output path must have .m3u8 or .m3u extension when HLS is requested");
        }
        else if (!format_requests_hls && (extLower != ".m3u8" && extLower != ".m3u"))
        {
            return;
        }
    }

    // Output format is HLS
    if (cfg->outputFormatName.empty())
        cfg->outputFormatName = "hls";

    hlsOpts.enabled = true;
    hlsOpts.overwrite = cfg->overwrite;
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

    out->hlsOptions = hlsOpts;
}

// Configure video encoder
AVBufferRef *configure_video_encoder(PipelineConfig &cfg, InputContext &in, OutputContext &out,
                                     bool inputIsHDR, bool use_cuda_path, int dstW, int dstH,
                                     const AVRational &fr, bool hls_enabled, bool mux_is_isobmff)
{
    LOG_DEBUG("Configuring HEVC encoder...");
    out.venc->codec_id = AV_CODEC_ID_HEVC;
    out.venc->width = dstW;
    out.venc->height = dstH;

    // Encoder timebase configuration:
    // - HLS: Use framerate-based timebase (already set in ffmpeg_utils.cpp as 1/fps)
    //        This matches vanilla FFmpeg behavior and ensures HLS muxer uses 1/24000
    // - Non-HLS: Use input stream timebase for -copyts compatibility (-enc_time_base -1)
    if (!hls_enabled)
    {
        // Non-HLS: Override to use input stream timebase
        out.venc->time_base = in.vst->time_base;
        out.vstream->time_base = out.venc->time_base;
        LOG_DEBUG("Non-HLS: Using input stream timebase %d/%d for encoder",
                  out.venc->time_base.num, out.venc->time_base.den);
    }
    else
    {
        // HLS: Keep the framerate-based timebase that was set in ffmpeg_utils.cpp
        // This ensures muxer will use 1/24000 (or similar) instead of 1/16000
        LOG_INFO("HLS: Using framerate-based encoder timebase %d/%d",
                 out.venc->time_base.num, out.venc->time_base.den);
    }
    out.venc->framerate = fr;

    bool outputHDR = cfg.rtxCfg.enableTHDR || inputIsHDR;
    if (use_cuda_path)
    {
        out.venc->pix_fmt = AV_PIX_FMT_CUDA;
    }
    else
    {
        out.venc->pix_fmt = outputHDR ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
    }

    // For HLS, align GOP with segment duration
    int gop_duration_sec = cfg.gop;
    int gop_size_frames = cfg.gop * fr.num / std::max(1, fr.den);

    if (hls_enabled && out.hlsOptions.hlsTime > 0)
    {
        gop_duration_sec = out.hlsOptions.hlsTime;

        // Calculate GOP size with proper rounding for fractional framerates
        // For 23.976fps (24000/1001): 3 * 24000 / 1001 = 71.928 → round to 72
        gop_size_frames = (int)round((double)gop_duration_sec * fr.num / fr.den);

        double fps = (double)fr.num / fr.den;
        if (fps > 50 && gop_duration_sec > 2)
        {
            LOG_WARN("High frame rate (%.1f fps) with %d-sec segments may cause periodic slowdowns",
                     fps, gop_duration_sec);
            LOG_WARN("Consider using -hls_time 2 for better performance at high FPS");
        }

        LOG_INFO("Aligning GOP size (%d sec = %d frames) with HLS segment duration",
                 gop_duration_sec, gop_size_frames);
    }

    out.venc->gop_size = gop_size_frames;
    out.venc->max_b_frames = cfg.bframes;
    out.venc->color_range = AVCOL_RANGE_MPEG;

    // For HLS: Set minimum keyframe interval equal to GOP size
    // This matches FFmpeg's -keyint_min behavior and ensures strict GOP alignment
    if (hls_enabled)
    {
        av_opt_set_int(out.venc->priv_data, "g", gop_size_frames, 0);
        // Note: NVENC doesn't have keyint_min, but strict_gop below achieves the same
    }

    // HDR color settings
    if (outputHDR)
    {
        if (inputIsHDR)
        {
            out.venc->color_trc = in.vst->codecpar->color_trc;
            out.venc->color_primaries = in.vst->codecpar->color_primaries;
            out.venc->colorspace = in.vst->codecpar->color_space;
        }
        else
        {
            out.venc->color_trc = AVCOL_TRC_SMPTE2084;
            out.venc->color_primaries = AVCOL_PRI_BT2020;
            out.venc->colorspace = AVCOL_SPC_BT2020_NCL;
        }
    }
    else
    {
        out.venc->color_trc = AVCOL_TRC_BT709;
        out.venc->color_primaries = AVCOL_PRI_BT709;
        out.venc->colorspace = AVCOL_SPC_BT709;
    }

    if ((out.fmt->oformat->flags & AVFMT_GLOBALHEADER) || !mux_is_isobmff || hls_enabled)
    {
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

    if (hls_enabled)
    {
        av_opt_set(out.venc->priv_data, "forced-idr", "1", 0);
        av_opt_set(out.venc->priv_data, "strict_gop", "1", 0);
        LOG_INFO("Enabled forced-idr + strict_gop for HLS segment alignment (ensures I-frame at every segment boundary)");
    }

    if (outputHDR)
    {
        av_opt_set(out.venc->priv_data, "profile", "main10", 0);
    }
    else
    {
        av_opt_set(out.venc->priv_data, "profile", "main", 0);
    }

    AVBufferRef *enc_hw_frames = nullptr;
    if (use_cuda_path)
    {
        enc_hw_frames = av_hwframe_ctx_alloc(in.hw_device_ctx);
        if (!enc_hw_frames)
            throw std::runtime_error("av_hwframe_ctx_alloc failed for encoder");
        AVHWFramesContext *fctx = (AVHWFramesContext *)enc_hw_frames->data;
        fctx->format = AV_PIX_FMT_CUDA;
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

// Configure stream metadata
void configure_stream_metadata(InputContext &in, OutputContext &out, PipelineConfig &cfg,
                               bool inputIsHDR, bool mux_is_isobmff, bool hls_enabled)
{
    bool outputHDR = cfg.rtxCfg.enableTHDR || inputIsHDR;

    ff_check(avcodec_parameters_from_context(out.vstream->codecpar, out.venc), "enc params to stream");
    if (out.vstream->codecpar->extradata_size == 0 || out.vstream->codecpar->extradata == nullptr)
    {
        throw std::runtime_error("HEVC encoder did not provide extradata; required for Matroska outputs");
    }
    LOG_DEBUG("Encoder extradata size: %d bytes", out.vstream->codecpar->extradata_size);
    if (out.vstream->codecpar->extradata_size >= 4)
    {
        const uint8_t *ed = out.vstream->codecpar->extradata;
        LOG_DEBUG("Extradata head: %02X %02X %02X %02X", ed[0], ed[1], ed[2], ed[3]);
    }

    if (out.vstream->codecpar)
    {
        if (mux_is_isobmff)
        {
            out.vstream->codecpar->codec_tag = MKTAG('h', 'v', 'c', '1');
        }
        else
        {
            out.vstream->codecpar->codec_tag = 0;
        }
        out.vstream->codecpar->color_range = AVCOL_RANGE_MPEG;
        if (outputHDR)
        {
            if (inputIsHDR)
            {
                out.vstream->codecpar->color_trc = in.vst->codecpar->color_trc;
                out.vstream->codecpar->color_primaries = in.vst->codecpar->color_primaries;
                out.vstream->codecpar->color_space = in.vst->codecpar->color_space;
            }
            else
            {
                out.vstream->codecpar->color_trc = AVCOL_TRC_SMPTE2084;
                out.vstream->codecpar->color_primaries = AVCOL_PRI_BT2020;
                out.vstream->codecpar->color_space = AVCOL_SPC_BT2020_NCL;
            }
        }
        else
        {
            out.vstream->codecpar->color_trc = AVCOL_TRC_BT709;
            out.vstream->codecpar->color_primaries = AVCOL_PRI_BT709;
            out.vstream->codecpar->color_space = AVCOL_SPC_BT709;
        }
    }

    if (outputHDR)
    {
        if (inputIsHDR)
        {
            LOG_INFO("Preserving HDR mastering metadata from input stream");
            add_mastering_and_cll(out.vstream, 4000);
        }
        else
        {
            add_mastering_and_cll(out.vstream, cfg.rtxCfg.thdrMaxLuminance);
        }
    }
}
