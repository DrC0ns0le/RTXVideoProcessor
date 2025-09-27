#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <queue>

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/packet.h>
#include <libavutil/avutil.h>
#include <libavutil/common.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/display.h>
#include <libavutil/rational.h>
#include <libswscale/swscale.h>
}

#include <cuda_runtime.h>

#include "ffmpeg_utils.h"
#include "rtx_processor.h"

// Global flag for logging
static bool g_verbose = false;
static bool g_debug = false;

// Verbose logging macro
#define LOG_VERBOSE(...)                   \
    do                                     \
    {                                      \
        if (g_verbose)                     \
        {                                  \
            fprintf(stderr, "[VERBOSE] "); \
            fprintf(stderr, __VA_ARGS__);  \
            fprintf(stderr, "\n");         \
        }                                  \
    } while (0)

// Error logging macro
#define LOG_ERROR(...)                \
    do                                \
    {                                 \
        fprintf(stderr, "[ERROR] ");  \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n");        \
    } while (0)

// Info logging macro
#define LOG_INFO(...)                 \
    do                                \
    {                                 \
        fprintf(stderr, "[INFO] ");   \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n");        \
    } while (0)

// Debug logging macro
#define LOG_DEBUG(...)                    \
    do                                    \
    {                                     \
        if (g_debug)                      \
        {                                 \
            fprintf(stderr, "[DEBUG] ");  \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n");        \
        }                                 \
    } while (0)

// Warning logging macro
#define LOG_WARN(...)                 \
    do                                \
    {                                 \
        fprintf(stderr, "[WARN] ");   \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n");        \
    } while (0)

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

using FramePtr = std::unique_ptr<AVFrame, void (*)(AVFrame *)>;
using PacketPtr = std::unique_ptr<AVPacket, void (*)(AVPacket *)>;

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

struct PipelineConfig
{
    bool verbose = false;
    bool cpuOnly = false;

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
};

static AVPixelFormat
get_cuda_sw_format(AVCodecContext *ctx, const AVPixelFormat *pix_fmts)
{
    while (*pix_fmts != AV_PIX_FMT_NONE)
    {
        if (*pix_fmts == AV_PIX_FMT_CUDA)
            return *pix_fmts;
        pix_fmts++;
    }
    return AV_PIX_FMT_NONE;
}

static bool open_input(const char *inPath, InputContext &in)
{
    LOG_DEBUG("Opening input file %s...", inPath);
    ff_check(avformat_open_input(&in.fmt, inPath, nullptr, nullptr), "open input");
    LOG_DEBUG("Finding stream info...");
    ff_check(avformat_find_stream_info(in.fmt, nullptr), "find stream info");
    LOG_DEBUG("Found %d streams", in.fmt->nb_streams);

    // Find best video stream
    LOG_DEBUG("Looking for video stream...");
    in.vstream = av_find_best_stream(in.fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (in.vstream < 0)
        throw std::runtime_error("no video stream");
    in.vst = in.fmt->streams[in.vstream];
    LOG_VERBOSE("Found video stream %d: %dx%d, codec: %s", in.vstream,
                in.vst->codecpar->width, in.vst->codecpar->height,
                avcodec_get_name(in.vst->codecpar->codec_id));

    const AVCodec *dec = avcodec_find_decoder(in.vst->codecpar->codec_id);
    if (!dec)
        throw std::runtime_error("decoder not found");

    in.vdec = avcodec_alloc_context3(dec);
    if (!in.vdec)
        throw std::runtime_error("alloc dec ctx");
    ff_check(avcodec_parameters_to_context(in.vdec, in.vst->codecpar), "copy dec params");

    // Try to enable CUDA hwaccel
    if (av_hwdevice_ctx_create(&in.hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) >= 0)
    {
        LOG_VERBOSE("CUDA hardware acceleration enabled for decoder");
        in.vdec->hw_device_ctx = av_buffer_ref(in.hw_device_ctx);
        in.vdec->get_format = [](AVCodecContext *ctx, const AVPixelFormat *pix_fmts)
        {
            return get_cuda_sw_format(ctx, pix_fmts);
        };
    }
    else
    {
        LOG_WARN("WARNING: CUDA hardware acceleration not available, using CPU decoder");
    }

    ff_check(avcodec_open2(in.vdec, dec, nullptr), "open decoder");
    return true;
}

static void close_input(InputContext &in)
{
    if (in.vdec)
        avcodec_free_context(&in.vdec);
    if (in.hw_device_ctx)
        av_buffer_unref(&in.hw_device_ctx);
    if (in.fmt)
        avformat_close_input(&in.fmt);
}

static void add_mastering_and_cll(AVStream *st)
{
    // Attach HDR mastering metadata to codec parameters coded_side_data (FFmpeg >= 8)
    AVPacketSideData *sd = av_packet_side_data_new(&st->codecpar->coded_side_data,
                                                   &st->codecpar->nb_coded_side_data,
                                                   AV_PKT_DATA_MASTERING_DISPLAY_METADATA,
                                                   sizeof(AVMasteringDisplayMetadata), 0);
    if (sd && sd->data && sd->size == sizeof(AVMasteringDisplayMetadata))
    {
        AVMasteringDisplayMetadata *mdm = (AVMasteringDisplayMetadata *)sd->data;
        memset(mdm, 0, sizeof(*mdm));
        // BT.2020 primaries and D65 white point
        mdm->display_primaries[0][0] = av_d2q(0.708, 100000); // R x
        mdm->display_primaries[0][1] = av_d2q(0.292, 100000); // R y
        mdm->display_primaries[1][0] = av_d2q(0.170, 100000); // G x
        mdm->display_primaries[1][1] = av_d2q(0.797, 100000); // G y
        mdm->display_primaries[2][0] = av_d2q(0.131, 100000); // B x
        mdm->display_primaries[2][1] = av_d2q(0.046, 100000); // B y
        mdm->white_point[0] = av_d2q(0.3127, 100000);
        mdm->white_point[1] = av_d2q(0.3290, 100000);

        // Luminance values must be encoded according to SMPTE ST 2086 standard
        // Values are in units of 0.0001 cd/m² (candelas per square meter)
        // For 1000 cd/m² max: 1000 * 10000 = 10000000 units
        // For 0.005 cd/m² min: 0.005 * 10000 = 50 units
        mdm->min_luminance = av_make_q(50, 10000);       // 0.005 cd/m² = 50/10000
        mdm->max_luminance = av_make_q(10000000, 10000); // 1000 cd/m² = 10000000/10000
        mdm->has_luminance = 1;
        mdm->has_primaries = 1;
    }

    sd = av_packet_side_data_new(&st->codecpar->coded_side_data,
                                 &st->codecpar->nb_coded_side_data,
                                 AV_PKT_DATA_CONTENT_LIGHT_LEVEL,
                                 sizeof(AVContentLightMetadata), 0);
    if (sd && sd->data && sd->size == sizeof(AVContentLightMetadata))
    {
        AVContentLightMetadata *cll = (AVContentLightMetadata *)sd->data;
        cll->MaxCLL = 1000;
        cll->MaxFALL = 400;
    }
}

static bool open_output(const char *outPath, const InputContext &in, OutputContext &out)
{
    LOG_DEBUG("Opening output file: %s", outPath);
    ff_check(avformat_alloc_output_context2(&out.fmt, nullptr, nullptr, outPath), "alloc out ctx");
    if (!out.fmt)
        throw std::runtime_error("cannot alloc out ctx");
    // Ensure packets are flushed promptly for progressive playback while encoding
    out.fmt->flags |= AVFMT_FLAG_FLUSH_PACKETS;

    // Map streams: copy all non-video as is (audio/subtitles). Create new video stream.
    out.map_streams.assign(in.fmt->nb_streams, -1);

    // Create video encoder stream
    LOG_DEBUG("Looking for HEVC encoder...");
    const AVCodec *hevc = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!hevc)
        hevc = avcodec_find_encoder(AV_CODEC_ID_HEVC);
    if (!hevc)
        throw std::runtime_error("hevc encoder not found");
    LOG_VERBOSE("Found HEVC encoder: %s", hevc->name);

    out.vstream = avformat_new_stream(out.fmt, hevc);
    if (!out.vstream)
        throw std::runtime_error("new video stream");
    out.vstream->id = (int)out.fmt->nb_streams - 1;

    out.venc = avcodec_alloc_context3(hevc);
    if (!out.venc)
        throw std::runtime_error("alloc enc ctx");

    // We'll fill details later after first input frame size/fps is known.

    // Copy other streams (audio/subs) with codec copy
    for (unsigned i = 0; i < in.fmt->nb_streams; ++i)
    {
        if ((int)i == in.vstream)
            continue;

        AVStream *ist = in.fmt->streams[i];

        // Skip unsupported subtitle formats for MP4
        if (ist->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE)
        {
            const AVCodecDescriptor *desc = avcodec_descriptor_get(ist->codecpar->codec_id);
            if (desc && strcmp(desc->name, "subrip") == 0)
            {
                fprintf(stderr, "Note: Skipping SRT subtitles (stream %d) - not supported in MP4 container\n", i);
                continue; // Skip this stream
            }
        }

        AVStream *ost = avformat_new_stream(out.fmt, nullptr);
        if (!ost)
            throw std::runtime_error("new stream (copy)");
        ff_check(avcodec_parameters_copy(ost->codecpar, ist->codecpar), "copy stream params");
        ost->codecpar->codec_tag = 0;
        ost->time_base = ist->time_base; // Preserve input stream time base to ensure correct timestamp scaling for copied streams
        out.map_streams[i] = ost->index;
    }

    // Open IO
    if (!(out.fmt->oformat->flags & AVFMT_NOFILE))
    {
        ff_check(avio_open(&out.fmt->pb, outPath, AVIO_FLAG_WRITE), "open output file");
    }

    return true;
}

static void close_output(OutputContext &out)
{
    if (out.venc)
        avcodec_free_context(&out.venc);
    if (out.fmt)
    {
        if (!(out.fmt->oformat->flags & AVFMT_NOFILE) && out.fmt->pb)
            avio_closep(&out.fmt->pb);
        avformat_free_context(out.fmt);
        out.fmt = nullptr;
    }
}

static void print_help(const char *argv0)
{
    fprintf(stderr, "Usage: %s input.mp4 output.mp4 [options]\\n", argv0);
    fprintf(stderr, "\\nOptions:\\n");
    fprintf(stderr, "  -v, --verbose Enable verbose logging\\n");
    fprintf(stderr, "  --cpu         Bypass GPU for video processing pipeline other than RTX processing\\n");
    fprintf(stderr, "\nVSR options:\n");
    fprintf(stderr, "  --no-vsr      Disable VSR\n");
    fprintf(stderr, "  --vsr-quality     Set VSR quality, default 4\n");
    fprintf(stderr, "\nTHDR options:\n");
    fprintf(stderr, "  --no-thdr     Disable THDR\n");
    fprintf(stderr, "  --thdr-contrast   Set THDR contrast, default 110\n");
    fprintf(stderr, "  --thdr-saturation Set THDR saturation, default 75\n");
    fprintf(stderr, "  --thdr-middle-gray Set THDR middle gray, default 30\n");
    fprintf(stderr, "  --thdr-max-luminance Set THDR max luminance, default 1000\n");
    fprintf(stderr, "\nNVENC options:\n");
    fprintf(stderr, "  --nvenc-tune        Set NVENC tune, default hq\n");
    fprintf(stderr, "  --nvenc-preset      Set NVENC preset, default p4\n");
    fprintf(stderr, "  --nvenc-rc          Set NVENC rate control, default constqp\n");
    fprintf(stderr, "  --nvenc-gop         Set NVENC GOP, default 1\n");
    fprintf(stderr, "  --nvenc-bframes     Set NVENC bframes, default 2\n");
    fprintf(stderr, "  --nvenc-qp          Set NVENC QP, default 21\n");
    fprintf(stderr, "  --nvenc-bitrate-multiplier Set NVENC bitrate multiplier, default 10\n");
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

    cfg->gop = 1;
    cfg->bframes = 0;
    cfg->qp = 21;
    cfg->targetBitrateMultiplier = 10;

    cfg->inputPath = argv[1];
    cfg->outputPath = argv[2];

    for (int i = 3; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v")
            cfg->verbose = true;
        else if (arg == "--cpu" || arg == "-cpu")
            cfg->cpuOnly = true;
        else if (arg == "--help" || arg == "-h")
        {
            print_help(argv[0]);
            exit(0);
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
            cfg->rtxCfg.enableTHDR = false;
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
        }
    }
}

int run_pipeline(PipelineConfig cfg)
{
    // Set global verbose flag
    g_verbose = cfg.verbose;

    LOG_VERBOSE("Starting video processing pipeline");
    LOG_DEBUG("Input: %s", cfg.inputPath);
    LOG_DEBUG("Output: %s", cfg.outputPath);
    LOG_VERBOSE("CPU-only mode: %s", cfg.cpuOnly ? "enabled" : "disabled");

    // Start pipeline
    InputContext in{};
    OutputContext out{};

    try
    {
        open_input(cfg.inputPath, in);
        open_output(cfg.outputPath, in, out);

        // Read input bitrate and fps
        AVRational fr = in.vst->avg_frame_rate.num ? in.vst->avg_frame_rate : in.vst->r_frame_rate;
        if (fr.num == 0 || fr.den == 0)
            fr = {in.vst->time_base.den, in.vst->time_base.num};

        // Get total duration and frames for progress tracking
        int64_t total_frames = 0;
        if (in.vst->nb_frames > 0)
        {
            total_frames = in.vst->nb_frames;
        }
        else
        {
            // Estimate total frames from duration if frame count is not available
            double duration_sec = 0.0;
            if (in.vst->duration > 0 && in.vst->duration != AV_NOPTS_VALUE)
            {
                duration_sec = in.vst->duration * av_q2d(in.vst->time_base);
            }
            else if (in.fmt->duration != AV_NOPTS_VALUE)
            {
                duration_sec = static_cast<double>(in.fmt->duration) / AV_TIME_BASE;
            }

            if (duration_sec > 0.0 && fr.num > 0 && fr.den > 0)
            {
                total_frames = static_cast<int64_t>(duration_sec * av_q2d(fr) + 0.5);
            }
        }

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
        int dstW = in.vdec->width * cfg.rtxCfg.scaleFactor;
        int dstH = in.vdec->height * cfg.rtxCfg.scaleFactor;
        SwsContext *sws_to_p010 = nullptr; // CPU fallback
        bool use_cuda_path = (in.vdec->hw_device_ctx != nullptr) && !cfg.cpuOnly;

        LOG_VERBOSE("Processing path: %s", use_cuda_path ? "GPU (CUDA)" : "CPU");
        LOG_VERBOSE("Output resolution: %dx%d (scale factor: %d)", dstW, dstH, cfg.rtxCfg.scaleFactor);

        // Configure encoder now that sizes are known
        LOG_DEBUG("Configuring HEVC encoder...");
        out.venc->codec_id = AV_CODEC_ID_HEVC;
        out.venc->width = dstW;
        out.venc->height = dstH;
        out.venc->time_base = av_inv_q(fr);
        out.venc->framerate = fr;
        // Prefer CUDA frames if decoder is CUDA-capable to avoid copies
        if (use_cuda_path)
        {
            out.venc->pix_fmt = AV_PIX_FMT_CUDA; // NVENC consumes CUDA frames via hw_frames_ctx
        }
        else
        {
            out.venc->pix_fmt = AV_PIX_FMT_P010LE; // CPU fallback
        }
        out.venc->gop_size = cfg.gop * fr.num / std::max(1, fr.den);
        out.venc->max_b_frames = 2;
        out.venc->color_range = AVCOL_RANGE_MPEG;
        out.venc->color_trc = AVCOL_TRC_SMPTE2084; // PQ
        out.venc->color_primaries = AVCOL_PRI_BT2020;
        out.venc->colorspace = AVCOL_SPC_BT2020_NCL;

        int64_t target_bitrate = (in.fmt->bit_rate > 0) ? (int64_t)(in.fmt->bit_rate * cfg.targetBitrateMultiplier) : (int64_t)25000000;
        LOG_VERBOSE("Input bitrate: %lld (Mbps), Target bitrate: %lld (Mbps)\n", in.fmt->bit_rate / 1000000, target_bitrate / 1000000);
        LOG_VERBOSE("Encoder settings - tune: %s, preset: %s, rc: %s, qp: %d",
                    cfg.tune.c_str(), cfg.preset.c_str(), cfg.rc.c_str(), cfg.qp);
        av_opt_set(out.venc->priv_data, "tune", cfg.tune.c_str(), 0);
        av_opt_set(out.venc->priv_data, "preset", cfg.preset.c_str(), 0);
        // Switch to constant QP rate control
        av_opt_set(out.venc->priv_data, "rc", cfg.rc.c_str(), 0);
        av_opt_set_int(out.venc->priv_data, "qp", cfg.qp, 0);
        av_opt_set(out.venc->priv_data, "profile", "main10", 0);
        ff_check(avcodec_parameters_from_context(out.vstream->codecpar, out.venc), "enc params to stream");
        // Prefer 'hvc1' brand to carry parameter sets (VPS/SPS/PPS) in-band, which improves fMP4 compatibility, and required by macOS
        if (out.vstream->codecpar)
        {
            out.vstream->codecpar->codec_tag = MKTAG('h', 'v', 'c', '1');
            out.vstream->codecpar->color_range = AVCOL_RANGE_MPEG;
            out.vstream->codecpar->color_trc = AVCOL_TRC_SMPTE2084;
            out.vstream->codecpar->color_primaries = AVCOL_PRI_BT2020;
        }
        add_mastering_and_cll(out.vstream);

        // If using CUDA path, create encoder hw_frames_ctx on the same device before opening encoder
        AVBufferRef *enc_hw_frames = nullptr;
        if (use_cuda_path)
        {
            enc_hw_frames = av_hwframe_ctx_alloc(in.hw_device_ctx);
            if (!enc_hw_frames)
                throw std::runtime_error("av_hwframe_ctx_alloc failed for encoder");
            AVHWFramesContext *fctx = (AVHWFramesContext *)enc_hw_frames->data;
            fctx->format = AV_PIX_FMT_CUDA;
            fctx->sw_format = AV_PIX_FMT_P010LE;
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
        if (enc_hw_frames)
            av_buffer_unref(&enc_hw_frames);

        // CUDA frame pool for optimized GPU processing
        std::vector<FramePtr> cuda_frame_pool;
        const int POOL_SIZE = 8; // Adjust based on your needs
        int pool_index = 0;

        // Frame buffering for handling processing spikes
        std::queue<std::pair<AVFrame *, int64_t>> frame_buffer; // frame and output_pts pairs
        const int MAX_BUFFER_SIZE = 4;

        // Pre-allocate CUDA frames if using CUDA path
        if (use_cuda_path)
        {
            cuda_frame_pool.reserve(POOL_SIZE);
            for (int i = 0; i < POOL_SIZE; i++)
            {
                FramePtr enc_hw(av_frame_alloc(), &av_frame_free_single);
                enc_hw->format = AV_PIX_FMT_CUDA;
                enc_hw->width = dstW;
                enc_hw->height = dstH;
                ff_check(av_hwframe_get_buffer(out.venc->hw_frames_ctx, enc_hw.get(), 0), "alloc enc hw frame pool");
                cuda_frame_pool.push_back(std::move(enc_hw));
            }
        }

        // Write header
        out.vstream->time_base = out.venc->time_base; // Ensure muxer stream uses the same time_base as the encoder for consistent PTS/DTS
        out.vstream->avg_frame_rate = fr;
        // Write header with MOV/MP4 muxer flags for broad compatibility
        // faststart: move moov atom to the beginning at finalize time; plays everywhere after encode completes
        AVDictionary *movopts = nullptr;
        // write_colr: ensure colr (nclx) atom is written from color_* fields (needed for HDR on macOS)
        av_dict_set(&movopts, "movflags", "+faststart+write_colr", 0);
        ff_check(avformat_write_header(out.fmt, &movopts), "write header");
        av_dict_free(&movopts);

        // Frame buffers
        FramePtr frame(av_frame_alloc(), &av_frame_free_single);
        FramePtr swframe(av_frame_alloc(), &av_frame_free_single);
        FramePtr bgra_frame(av_frame_alloc(), &av_frame_free_single);
        FramePtr p010_frame(av_frame_alloc(), &av_frame_free_single);

        bgra_frame->format = AV_PIX_FMT_RGBA;
        bgra_frame->width = in.vdec->width;
        bgra_frame->height = in.vdec->height;
        ff_check(av_frame_get_buffer(bgra_frame.get(), 32), "alloc bgra");

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

        PacketPtr pkt(av_packet_alloc(), &av_packet_free_single);
        PacketPtr opkt(av_packet_alloc(), &av_packet_free_single);

        // PTS handling: derive video PTS from input timestamps to avoid A/V drift
        int64_t v_start_pts = AV_NOPTS_VALUE;              // first input video pts
        int64_t last_output_pts = AV_NOPTS_VALUE;          // last emitted video pts in encoder tb
        AVRational output_time_base = out.venc->time_base; // encoder time base

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

        RTXProcessor rtx_gpu;
        RTXProcessor rtx_cpu; // CPU-path RTX instance; initialize lazily only if CPU path is used
        bool rtx_cpu_init = false;
        bool rtx_gpu_init = false;

        // Read packets
        while (true)
        {
            int ret = av_read_frame(in.fmt, pkt.get());
            if (ret == AVERROR_EOF)
                break;
            ff_check(ret, "read frame");

            // Skip non-video packets
            if (pkt->stream_index == in.vstream)
            {
                ff_check(avcodec_send_packet(in.vdec, pkt.get()), "send packet");
                av_packet_unref(pkt.get());

                while (true)
                {
                    ret = avcodec_receive_frame(in.vdec, frame.get());
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

                    // Compute output PTS by rescaling input timestamps to encoder time base
                    int64_t in_pts = (decframe->pts != AV_NOPTS_VALUE)
                                         ? decframe->pts
                                         : decframe->best_effort_timestamp;
                    // Establish a zero-based timeline from the first valid input PTS
                    if (in_pts != AV_NOPTS_VALUE)
                    {
                        if (v_start_pts == AV_NOPTS_VALUE)
                            v_start_pts = in_pts;
                        if (v_start_pts != AV_NOPTS_VALUE)
                            in_pts -= v_start_pts;
                    }
                    // Fallback to frame counter if input PTS is unavailable
                    static int64_t synthetic_counter = 0;
                    int64_t output_pts = (in_pts != AV_NOPTS_VALUE)
                                             ? av_rescale_q(in_pts, in.vst->time_base, out.venc->time_base)
                                             : synthetic_counter++;
                    // Ensure strict monotonicity to satisfy encoder/muxer
                    // if (last_output_pts != AV_NOPTS_VALUE && output_pts <= last_output_pts) {
                    //     output_pts = last_output_pts + 1;
                    // }
                    // last_output_pts = output_pts;

                    AVFrame *frame_to_send = nullptr;
                    if (frame_is_cuda && use_cuda_path)
                    {
                        // GPU path: reuse pre-allocated frame from pool instead of allocating new one
                        AVFrame *enc_hw = cuda_frame_pool[pool_index].get();
                        pool_index = (pool_index + 1) % POOL_SIZE;

                        // Clear previous frame data
                        av_frame_unref(enc_hw);
                        enc_hw->format = AV_PIX_FMT_CUDA;
                        enc_hw->width = dstW;
                        enc_hw->height = dstH;

                        // Get buffer for this frame (much faster than full allocation)
                        ff_check(av_hwframe_get_buffer(out.venc->hw_frames_ctx, enc_hw, 0), "get enc hw frame buffer");

                        if (!rtx_gpu_init)
                        {
                            AVHWDeviceContext *devctx = (AVHWDeviceContext *)in.hw_device_ctx->data;
                            AVCUDADeviceContext *cudactx = (AVCUDADeviceContext *)devctx->hwctx;
                            CUcontext cu = cudactx->cuda_ctx;
                            if (!rtx_gpu.initializeWithContext(cu, cfg.rtxCfg, in.vdec->width, in.vdec->height))
                            {
                                std::string detail = rtx_gpu.lastError();
                                if (detail.empty())
                                    detail = "unknown error";
                                throw std::runtime_error(std::string("Failed to init RTX GPU path: ") + detail);
                            }
                            rtx_gpu_init = true;
                        }

                        bool bt2020 = (in.vdec->colorspace == AVCOL_SPC_BT2020_NCL);
                        if (rtx_gpu.processGpuNV12ToP010(decframe->data[0], decframe->linesize[0],
                                                         decframe->data[1], decframe->linesize[1],
                                                         enc_hw, bt2020))
                        {
                            frame_to_send = enc_hw;
                            // Update progress prior to encoding
                            processed_frames++;
                            show_progress();

                            // Set consistent PTS to prevent stuttering
                            frame_to_send->pts = output_pts;

                            // Encode and then release enc_hw by letting enc_hw FramePtr go out of scope after send
                            // Ensure all GPU work for this frame completes before encoding and moving to next frame
                            cudaStreamSynchronize(0);
                            ff_check(avcodec_send_frame(out.venc, frame_to_send), "send frame to encoder (CUDA)");
                            while (true)
                            {
                                ret = avcodec_receive_packet(out.venc, opkt.get());
                                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                                    break;
                                ff_check(ret, "receive encoded packet");
                                opkt->stream_index = out.vstream->index;
                                av_packet_rescale_ts(opkt.get(), out.venc->time_base, out.vstream->time_base);
                                ff_check(av_interleaved_write_frame(out.fmt, opkt.get()), "write video packet");
                                av_packet_unref(opkt.get());
                            }
                            av_frame_unref(frame.get());
                            if (swframe)
                                av_frame_unref(swframe.get());
                            continue; // handled this frame on GPU
                        }

                        // GPU path failed; fall back: transfer to SW, run CPU path, convert to P010, then upload into enc_hw
                        if (!swframe)
                            swframe.reset(av_frame_alloc());
                        ff_check(av_hwframe_transfer_data(swframe.get(), decframe, 0), "hwframe transfer fallback");

                        // Convert to ARGB
                        if (!sws_to_argb || last_src_format != swframe->format)
                        {
                            if (sws_to_argb)
                                sws_freeContext(sws_to_argb);
                            sws_to_argb = sws_getContext(
                                swframe->width, swframe->height, (AVPixelFormat)swframe->format,
                                swframe->width, swframe->height, AV_PIX_FMT_RGBA,
                                SWS_BILINEAR, nullptr, nullptr, nullptr);
                            if (!sws_to_argb)
                                throw std::runtime_error("sws_to_argb alloc failed");
                            const int *coeffs = (in.vdec->colorspace == AVCOL_SPC_BT2020_NCL)
                                                    ? sws_getCoefficients(SWS_CS_BT2020)
                                                    : sws_getCoefficients(SWS_CS_ITU709);
                            sws_setColorspaceDetails(sws_to_argb, coeffs, 0, coeffs, 1, 0, 1 << 16, 1 << 16);
                            last_src_format = swframe->format;
                        }
                        const uint8_t *srcData2[AV_NUM_DATA_POINTERS] = {swframe->data[0], swframe->data[1], swframe->data[2], swframe->data[3]};
                        int srcLines2[AV_NUM_DATA_POINTERS] = {swframe->linesize[0], swframe->linesize[1], swframe->linesize[2], swframe->linesize[3]};
                        ff_check(av_frame_make_writable(bgra_frame.get()), "argb make writable (fallback)");
                        sws_scale(sws_to_argb, srcData2, srcLines2, 0, swframe->height, bgra_frame->data, bgra_frame->linesize);

                        // Lazy-initialize CPU RTX processor when first needed (fallback)
                        if (!rtx_cpu_init)
                        {
                            if (!rtx_cpu.initialize(0, cfg.rtxCfg, in.vdec->width, in.vdec->height))
                            {
                                std::string detail = rtx_cpu.lastError();
                                if (detail.empty())
                                    detail = "unknown error";
                                throw std::runtime_error(std::string("Failed to initialize RTX CPU path (fallback): ") + detail);
                            }
                            rtx_cpu_init = true;
                        }
                        // RTX CPU process
                        if (!rtx_cpu.process(bgra_frame->data[0], (size_t)bgra_frame->linesize[0], rtx_data, rtxW, rtxH, rtxPitch))
                            throw std::runtime_error("RTX CPU processing failed (fallback)");
                        if (rtxW != (uint32_t)dstW || rtxH != (uint32_t)dstH)
                            throw std::runtime_error("Unexpected RTX output dimensions (fallback)");

                        // ABGR10 -> P010 on CPU
                        ff_check(av_frame_make_writable(p010_frame.get()), "p010 make writable (fallback)");
                        const uint8_t *abgr_planes2[1] = {rtx_data};
                        int abgr_lines2[1] = {static_cast<int>(rtxPitch)};
                        sws_scale(sws_to_p010, abgr_planes2, abgr_lines2, 0, dstH, p010_frame->data, p010_frame->linesize);

                        // Upload P010 to encoder CUDA frame
                        ff_check(av_hwframe_transfer_data(enc_hw, p010_frame.get(), 0), "upload P010 to CUDA frame");

                        // Ensure GPU upload is complete before encoding
                        cudaStreamSynchronize(0);

                        // Set PTS and encode
                        frame_to_send = enc_hw;
                        // Set consistent PTS to prevent stuttering
                        frame_to_send->pts = output_pts;
                        // progress
                        processed_frames++;
                        show_progress();

                        ff_check(avcodec_send_frame(out.venc, frame_to_send), "send frame to encoder (fallback CUDA)");
                        while (true)
                        {
                            ret = avcodec_receive_packet(out.venc, opkt.get());
                            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                                break;
                            ff_check(ret, "receive encoded packet");
                            opkt->stream_index = out.vstream->index;
                            av_packet_rescale_ts(opkt.get(), out.venc->time_base, out.vstream->time_base);
                            ff_check(av_interleaved_write_frame(out.fmt, opkt.get()), "write video packet");
                            av_packet_unref(opkt.get());
                        }
                        av_frame_unref(frame.get());
                        if (swframe)
                            av_frame_unref(swframe.get());
                        continue; // handled this frame via fallback
                    }

                    // CPU path: Convert to RGBA for RTX input
                    if (!sws_to_argb || last_src_format != decframe->format)
                    {
                        if (sws_to_argb)
                            sws_freeContext(sws_to_argb);
                        sws_to_argb = sws_getContext(
                            decframe->width, decframe->height, (AVPixelFormat)decframe->format,
                            decframe->width, decframe->height, AV_PIX_FMT_RGBA, // ARGB for RTX
                            SWS_BILINEAR, nullptr, nullptr, nullptr);
                        if (!sws_to_argb)
                            throw std::runtime_error("sws_to_argb alloc failed");
                        const int *coeffs = (in.vdec->colorspace == AVCOL_SPC_BT2020_NCL)
                                                ? sws_getCoefficients(SWS_CS_BT2020)
                                                : sws_getCoefficients(SWS_CS_ITU709);
                        sws_setColorspaceDetails(sws_to_argb, coeffs, 0, coeffs, 1, 0, 1 << 16, 1 << 16);
                        last_src_format = decframe->format;
                    }
                    const uint8_t *srcData[AV_NUM_DATA_POINTERS] = {decframe->data[0], decframe->data[1], decframe->data[2], decframe->data[3]};
                    int srcLines[AV_NUM_DATA_POINTERS] = {decframe->linesize[0], decframe->linesize[1], decframe->linesize[2], decframe->linesize[3]};
                    ff_check(av_frame_make_writable(bgra_frame.get()), "argb make writable");
                    sws_scale(sws_to_argb, srcData, srcLines, 0, decframe->height, bgra_frame->data, bgra_frame->linesize);

                    // Update progress counter before processing
                    processed_frames++;
                    show_progress();

                    // Lazy-initialize CPU RTX processor when first needed
                    if (!rtx_cpu_init)
                    {
                        if (!rtx_cpu.initialize(0, cfg.rtxCfg, in.vdec->width, in.vdec->height))
                        {
                            std::string detail = rtx_cpu.lastError();
                            if (detail.empty())
                                detail = "unknown error";
                            throw std::runtime_error(std::string("Failed to initialize RTX CPU path: ") + detail);
                        }
                        rtx_cpu_init = true;
                    }
                    // Feed RTX (CPU path)
                    if (!rtx_cpu.process(bgra_frame->data[0], (size_t)bgra_frame->linesize[0], rtx_data, rtxW, rtxH, rtxPitch))
                        throw std::runtime_error("RTX processing failed");

                    if (rtxW != (uint32_t)dstW || rtxH != (uint32_t)dstH)
                    {
                        throw std::runtime_error("Unexpected RTX output dimensions");
                    }

                    // Convert RTX ABGR10 output directly into encoder P010 buffer (CPU path)
                    ff_check(av_frame_make_writable(p010_frame.get()), "p010 make writable");
                    const uint8_t *abgr_planes[1] = {rtx_data};
                    int abgr_lines[1] = {static_cast<int>(rtxPitch)};
                    sws_scale(sws_to_p010, abgr_planes, abgr_lines, 0, dstH, p010_frame->data, p010_frame->linesize);

                    // Set consistent PTS to prevent stuttering
                    p010_frame->pts = output_pts;

                    // Encode
                    ff_check(avcodec_send_frame(out.venc, p010_frame.get()), "send frame to encoder");
                    while (true)
                    {
                        ret = avcodec_receive_packet(out.venc, opkt.get());
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                            break;
                        ff_check(ret, "receive encoded packet");
                        opkt->stream_index = out.vstream->index;
                        av_packet_rescale_ts(opkt.get(), out.venc->time_base, out.vstream->time_base);
                        ff_check(av_interleaved_write_frame(out.fmt, opkt.get()), "write video packet");
                        av_packet_unref(opkt.get());
                    }
                    av_frame_unref(frame.get());
                    if (swframe)
                        av_frame_unref(swframe.get());
                }
            }
            else
            {
                // Copy other streams
                int out_index = out.map_streams[pkt->stream_index];
                if (out_index >= 0)
                {
                    AVStream *ist = in.fmt->streams[pkt->stream_index];
                    AVStream *ost = out.fmt->streams[out_index];
                    av_packet_rescale_ts(pkt.get(), ist->time_base, ost->time_base);
                    pkt->stream_index = out_index;
                    ff_check(av_interleaved_write_frame(out.fmt, pkt.get()), "write copied packet");
                }
                av_packet_unref(pkt.get());
            }
        }

        // Flush encoder
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

        ff_check(av_write_trailer(out.fmt), "write trailer");

        // Print final progress
        if (total_frames > 0)
        {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            double total_sec = total_ms / 1000.0;
            double avg_fps = (total_sec > 0) ? processed_frames / total_sec : 0.0;

            fprintf(stdout, "\nProcessing completed in %.1f seconds (%.1f fps)\n",
                    total_sec, avg_fps);
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
        if (rtx_cpu_init)
        {
            LOG_VERBOSE("Shutting down RTX CPU...");
            rtx_cpu.shutdown();
            LOG_VERBOSE("RTX CPU shutdown complete.");
        }
        if (rtx_gpu_init)
        {
            LOG_VERBOSE("Shutting down RTX GPU...");
            rtx_gpu.shutdown();
            LOG_VERBOSE("RTX GPU shutdown complete.");
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

    PipelineConfig cfg;

    init_setup(argc, argv, &cfg);

    // Set log level
    av_log_set_level(AV_LOG_WARNING);

    int ret = run_pipeline(cfg);
}