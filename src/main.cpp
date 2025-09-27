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
#include "frame_pool.h"
#include "ts_utils.h"
#include "processor.h"
#include "logger.h"


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

// FFmpeg setup helpers moved to ffmpeg_utils.cpp

static void print_help(const char *argv0)
{
    fprintf(stderr, "Usage: %s input.mp4 output.{mp4|mkv} [options]\n", argv0);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -v, --verbose Enable verbose logging\n");
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
    fprintf(stderr, "  --nvenc-preset      Set NVENC preset, default p7\n");
    fprintf(stderr, "  --nvenc-rc          Set NVENC rate control, default constqp\n");
    fprintf(stderr, "  --nvenc-gop         Set NVENC GOP, default 1\n");
    fprintf(stderr, "  --nvenc-bframes     Set NVENC bframes, default 2\n");
    fprintf(stderr, "  --nvenc-qp          Set NVENC QP, default 21\n");
    fprintf(stderr, "  --nvenc-bitrate-multiplier Set NVENC bitrate multiplier, default 5\n");
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
    cfg->preset = "p7";
    cfg->rc = "constqp";

    cfg->gop = 1;
    cfg->bframes = 2;
    cfg->qp = 21;
    cfg->targetBitrateMultiplier = 5;

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

int run_pipeline(PipelineConfig cfg)
{
    Logger::instance().setVerbose(cfg.verbose);
    Logger::instance().setDebug(false);

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
        // Auto-disable VSR for inputs >= 1920x1080 in either orientation
        if (cfg.rtxCfg.enableVSR) {
            bool ge1080p = (in.vdec->width >= 1920 && in.vdec->height >= 1080) ||
                           (in.vdec->width >= 1080 && in.vdec->height >= 1920);
            if (ge1080p) {
                LOG_INFO("Input resolution is %dx%d (>=1080p). Disabling VSR.", in.vdec->width, in.vdec->height);
                cfg.rtxCfg.enableVSR = false;
            }
        }
        int effScale = cfg.rtxCfg.enableVSR ? cfg.rtxCfg.scaleFactor : 1;
        int dstW = in.vdec->width * effScale;
        int dstH = in.vdec->height * effScale;
        SwsContext *sws_to_p010 = nullptr; // CPU fallback
        bool use_cuda_path = (in.vdec->hw_device_ctx != nullptr) && !cfg.cpuOnly;

        LOG_VERBOSE("Processing path: %s", use_cuda_path ? "GPU (CUDA)" : "CPU");
        LOG_VERBOSE("Output resolution: %dx%d (scale factor: %d)", dstW, dstH, effScale);
        std::string muxer_name = (out.fmt && out.fmt->oformat && out.fmt->oformat->name)
                                     ? out.fmt->oformat->name
                                     : "";
        bool mux_is_isobmff = muxer_name.find("mp4") != std::string::npos ||
                              muxer_name.find("mov") != std::string::npos;
        LOG_VERBOSE("Output container: %s",
                    muxer_name.empty() ? "unknown" : muxer_name.c_str());

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
        if (cfg.rtxCfg.enableTHDR) {
            out.venc->color_trc = AVCOL_TRC_SMPTE2084; // PQ
            out.venc->color_primaries = AVCOL_PRI_BT2020;
            out.venc->colorspace = AVCOL_SPC_BT2020_NCL;
        } else {
            out.venc->color_trc = AVCOL_TRC_BT709;
            out.venc->color_primaries = AVCOL_PRI_BT709;
            out.venc->colorspace = AVCOL_SPC_BT709;
        }

        // Ensure muxers that require extradata (e.g., Matroska) receive global headers
        if ((out.fmt->oformat->flags & AVFMT_GLOBALHEADER) || !mux_is_isobmff)
        {
            out.venc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        int64_t target_bitrate = (in.fmt->bit_rate > 0) ? (int64_t)(in.fmt->bit_rate * cfg.targetBitrateMultiplier) : (int64_t)25000000;
        LOG_VERBOSE("Input bitrate: %.2f (Mbps), Target bitrate: %.2f (Mbps)\n", in.fmt->bit_rate / 1000000.0, target_bitrate / 1000000.0);
        LOG_VERBOSE("Encoder settings - tune: %s, preset: %s, rc: %s, qp: %d, gop: %d, bframes: %d",
                    cfg.tune.c_str(), cfg.preset.c_str(), cfg.rc.c_str(), cfg.qp, cfg.gop, cfg.bframes);
        av_opt_set(out.venc->priv_data, "tune", cfg.tune.c_str(), 0);
        av_opt_set(out.venc->priv_data, "preset", cfg.preset.c_str(), 0);
        // Switch to constant QP rate control
        av_opt_set(out.venc->priv_data, "rc", cfg.rc.c_str(), 0);
        av_opt_set_int(out.venc->priv_data, "qp", cfg.qp, 0);
        av_opt_set(out.venc->priv_data, "profile", "main10", 0);

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
        // Prefer 'hvc1' brand to carry parameter sets (VPS/SPS/PPS) in-band, which improves fMP4 compatibility, and required by macOS
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
            if (cfg.rtxCfg.enableTHDR) {
                out.vstream->codecpar->color_trc = AVCOL_TRC_SMPTE2084;
                out.vstream->codecpar->color_primaries = AVCOL_PRI_BT2020;
                out.vstream->codecpar->color_space = AVCOL_SPC_BT2020_NCL;
            } else {
                out.vstream->codecpar->color_trc = AVCOL_TRC_BT709;
                out.vstream->codecpar->color_primaries = AVCOL_PRI_BT709;
                out.vstream->codecpar->color_space = AVCOL_SPC_BT709;
            }
        }
        if (cfg.rtxCfg.enableTHDR) {
            add_mastering_and_cll(out.vstream, cfg.rtxCfg.thdrMaxLuminance);
        }

        // CUDA frame pool for optimized GPU processing
        CudaFramePool cuda_pool;
        const int POOL_SIZE = 8; // Adjust based on your needs

        // Frame buffering for handling processing spikes
        std::queue<std::pair<AVFrame *, int64_t>> frame_buffer; // frame and output_pts pairs
        const int MAX_BUFFER_SIZE = 4;

        // Initialize CUDA frame pool if using CUDA path
        if (use_cuda_path)
        {
            cuda_pool.initialize(out.venc->hw_frames_ctx, dstW, dstH, POOL_SIZE);
        }

        // Write header
        out.vstream->time_base = out.venc->time_base; // Ensure muxer stream uses the same time_base as the encoder for consistent PTS/DTS
        out.vstream->avg_frame_rate = fr;
        // Write header with MOV/MP4 muxer flags for broad compatibility
        // faststart: move moov atom to the beginning at finalize time; plays everywhere after encode completes
        AVDictionary *muxopts = nullptr;
        // write_colr: ensure colr (nclx) atom is written from color_* fields (needed for HDR on macOS)
        if (mux_is_isobmff)
        {
            av_dict_set(&muxopts, "movflags", "+faststart+write_colr", 0);
        }
        ff_check(avformat_write_header(out.fmt, &muxopts), "write header");
        av_dict_free(&muxopts);

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

        // Single processor instance; choose CPU or GPU at initialization and stick with it
        RTXProcessor rtx;
        bool rtx_init = false;

        // Initialize the chosen processing path (GPU or CPU) once
        if (use_cuda_path)
        {
            AVHWDeviceContext *devctx = (AVHWDeviceContext *)in.hw_device_ctx->data;
            AVCUDADeviceContext *cudactx = (AVCUDADeviceContext *)devctx->hwctx;
            CUcontext cu = cudactx->cuda_ctx;
            if (!rtx.initializeWithContext(cu, cfg.rtxCfg, in.vdec->width, in.vdec->height))
            {
                std::string detail = rtx.lastError();
                if (detail.empty()) detail = "unknown error";
                throw std::runtime_error(std::string("Failed to init RTX GPU path: ") + detail);
            }
            rtx_init = true;
        }
        else
        {
            if (!rtx.initialize(0, cfg.rtxCfg, in.vdec->width, in.vdec->height))
            {
                std::string detail = rtx.lastError();
                if (detail.empty()) detail = "unknown error";
                throw std::runtime_error(std::string("Failed to initialize RTX CPU path: ") + detail);
            }
            rtx_init = true;
        }

        // Build a processor abstraction for the loop
        std::unique_ptr<IProcessor> processor;
        if (use_cuda_path)
        {
            processor = std::make_unique<GpuProcessor>(rtx, cuda_pool, in.vdec->colorspace);
        }
        else
        {
            auto cpuProc = std::make_unique<CpuProcessor>(rtx, in.vdec->width, in.vdec->height, dstW, dstH);
            cpuProc->setConfig(cfg.rtxCfg);
            processor = std::move(cpuProc);
        }

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
                    int64_t output_pts = derive_output_pts(v_start_pts, decframe, in.vst->time_base, out.venc->time_base);
                    // Ensure strict monotonicity to satisfy encoder/muxer
                    // if (last_output_pts != AV_NOPTS_VALUE && output_pts <= last_output_pts) {
                    //     output_pts = last_output_pts + 1;
                    // }
                    // last_output_pts = output_pts;

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

                    // Set consistent PTS to prevent stuttering
                    outFrame->pts = output_pts;

                    // Ensure GPU work completes before encoding when on CUDA path
                    if (use_cuda_path)
                        cudaStreamSynchronize(0);

                    // Encode
                    encode_and_write(out.venc, out.vstream, out.fmt, outFrame, opkt, "send frame to encoder");
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

    PipelineConfig cfg;

    init_setup(argc, argv, &cfg);

    // Set log level
    av_log_set_level(AV_LOG_WARNING);

    int ret = run_pipeline(cfg);
}