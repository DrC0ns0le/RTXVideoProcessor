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
#include "timestamp_manager.h"
#include "processor.h"
#include "logger.h"
#include "audio_config.h"
#include "async_demuxer.h"
#include "ffmpeg_passthrough.h"
#include "config_parser.h"
#include "input_config.h"
#include "output_config.h"
#include "utils.h"

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

using FramePtr = std::unique_ptr<AVFrame, void (*)(AVFrame *)>;
using PacketPtr = std::unique_ptr<AVPacket, void (*)(AVPacket *)>;

// Pipeline types are provided by pipeline_types.h via ffmpeg_utils.h

// Unified helper to send a frame to the encoder and interleaved-write all produced packets
static inline void encode_and_write(AVCodecContext *enc,
                                    AVStream *vstream,
                                    AVFormatContext *ofmt,
                                    OutputContext &out,
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

        // DTS monotonicity fix (mirrors fftools/ffmpeg_mux.c:mux_fixup_ts)
        // NVENC generates duplicate DTS at GOP boundaries with forced-idr + strict_gop (HLS alignment).
        // This is standard FFmpeg behavior - the CLI tool applies the same correction before muxing.
        // Without this, HLS muxer fails with "non monotonically increasing dts" errors.
        if (out.last_video_dts != AV_NOPTS_VALUE && opkt->dts != AV_NOPTS_VALUE)
        {
            int64_t max_dts = out.last_video_dts + 1; // Minimum acceptable DTS
            if (opkt->dts < max_dts)
            {
                LOG_DEBUG("DTS monotonicity: adjusting packet DTS from %lld to %lld", opkt->dts, max_dts);
                opkt->dts = max_dts;

                // Ensure PTS >= DTS after adjustment (FFmpeg requirement)
                if (opkt->pts != AV_NOPTS_VALUE && opkt->pts < opkt->dts)
                {
                    opkt->pts = opkt->dts;
                }
            }
        }
        out.last_video_dts = opkt->dts;

        ff_check(av_interleaved_write_frame(ofmt, opkt.get()), "write video packet");
        av_packet_unref(opkt.get());
    }
}

// Helper function for frame buffer and context initialization
static void initialize_frame_buffers_and_contexts(bool use_cuda_path, int dstW, int dstH,
                                                  CudaFramePool &cuda_pool, SwsContext *&sws_to_p010,
                                                  FramePtr &bgra_frame, FramePtr &p010_frame,
                                                  const InputContext &in, const OutputContext &out)
{
    if (use_cuda_path)
    {
        // Initialize CUDA frame pool for GPU path
        // Increased from 8 to 16 to prevent pool exhaustion during high-throughput processing (90+ fps)
        // This provides more working buffers for the RTX processor and encoder pipeline
        const int POOL_SIZE = 16;
        cuda_pool.initialize(out.venc->hw_frames_ctx, dstW, dstH, POOL_SIZE);
    }
    else
    {
        // Allocate CPU resources only when using CPU path
        // CpuProcessor manages its own internal buffers, but these are needed for the main pipeline
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
}

// Helper function for RTX processor initialization
static void initialize_rtx_processor(RTXProcessor &rtx, bool &rtx_init, bool use_cuda_path,
                                     const PipelineConfig &cfg, const InputContext &in)
{
    if (use_cuda_path)
    {
        AVHWDeviceContext *devctx = (AVHWDeviceContext *)in.hw_device_ctx->data;
        AVCUDADeviceContext *cudactx = (AVCUDADeviceContext *)devctx->hwctx;
        CUcontext cu = cudactx->cuda_ctx;
        if (!rtx.initializeWithContext(cu, cfg.rtxCfg, in.vdec->width, in.vdec->height))
        {
            std::string detail = rtx.lastError();
            if (detail.empty())
                detail = "unknown error";
            throw std::runtime_error(std::string("Failed to init RTX GPU path: ") + detail);
        }
        rtx_init = true;
    }
    else
    {
        if (!rtx.initialize(0, cfg.rtxCfg, in.vdec->width, in.vdec->height))
        {
            std::string detail = rtx.lastError();
            if (detail.empty())
                detail = "unknown error";
            throw std::runtime_error(std::string("Failed to init RTX CPU path: ") + detail);
        }
        rtx_init = true;
    }
}

static void apply_movflags(AVDictionary **muxopts, bool is_pipe, bool hls_enabled, const PipelineConfig &cfg)
{
    // User-specified flags take priority (FFmpeg-compatible)
    if (!cfg.movflags.empty())
    {
        av_dict_set(muxopts, "movflags", cfg.movflags.c_str(), 0);
        LOG_DEBUG("Applied user movflags: %s", cfg.movflags.c_str());
        return;
    }

    // Legacy mode: Auto-apply flags for compatibility
    if (cfg.ffCompatible)
        return;

    if (is_pipe)
    {
        av_dict_set(muxopts, "movflags", "+empty_moov+default_base_moof+delay_moov+dash+write_colr", 0);
    }
    else if (!hls_enabled)
    {
        av_dict_set(muxopts, "movflags", "+faststart+write_colr", 0);
    }
    else
    {
        av_dict_set(muxopts, "movflags", "+frag_keyframe+delay_moov+faststart+write_colr", 0);
    }
}

static void apply_fragment_options(AVDictionary **muxopts, const PipelineConfig &cfg)
{
    if (cfg.fragDuration > 0)
    {
        av_dict_set(muxopts, "frag_duration", std::to_string(cfg.fragDuration).c_str(), 0);
    }
    if (cfg.fragmentIndex >= 0)
    {
        av_dict_set(muxopts, "fragment_index", std::to_string(cfg.fragmentIndex).c_str(), 0);
    }
    if (cfg.useEditlist >= 0)
    {
        av_dict_set(muxopts, "use_editlist", std::to_string(cfg.useEditlist).c_str(), 0);
    }
}

static void write_muxer_header(InputContext &in, OutputContext &out, bool hls_enabled, bool mux_is_isobmff,
                               const AVRational &fr, bool is_pipe, const PipelineConfig &cfg)
{
    out.vstream->time_base = out.venc->time_base;
    out.vstream->avg_frame_rate = fr;
    out.vstream->r_frame_rate = fr;

    AVDictionary *muxopts = out.muxOptions;

    if (mux_is_isobmff)
    {
        apply_movflags(&muxopts, is_pipe, hls_enabled, cfg);
        apply_fragment_options(&muxopts, cfg);
    }

    // Apply FFmpeg-compatible timestamp handling to muxer (affects all formats)
    av_dict_set(&muxopts, "avoid_negative_ts", cfg.avoidNegativeTs.c_str(), 0);
    LOG_DEBUG("Muxer avoid_negative_ts: %s", cfg.avoidNegativeTs.c_str());

    // Apply output_ts_offset to muxer (FFmpeg-compatible behavior)
    // libavformat/mux.c applies this offset to ALL packet timestamps during av_interleaved_write_frame()
    // Works in BOTH copyts and non-copyts modes:
    // - Non-copyts mode: Adds offset to zero-based timestamps (e.g., 0s → 24s output)
    // - Copyts mode: Adds offset to normalized timestamps (e.g., normalized to 0s → 0s+1076s offset = 1076s output)
    //
    // CRITICAL: Do NOT apply output_ts_offset for HLS muxer!
    // HLS segments require timestamps starting near zero for proper playback in hls.js and other players.
    // The HLS muxer handles timing internally via baseMediaDecodeTime in fMP4 fragments.
    // Applying output_ts_offset causes timestamps like 60s in the first segment, breaking hls.js playback (causes looping).
    // Verified: vanilla FFmpeg with -copyts produces HLS starting at 0.041s, not 60s.
    if (!cfg.outputTsOffset.empty() && !hls_enabled)
    {
        int64_t offset_us = 0;
        int ret = av_parse_time(&offset_us, cfg.outputTsOffset.c_str(), 1);
        if (ret >= 0)
        {
            out.fmt->output_ts_offset = offset_us;
            LOG_DEBUG("Set muxer output_ts_offset: %lld us (%.3fs) - FFmpeg muxer will ADD this to all timestamps",
                      offset_us, offset_us / 1000000.0);
            if (cfg.copyts)
            {
                LOG_DEBUG("COPYTS + output_ts_offset: preserving timestamps then offsetting (non-HLS mode)");
            }
        }
        else
        {
            LOG_ERROR("Failed to parse output_ts_offset: %s", cfg.outputTsOffset.c_str());
        }
    }
    else if (!cfg.outputTsOffset.empty() && hls_enabled)
    {
        LOG_DEBUG("HLS mode: output_ts_offset handled by TimestampManager, not muxer (Stremio compatibility)");
    }

    if (cfg.maxMuxingQueueSize > 0)
    {
        out.fmt->max_interleave_delta = cfg.maxMuxingQueueSize;
    }

    ff_check(avformat_write_header(out.fmt, &muxopts), "write header");
    if (muxopts)
        av_dict_free(&muxopts);
    out.muxOptions = nullptr;
}

int run_pipeline(PipelineConfig cfg)
{
    Logger::instance().setVerbose(cfg.verbose || cfg.debug);
    Logger::instance().setDebug(cfg.debug);

    // Set FFmpeg log level to match application log level
    // This allows us to see internal FFmpeg messages (e.g., HLS muxer temp_file operations)
    if (cfg.debug)
    {
        av_log_set_level(AV_LOG_VERBOSE);  // Show detailed FFmpeg internal logs
    }
    else if (cfg.verbose)
    {
        av_log_set_level(AV_LOG_INFO);     // Show FFmpeg info messages
    }
    else
    {
        av_log_set_level(AV_LOG_WARNING);  // Default: only warnings and errors
    }

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
        inputOpts.enableErrorConcealment = !cfg.ffCompatible; // FFmpeg doesn't enable error concealment by default

        inputOpts.flushOnSeek = false; // FFmpeg never flushes decoder on seek
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
        if (willReencodeAudio)
        {
            out.audioConfig.enabled = true;
            out.audioConfig.codec = cfg.audioCodec.empty() ? "aac" : cfg.audioCodec;
            out.audioConfig.applyToAllAudioStreams = cfg.audioCodecApplyToAll;
            out.audioConfig.copyts = cfg.copyts; // Pass copyts mode to audio encoder
        }

        LOG_DEBUG("Opening output...");
        open_output(cfg.outputPath, in, out, cfg.streamMaps, cfg.outputFormatName);
        LOG_DEBUG("Output opened successfully");

        // Apply metadata and chapter settings (Jellyfin compatibility)
        apply_metadata_chapter_settings(out, cfg, in);

        // Stage 4: Configure audio processing (complete the audio setup)
        configure_audio_processing(cfg, in, out);

        // Stage 5: Calculate frame rate and setup progress tracking
        const bool hls_enabled = out.hlsOptions.enabled;
        const bool hls_segments_are_fmp4 = hls_enabled && lowercase_copy(out.hlsOptions.segmentType) == "fmp4";

        // Read nominal fps (FFmpeg-like priority): guess -> r_frame_rate -> avg_frame_rate -> inverse time_base
        AVRational fr = av_guess_frame_rate(in.fmt, in.vst, nullptr);
        if (fr.num == 0 || fr.den == 0) fr = in.vst->r_frame_rate;
        if (fr.num == 0 || fr.den == 0) fr = in.vst->avg_frame_rate;
        if (fr.num == 0 || fr.den == 0) fr = AVRational{in.vst->time_base.den, in.vst->time_base.num};

        // Override framerate if -r or -r:v flag was specified (FFmpeg compatibility)
        if (!cfg.outputFrameRate.empty())
        {
            AVRational override_fr;
            int ret = av_parse_video_rate(&override_fr, cfg.outputFrameRate.c_str());
            if (ret < 0)
            {
                char errbuf[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, errbuf, sizeof(errbuf));
                throw std::runtime_error("Invalid framerate format for -r: " + cfg.outputFrameRate + " (" + std::string(errbuf) + ")");
            }
            LOG_INFO("Output framerate override: %d/%d (%.3f fps) via -r flag",
                     override_fr.num, override_fr.den, av_q2d(override_fr));
            fr = override_fr;
        }

        int64_t total_frames = setup_progress_tracking(in, fr);

        // Progress tracking variables
        int64_t processed_frames = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto last_update = start_time;
        const int update_interval_ms = 500; // Update progress every 500ms
        std::string progress_bar(50, ' ');

        // Prepare sws contexts (created on first decoded frame when actual format is known)

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
        if (hls_segments_are_fmp4)
        {
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

        // Stage 10: Write muxer header
        bool isPipeOutput = is_pipe_output(cfg.outputPath);
        write_muxer_header(in, out, hls_enabled, mux_is_isobmff, fr, isPipeOutput, cfg);

        PacketPtr pkt(av_packet_alloc(), &av_packet_free_single);
        PacketPtr opkt(av_packet_alloc(), &av_packet_free_single);

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

        // Initialize simplified timestamp manager - FFmpeg 8 compatible
        // Philosophy: Set AVFrame->pts correctly, let FFmpeg handle the rest
        TimestampManager::Config ts_config;
        ts_config.mode = cfg.copyts ? TimestampManager::Mode::COPYTS : TimestampManager::Mode::NORMAL;
        ts_config.input_seek_us = in.seek_offset_us;
        ts_config.start_at_zero = cfg.startAtZero;
        // Respect -avoid_negative_ts for COPYTS: if disabled, don't clamp negatives
        // Valid values: "auto", "make_zero", "make_non_negative", "disabled"
        if (cfg.copyts && !cfg.avoidNegativeTs.empty())
        {
            std::string ant = lowercase_copy(cfg.avoidNegativeTs);
            if (ant == "disabled")
            {
                ts_config.clamp_negative_copyts = false;
            }
        }

        // Configure CFR mode if -vsync cfr is specified
        if (cfg.vsync == "cfr" || cfg.vsync == "0")
        {
            ts_config.vsync_cfr = true;
            ts_config.cfr_frame_rate = fr;
            LOG_INFO("CFR mode enabled: constant frame rate at %d/%d fps", fr.num, fr.den);
        }

        // Parse output seeking time (optional frame dropping - could use muxer's -t duration instead)
        if (!cfg.outputSeekTime.empty())
        {
            int ret = av_parse_time(&ts_config.output_seek_target_us, cfg.outputSeekTime.c_str(), 1);
            if (ret < 0)
            {
                throw std::runtime_error("Invalid output seek time format: " + cfg.outputSeekTime);
            }
            LOG_DEBUG("Output seeking enabled: target = %.3fs", ts_config.output_seek_target_us / 1000000.0);

            // Edge case: Warn if output seek < input seek
            if (in.seek_offset_us > 0 && ts_config.output_seek_target_us < in.seek_offset_us)
            {
                LOG_WARN("Output seek target (%.3fs) < input seek (%.3fs) - may cause unexpected behavior",
                         ts_config.output_seek_target_us / 1000000.0, in.seek_offset_us / 1000000.0);
            }
        }

        // Parse output timestamp offset (Stremio compatibility - sets first frame PTS to offset value)
        // Used with HLS/DASH for maintaining timeline position during seeking
        if (!cfg.outputTsOffset.empty())
        {
            int ret = av_parse_time(&ts_config.output_ts_offset_us, cfg.outputTsOffset.c_str(), 1);
            if (ret < 0)
            {
                throw std::runtime_error("Invalid output timestamp offset format: " + cfg.outputTsOffset);
            }
            LOG_DEBUG("Output timestamp offset enabled: offset = %.3fs", ts_config.output_ts_offset_us / 1000000.0);
        }

        // Enable HLS mode for proper tfdt (baseMediaDecodeTime) handling
        // HLS fMP4 segments must preserve timeline position in tfdt for A/V sync
        ts_config.hls_mode = hls_enabled;

        // Share HLS and output seeking config with audio for A/V sync
        out.hls_mode = hls_enabled;
        out.output_seek_target_us = ts_config.output_seek_target_us;

        // NOTE: output_ts_offset is applied directly to AVFormatContext->output_ts_offset in
        // write_muxer_header() to match vanilla FFmpeg behavior. The muxer (libavformat/mux.c)
        // handles timestamp offsetting during av_interleaved_write_frame(), ensuring correct
        // behavior for HLS fragmented streaming in dual-process deployments where RTXVideoProcessor
        // handles video and vanilla FFmpeg handles audio with different seek positions.

        // NOTE: output_ts_offset is applied directly to AVFormatContext->output_ts_offset in
        // write_muxer_header() to match vanilla FFmpeg behavior. The muxer (libavformat/mux.c)
        // handles timestamp offsetting during av_interleaved_write_frame(), ensuring correct
        // behavior for HLS fragmented streaming in dual-process deployments where RTXVideoProcessor
        // handles video and vanilla FFmpeg handles audio with different seek positions.

        // Create timestamp manager (simplified)
        TimestampManager ts_manager(ts_config);

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
        if (!async_demuxer.start())
        {
            throw std::runtime_error("Failed to start async demuxer");
        }

        // Read packets
        LOG_DEBUG("Starting frame processing loop with async demuxing...");
        LOG_DEBUG("Video stream index: %d, Primary audio stream index: %d", in.vstream, in.primary_audio_stream);
        LOG_DEBUG("Audio config enabled: %s", out.audioConfig.enabled ? "true" : "false");
        LOG_DEBUG("Copyts mode: %s", cfg.copyts ? "enabled" : "disabled");
        LOG_DEBUG("FFmpeg compatibility: avoid_negative_ts=%s, start_at_zero=%s",
                  cfg.avoidNegativeTs.c_str(), cfg.startAtZero ? "enabled" : "disabled");
        LOG_DEBUG("Starting processing with seek_offset_us = %lld (%.3fs)", in.seek_offset_us, in.seek_offset_us / 1000000.0);

        // Parse -t duration limit if provided
        int64_t duration_us = 0;
        if (!cfg.duration.empty())
        {
            int ret = av_parse_time(&duration_us, cfg.duration.c_str(), 1);
            if (ret >= 0)
            {
                LOG_INFO("Duration limit: %lld us (%.3fs)", duration_us, duration_us / 1000000.0);
            }
            else
            {
                LOG_ERROR("Failed to parse duration: %s", cfg.duration.c_str());
                duration_us = 0;
            }
        }

        int packet_count = 0;
        while (true)
        {
            // Get packet from async demuxer (non-blocking I/O)
            AVPacket *raw_pkt = async_demuxer.getPacket();
            if (!raw_pkt)
            {
                // Check for EOF or error
                if (async_demuxer.isEOF())
                {
                    break;
                }
                int err = async_demuxer.getError();
                if (err != 0)
                {
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

            // Audio/video sync handled by FFmpeg's muxer via output_ts_offset
            // No manual baseline tracking needed - delegate to FFmpeg

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

                    // Accurate seeking: Discard frames before seek target (only if accurate seek is enabled)
                    // When seeking to a timestamp, avformat_seek_file() seeks to the nearest keyframe BEFORE the target.
                    // The decoder then outputs all frames from that keyframe onwards.
                    // FFmpeg's default behavior (accurate_seek) is to decode but discard frames before the target.
                    // Respect user's -noaccurate_seek or -seek2any flags - don't discard if they disabled accurate seeking.
                    if (in.seek_offset_us > 0 && !cfg.noAccurateSeek && !cfg.seek2any && decframe->pts != AV_NOPTS_VALUE)
                    {
                        int64_t frame_time_us = av_rescale_q(decframe->pts, in.vst->time_base, {1, AV_TIME_BASE});
                        if (frame_time_us < in.seek_offset_us)
                        {
                            // Frame is before seek target - discard it (accurate seeking behavior)
                            av_frame_unref(frame.get());
                            continue;
                        }
                    }

                    // Duration limit: Stop processing when we've reached the requested duration
                    if (duration_us > 0 && decframe->pts != AV_NOPTS_VALUE)
                    {
                        // Convert frame PTS to microseconds
                        int64_t frame_time_us = av_rescale_q(decframe->pts, in.vst->time_base, {1, 1000000});

                        // Calculate elapsed time from start
                        // FFmpeg behavior: With output seeking (-ss after -i), duration is measured from
                        // the output seek point, not the input seek point. This ensures proper clip duration.
                        int64_t reference_time_us = ts_config.output_seek_target_us > 0
                                                     ? ts_config.output_seek_target_us
                                                     : in.seek_offset_us;
                        int64_t elapsed_us = frame_time_us - reference_time_us;

                        // Stop when we exceed the duration limit
                        if (elapsed_us >= duration_us)
                        {
                            LOG_INFO("Reached duration limit: %.3fs (elapsed from %.3fs)",
                                     elapsed_us / 1000000.0, reference_time_us / 1000000.0);
                            goto done_processing;
                        }
                    }

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
                    if (ts_manager.shouldDropFrameForOutputSeek(decframe, in.vst->time_base))
                    {
                        av_frame_unref(frame.get());
                        if (swframe)
                            av_frame_unref(swframe.get());
                        continue;
                    }

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

                    bool cfr_active = (cfg.vsync == "cfr" || cfg.vsync == "0");
                    if (cfr_active && ts_manager.cfrActive())
                    {
                        // Obtain input PTS
                        int64_t in_pts = (decframe->pts != AV_NOPTS_VALUE) ? decframe->pts : decframe->best_effort_timestamp;
                        if (in_pts == AV_NOPTS_VALUE)
                        {
                            // Fallback to previous behavior when no input PTS is available
                            outFrame->pts = ts_manager.deriveVideoPTS(decframe, in.vst->time_base, out.venc->time_base);
                            if (outFrame->pts == AV_NOPTS_VALUE)
                            {
                                av_frame_unref(frame.get());
                                if (swframe)
                                    av_frame_unref(swframe.get());
                                continue;
                            }
                            // duration in CFR
                            int64_t tpf = av_rescale_q(1, av_inv_q(fr), out.venc->time_base);
                            if (tpf <= 0) tpf = 1;
                            outFrame->duration = (int)tpf;

                            // Encode current frame
                            if (use_cuda_path)
                                rtx.syncStream();
                            encode_and_write(out.venc, out.vstream, out.fmt, out, outFrame, opkt, "send frame to encoder");
                        }
                        else
                        {
                            // CFR synchronization (FFmpeg-compliant delta-based approach)
                            // IMPORTANT: Apply COPYTS baseline normalization BEFORE CFR sync
                            // FFmpeg's CFR works with normalized timestamps (starting at 0), not raw input timestamps

                            // First, get baseline-normalized PTS using deriveVideoPTS
                            // This handles COPYTS baseline, output seeking, etc.
                            int64_t normalized_pts = ts_manager.deriveVideoPTS(decframe, in.vst->time_base, out.venc->time_base);
                            if (normalized_pts == AV_NOPTS_VALUE)
                            {
                                av_frame_unref(frame.get());
                                if (swframe)
                                    av_frame_unref(swframe.get());
                                continue;
                            }

                            // Create a temporary frame with normalized PTS for CFR sync
                            // CFR sync needs to see timestamps starting at ~0, not raw input timeline
                            AVFrame temp_frame = *decframe;
                            temp_frame.pts = normalized_pts;
                            temp_frame.time_base = out.venc->time_base;

                            int64_t cfr_pts;  // PTS in CFR timebase (av_inv_q(framerate))
                            double cfr_duration;
                            int64_t nb_frames = ts_manager.cfrSync(&temp_frame, out.venc->time_base, out.venc->time_base, &cfr_pts, &cfr_duration);

                            if (nb_frames == 0)
                            {
                                // Drop frame (delta < -1.1 or below threshold)
                                av_frame_unref(frame.get());
                                if (swframe)
                                    av_frame_unref(swframe.get());
                                continue;
                            }

                            // Calculate duration in encoder timebase ticks
                            int64_t duration_ticks = (int64_t)llrint(cfr_duration);
                            if (duration_ticks <= 0) duration_ticks = 1;

                            // CFR timebase: 1 tick = 1 frame (e.g., 21/500 for 500/21 fps)
                            AVRational cfr_tb = av_inv_q(fr);

                            // Output nb_frames times (1 for normal, >1 for duplication)
                            for (int64_t i = 0; i < nb_frames; i++)
                            {
                                // Convert from CFR timebase to encoder timebase
                                // FFmpeg equivalently uses filter output tb = av_inv_q(framerate)
                                int64_t encoder_pts = av_rescale_q(cfr_pts + i, cfr_tb, out.venc->time_base);
                                outFrame->pts = encoder_pts;
                                outFrame->duration = duration_ticks;

                                if (use_cuda_path)
                                    rtx.syncStream();
                                encode_and_write(out.venc, out.vstream, out.fmt, out, outFrame, opkt, "send CFR frame to encoder");
                            }
                        }

                        // Share the COPYTS baseline with audio (same as generic path)
                        if (ts_manager.hasCopytsBaseline() && out.copyts_baseline_pts == AV_NOPTS_VALUE)
                        {
                            int64_t baseline_ticks = ts_manager.getCopytsBaseline();
                            out.copyts_baseline_pts = av_rescale_q(baseline_ticks, out.venc->time_base, {1, AV_TIME_BASE});
                            LOG_DEBUG("Shared COPYTS baseline with audio: %lld ticks (video tb) = %lld us",
                                      baseline_ticks, out.copyts_baseline_pts);
                        }

                        // Done with frames (already encoded above)
                        av_frame_unref(frame.get());
                        if (swframe)
                            av_frame_unref(swframe.get());
                        continue; // Skip the generic encode path below
                    }
                    else
                    {
                        // Non-CFR: previous behavior
                        outFrame->pts = ts_manager.deriveVideoPTS(decframe, in.vst->time_base, out.venc->time_base);
                        if (outFrame->pts == AV_NOPTS_VALUE)
                        {
                            av_frame_unref(frame.get());
                            if (swframe)
                                av_frame_unref(swframe.get());
                            continue;
                        }

                        // Set frame duration from framerate (FFmpeg-compliant behavior)
                        // Prefer per-frame duration from decoder for VFR content, fallback to fixed duration
                        int64_t frame_duration;
                        if (decframe->duration > 0) {
                            // Use decoder's per-frame duration (handles VFR correctly)
                            frame_duration = av_rescale_q(decframe->duration, in.vst->time_base, out.venc->time_base);
                        } else {
                            // Fallback to fixed duration calculated from framerate
                            frame_duration = av_rescale_q(1, av_inv_q(fr), out.venc->time_base);
                        }
                        if (frame_duration <= 0) frame_duration = 1;  // Safety clamp
                        outFrame->duration = frame_duration;

                        if (use_cuda_path)
                            rtx.syncStream();
                        encode_and_write(out.venc, out.vstream, out.fmt, out, outFrame, opkt, "send frame to encoder");
                        av_frame_unref(frame.get());
                        if (swframe)
                            av_frame_unref(swframe.get());
                        continue;
                    }
                    // Share the COPYTS baseline with audio for A/V sync in HLS segments
                    // After the first video frame, the baseline is established and audio must use it
                    if (ts_manager.hasCopytsBaseline() && out.copyts_baseline_pts == AV_NOPTS_VALUE)
                    {
                        // Convert baseline from video timebase to microseconds for audio to use
                        int64_t baseline_ticks = ts_manager.getCopytsBaseline();
                        out.copyts_baseline_pts = av_rescale_q(baseline_ticks, out.venc->time_base, {1, AV_TIME_BASE});
                        LOG_DEBUG("Shared COPYTS baseline with audio: %lld ticks (video tb) = %lld us",
                                  baseline_ticks, out.copyts_baseline_pts);
                    }

                    if (use_cuda_path)
                        rtx.syncStream();
                    encode_and_write(out.venc, out.vstream, out.fmt, out, outFrame, opkt, "send frame to encoder");
                    av_frame_unref(frame.get());
                    if (swframe)
                        av_frame_unref(swframe.get());
                }
                // Video packet fully processed, continue to next packet
                continue;
            }
            else if (cfg.ffCompatible && out.audioConfig.enabled && pkt->stream_index < (int)out.stream_decisions.size() &&
                     out.stream_decisions[pkt->stream_index] == StreamMapDecision::PROCESS_AUDIO)
            {
                // Capture stream index BEFORE unref (critical: packet data is invalid after unref)
                int audio_stream_idx = pkt->stream_index;

                // Find decoder for this stream
                auto decoder_it = in.audio_decoders.find(audio_stream_idx);
                if (decoder_it == in.audio_decoders.end())
                {
                    LOG_WARN("No decoder for audio stream %d", audio_stream_idx);
                    av_packet_unref(pkt.get());
                    continue;
                }

                AVCodecContext *decoder = decoder_it->second;

                if (decoder)
                {
                    ff_check(avcodec_send_packet(decoder, pkt.get()), "send audio packet");
                    av_packet_unref(pkt.get());

                    FramePtr audio_frame(av_frame_alloc(), &av_frame_free_single);
                    while (true)
                    {
                        int ret = avcodec_receive_frame(decoder, audio_frame.get());
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                            break;
                        ff_check(ret, "receive audio frame");

                        // Get the audio stream for this packet
                        AVStream *ast = in.fmt->streams[audio_stream_idx];

                        // Accurate seeking: Discard audio frames before seek target (only if accurate seek enabled)
                        if (in.seek_offset_us > 0 && !cfg.noAccurateSeek && !cfg.seek2any && audio_frame->pts != AV_NOPTS_VALUE)
                        {
                            int64_t frame_time_us = av_rescale_q(audio_frame->pts, ast->time_base, {1, AV_TIME_BASE});
                            if (frame_time_us < in.seek_offset_us)
                            {
                                // Audio frame is before seek target - discard it
                                av_frame_unref(audio_frame.get());
                                continue;
                            }
                        }

                        // Output seeking: Discard audio frames before output seek target
                        // This ensures audio and video segments have the same content range for proper A/V sync
                        // Like video's TimestampManager, stop checking after first valid frame passes
                        if (!out.audio_output_seek_complete && ts_config.output_seek_target_us > 0 && audio_frame->pts != AV_NOPTS_VALUE)
                        {
                            int64_t frame_time_us = av_rescale_q(audio_frame->pts, ast->time_base, {1, AV_TIME_BASE});
                            if (frame_time_us < ts_config.output_seek_target_us)
                            {
                                // Audio frame is before output seek target - discard it
                                LOG_DEBUG("Dropping audio frame before output seek target: %.3fs < %.3fs",
                                          frame_time_us / 1000000.0, ts_config.output_seek_target_us / 1000000.0);
                                av_frame_unref(audio_frame.get());
                                continue;
                            }
                            // First valid frame passed - stop checking subsequent frames
                            out.audio_output_seek_complete = true;
                            LOG_DEBUG("Audio output seeking complete: first frame at %.3fs",
                                      frame_time_us / 1000000.0);
                        }

                        // Set correct source time_base for timestamp rescaling
                        audio_frame->time_base = ast->time_base;

                        // Process frame using multi-stream encoder
                        if (!process_audio_frame_multi(audio_frame.get(), audio_stream_idx, out))
                        {
                            LOG_WARN("Failed to process audio frame for stream %d", audio_stream_idx);
                        }

                        av_frame_unref(audio_frame.get());
                    }
                }
                else
                {
                    // Fallback: copy audio packet without re-encoding
                    int out_index = out.input_to_output_map[audio_stream_idx];

                    if (out_index >= 0)
                    {
                        AVStream *ist = in.fmt->streams[audio_stream_idx];
                        AVStream *ost = out.fmt->streams[out_index];

                        // Accurate seeking: Discard audio packets before seek target (only if accurate seek enabled)
                        if (in.seek_offset_us > 0 && !cfg.noAccurateSeek && !cfg.seek2any && pkt->pts != AV_NOPTS_VALUE)
                        {
                            int64_t pkt_time_us = av_rescale_q(pkt->pts, ist->time_base, {1, AV_TIME_BASE});
                            if (pkt_time_us < in.seek_offset_us)
                            {
                                // Audio packet is before seek target - discard it
                                av_packet_unref(pkt.get());
                                continue;
                            }
                        }

                        // Output seeking: Discard audio packets before output seek target
                        // This ensures audio and video segments have the same content range for proper A/V sync
                        // Like video's TimestampManager, stop checking after first valid packet passes
                        if (!out.audio_output_seek_complete && ts_config.output_seek_target_us > 0 && pkt->pts != AV_NOPTS_VALUE)
                        {
                            int64_t pkt_time_us = av_rescale_q(pkt->pts, ist->time_base, {1, AV_TIME_BASE});
                            if (pkt_time_us < ts_config.output_seek_target_us)
                            {
                                // Audio packet is before output seek target - discard it
                                LOG_DEBUG("Dropping audio packet before output seek target: %.3fs < %.3fs",
                                          pkt_time_us / 1000000.0, ts_config.output_seek_target_us / 1000000.0);
                                av_packet_unref(pkt.get());
                                continue;
                            }
                            // First valid packet passed - stop checking subsequent packets
                            out.audio_output_seek_complete = true;
                            LOG_DEBUG("Audio output seeking complete: first packet at %.3fs",
                                      pkt_time_us / 1000000.0);
                        }

                        // Only adjust timestamps in FFmpeg mode without copyts (advanced handling)
                        // Default mode and FFmpeg+copyts use simple passthrough
                        // Simple rescaling - muxer handles the rest
                        ts_manager.rescalePacketTimestamps(pkt.get(), ist->time_base, ost->time_base);
                        pkt->stream_index = out_index;
                        ff_check(av_interleaved_write_frame(out.fmt, pkt.get()), "write copied audio packet");
                    }
                    av_packet_unref(pkt.get());
                }
            }
            else
            {
                // Copy other streams
                int out_index = out.input_to_output_map[pkt->stream_index];

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

                    // Simple rescaling - muxer handles the rest
                    ts_manager.rescalePacketTimestamps(pkt.get(), ist->time_base, ost->time_base);
                    pkt->stream_index = out_index;
                    ff_check(av_interleaved_write_frame(out.fmt, pkt.get()), "write copied packet");
                }
                av_packet_unref(pkt.get());
            }
        }

    done_processing:
        // Flush encoder (uses encode_and_write to ensure DTS monotonicity fix is applied)
        LOG_DEBUG("Finished processing all frames, flushing encoder...");
        encode_and_write(out.venc, out.vstream, out.fmt, out, nullptr, opkt, "flush encoder");

        // Flush audio encoders if enabled
        if (cfg.ffCompatible && out.audioConfig.enabled && !out.audio_encoders.empty())
        {
            // Multi-stream flushing
            LOG_DEBUG("Flushing %zu multi-stream audio encoders...", out.audio_encoders.size());
            for (auto &pair : out.audio_encoders)
            {
                int stream_idx = pair.first;
                AudioEncoderContext &enc_ctx = pair.second;

                    if (!enc_ctx.encoder || !enc_ctx.output_stream)
                        continue;

                    LOG_DEBUG("Flushing audio encoder for stream %d...", stream_idx);

                    // Flush any remaining samples in FIFO (loop until empty)
                    while (enc_ctx.fifo && av_audio_fifo_size(enc_ctx.fifo) > 0)
                    {
                        int remaining = av_audio_fifo_size(enc_ctx.fifo);
                        int frame_sz = enc_ctx.encoder->frame_size;

                        AVFrame *encoder_frame = av_frame_alloc();
                        if (!encoder_frame)
                            break;

                        encoder_frame->nb_samples = frame_sz;
                        encoder_frame->format = enc_ctx.encoder->sample_fmt;
                        av_channel_layout_copy(&encoder_frame->ch_layout, &enc_ctx.encoder->ch_layout);
                        encoder_frame->sample_rate = enc_ctx.encoder->sample_rate;

                        if (av_frame_get_buffer(encoder_frame, 0) < 0)
                        {
                            av_frame_free(&encoder_frame);
                            break;
                        }

                        int to_read = remaining < frame_sz ? remaining : frame_sz;
                        if (to_read > 0)
                        {
                            av_audio_fifo_read(enc_ctx.fifo, (void **)encoder_frame->data, to_read);
                        }
                        if (to_read < frame_sz)
                        {
                            // Zero-pad the rest
                            for (int ch = 0; ch < encoder_frame->ch_layout.nb_channels; ++ch)
                            {
                                uint8_t *dst = encoder_frame->data[ch] + to_read * av_get_bytes_per_sample((AVSampleFormat)encoder_frame->format);
                                int pad_bytes = (frame_sz - to_read) * av_get_bytes_per_sample((AVSampleFormat)encoder_frame->format);
                                memset(dst, 0, pad_bytes);
                            }
                        }

                        encoder_frame->pts = enc_ctx.accumulated_samples;
                        enc_ctx.accumulated_samples += to_read;

                        avcodec_send_frame(enc_ctx.encoder, encoder_frame);
                        av_frame_free(&encoder_frame);

                        // Receive and write packets
                        while (true)
                        {
                            int ret = avcodec_receive_packet(enc_ctx.encoder, opkt.get());
                            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                                break;
                            if (ret < 0)
                                break;

                            opkt->stream_index = enc_ctx.output_stream->index;
                            av_packet_rescale_ts(opkt.get(), enc_ctx.encoder->time_base, enc_ctx.output_stream->time_base);
                            av_interleaved_write_frame(out.fmt, opkt.get());
                            av_packet_unref(opkt.get());
                        }
                    }

                    // Flush encoder
                    avcodec_send_frame(enc_ctx.encoder, nullptr);
                    while (true)
                    {
                        int ret = avcodec_receive_packet(enc_ctx.encoder, opkt.get());
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                            break;
                        if (ret < 0)
                            break;

                        opkt->stream_index = enc_ctx.output_stream->index;
                        av_packet_rescale_ts(opkt.get(), enc_ctx.encoder->time_base, enc_ctx.output_stream->time_base);
                        av_interleaved_write_frame(out.fmt, opkt.get());
                        av_packet_unref(opkt.get());
                    }
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

            // Simple timestamp stats
            LOG_DEBUG("Timestamp stats: frames_processed=%lld, frames_dropped=%d",
                      ts_manager.getFrameCount(), ts_manager.getDroppedFrames());

            // Overall health assessment
            bool healthy = true;
            std::string warnings;
            if (demux_stats.queue_full_waits > demux_stats.total_reads * 0.1)
            {
                warnings += "demux_backpressure ";
                healthy = false;
            }
            LOG_DEBUG("Pipeline health: %s%s", healthy ? "OK" : "WARN",
                      warnings.empty() ? "" : (" - " + warnings).c_str());
        }
        if (sws_to_p010)
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
    // Match ffmpeg behavior: minimal console handling
    setvbuf(stderr, NULL, _IONBF, 0); /* win32 runtime needs this */
#endif

    if (passthrough_required(argc, argv))
    {
        return run_ffmpeg_passthrough(argc, argv);
    }

    PipelineConfig cfg;

    parse_arguments(argc, argv, &cfg);

    // FFmpeg log level is now set dynamically in run_pipeline() based on cfg.verbose/debug

    int ret = run_pipeline(cfg);
}
