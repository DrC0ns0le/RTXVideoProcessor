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

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/packet.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/display.h>
#include <libavutil/rational.h>
#include <libswscale/swscale.h>
}

#include "ffmpeg_utils.h"
#include "rtx_processor.h"

// RAII deleters for FFmpeg types that require ** double-pointer frees
static inline void av_frame_free_single(AVFrame* f) { if (f) av_frame_free(&f); }
static inline void av_packet_free_single(AVPacket* p) { if (p) av_packet_free(&p); }

using FramePtr  = std::unique_ptr<AVFrame,  void(*)(AVFrame*)>;
using PacketPtr = std::unique_ptr<AVPacket, void(*)(AVPacket*)>;

struct InputContext {
    AVFormatContext* fmt = nullptr;
    int vstream = -1;
    AVStream* vst = nullptr;
    AVCodecContext* vdec = nullptr;
    AVBufferRef* hw_device_ctx = nullptr; // CUDA device
};

struct OutputContext {
    AVFormatContext* fmt = nullptr;
    AVStream* vstream = nullptr;
    AVCodecContext* venc = nullptr;
    std::vector<int> map_streams; // input->output map, -1 for unmapped
};

static AVPixelFormat get_cuda_sw_format(AVCodecContext* ctx, const AVPixelFormat* pix_fmts) {
    while (*pix_fmts != AV_PIX_FMT_NONE) {
        if (*pix_fmts == AV_PIX_FMT_CUDA) return *pix_fmts;
        pix_fmts++;
    }
    return AV_PIX_FMT_NONE;
}

static bool open_input(const char* inPath, InputContext& in)
{
    ff_check(avformat_open_input(&in.fmt, inPath, nullptr, nullptr), "open input");
    ff_check(avformat_find_stream_info(in.fmt, nullptr), "find stream info");

    // Find best video stream
    in.vstream = av_find_best_stream(in.fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (in.vstream < 0) throw std::runtime_error("no video stream");
    in.vst = in.fmt->streams[in.vstream];

    const AVCodec* dec = avcodec_find_decoder(in.vst->codecpar->codec_id);
    if (!dec) throw std::runtime_error("decoder not found");

    in.vdec = avcodec_alloc_context3(dec);
    if (!in.vdec) throw std::runtime_error("alloc dec ctx");
    ff_check(avcodec_parameters_to_context(in.vdec, in.vst->codecpar), "copy dec params");

    // Try to enable CUDA hwaccel
    if (av_hwdevice_ctx_create(&in.hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) >= 0) {
        in.vdec->hw_device_ctx = av_buffer_ref(in.hw_device_ctx);
        in.vdec->get_format = [](AVCodecContext* ctx, const AVPixelFormat* pix_fmts) {
            return get_cuda_sw_format(ctx, pix_fmts);
        };
    }

    ff_check(avcodec_open2(in.vdec, dec, nullptr), "open decoder");
    return true;
}

static void close_input(InputContext& in)
{
    if (in.vdec) avcodec_free_context(&in.vdec);
    if (in.hw_device_ctx) av_buffer_unref(&in.hw_device_ctx);
    if (in.fmt) avformat_close_input(&in.fmt);
}

static void add_mastering_and_cll(AVStream* st)
{
    // Attach HDR mastering metadata to codec parameters coded_side_data (FFmpeg >= 8)
    AVPacketSideData* sd = av_packet_side_data_new(&st->codecpar->coded_side_data,
                                                   &st->codecpar->nb_coded_side_data,
                                                   AV_PKT_DATA_MASTERING_DISPLAY_METADATA,
                                                   sizeof(AVMasteringDisplayMetadata), 0);
    if (sd && sd->data && sd->size == sizeof(AVMasteringDisplayMetadata)) {
        AVMasteringDisplayMetadata* mdm = (AVMasteringDisplayMetadata*)sd->data;
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
        mdm->min_luminance = av_d2q(0.005, 10000);
        mdm->max_luminance = av_d2q(1000.0, 1);
        mdm->has_luminance = 1;
        mdm->has_primaries = 1;
    }

    sd = av_packet_side_data_new(&st->codecpar->coded_side_data,
                                 &st->codecpar->nb_coded_side_data,
                                 AV_PKT_DATA_CONTENT_LIGHT_LEVEL,
                                 sizeof(AVContentLightMetadata), 0);
    if (sd && sd->data && sd->size == sizeof(AVContentLightMetadata)) {
        AVContentLightMetadata* cll = (AVContentLightMetadata*)sd->data;
        cll->MaxCLL = 1000;
        cll->MaxFALL = 400;
    }
}

static bool open_output(const char* outPath, const InputContext& in, OutputContext& out)
{
    ff_check(avformat_alloc_output_context2(&out.fmt, nullptr, nullptr, outPath), "alloc out ctx");
    if (!out.fmt) throw std::runtime_error("cannot alloc out ctx");
    // Ensure packets are flushed promptly for progressive playback while encoding
    out.fmt->flags |= AVFMT_FLAG_FLUSH_PACKETS;

    // Map streams: copy all non-video as is (audio/subtitles). Create new video stream.
    out.map_streams.assign(in.fmt->nb_streams, -1);

    // Create video encoder stream
    const AVCodec* hevc = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!hevc) hevc = avcodec_find_encoder(AV_CODEC_ID_HEVC);
    if (!hevc) throw std::runtime_error("hevc encoder not found");

    out.vstream = avformat_new_stream(out.fmt, hevc);
    if (!out.vstream) throw std::runtime_error("new video stream");
    out.vstream->id = (int)out.fmt->nb_streams - 1;

    out.venc = avcodec_alloc_context3(hevc);
    if (!out.venc) throw std::runtime_error("alloc enc ctx");

    // We'll fill details later after first input frame size/fps is known.

    // Copy other streams (audio/subs) with codec copy
    for (unsigned i = 0; i < in.fmt->nb_streams; ++i) {
        if ((int)i == in.vstream) continue;
        
        AVStream* ist = in.fmt->streams[i];
        
        // Skip unsupported subtitle formats for MP4
        if (ist->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE) {
            const AVCodecDescriptor* desc = avcodec_descriptor_get(ist->codecpar->codec_id);
            if (desc && strcmp(desc->name, "subrip") == 0) {
                fprintf(stderr, "Note: Skipping SRT subtitles (stream %d) - not supported in MP4 container\n", i);
                continue;  // Skip this stream
            }
        }
        
        AVStream* ost = avformat_new_stream(out.fmt, nullptr);
        if (!ost) throw std::runtime_error("new stream (copy)");
        ff_check(avcodec_parameters_copy(ost->codecpar, ist->codecpar), "copy stream params");
        ost->codecpar->codec_tag = 0;
        ost->time_base = ist->time_base; // Preserve input stream time base to ensure correct timestamp scaling for copied streams
        out.map_streams[i] = ost->index;
    }

    // Open IO
    if (!(out.fmt->oformat->flags & AVFMT_NOFILE)) {
        ff_check(avio_open(&out.fmt->pb, outPath, AVIO_FLAG_WRITE), "open output file");
    }

    return true;
}

static void close_output(OutputContext& out)
{
    if (out.venc) avcodec_free_context(&out.venc);
    if (out.fmt) {
        if (!(out.fmt->oformat->flags & AVFMT_NOFILE) && out.fmt->pb) avio_closep(&out.fmt->pb);
        avformat_free_context(out.fmt);
        out.fmt = nullptr;
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.mp4 output.mp4 [--sws-colorspace|--cpu-colorspace]\n", argv[0]);
        fprintf(stderr, "\nOptions:\n");
        fprintf(stderr, "  --sws-colorspace, --cpu-colorspace  Bypass GPU color conversion pipeline and use libswscale instead.\n");
        return 1;
    }

    const char* inPath = argv[1];
    const char* outPath = argv[2];
    bool force_sws_colorspace = false; // when true, bypass GPU color conversion and use SWS

    // Parse optional flags
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sws-colorspace" || arg == "--cpu-colorspace") {
            force_sws_colorspace = true;
        }
    }

    av_log_set_level(AV_LOG_WARNING);

    InputContext in{};
    OutputContext out{};

    try {
        open_input(inPath, in);
        open_output(outPath, in, out);

        // Read input bitrate and fps
        int64_t in_bitrate = in.fmt->bit_rate;
        AVRational fr = in.vst->avg_frame_rate.num ? in.vst->avg_frame_rate : in.vst->r_frame_rate;
        if (fr.num == 0 || fr.den == 0) fr = {in.vst->time_base.den, in.vst->time_base.num};

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

        // Progress tracking variables
        int64_t processed_frames = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto last_update = start_time;
        const int update_interval_ms = 100; // Update progress every 100ms
        std::string progress_bar(50, ' ');

        // Initialize RTX processor when we know input size
        RtxProcessConfig cfg;
        cfg.enableVSR = true;
        cfg.enableTHDR = true;
        cfg.vsrQuality = 4;
        cfg.scaleFactor = 2;
        cfg.thdrContrast = 100;
        cfg.thdrSaturation = 100;
        cfg.thdrMiddleGray = 50;
        cfg.thdrMaxLuminance = 1000;

        RtxProcessor rtx; // CPU-path RTX instance; initialize lazily only if CPU path is used
        bool rtx_cpu_inited = false;

        // Prepare sws contexts (created on first decoded frame when actual format is known)
        SwsContext* sws_to_argb = nullptr;
        int last_src_format = AV_PIX_FMT_NONE;

        // A2R10G10B10 -> P010 for NVENC
        // In this pipeline, the RTX output maps to A2R10G10B10; prefer doing P010 conversion on GPU when possible.
        int dstW = in.vdec->width * cfg.scaleFactor;
        int dstH = in.vdec->height * cfg.scaleFactor;
        SwsContext* sws_to_p010 = nullptr; // CPU fallback
        // If force_sws_colorspace is enabled, we still allow GPU decode but bypass GPU color conversion pipeline
        bool use_cuda_path = (in.vdec->hw_device_ctx != nullptr) && !force_sws_colorspace;

        // Configure encoder now that sizes are known
        out.venc->codec_id = AV_CODEC_ID_HEVC;
        out.venc->width = dstW;
        out.venc->height = dstH;
        out.venc->time_base = av_inv_q(fr);
        out.venc->framerate = fr;
        // Prefer CUDA frames if decoder is CUDA-capable to avoid copies
        if (use_cuda_path) {
            out.venc->pix_fmt = AV_PIX_FMT_CUDA; // NVENC consumes CUDA frames via hw_frames_ctx
        } else {
            out.venc->pix_fmt = AV_PIX_FMT_P010LE; // CPU fallback
        }
        out.venc->gop_size = 2 * fr.num / std::max(1, fr.den);
        out.venc->max_b_frames = 2;
        out.venc->color_range = AVCOL_RANGE_MPEG;
        out.venc->color_trc = AVCOL_TRC_SMPTE2084;        // PQ
        out.venc->color_primaries = AVCOL_PRI_BT2020;
        out.venc->colorspace = AVCOL_SPC_BT2020_NCL;

        int64_t target_bitrate = (in_bitrate > 0) ? in_bitrate * 10 : (int64_t)25000000; // fallback 25Mbps
        av_opt_set(out.venc->priv_data, "preset", "p4", 0);
        av_opt_set(out.venc->priv_data, "rc", "cbr", 0);
        av_opt_set_int(out.venc->priv_data, "bitrate", target_bitrate, 0);
        av_opt_set_int(out.venc->priv_data, "maxrate", target_bitrate * 2, 0);
        // Set HEVC profile via string for compatibility
        av_opt_set(out.venc->priv_data, "profile", "main10", 0);
        // Ensure parameter sets are repeated and AUDs are present for better player compatibility
        av_opt_set_int(out.venc->priv_data, "repeat-headers", 1, 0);
        av_opt_set_int(out.venc->priv_data, "aud", 1, 0);

        ff_check(avcodec_parameters_from_context(out.vstream->codecpar, out.venc), "enc params to stream");
        // Prefer 'hvc1' brand to carry parameter sets (VPS/SPS/PPS) in-band, which improves fMP4 compatibility
        if (out.vstream->codecpar) {
            out.vstream->codecpar->codec_tag = MKTAG('h','v','c','1');
        }
        add_mastering_and_cll(out.vstream);

        // If using CUDA path, create encoder hw_frames_ctx on the same device before opening encoder
        AVBufferRef* enc_hw_frames = nullptr;
        if (use_cuda_path) {
            enc_hw_frames = av_hwframe_ctx_alloc(in.hw_device_ctx);
            if (!enc_hw_frames) throw std::runtime_error("av_hwframe_ctx_alloc failed for encoder");
            AVHWFramesContext* fctx = (AVHWFramesContext*)enc_hw_frames->data;
            fctx->format = AV_PIX_FMT_CUDA;
            fctx->sw_format = AV_PIX_FMT_P010LE;
            fctx->width = dstW;
            fctx->height = dstH;
            fctx->initial_pool_size = 16;
            ff_check(av_hwframe_ctx_init(enc_hw_frames), "init encoder hwframe ctx");
            out.venc->hw_frames_ctx = av_buffer_ref(enc_hw_frames);
        }

        ff_check(avcodec_open2(out.venc, out.venc->codec, nullptr), "open encoder");
        fprintf(stderr, "Pipeline: decode=%s, colorspace+scale+pack=%s\n",
                (in.vdec->hw_device_ctx ? "GPU(NVDEC)" : "CPU"),
                (use_cuda_path ? "GPU(RTX/CUDA)" : "CPU(SWS)"));
        if (enc_hw_frames) av_buffer_unref(&enc_hw_frames);

        // Write header
        out.vstream->time_base = out.venc->time_base; // Ensure muxer stream uses the same time_base as the encoder for consistent PTS/DTS
        out.vstream->avg_frame_rate = fr;
        // Write header with MOV/MP4 muxer flags for broad compatibility
        // faststart: move moov atom to the beginning at finalize time; plays everywhere after encode completes
        AVDictionary* movopts = nullptr;
        av_dict_set(&movopts, "movflags", "+faststart", 0);
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
        if (!sws_to_p010) throw std::runtime_error("sws_to_p010 alloc failed");
        const int* coeffs_bt2020 = sws_getCoefficients(SWS_CS_BT2020);
        sws_setColorspaceDetails(sws_to_p010,
                                 coeffs_bt2020, 1,
                                 coeffs_bt2020, 0,
                                 0, 1 << 16, 1 << 16);

        PacketPtr pkt(av_packet_alloc(), &av_packet_free_single);
        PacketPtr opkt(av_packet_alloc(), &av_packet_free_single);

        const uint8_t* rtx_data = nullptr;
        uint32_t rtxW=0, rtxH=0; size_t rtxPitch=0;

        // Progress display function
        auto show_progress = [&]() {
            if (total_frames <= 0) return; // Skip if we can't determine total frames
            
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            auto time_since_last_update = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
            
            if (time_since_last_update < update_interval_ms && processed_frames < total_frames) return;
            
            last_update = now;
            
            double progress = static_cast<double>(processed_frames) / total_frames;
            int bar_width = 50;
            int pos = static_cast<int>(bar_width * progress);
            
            std::string bar;
            bar.reserve(bar_width + 10);
            bar = "[";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) bar += "=";
                else if (i == pos) bar += ">";
                else bar += " ";
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

        // Read packets
        while (true) {
            int ret = av_read_frame(in.fmt, pkt.get());
            if (ret == AVERROR_EOF) break;
            ff_check(ret, "read frame");

            if (pkt->stream_index == in.vstream) {
                ff_check(avcodec_send_packet(in.vdec, pkt.get()), "send packet");
                av_packet_unref(pkt.get());

                while (true) {
                    ret = avcodec_receive_frame(in.vdec, frame.get());
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                    ff_check(ret, "receive frame");

                    AVFrame* decframe = frame.get();
                    FramePtr tmp(nullptr, &av_frame_free_single);
                    bool frame_is_cuda = (decframe->format == AV_PIX_FMT_CUDA);
                    if (frame_is_cuda && !use_cuda_path) {
                        // Decoder produced CUDA but encoder path is CPU: transfer to SW
                        if (!swframe) swframe.reset(av_frame_alloc());
                        ff_check(av_hwframe_transfer_data(swframe.get(), decframe, 0), "hwframe transfer");
                        swframe->pts = decframe->pts;
                        decframe = swframe.get();
                        frame_is_cuda = false;
                    }

                    // Select a stable input timestamp
                    int64_t in_pts = (decframe->best_effort_timestamp != AV_NOPTS_VALUE)
                        ? decframe->best_effort_timestamp
                        : decframe->pts;

                    AVFrame* frame_to_send = nullptr;
                    if (frame_is_cuda && use_cuda_path) {
                        // GPU path: allocate an encoder CUDA frame and fill it entirely on GPU via RTX
                        FramePtr enc_hw(av_frame_alloc(), &av_frame_free_single);
                        enc_hw->format = AV_PIX_FMT_CUDA;
                        enc_hw->width = dstW;
                        enc_hw->height = dstH;
                        ff_check(av_hwframe_get_buffer(out.venc->hw_frames_ctx, enc_hw.get(), 0), "alloc enc hw frame");

                        static bool rtx_gpu_init = false;
                        static RtxProcessor rtx_gpu;
                        if (!rtx_gpu_init) {
                            AVHWDeviceContext* devctx = (AVHWDeviceContext*)in.hw_device_ctx->data;
                            AVCUDADeviceContext* cudactx = (AVCUDADeviceContext*)devctx->hwctx;
                            CUcontext cu = cudactx->cuda_ctx;
                            if (!rtx_gpu.initializeWithContext(cu, cfg, in.vdec->width, in.vdec->height)) {
                                std::string detail = rtx_gpu.lastError();
                                if (detail.empty()) detail = "unknown error";
                                throw std::runtime_error(std::string("Failed to init RTX GPU path: ") + detail);
                            }
                            rtx_gpu_init = true;
                        }

                        bool bt2020 = (in.vdec->colorspace == AVCOL_SPC_BT2020_NCL);
                        if (rtx_gpu.processGpuNV12ToP010(decframe->data[0], decframe->linesize[0],
                                                         decframe->data[1], decframe->linesize[1],
                                                         enc_hw.get(), bt2020)) {
                            frame_to_send = enc_hw.get();
                            // Update progress prior to encoding
                            processed_frames++;
                            show_progress();

                            // Set PTS on the frame
                            if (in_pts != AV_NOPTS_VALUE) frame_to_send->pts = av_rescale_q(in_pts, in.vst->time_base, out.venc->time_base);
                            else frame_to_send->pts = AV_NOPTS_VALUE;

                            // Encode and then release enc_hw by letting enc_hw FramePtr go out of scope after send
                            ff_check(avcodec_send_frame(out.venc, frame_to_send), "send frame to encoder (CUDA)");
                            while (true) {
                                ret = avcodec_receive_packet(out.venc, opkt.get());
                                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                                ff_check(ret, "receive encoded packet");
                                opkt->stream_index = out.vstream->index;
                                av_packet_rescale_ts(opkt.get(), out.venc->time_base, out.vstream->time_base);
                                ff_check(av_interleaved_write_frame(out.fmt, opkt.get()), "write video packet");
                                av_packet_unref(opkt.get());
                            }
                            av_frame_unref(frame.get());
                            if (swframe) av_frame_unref(swframe.get());
                            continue; // handled this frame on GPU
                        }

                        // GPU path failed; fall back: transfer to SW, run CPU path, convert to P010, then upload into enc_hw
                        if (!swframe) swframe.reset(av_frame_alloc());
                        ff_check(av_hwframe_transfer_data(swframe.get(), decframe, 0), "hwframe transfer fallback");
                        swframe->pts = decframe->pts;

                        // Convert to ARGB
                        if (!sws_to_argb || last_src_format != swframe->format) {
                            if (sws_to_argb) sws_freeContext(sws_to_argb);
                            sws_to_argb = sws_getContext(
                                swframe->width, swframe->height, (AVPixelFormat)swframe->format,
                                swframe->width, swframe->height, AV_PIX_FMT_RGBA,
                                SWS_BILINEAR, nullptr, nullptr, nullptr);
                            if (!sws_to_argb) throw std::runtime_error("sws_to_argb alloc failed");
                            const int* coeffs = (in.vdec->colorspace == AVCOL_SPC_BT2020_NCL)
                                ? sws_getCoefficients(SWS_CS_BT2020)
                                : sws_getCoefficients(SWS_CS_ITU709);
                            sws_setColorspaceDetails(sws_to_argb, coeffs, 0, coeffs, 1, 0, 1 << 16, 1 << 16);
                            last_src_format = swframe->format;
                        }
                        const uint8_t* srcData2[AV_NUM_DATA_POINTERS] = { swframe->data[0], swframe->data[1], swframe->data[2], swframe->data[3] };
                        int srcLines2[AV_NUM_DATA_POINTERS] = { swframe->linesize[0], swframe->linesize[1], swframe->linesize[2], swframe->linesize[3] };
                        ff_check(av_frame_make_writable(bgra_frame.get()), "argb make writable (fallback)");
                        sws_scale(sws_to_argb, srcData2, srcLines2, 0, swframe->height, bgra_frame->data, bgra_frame->linesize);

                        // Lazy-initialize CPU RTX processor when first needed (fallback)
                        if (!rtx_cpu_inited) {
                            if (!rtx.initialize(0, cfg, in.vdec->width, in.vdec->height)) {
                                std::string detail = rtx.lastError();
                                if (detail.empty()) detail = "unknown error";
                                throw std::runtime_error(std::string("Failed to initialize RTX CPU path (fallback): ") + detail);
                            }
                            rtx_cpu_inited = true;
                        }
                        // RTX CPU process
                        if (!rtx.process(bgra_frame->data[0], (size_t)bgra_frame->linesize[0], rtx_data, rtxW, rtxH, rtxPitch))
                            throw std::runtime_error("RTX CPU processing failed (fallback)");
                        if (rtxW != (uint32_t)dstW || rtxH != (uint32_t)dstH)
                            throw std::runtime_error("Unexpected RTX output dimensions (fallback)");

                        // ABGR10 -> P010 on CPU
                        ff_check(av_frame_make_writable(p010_frame.get()), "p010 make writable (fallback)");
                        const uint8_t* abgr_planes2[1] = { rtx_data };
                        int abgr_lines2[1] = { static_cast<int>(rtxPitch) };
                        sws_scale(sws_to_p010, abgr_planes2, abgr_lines2, 0, dstH, p010_frame->data, p010_frame->linesize);

                        // Upload P010 to encoder CUDA frame
                        ff_check(av_hwframe_transfer_data(enc_hw.get(), p010_frame.get(), 0), "upload P010 to CUDA frame");

                        // Set PTS and encode
                        frame_to_send = enc_hw.get();
                        if (in_pts != AV_NOPTS_VALUE) frame_to_send->pts = av_rescale_q(in_pts, in.vst->time_base, out.venc->time_base);
                        else frame_to_send->pts = AV_NOPTS_VALUE;
                        // progress
                        processed_frames++;
                        show_progress();

                        ff_check(avcodec_send_frame(out.venc, frame_to_send), "send frame to encoder (fallback CUDA)");
                        while (true) {
                            ret = avcodec_receive_packet(out.venc, opkt.get());
                            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                            ff_check(ret, "receive encoded packet");
                            opkt->stream_index = out.vstream->index;
                            av_packet_rescale_ts(opkt.get(), out.venc->time_base, out.vstream->time_base);
                            ff_check(av_interleaved_write_frame(out.fmt, opkt.get()), "write video packet");
                            av_packet_unref(opkt.get());
                        }
                        av_frame_unref(frame.get());
                        if (swframe) av_frame_unref(swframe.get());
                        continue; // handled this frame via fallback
                    }

                    // CPU path: Convert to RGBA for RTX input
                    if (!sws_to_argb || last_src_format != decframe->format) {
                        if (sws_to_argb) sws_freeContext(sws_to_argb);
                        sws_to_argb = sws_getContext(
                            decframe->width, decframe->height, (AVPixelFormat)decframe->format,
                            decframe->width, decframe->height, AV_PIX_FMT_RGBA,  // ARGB for RTX
                            SWS_BILINEAR, nullptr, nullptr, nullptr);
                        if (!sws_to_argb) throw std::runtime_error("sws_to_argb alloc failed");
                        const int* coeffs = (in.vdec->colorspace == AVCOL_SPC_BT2020_NCL)
                            ? sws_getCoefficients(SWS_CS_BT2020)
                            : sws_getCoefficients(SWS_CS_ITU709);
                        sws_setColorspaceDetails(sws_to_argb, coeffs, 0, coeffs, 1, 0, 1 << 16, 1 << 16);
                        last_src_format = decframe->format;
                    }
                    const uint8_t* srcData[AV_NUM_DATA_POINTERS] = { decframe->data[0], decframe->data[1], decframe->data[2], decframe->data[3] };
                    int srcLines[AV_NUM_DATA_POINTERS] = { decframe->linesize[0], decframe->linesize[1], decframe->linesize[2], decframe->linesize[3] };
                    ff_check(av_frame_make_writable(bgra_frame.get()), "argb make writable");
                    sws_scale(sws_to_argb, srcData, srcLines, 0, decframe->height, bgra_frame->data, bgra_frame->linesize);

                    // Update progress counter before processing
                    processed_frames++;
                    show_progress();
                    
                    // Lazy-initialize CPU RTX processor when first needed
                    if (!rtx_cpu_inited) {
                        if (!rtx.initialize(0, cfg, in.vdec->width, in.vdec->height)) {
                            std::string detail = rtx.lastError();
                            if (detail.empty()) detail = "unknown error";
                            throw std::runtime_error(std::string("Failed to initialize RTX CPU path: ") + detail);
                        }
                        rtx_cpu_inited = true;
                    }
                    // Feed RTX (CPU path)
                    if (!rtx.process(bgra_frame->data[0], (size_t)bgra_frame->linesize[0], rtx_data, rtxW, rtxH, rtxPitch))
                        throw std::runtime_error("RTX processing failed");

                    if (rtxW != (uint32_t)dstW || rtxH != (uint32_t)dstH) {
                        throw std::runtime_error("Unexpected RTX output dimensions");
                    }

                    // Convert RTX ABGR10 output directly into encoder P010 buffer (CPU path)
                    ff_check(av_frame_make_writable(p010_frame.get()), "p010 make writable");
                    const uint8_t* abgr_planes[1] = { rtx_data };
                    int abgr_lines[1] = { static_cast<int>(rtxPitch) };
                    sws_scale(sws_to_p010, abgr_planes, abgr_lines, 0, dstH, p010_frame->data, p010_frame->linesize);
                    if (in_pts != AV_NOPTS_VALUE) {
                        p010_frame->pts = av_rescale_q(in_pts, in.vst->time_base, out.venc->time_base);
                    } else {
                        p010_frame->pts = AV_NOPTS_VALUE;
                    }

                    // Encode
                    ff_check(avcodec_send_frame(out.venc, p010_frame.get()), "send frame to encoder");
                    while (true) {
                        ret = avcodec_receive_packet(out.venc, opkt.get());
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                        ff_check(ret, "receive encoded packet");
                        opkt->stream_index = out.vstream->index;
                        av_packet_rescale_ts(opkt.get(), out.venc->time_base, out.vstream->time_base);
                        ff_check(av_interleaved_write_frame(out.fmt, opkt.get()), "write video packet");
                        av_packet_unref(opkt.get());
                    }
                    av_frame_unref(frame.get());
                    if (swframe) av_frame_unref(swframe.get());
                }
            } else {
                // Copy other streams
                int out_index = out.map_streams[pkt->stream_index];
                if (out_index >= 0) {
                    AVStream* ist = in.fmt->streams[pkt->stream_index];
                    AVStream* ost = out.fmt->streams[out_index];
                    av_packet_rescale_ts(pkt.get(), ist->time_base, ost->time_base);
                    pkt->stream_index = out_index;
                    ff_check(av_interleaved_write_frame(out.fmt, pkt.get()), "write copied packet");
                }
                av_packet_unref(pkt.get());
            }
        }

        // Flush encoder
        ff_check(avcodec_send_frame(out.venc, nullptr), "send flush");
        while (true) {
            int ret = avcodec_receive_packet(out.venc, opkt.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            ff_check(ret, "receive packet flush");
            opkt->stream_index = out.vstream->index;
            av_packet_rescale_ts(opkt.get(), out.venc->time_base, out.vstream->time_base);
            ff_check(av_interleaved_write_frame(out.fmt, opkt.get()), "write packet flush");
            av_packet_unref(opkt.get());
        }

        ff_check(av_write_trailer(out.fmt), "write trailer");
        
        // Print final progress
        if (total_frames > 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            double total_sec = total_ms / 1000.0;
            double avg_fps = (total_sec > 0) ? processed_frames / total_sec : 0.0;
            
            fprintf(stderr, "\nProcessing completed in %.1f seconds (%.1f fps)\n", 
                   total_sec, avg_fps);
        }

        if (sws_to_argb) sws_freeContext(sws_to_argb);
        sws_freeContext(sws_to_p010);

        close_output(out);
        close_input(in);

        return 0;

    } catch (const std::exception& ex) {
        fprintf(stderr, "Error: %s\n", ex.what());
        close_output(out);
        close_input(in);
        return 2;
    }
}
