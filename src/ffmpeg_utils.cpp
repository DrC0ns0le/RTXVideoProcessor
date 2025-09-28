#include "ffmpeg_utils.h"
#include "logger.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

extern "C" {
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/opt.h>
}

static AVPixelFormat get_cuda_hw_format(AVCodecContext *, const AVPixelFormat *pix_fmts);
static AVPixelFormat get_cuda_sw_format(AVCodecContext *ctx, const AVPixelFormat *pix_fmts);

bool open_input(const char *inPath, InputContext &in, const InputOpenOptions *options)
{
    if (!inPath)
        throw std::invalid_argument("open_input: null path");

    if (in.fmt)
        close_input(in);

    AVDictionary *openOpts = nullptr;
    if (options && !options->fflags.empty())
    {
        av_dict_set(&openOpts, "fflags", options->fflags.c_str(), 0);
    }

    int err = avformat_open_input(&in.fmt, inPath, nullptr, &openOpts);
    if (openOpts)
        av_dict_free(&openOpts);
    ff_check(err, "open input");
    ff_check(avformat_find_stream_info(in.fmt, nullptr), "find stream info");

    int vstream = av_find_best_stream(in.fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (vstream < 0)
        throw std::runtime_error("Failed to find video stream in input");

    in.vstream = vstream;
    in.vst = in.fmt->streams[vstream];

    const AVCodec *decoder = avcodec_find_decoder(in.vst->codecpar->codec_id);
    if (!decoder)
        throw std::runtime_error("Failed to find decoder for input video");

    in.vdec = avcodec_alloc_context3(decoder);
    if (!in.vdec)
        throw std::runtime_error("Failed to allocate decoder context");

    ff_check(avcodec_parameters_to_context(in.vdec, in.vst->codecpar), "copy decoder parameters");
    in.vdec->pkt_timebase = in.vst->time_base;
    in.vdec->framerate = av_guess_frame_rate(in.fmt, in.vst, nullptr);

    // Try to enable CUDA hardware decoding
    err = av_hwdevice_ctx_create(&in.hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (err >= 0 && in.hw_device_ctx)
    {
        // Configure decoder hardware frames context for P010 output if requested
        if (options && options->preferP010ForHDR)
        {
            AVBufferRef *dec_hw_frames = av_hwframe_ctx_alloc(in.hw_device_ctx);
            if (dec_hw_frames)
            {
                AVHWFramesContext *dec_fctx = (AVHWFramesContext *)dec_hw_frames->data;
                dec_fctx->format = AV_PIX_FMT_CUDA;
                dec_fctx->sw_format = AV_PIX_FMT_P010LE; // Request P010 for HDR content
                dec_fctx->width = in.vst->codecpar->width;
                dec_fctx->height = in.vst->codecpar->height;
                dec_fctx->initial_pool_size = 64;

                if (av_hwframe_ctx_init(dec_hw_frames) >= 0)
                {
                    in.vdec->hw_frames_ctx = av_buffer_ref(dec_hw_frames);
                }
                av_buffer_unref(&dec_hw_frames);
            }
        }

        in.vdec->hw_device_ctx = av_buffer_ref(in.hw_device_ctx);
        in.vdec->get_format = get_cuda_hw_format;
    }
    else
    {
        in.hw_device_ctx = nullptr;
    }

    err = avcodec_open2(in.vdec, decoder, nullptr);
    if (err < 0)
    {
        // Fallback to software decoding if hardware path failed
        if (in.vdec->hw_device_ctx)
            av_buffer_unref(&in.vdec->hw_device_ctx);
        if (in.hw_device_ctx)
            av_buffer_unref(&in.hw_device_ctx);
        in.hw_device_ctx = nullptr;
        in.vdec->get_format = nullptr;
        ff_check(avcodec_open2(in.vdec, decoder, nullptr), "open decoder");
    }

    return true;
}

void close_input(InputContext &in)
{
    if (in.vdec)
    {
        avcodec_free_context(&in.vdec);
        in.vdec = nullptr;
    }

    if (in.hw_device_ctx)
    {
        av_buffer_unref(&in.hw_device_ctx);
        in.hw_device_ctx = nullptr;
    }

    if (in.fmt)
    {
        avformat_close_input(&in.fmt);
        in.fmt = nullptr;
    }

    in.vstream = -1;
    in.vst = nullptr;
}

bool open_output(const char *outPath, const InputContext &in, OutputContext &out)
{
    if (!outPath)
        throw std::invalid_argument("open_output: null path");
    if (!in.fmt || !in.vst)
        throw std::runtime_error("open_output: input context not initialized");

    if (out.fmt)
        close_output(out);


    const char *effectiveFormat = nullptr;
    if (out.hlsOptions.enabled)
    {
        effectiveFormat = "hls";
    }

    ff_check(avformat_alloc_output_context2(&out.fmt, nullptr, effectiveFormat, outPath), "alloc output context");
    if (!out.fmt)
        throw std::runtime_error("Failed to allocate output format context");
    
    fprintf(stderr, "DEBUG: Output format context allocated successfully, muxer='%s'\n", 
            out.fmt->oformat ? out.fmt->oformat->name : "unknown");

    if (out.hlsOptions.enabled)
    {
        fprintf(stderr, "DEBUG: HLS muxer enabled, setting up options\n");
        AVDictionary *muxOpts = nullptr;
        if (out.hlsOptions.hlsTime > 0)
        {
            std::string value = std::to_string(out.hlsOptions.hlsTime);
            av_dict_set(&muxOpts, "hls_time", value.c_str(), 0);
            fprintf(stderr, "DEBUG: Set hls_time = %s\n", value.c_str());
        }
        if (!out.hlsOptions.segmentFilename.empty())
        {
            av_dict_set(&muxOpts, "hls_segment_filename", out.hlsOptions.segmentFilename.c_str(), 0);
            fprintf(stderr, "DEBUG: Set hls_segment_filename = %s\n", out.hlsOptions.segmentFilename.c_str());
        }
        if (!out.hlsOptions.segmentType.empty())
        {
            av_dict_set(&muxOpts, "hls_segment_type", out.hlsOptions.segmentType.c_str(), 0);
            fprintf(stderr, "DEBUG: Set hls_segment_type = %s\n", out.hlsOptions.segmentType.c_str());
        }
        if (!out.hlsOptions.initFilename.empty())
        {
            av_dict_set(&muxOpts, "hls_fmp4_init_filename", out.hlsOptions.initFilename.c_str(), 0);
            fprintf(stderr, "DEBUG: Set hls_fmp4_init_filename = %s\n", out.hlsOptions.initFilename.c_str());
        }
        if (out.hlsOptions.startNumber >= 0)
        {
            std::string value = std::to_string(out.hlsOptions.startNumber);
            av_dict_set(&muxOpts, "start_number", value.c_str(), 0);
            fprintf(stderr, "DEBUG: Set start_number = %s\n", value.c_str());
        }
        if (!out.hlsOptions.playlistType.empty())
        {
            av_dict_set(&muxOpts, "hls_playlist_type", out.hlsOptions.playlistType.c_str(), 0);
            fprintf(stderr, "DEBUG: Set hls_playlist_type = %s\n", out.hlsOptions.playlistType.c_str());
        }
        if (out.hlsOptions.listSize >= 0)
        {
            std::string value = std::to_string(out.hlsOptions.listSize);
            av_dict_set(&muxOpts, "hls_list_size", value.c_str(), 0);
            fprintf(stderr, "DEBUG: Set hls_list_size = %s\n", value.c_str());
        }
        if (out.hlsOptions.maxDelay >= 0)
        {
            std::string value = std::to_string(out.hlsOptions.maxDelay);
            av_dict_set(&muxOpts, "max_delay", value.c_str(), 0);
            fprintf(stderr, "DEBUG: Set max_delay = %s\n", value.c_str());
        }

        std::string hlsFlags;
        if (out.hlsOptions.segmentType == "fmp4")
        {
            // Enhanced flags for better browser compatibility with fMP4 segments
            // append_list: Create playlist immediately after header
            hlsFlags = "+append_list";
        }
        else
        {
            hlsFlags = "split_by_time+append_list";
        }
        if (out.hlsOptions.listSize > 0)
            hlsFlags += "+delete_segments";
        if (!hlsFlags.empty())
        {
            av_dict_set(&muxOpts, "hls_flags", hlsFlags.c_str(), 0);
            fprintf(stderr, "DEBUG: Set hls_flags = %s\n", hlsFlags.c_str());
        }

        // Debug: Print all HLS options that will be passed to the muxer
        fprintf(stderr, "DEBUG: Final HLS muxer options:\n");
        AVDictionaryEntry *entry = nullptr;
        while ((entry = av_dict_get(muxOpts, "", entry, AV_DICT_IGNORE_SUFFIX))) {
            fprintf(stderr, "DEBUG:   %s = %s\n", entry->key, entry->value);
        }

        out.muxOptions = muxOpts;
    }
    else
    {
        out.hlsOptions = HlsMuxOptions{};
        out.muxOptions = nullptr;
    }

    const AVCodec *encoder = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!encoder)
        encoder = avcodec_find_encoder(AV_CODEC_ID_HEVC);
    if (!encoder)
        throw std::runtime_error("Failed to find HEVC encoder (hevc_nvenc/AV_CODEC_ID_HEVC)");

    out.venc = avcodec_alloc_context3(encoder);
    if (!out.venc)
        throw std::runtime_error("Failed to allocate encoder context");

    out.venc->codec_id = encoder->id;
    out.venc->codec_type = AVMEDIA_TYPE_VIDEO;
    out.venc->time_base = av_inv_q(av_guess_frame_rate(in.fmt, in.vst, nullptr));
    if (out.venc->time_base.num == 0 || out.venc->time_base.den == 0)
        out.venc->time_base = {1, 60};

    out.vstream = avformat_new_stream(out.fmt, nullptr);
    if (!out.vstream)
        throw std::runtime_error("Failed to allocate output video stream");
    out.vstream->time_base = out.venc->time_base;

    out.map_streams.assign(in.fmt->nb_streams, -1);
    out.map_streams[in.vstream] = out.vstream->index;

    for (unsigned int i = 0; i < in.fmt->nb_streams; ++i)
    {
        if ((int)i == in.vstream)
            continue;

        AVStream *ist = in.fmt->streams[i];
        // Drop unsupported subtitle codecs for HLS outputs (only WebVTT is supported).
        if (out.hlsOptions.enabled && ist->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE &&
            ist->codecpar->codec_id != AV_CODEC_ID_WEBVTT)
        {
            const char *codec_name = avcodec_get_name(ist->codecpar->codec_id);
            LOG_WARN("Dropping subtitle stream %u (%s): HLS output requires WebVTT subtitles", i,
                     codec_name ? codec_name : "unknown");
            out.map_streams[i] = -1;
            continue;
        }

        // Ensure the output container supports the codec when stream copying.
        if (!avformat_query_codec(out.fmt->oformat, ist->codecpar->codec_id, FF_COMPLIANCE_NORMAL))
        {
            const char *codec_name = avcodec_get_name(ist->codecpar->codec_id);
            LOG_WARN("Dropping stream %u (%s): not supported by output container", i,
                     codec_name ? codec_name : "unknown");
            out.map_streams[i] = -1;
            continue;
        }

        AVStream *ost = avformat_new_stream(out.fmt, nullptr);
        if (!ost)
            throw std::runtime_error("Failed to allocate output stream");

        ff_check(avcodec_parameters_copy(ost->codecpar, ist->codecpar), "copy stream parameters");
        ost->time_base = ist->time_base;
        out.map_streams[i] = ost->index;
    }

    if (!(out.fmt->oformat->flags & AVFMT_NOFILE))
    {
        int avioFlags = AVIO_FLAG_WRITE;
        AVDictionary *avioOpts = nullptr;
        if (out.hlsOptions.enabled && out.hlsOptions.overwrite)
            av_dict_set(&avioOpts, "truncate", "1", 0);
        int openErr = avio_open2(&out.fmt->pb, outPath, avioFlags, nullptr, &avioOpts);
        if (avioOpts)
            av_dict_free(&avioOpts);
        ff_check(openErr, "open output file");
    }

    return true;
}

void close_output(OutputContext &out)
{
    if (out.venc)
    {
        avcodec_free_context(&out.venc);
        out.venc = nullptr;
    }

    if (out.fmt)
    {
        if (!(out.fmt->oformat->flags & AVFMT_NOFILE) && out.fmt->pb)
            avio_closep(&out.fmt->pb);
        avformat_free_context(out.fmt);
        out.fmt = nullptr;
    }

    out.vstream = nullptr;
    out.map_streams.clear();
}

static AVPixelFormat get_cuda_sw_format(AVCodecContext *ctx, const AVPixelFormat *pix_fmts)
{
    while (*pix_fmts != AV_PIX_FMT_NONE)
    {
        if (*pix_fmts == AV_PIX_FMT_CUDA)
            return *pix_fmts;
        pix_fmts++;
    }
    return AV_PIX_FMT_NONE;
}

static AVPixelFormat get_cuda_hw_format(AVCodecContext *, const AVPixelFormat *pix_fmts)
{
    const AVPixelFormat *p = pix_fmts;
    AVPixelFormat fallback = AV_PIX_FMT_NONE;
    if (p && *p != AV_PIX_FMT_NONE)
        fallback = *p;

    while (p && *p != AV_PIX_FMT_NONE)
    {
        if (*p == AV_PIX_FMT_CUDA)
            return *p;
        ++p;
    }

    return fallback;
}

// Attach HDR mastering metadata to codecpar coded_side_data and content light level.
void add_mastering_and_cll(AVStream *st, int max_luminance_nits)
{
    if (!st || !st->codecpar) return;

    constexpr int kMinNit = 1;
    constexpr int kMaxNit = 10000; // HDR10 mastering metadata limit
    int max_luminance = std::min(std::max(max_luminance_nits, kMinNit), kMaxNit);

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
        // Luminance
        mdm->min_luminance = av_make_q(50, 10000); // 0.005 cd/m²
        mdm->max_luminance = av_make_q(max_luminance * max_luminance, 10000); // 1000 cd/m² = 1000 * 1000 = 1000000
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
        cll->MaxCLL = max_luminance;
        cll->MaxFALL = std::max(kMinNit, max_luminance / 2);
    }
}
