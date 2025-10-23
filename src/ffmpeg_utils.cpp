#include "ffmpeg_utils.h"
#include "logger.h"
#include "audio_config.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

extern "C"
{
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/opt.h>
#include <libavutil/parseutils.h>
#include <libavutil/error.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersrc.h>
#include <libavfilter/buffersink.h>
}

// Use AudioParameters from audio_config.h to avoid struct duplication issues

static AVPixelFormat get_cuda_hw_format(AVCodecContext *, const AVPixelFormat *pix_fmts);
static AVPixelFormat get_cuda_sw_format(AVCodecContext *ctx, const AVPixelFormat *pix_fmts);

bool open_input(const char *inPath, InputContext &in, const InputOpenOptions *options)
{
    if (!inPath)
        throw std::invalid_argument("open_input: null path");

    if (in.fmt)
        close_input(in);

    // Initialize seek offset
    in.seek_offset_us = 0;

    AVDictionary *openOpts = nullptr;
    if (options && !options->fflags.empty())
    {
        av_dict_set(&openOpts, "fflags", options->fflags.c_str(), 0);
    }

    // Check if input is a network URL
    bool isNetworkInput = (std::strncmp(inPath, "http://", 7) == 0 ||
                           std::strncmp(inPath, "https://", 8) == 0 ||
                           std::strncmp(inPath, "rtmp://", 7) == 0 ||
                           std::strncmp(inPath, "rtsp://", 7) == 0);

    // For network inputs, set some helpful options
    if (isNetworkInput)
    {
        // Timeout for network operations (30 seconds)
        av_dict_set(&openOpts, "timeout", "30000000", 0);
        // Reconnect settings
        av_dict_set(&openOpts, "reconnect", "1", 0);
        av_dict_set(&openOpts, "reconnect_streamed", "1", 0);
        av_dict_set(&openOpts, "reconnect_delay_max", "5", 0);
        LOG_VERBOSE("Opening network input: %s", inPath);
    }

    int err = avformat_open_input(&in.fmt, inPath, nullptr, &openOpts);
    if (openOpts)
        av_dict_free(&openOpts);
    ff_check(err, "open input");
    ff_check(avformat_find_stream_info(in.fmt, nullptr), "find stream info");

    // Seek to start time if specified (equivalent to FFmpeg -ss option)
    if (options && !options->seekTime.empty())
    {
        int64_t seek_target = 0;
        int ret = av_parse_time(&seek_target, options->seekTime.c_str(), 1);
        if (ret < 0)
        {
            throw std::runtime_error("Invalid seek time format: " + options->seekTime);
        }

        in.seek_offset_us = seek_target;

        int seek_flags = AVSEEK_FLAG_BACKWARD;

        // Apply -noaccurate_seek or -seek2any: Allow seeking to non-keyframes
        // This enables fast seeking by using AVSEEK_FLAG_ANY
        if (options->noAccurateSeek || options->seek2any)
        {
            seek_flags |= AVSEEK_FLAG_ANY;
            LOG_DEBUG("Fast/inaccurate seeking enabled: AVSEEK_FLAG_BACKWARD | AVSEEK_FLAG_ANY");
        }

        if (options->seekTimestamp)
            seek_flags |= AVSEEK_FLAG_FRAME;

        ret = avformat_seek_file(in.fmt, -1, INT64_MIN, seek_target, INT64_MAX, seek_flags);
        if (ret < 0)
        {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, sizeof(errbuf), ret);
            LOG_WARN("Failed to seek to time %s: %s", options->seekTime.c_str(), errbuf);
            in.seek_offset_us = 0;
        }
        else
        {
            LOG_VERBOSE("Seeked to time: %s", options->seekTime.c_str());
        }
    }

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

    // Enable error concealment if requested
    // FFmpeg default: decoder drops corrupted frames automatically
    // With these flags: decoder outputs corrupted frames (may cause green frames after seeking)
    if (options && options->enableErrorConcealment)
    {
        in.vdec->flags2 |= AV_CODEC_FLAG2_SHOW_ALL;     // Show all frames even if corrupted
        in.vdec->flags |= AV_CODEC_FLAG_OUTPUT_CORRUPT; // Output potentially corrupted frames
        LOG_DEBUG("Decoder error concealment enabled");
    }

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

    // Find and set up audio stream if available
    int astream = av_find_best_stream(in.fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (astream >= 0)
    {
        in.astream = astream;
        in.ast = in.fmt->streams[astream];

        const AVCodec *audio_decoder = avcodec_find_decoder(in.ast->codecpar->codec_id);
        if (audio_decoder)
        {
            in.adec = avcodec_alloc_context3(audio_decoder);
            if (in.adec)
            {
                ff_check(avcodec_parameters_to_context(in.adec, in.ast->codecpar), "copy audio decoder parameters");
                in.adec->pkt_timebase = in.ast->time_base;

                err = avcodec_open2(in.adec, audio_decoder, nullptr);
                if (err < 0)
                {
                    char errbuf[AV_ERROR_MAX_STRING_SIZE];
                    av_make_error_string(errbuf, sizeof(errbuf), err);
                    LOG_WARN("Failed to open audio decoder (%s), audio will be copied without re-encoding", errbuf);
                    avcodec_free_context(&in.adec);
                    in.adec = nullptr;
                }
            }
        }
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

    if (in.adec)
    {
        avcodec_free_context(&in.adec);
        in.adec = nullptr;
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
    in.astream = -1;
    in.ast = nullptr;
    in.seek_offset_us = 0;
}

bool open_output(const char *outPath, const InputContext &in, OutputContext &out, const std::vector<std::string> &streamMaps)
{
    if (!outPath)
        throw std::invalid_argument("open_output: null path");
    if (!in.fmt || !in.vst)
        throw std::runtime_error("open_output: input context not initialized");

    if (out.fmt)
        close_output(out);

    // Detect pipe/stdout output
    bool isPipe = (std::strcmp(outPath, "-") == 0) ||
                  (std::strcmp(outPath, "pipe:") == 0) ||
                  (std::strcmp(outPath, "pipe:1") == 0);

    const char *effectiveFormat = nullptr;
    if (out.hlsOptions.enabled)
    {
        effectiveFormat = "hls";
    }
    else if (isPipe)
    {
        // For pipe output, we need to specify format explicitly since we can't infer from filename
        // Default to mp4 for pipe output (most compatible for streaming)
        effectiveFormat = "mp4";
    }

    ff_check(avformat_alloc_output_context2(&out.fmt, nullptr, effectiveFormat, isPipe ? nullptr : outPath), "alloc output context");
    if (!out.fmt)
        throw std::runtime_error("Failed to allocate output format context");

    LOG_DEBUG("Output format context allocated successfully, muxer='%s'\n",
              out.fmt->oformat ? out.fmt->oformat->name : "unknown");

    if (out.hlsOptions.enabled)
    {
        LOG_DEBUG("HLS muxer enabled, setting up options\n");
        AVDictionary *muxOpts = nullptr;
        if (out.hlsOptions.hlsTime > 0)
        {
            std::string value = std::to_string(out.hlsOptions.hlsTime);
            av_dict_set(&muxOpts, "hls_time", value.c_str(), 0);
            LOG_DEBUG("Set hls_time = %s\n", value.c_str());
        }
        if (!out.hlsOptions.segmentFilename.empty())
        {
            av_dict_set(&muxOpts, "hls_segment_filename", out.hlsOptions.segmentFilename.c_str(), 0);
            LOG_DEBUG("Set hls_segment_filename = %s\n", out.hlsOptions.segmentFilename.c_str());
        }
        if (!out.hlsOptions.segmentType.empty())
        {
            av_dict_set(&muxOpts, "hls_segment_type", out.hlsOptions.segmentType.c_str(), 0);
            LOG_DEBUG("Set hls_segment_type = %s\n", out.hlsOptions.segmentType.c_str());
        }
        if (!out.hlsOptions.initFilename.empty())
        {
            av_dict_set(&muxOpts, "hls_fmp4_init_filename", out.hlsOptions.initFilename.c_str(), 0);
            LOG_DEBUG("Set hls_fmp4_init_filename = %s\n", out.hlsOptions.initFilename.c_str());
        }
        if (out.hlsOptions.startNumber >= 0)
        {
            std::string value = std::to_string(out.hlsOptions.startNumber);
            av_dict_set(&muxOpts, "start_number", value.c_str(), 0);
            LOG_DEBUG("Set start_number = %s\n", value.c_str());
        }
        if (!out.hlsOptions.playlistType.empty())
        {
            av_dict_set(&muxOpts, "hls_playlist_type", out.hlsOptions.playlistType.c_str(), 0);
            LOG_DEBUG("Set hls_playlist_type = %s\n", out.hlsOptions.playlistType.c_str());
        }
        if (out.hlsOptions.listSize >= 0)
        {
            std::string value = std::to_string(out.hlsOptions.listSize);
            av_dict_set(&muxOpts, "hls_list_size", value.c_str(), 0);
            LOG_DEBUG("Set hls_list_size = %s\n", value.c_str());
        }
        if (out.hlsOptions.maxDelay >= 0)
        {
            std::string value = std::to_string(out.hlsOptions.maxDelay);
            av_dict_set(&muxOpts, "max_delay", value.c_str(), 0);
            LOG_DEBUG("Set max_delay = %s\n", value.c_str());
        }

        // Set hls_flags: user-specified flags take precedence over automatic flags
        std::string hlsFlags;
        if (!out.hlsOptions.customFlags.empty())
        {
            // User explicitly specified hls_flags via -hls_flags option
            hlsFlags = out.hlsOptions.customFlags;
            LOG_DEBUG("Using user-specified hls_flags = %s\n", hlsFlags.c_str());
        }
        else if (!out.hlsOptions.autoDiscontinuity) // autoDiscontinuity = !ffCompatible
        {
            // FFmpeg-compatible mode: Use minimal flags, let FFmpeg handle defaults
            if (out.hlsOptions.listSize > 0)
                hlsFlags = "+delete_segments";
        }
        else
        {
            // Legacy mode: Use custom flags for enhanced compatibility
            if (out.hlsOptions.segmentType == "fmp4")
            {
                hlsFlags = "+append_list";
            }
            else
            {
                hlsFlags = "split_by_time+append_list";
            }
            if (out.hlsOptions.listSize > 0)
                hlsFlags += "+delete_segments";
        }
        if (!hlsFlags.empty())
        {
            av_dict_set(&muxOpts, "hls_flags", hlsFlags.c_str(), 0);
            LOG_DEBUG("Set hls_flags = %s\n", hlsFlags.c_str());
        }

        // Conditional HLS discontinuity marking (FFmpeg does NOT do this automatically)
        // Only mark discontinuity if explicitly requested (legacy mode)
        if (in.seek_offset_us > 0 && out.hlsOptions.autoDiscontinuity)
        {
            LOG_DEBUG("Seek offset detected (%lld us), marking HLS discontinuity (non-FFmpeg default)\n", in.seek_offset_us);

            // Add discont_start flag to mark the beginning as a discontinuity
            // This tells HLS players that this is a new starting point
            av_dict_set(&muxOpts, "hls_flags", "+discont_start", AV_DICT_APPEND);
            LOG_DEBUG("Added discont_start to hls_flags for seek\n");
        }

        // Apply user-specified segment options (e.g., movflags=+frag_discont for fMP4)
        // This is required for proper hls.js playback with fMP4 segments
        if (!out.hlsOptions.segmentOptions.empty())
        {
            av_dict_set(&muxOpts, "hls_segment_options", out.hlsOptions.segmentOptions.c_str(), 0);
            LOG_INFO("HLS: Applied segment options: %s (passed to individual segment muxers)\n", out.hlsOptions.segmentOptions.c_str());
        }
        else if (out.hlsOptions.segmentType == "fmp4")
        {
            LOG_INFO("HLS: No custom segment options specified. Using default movflags (+frag_discont+frag_keyframe) from main muxer.\n");
        }

        // Debug: Print all HLS options that will be passed to the muxer
        LOG_DEBUG("Final HLS muxer options:\n");
        AVDictionaryEntry *entry = nullptr;
        while ((entry = av_dict_get(muxOpts, "", entry, AV_DICT_IGNORE_SUFFIX)))
        {
            LOG_DEBUG("  %s = %s\n", entry->key, entry->value);
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
    {
        // Try software libx265 encoder instead of hardware encoders
        encoder = avcodec_find_encoder_by_name("libx265");
        if (!encoder)
        {
            // Last resort: try any HEVC encoder (may select hardware encoders like hevc_d3d12va)
            encoder = avcodec_find_encoder(AV_CODEC_ID_HEVC);
        }
    }
    if (!encoder)
        throw std::runtime_error("Failed to find HEVC encoder (hevc_nvenc/libx265/AV_CODEC_ID_HEVC)");

    out.venc = avcodec_alloc_context3(encoder);
    if (!out.venc)
        throw std::runtime_error("Failed to allocate encoder context");

    out.venc->codec_id = encoder->id;
    out.venc->codec_type = AVMEDIA_TYPE_VIDEO;
    out.venc->time_base = av_inv_q(av_guess_frame_rate(in.fmt, in.vst, nullptr));
    if (out.venc->time_base.num == 0 || out.venc->time_base.den == 0)
        out.venc->time_base = {1, 60};

    // Decide which streams to include based on -map arguments
    if (!streamMaps.empty())
    {
        apply_stream_mappings(streamMaps, in, out);
    }
    else
    {
        // No explicit mappings - include all streams by default
        out.stream_decisions.assign(in.fmt->nb_streams, StreamMapDecision::COPY);
    }

    // Initialize input-to-output mapping
    out.input_to_output_map.assign(in.fmt->nb_streams, -1);

    // Mark video stream for processing (it's always processed, not just copied)
    out.stream_decisions[in.vstream] = StreamMapDecision::PROCESS_VIDEO;

    // Create output video stream (always created)
    out.vstream = avformat_new_stream(out.fmt, nullptr);
    if (!out.vstream)
        throw std::runtime_error("Failed to allocate output video stream");
    out.vstream->time_base = out.venc->time_base;
    out.input_to_output_map[in.vstream] = out.vstream->index;

    // Check if output is a pipe
    bool isPipeOutput = (std::strcmp(outPath, "-") == 0) ||
                        (std::strcmp(outPath, "pipe:") == 0) ||
                        (std::strcmp(outPath, "pipe:1") == 0);

    // Create output streams based on decisions
    for (unsigned int i = 0; i < in.fmt->nb_streams; ++i)
    {
        // Skip video stream (already handled)
        if ((int)i == in.vstream)
            continue;

        // Skip excluded streams
        if (out.stream_decisions[i] == StreamMapDecision::EXCLUDE)
            continue;

        AVStream *ist = in.fmt->streams[i];

        // Apply output-specific filters (overrides user mapping decisions if needed)

        // Drop subtitle streams when outputting to pipe to avoid text contamination
        if (isPipeOutput && ist->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE)
        {
            const char *codec_name = avcodec_get_name(ist->codecpar->codec_id);
            LOG_INFO("Dropping subtitle stream %u (%s): not supported for pipe output", i,
                     codec_name ? codec_name : "unknown");
            out.stream_decisions[i] = StreamMapDecision::EXCLUDE;
            continue;
        }

        // Drop unsupported subtitle codecs for HLS outputs (only WebVTT is supported)
        if (out.hlsOptions.enabled && ist->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE &&
            ist->codecpar->codec_id != AV_CODEC_ID_WEBVTT)
        {
            const char *codec_name = avcodec_get_name(ist->codecpar->codec_id);
            LOG_WARN("Dropping subtitle stream %u (%s): HLS output requires WebVTT subtitles", i,
                     codec_name ? codec_name : "unknown");
            out.stream_decisions[i] = StreamMapDecision::EXCLUDE;
            continue;
        }

        // Ensure the output container supports the codec when stream copying
        if (!avformat_query_codec(out.fmt->oformat, ist->codecpar->codec_id, FF_COMPLIANCE_NORMAL))
        {
            const char *codec_name = avcodec_get_name(ist->codecpar->codec_id);
            LOG_WARN("Dropping stream %u (%s): not supported by output container", i,
                     codec_name ? codec_name : "unknown");
            out.stream_decisions[i] = StreamMapDecision::EXCLUDE;
            continue;
        }

        // Skip creating output streams for audio that will be re-encoded
        if (out.stream_decisions[i] == StreamMapDecision::PROCESS_AUDIO)
        {
            const char *codec_name = avcodec_get_name(ist->codecpar->codec_id);
            LOG_INFO("Audio stream %u (%s) will be re-encoded", i,
                     codec_name ? codec_name : "unknown");
            // Output stream will be created later in setup_audio_encoder
            continue;
        }

        // Create output stream for copying
        AVStream *ost = avformat_new_stream(out.fmt, nullptr);
        if (!ost)
            throw std::runtime_error("Failed to allocate output stream");

        ff_check(avcodec_parameters_copy(ost->codecpar, ist->codecpar), "copy stream parameters");

        // For audio streams, set timescale to sample rate for MP4 parser compatibility
        if (ist->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && ist->codecpar->sample_rate > 0)
        {
            ost->time_base = {1, ist->codecpar->sample_rate};
            // Track the first audio stream created (for audio passthrough/copy)
            if (!out.astream)
            {
                out.astream = ost;
            }
        }
        else
        {
            ost->time_base = ist->time_base;
        }

        out.input_to_output_map[i] = ost->index;
    }

    // Log final stream mapping for debugging
    LOG_DEBUG("Stream mapping summary: %u input streams -> %u output streams",
              in.fmt->nb_streams, out.fmt->nb_streams);
    for (unsigned int i = 0; i < out.input_to_output_map.size(); ++i)
    {
        if (out.input_to_output_map[i] >= 0)
        {
            AVStream *ist = in.fmt->streams[i];
            const char *type_name = av_get_media_type_string(ist->codecpar->codec_type);
            LOG_DEBUG("  Input stream %u (%s) -> Output stream %d", i,
                      type_name ? type_name : "unknown", out.input_to_output_map[i]);
        }
    }

    if (!(out.fmt->oformat->flags & AVFMT_NOFILE))
    {
        int avioFlags = AVIO_FLAG_WRITE;
        AVDictionary *avioOpts = nullptr;
        if (out.hlsOptions.enabled && out.hlsOptions.overwrite)
            av_dict_set(&avioOpts, "truncate", "1", 0);

        // Handle pipe/stdout output
        bool isPipe = (std::strcmp(outPath, "-") == 0) ||
                      (std::strcmp(outPath, "pipe:") == 0) ||
                      (std::strcmp(outPath, "pipe:1") == 0);

        int openErr;
        if (isPipe)
        {
#ifdef _WIN32
            // Set stdout to binary mode on Windows to prevent newline conversion
            _setmode(_fileno(stdout), _O_BINARY);
#endif
            // Use pipe:1 to write to stdout
            openErr = avio_open2(&out.fmt->pb, "pipe:1", avioFlags, nullptr, &avioOpts);
        }
        else
        {
            openErr = avio_open2(&out.fmt->pb, outPath, avioFlags, nullptr, &avioOpts);
        }

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

    if (out.aenc)
    {
        avcodec_free_context(&out.aenc);
        out.aenc = nullptr;
    }

    if (out.filter_graph)
    {
        avfilter_graph_free(&out.filter_graph);
        out.filter_graph = nullptr;
        out.buffersrc_ctx = nullptr;
        out.buffersink_ctx = nullptr;
    }

    if (out.swr_ctx)
    {
        swr_free(&out.swr_ctx);
        out.swr_ctx = nullptr;
    }

    if (out.audio_fifo)
    {
        av_audio_fifo_free(out.audio_fifo);
        out.audio_fifo = nullptr;
    }

    if (out.fmt)
    {
        if (!(out.fmt->oformat->flags & AVFMT_NOFILE) && out.fmt->pb)
            avio_closep(&out.fmt->pb);
        avformat_free_context(out.fmt);
        out.fmt = nullptr;
    }

    out.vstream = nullptr;
    out.astream = nullptr;
    out.next_audio_pts = 0;
    out.a_start_pts = AV_NOPTS_VALUE;
    out.stream_decisions.clear();
    out.input_to_output_map.clear();
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
    if (!st || !st->codecpar)
        return;

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
        mdm->min_luminance = av_make_q(50, 10000);                            // 0.005 cd/m²
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

// Helper: Check if stream mappings include audio streams
static bool has_audio_in_mappings(const std::vector<std::string> &mappings)
{
    for (const auto &mapping : mappings)
    {
        // Skip empty or exclusion mappings
        if (mapping.empty() || mapping[0] == '-')
            continue;

        // Parse using same logic as parse_stream_mapping
        size_t colonPos = mapping.find(':');
        if (colonPos == std::string::npos)
            continue;

        std::string stream_part = mapping.substr(colonPos + 1);
        // Remove trailing '?' if present
        if (!stream_part.empty() && stream_part.back() == '?')
            stream_part.pop_back();

        // Check for audio: stream index 1 or type specifier 'a'
        if (stream_part == "1" || stream_part == "a")
        {
            return true;
        }
    }
    return false;
}

void configure_audio_from_params(const AudioParameters &params, OutputContext &out)
{
    // Configure audio settings from parsed parameters
    out.audioConfig.codec = params.codec;
    out.audioConfig.channels = params.channels;
    out.audioConfig.bitrate = params.bitrate;
    out.audioConfig.sampleRate = params.sampleRate;
    out.audioConfig.filter = params.filter;

    // Only enable audio processing if there are actual audio parameters specified
    // Exclude "copy" codec as it doesn't require processing pipeline (FFmpeg compatibility)
    bool hasAudioParams = (!params.codec.empty() && params.codec != "copy") ||
                          params.channels > 0 ||
                          params.bitrate > 0 ||
                          params.sampleRate > 0 ||
                          !params.filter.empty();

    // Check if stream mappings include audio
    bool hasAudioMapping = has_audio_in_mappings(params.streamMaps);

    out.audioConfig.enabled = hasAudioParams || hasAudioMapping;

    if (out.audioConfig.enabled)
    {
        LOG_INFO("Audio processing enabled with codec=%s, channels=%d, bitrate=%d, filter=%s",
                 params.codec.c_str(), params.channels, params.bitrate, params.filter.c_str());
    }
}

// Helper structure for parsing -map arguments
struct StreamMapping
{
    bool exclude;                                  // true if mapping starts with '-'
    int stream_index = -1;                         // specific stream index (e.g., 0 from "0:0"), or -1 for type-based
    AVMediaType media_type = AVMEDIA_TYPE_UNKNOWN; // type specifier (v/a/s/d/t)
};

static StreamMapping parse_stream_mapping(const std::string &mapping)
{
    StreamMapping result{};

    std::string clean = mapping;

    // Check for exclusion prefix
    result.exclude = (!clean.empty() && clean[0] == '-');
    if (result.exclude)
    {
        clean = clean.substr(1);
    }

    // Remove optional suffix '?' if present (not used but needs to be stripped for parsing)
    if (!clean.empty() && clean.back() == '?')
    {
        clean.pop_back();
    }

    // Parse "file:stream" format (we only support file 0)
    size_t colonPos = clean.find(':');
    if (colonPos == std::string::npos)
    {
        LOG_WARN("Invalid stream mapping format (missing ':'): %s", mapping.c_str());
        return result;
    }

    std::string file_part = clean.substr(0, colonPos);
    std::string stream_part = clean.substr(colonPos + 1);

    if (file_part != "0")
    {
        LOG_WARN("Only file index 0 is supported: %s", mapping.c_str());
        return result;
    }

    // Parse stream specifier: can be index (0, 1, 2) or type (v, a, s, d, t)
    if (!stream_part.empty() && std::isdigit(stream_part[0]))
    {
        // Numeric stream index
        result.stream_index = std::stoi(stream_part);
    }
    else if (stream_part == "v")
    {
        result.media_type = AVMEDIA_TYPE_VIDEO;
    }
    else if (stream_part == "a")
    {
        result.media_type = AVMEDIA_TYPE_AUDIO;
    }
    else if (stream_part == "s")
    {
        result.media_type = AVMEDIA_TYPE_SUBTITLE;
    }
    else if (stream_part == "d")
    {
        result.media_type = AVMEDIA_TYPE_DATA;
    }
    else if (stream_part == "t")
    {
        result.media_type = AVMEDIA_TYPE_ATTACHMENT;
    }
    else
    {
        LOG_WARN("Unknown stream specifier: %s", stream_part.c_str());
    }

    return result;
}

// Decide which streams should be included in output based on -map arguments
// This is the SINGLE SOURCE OF TRUTH for all stream mapping decisions
// Handles: inclusion, exclusion, copy vs process
static void decide_stream_mappings(const std::vector<std::string> &mappings,
                                   const InputContext &in,
                                   OutputContext &out)
{
    unsigned int nb_streams = in.fmt->nb_streams;

    // Initialize all streams as EXCLUDE
    out.stream_decisions.assign(nb_streams, StreamMapDecision::EXCLUDE);

    // Determine if audio should be re-encoded (not just copied)
    bool audio_needs_processing = out.audioConfig.enabled &&
                                  !out.audioConfig.codec.empty() &&
                                  out.audioConfig.codec != "copy";

    // Determine if we have explicit inclusions (non-exclusion -map directives)
    bool has_explicit_inclusions = false;
    for (const auto &mapping : mappings)
    {
        if (mapping.empty())
            continue;
        if (mapping[0] != '-')
        { // Not an exclusion
            has_explicit_inclusions = true;
            break;
        }
    }

    // FFmpeg behavior: if NO explicit inclusions, include all streams by default
    if (!has_explicit_inclusions)
    {
        for (unsigned int i = 0; i < nb_streams; ++i)
        {
            out.stream_decisions[i] = StreamMapDecision::COPY;
        }
    }

    // Process all -map directives in order
    for (const auto &mapping : mappings)
    {
        if (mapping.empty())
            continue;

        StreamMapping parsed = parse_stream_mapping(mapping);

        // Apply the mapping
        for (unsigned int i = 0; i < nb_streams; ++i)
        {
            AVStream *stream = in.fmt->streams[i];
            bool matches = false;

            // Check if this stream matches the mapping specifier
            if (parsed.stream_index >= 0)
            {
                // Index-based mapping
                matches = (i == (unsigned int)parsed.stream_index);
            }
            else if (parsed.media_type != AVMEDIA_TYPE_UNKNOWN)
            {
                // Type-based mapping
                matches = (stream->codecpar->codec_type == parsed.media_type);
            }

            if (matches)
            {
                if (parsed.exclude)
                {
                    out.stream_decisions[i] = StreamMapDecision::EXCLUDE;
                }
                else
                {
                    // Decide between COPY and PROCESS based on stream type and config
                    AVMediaType codec_type = stream->codecpar->codec_type;
                    if (codec_type == AVMEDIA_TYPE_AUDIO && audio_needs_processing)
                    {
                        out.stream_decisions[i] = StreamMapDecision::PROCESS_AUDIO;
                    }
                    else
                    {
                        out.stream_decisions[i] = StreamMapDecision::COPY;
                    }
                }
            }
        }
    }

    LOG_DEBUG("Stream mapping decisions:");
    for (unsigned int i = 0; i < nb_streams; ++i)
    {
        const char *type_name = av_get_media_type_string(in.fmt->streams[i]->codecpar->codec_type);
        const char *decision_name = "UNKNOWN";
        switch (out.stream_decisions[i])
        {
        case StreamMapDecision::EXCLUDE:
            decision_name = "EXCLUDE";
            break;
        case StreamMapDecision::COPY:
            decision_name = "COPY";
            break;
        case StreamMapDecision::PROCESS_VIDEO:
            decision_name = "PROCESS_VIDEO";
            break;
        case StreamMapDecision::PROCESS_AUDIO:
            decision_name = "PROCESS_AUDIO";
            break;
        }
        LOG_DEBUG("  Stream %u (%s): %s", i, type_name ? type_name : "unknown", decision_name);
    }
}

bool apply_stream_mappings(const std::vector<std::string> &mappings, const InputContext &in, OutputContext &out)
{
    // All logic consolidated into decide_stream_mappings for clarity
    decide_stream_mappings(mappings, in, out);
    return true;
}

bool setup_audio_encoder(const InputContext &in, OutputContext &out)
{
    if (!out.audioConfig.enabled)
    {
        return true; // No audio processing needed
    }

    // Check if we have a valid input context
    if (!in.fmt)
    {
        LOG_WARN("Invalid input format context for audio setup");
        return false;
    }

    // Find audio stream in input
    int audio_stream_idx = av_find_best_stream(in.fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (audio_stream_idx < 0)
    {
        LOG_WARN("No audio stream found in input for audio encoding");
        return false;
    }

    // Validate that we have a valid audio stream
    if (audio_stream_idx >= (int)in.fmt->nb_streams || !in.fmt->streams[audio_stream_idx])
    {
        LOG_WARN("Invalid audio stream index: %d", audio_stream_idx);
        return false;
    }

    // Set up audio codec
    std::string codec = out.audioConfig.codec.empty() ? "aac" : out.audioConfig.codec;
    const AVCodec *audio_encoder = avcodec_find_encoder_by_name(codec.c_str());
    if (!audio_encoder)
    {
        if (!out.audioConfig.codec.empty())
        {
            LOG_WARN("Requested audio codec '%s' not found, falling back to AAC", codec.c_str());
        }
        audio_encoder = avcodec_find_encoder(AV_CODEC_ID_AAC);
        codec = "aac"; // Update for logging
    }
    if (!audio_encoder)
    {
        LOG_WARN("Failed to find any audio encoder (tried: %s, aac)", out.audioConfig.codec.c_str());
        return false;
    }

    out.aenc = avcodec_alloc_context3(audio_encoder);
    if (!out.aenc)
    {
        LOG_WARN("Failed to allocate audio encoder context");
        return false;
    }

    // Configure audio encoder with new channel layout API
    AVStream *input_audio = in.fmt->streams[audio_stream_idx];
    out.aenc->codec_id = audio_encoder->id;
    out.aenc->codec_type = AVMEDIA_TYPE_AUDIO;

    // Use configured sample rate if specified, otherwise use input sample rate
    if (out.audioConfig.sampleRate > 0)
    {
        out.aenc->sample_rate = out.audioConfig.sampleRate;
    }
    else
    {
        out.aenc->sample_rate = input_audio->codecpar->sample_rate;
    }

    // Use new AVChannelLayout API instead of deprecated channels/channel_layout
    if (out.audioConfig.channels > 0)
    {
        av_channel_layout_default(&out.aenc->ch_layout, out.audioConfig.channels);
    }
    else
    {
        // Copy the channel layout from input
        int ret = av_channel_layout_copy(&out.aenc->ch_layout, &input_audio->codecpar->ch_layout);
        if (ret < 0)
        {
            LOG_WARN("Failed to copy channel layout from input");
            return false;
        }
    }

    out.aenc->bit_rate = (out.audioConfig.bitrate > 0) ? out.audioConfig.bitrate : 128000;
    out.aenc->time_base = {1, out.aenc->sample_rate};

    // Set sample format
    if (audio_encoder->sample_fmts)
    {
        out.aenc->sample_fmt = audio_encoder->sample_fmts[0];
    }
    else
    {
        out.aenc->sample_fmt = AV_SAMPLE_FMT_FLTP;
    }

    // Set frame size if encoder requires it
    if (audio_encoder->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)
    {
        out.aenc->frame_size = 0; // Variable frame size
    }
    else
    {
        // For AAC, frame size is typically 1024 samples
        out.aenc->frame_size = 1024;
    }

    // Create output audio stream (only if not already created during stream mapping)
    if (!out.astream)
    {
        out.astream = avformat_new_stream(out.fmt, nullptr);
        if (!out.astream)
        {
            LOG_WARN("Failed to create output audio stream");
            return false;
        }
    }

    // Open encoder
    int ret = avcodec_open2(out.aenc, audio_encoder, nullptr);
    if (ret < 0)
    {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, sizeof(errbuf), ret);
        LOG_WARN("Failed to open audio encoder: %s", errbuf);
        return false;
    }

    // Copy encoder parameters to stream
    ret = avcodec_parameters_from_context(out.astream->codecpar, out.aenc);
    if (ret < 0)
    {
        LOG_WARN("Failed to copy audio encoder parameters to stream");
        return false;
    }

    // Set audio stream time_base to match sample rate for MP4 muxer compatibility
    // This is required for fragmented MP4 formats (dash, fmp4) where timescale must equal sample rate
    out.astream->time_base = {1, out.aenc->sample_rate};

    // Initialize audio PTS tracking (will be adjusted after seeking if needed)
    init_audio_pts_after_seek(in, out);

    // Set up audio resampler for format conversion
    out.swr_ctx = swr_alloc();
    if (!out.swr_ctx)
    {
        LOG_WARN("Failed to allocate resampler context");
        return false;
    }

    // Configure resampler from input to encoder format
    av_opt_set_chlayout(out.swr_ctx, "in_chlayout", &input_audio->codecpar->ch_layout, 0);
    av_opt_set_int(out.swr_ctx, "in_sample_rate", input_audio->codecpar->sample_rate, 0);
    av_opt_set_sample_fmt(out.swr_ctx, "in_sample_fmt", (AVSampleFormat)input_audio->codecpar->format, 0);

    av_opt_set_chlayout(out.swr_ctx, "out_chlayout", &out.aenc->ch_layout, 0);
    av_opt_set_int(out.swr_ctx, "out_sample_rate", out.aenc->sample_rate, 0);
    av_opt_set_sample_fmt(out.swr_ctx, "out_sample_fmt", out.aenc->sample_fmt, 0);

    ret = swr_init(out.swr_ctx);
    if (ret < 0)
    {
        LOG_WARN("Failed to initialize resampler");
        swr_free(&out.swr_ctx);
        return false;
    }

    // Create audio FIFO for buffering samples to create fixed-size frames
    out.audio_fifo = av_audio_fifo_alloc(out.aenc->sample_fmt, out.aenc->ch_layout.nb_channels, out.aenc->frame_size * 2);
    if (!out.audio_fifo)
    {
        LOG_WARN("Failed to allocate audio FIFO");
        swr_free(&out.swr_ctx);
        return false;
    }

    return true;
}

bool setup_audio_filter(const InputContext &in, OutputContext &out)
{
    if (!out.audioConfig.enabled || out.audioConfig.filter.empty() || !out.aenc || in.astream < 0)
    {
        return true; // No filtering needed
    }

    // Additional validation
    if (!in.fmt || in.astream >= (int)in.fmt->nb_streams || !in.fmt->streams[in.astream])
    {
        LOG_WARN("Invalid audio stream for filter setup");
        return false;
    }

    int ret;
    char args[512];
    char ch_layout_str[128];
    const AVFilter *abuffersrc = avfilter_get_by_name("abuffer");
    const AVFilter *abuffersink = avfilter_get_by_name("abuffersink");
    AVFilterInOut *outputs = avfilter_inout_alloc();
    AVFilterInOut *inputs = avfilter_inout_alloc();

    if (!outputs || !inputs || !abuffersrc || !abuffersink)
    {
        LOG_WARN("Failed to allocate audio filter components");
        return false;
    }

    out.filter_graph = avfilter_graph_alloc();
    if (!out.filter_graph)
    {
        LOG_WARN("Failed to allocate audio filter graph");
        avfilter_inout_free(&inputs);
        avfilter_inout_free(&outputs);
        return false;
    }

    // Create buffer source with new channel layout API
    AVStream *input_audio = in.fmt->streams[in.astream];

    // Get channel layout description for the input
    av_channel_layout_describe(&input_audio->codecpar->ch_layout, ch_layout_str, sizeof(ch_layout_str));

    snprintf(args, sizeof(args),
             "time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%s",
             input_audio->time_base.num, input_audio->time_base.den,
             input_audio->codecpar->sample_rate,
             av_get_sample_fmt_name((AVSampleFormat)input_audio->codecpar->format),
             ch_layout_str);

    ret = avfilter_graph_create_filter(&out.buffersrc_ctx, abuffersrc, "in",
                                       args, nullptr, out.filter_graph);
    if (ret < 0)
    {
        LOG_WARN("Failed to create audio buffer source");
        goto cleanup;
    }

    // Create buffer sink
    ret = avfilter_graph_create_filter(&out.buffersink_ctx, abuffersink, "out",
                                       nullptr, nullptr, out.filter_graph);
    if (ret < 0)
    {
        LOG_WARN("Failed to create audio buffer sink");
        goto cleanup;
    }

    // Note: Setting format constraints on buffersink is optional for most cases
    // The encoder will handle format conversion if needed

    // Set up the filter chain
    outputs->name = av_strdup("in");
    outputs->filter_ctx = out.buffersrc_ctx;
    outputs->pad_idx = 0;
    outputs->next = nullptr;

    inputs->name = av_strdup("out");
    inputs->filter_ctx = out.buffersink_ctx;
    inputs->pad_idx = 0;
    inputs->next = nullptr;

    // Parse and configure the filter graph
    ret = avfilter_graph_parse_ptr(out.filter_graph, out.audioConfig.filter.c_str(),
                                   &inputs, &outputs, nullptr);
    if (ret < 0)
    {
        LOG_WARN("Failed to parse audio filter graph: %s", out.audioConfig.filter.c_str());
        goto cleanup;
    }

    ret = avfilter_graph_config(out.filter_graph, nullptr);
    if (ret < 0)
    {
        LOG_WARN("Failed to configure audio filter graph");
        goto cleanup;
    }

    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);
    return true;

cleanup:
    if (out.filter_graph)
    {
        avfilter_graph_free(&out.filter_graph);
        out.filter_graph = nullptr;
    }
    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);
    return false;
}

bool process_audio_frame(AVFrame *input_frame, OutputContext &out, AVPacket *output_packet)
{
    if (!input_frame || !out.aenc || !output_packet)
    {
        return false;
    }

    int ret;
    AVFrame *processed_frame = input_frame;

    // Handle first audio frame after seeking to establish baseline
    if (out.a_start_pts == AV_NOPTS_VALUE && input_frame->pts != AV_NOPTS_VALUE)
    {
        out.a_start_pts = input_frame->pts;

        // COPYTS mode: Initialize next_audio_pts from input timestamps to maintain A/V sync
        // This ensures audio timestamps match video timeline (e.g., both start at 24s)
        // Required for proper HLS tfdt calculation: tfdt = cluster[0].dts - start_dts
        if (out.audioConfig.copyts)
        {
            // Rescale input PTS to output timebase
            int64_t rescaled_pts = av_rescale_q(input_frame->pts,
                                                 input_frame->time_base,
                                                 out.aenc->time_base);
            out.next_audio_pts = rescaled_pts;
            LOG_DEBUG("COPYTS: Initialized audio PTS from first frame: %lld (input) -> %lld (output timebase)",
                      input_frame->pts, rescaled_pts);
        }
    }

    // Apply audio filter if configured
    if (out.filter_graph && out.buffersrc_ctx && out.buffersink_ctx)
    {
        // Send frame to filter
        ret = av_buffersrc_add_frame_flags(out.buffersrc_ctx, input_frame, AV_BUFFERSRC_FLAG_KEEP_REF);
        if (ret < 0)
        {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, sizeof(errbuf), ret);
            LOG_WARN("Failed to send frame to audio filter: %s", errbuf);
            return false;
        }

        // Get filtered frame
        AVFrame *filtered_frame = av_frame_alloc();
        if (!filtered_frame)
        {
            LOG_WARN("Failed to allocate filtered frame");
            return false;
        }

        ret = av_buffersink_get_frame_flags(out.buffersink_ctx, filtered_frame, 0);
        if (ret >= 0)
        {
            processed_frame = filtered_frame;
        }
        else if (ret == AVERROR(EAGAIN))
        {
            // No frame available yet, this is normal for filters that buffer frames
            av_frame_free(&filtered_frame);
            return true;
        }
        else if (ret == AVERROR_EOF)
        {
            // End of stream reached
            av_frame_free(&filtered_frame);
            return false;
        }
        else
        {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, sizeof(errbuf), ret);
            LOG_WARN("Failed to get frame from audio filter: %s", errbuf);
            av_frame_free(&filtered_frame);
            return false;
        }
    }

    // Resample frame to encoder format if resampler is available
    AVFrame *resampled_frame = nullptr;
    if (out.swr_ctx)
    {
        resampled_frame = av_frame_alloc();
        if (!resampled_frame)
        {
            if (processed_frame != input_frame)
                av_frame_free(&processed_frame);
            return false;
        }

        // Set up output frame properties
        resampled_frame->format = out.aenc->sample_fmt;
        av_channel_layout_copy(&resampled_frame->ch_layout, &out.aenc->ch_layout);
        resampled_frame->sample_rate = out.aenc->sample_rate;

        // Calculate output samples
        int max_out_samples = swr_get_out_samples(out.swr_ctx, processed_frame->nb_samples);
        resampled_frame->nb_samples = max_out_samples;

        ret = av_frame_get_buffer(resampled_frame, 0);
        if (ret < 0)
        {
            LOG_WARN("Failed to allocate resampled frame buffer");
            av_frame_free(&resampled_frame);
            if (processed_frame != input_frame)
                av_frame_free(&processed_frame);
            return false;
        }

        // Resample
        int samples_out = swr_convert(out.swr_ctx,
                                      resampled_frame->data, resampled_frame->nb_samples,
                                      (const uint8_t **)processed_frame->data, processed_frame->nb_samples);
        if (samples_out < 0)
        {
            LOG_WARN("Failed to resample audio frame");
            av_frame_free(&resampled_frame);
            if (processed_frame != input_frame)
                av_frame_free(&processed_frame);
            return false;
        }

        resampled_frame->nb_samples = samples_out;
        resampled_frame->pts = processed_frame->pts;

        // Clean up intermediate frame
        if (processed_frame != input_frame)
            av_frame_free(&processed_frame);
        processed_frame = resampled_frame;
    }

    // Add samples to FIFO
    if (out.audio_fifo)
    {
        ret = av_audio_fifo_write(out.audio_fifo, (void **)processed_frame->data, processed_frame->nb_samples);
        if (ret < 0)
        {
            LOG_WARN("Failed to write samples to audio FIFO");
            if (processed_frame != input_frame)
                av_frame_free(&processed_frame);
            return false;
        }
    }

    // Clean up processed frame
    if (processed_frame != input_frame)
        av_frame_free(&processed_frame);

    // Try to encode frames while we have enough samples
    while (out.audio_fifo && av_audio_fifo_size(out.audio_fifo) >= out.aenc->frame_size)
    {
        AVFrame *encoder_frame = av_frame_alloc();
        if (!encoder_frame)
        {
            return false;
        }

        encoder_frame->format = out.aenc->sample_fmt;
        av_channel_layout_copy(&encoder_frame->ch_layout, &out.aenc->ch_layout);
        encoder_frame->sample_rate = out.aenc->sample_rate;
        encoder_frame->nb_samples = out.aenc->frame_size;

        ret = av_frame_get_buffer(encoder_frame, 0);
        if (ret < 0)
        {
            av_frame_free(&encoder_frame);
            return false;
        }

        // Read samples from FIFO
        ret = av_audio_fifo_read(out.audio_fifo, (void **)encoder_frame->data, out.aenc->frame_size);
        if (ret < out.aenc->frame_size)
        {
            av_frame_free(&encoder_frame);
            break;
        }

        // Set proper timestamp for the encoder frame
        encoder_frame->pts = out.next_audio_pts;

        // CRITICAL: Advance PTS counter immediately after assignment to prevent duplicate timestamps
        // The encoder may buffer frames (EAGAIN) before producing packets, so if we wait until
        // receiving a packet to increment, multiple encoder frames can end up with the same PTS.
        // This causes duplicate audio timestamps in fMP4 segments, breaking hls.js playback with
        // BUFFER_APPENDING errors showing undefined start/end times.
        int64_t frame_pts = out.next_audio_pts;
        out.next_audio_pts += out.aenc->frame_size;

        // Encode frame
        ret = avcodec_send_frame(out.aenc, encoder_frame);
        av_frame_free(&encoder_frame);

        if (ret < 0)
        {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, sizeof(errbuf), ret);
            LOG_WARN("Failed to send frame to audio encoder: %s", errbuf);
            return false;
        }

        // Get encoded packets
        while (true)
        {
            ret = avcodec_receive_packet(out.aenc, output_packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            {
                break;
            }
            if (ret < 0)
            {
                char errbuf[AV_ERROR_MAX_STRING_SIZE];
                av_make_error_string(errbuf, sizeof(errbuf), ret);
                LOG_WARN("Failed to receive encoded audio packet: %s", errbuf);
                return false;
            }

            // Packet is ready for writing
            output_packet->stream_index = out.astream->index;

            // Ensure timestamps are set properly
            if (output_packet->pts == AV_NOPTS_VALUE)
            {
                // Generate timestamps if encoder didn't set them (use the frame's PTS that we assigned)
                output_packet->pts = frame_pts;
                output_packet->dts = frame_pts;
            }

            av_packet_rescale_ts(output_packet, out.aenc->time_base, out.astream->time_base);

            // FFmpeg 8 compatibility: Trust encoder and muxer for DTS handling
            // - Encoder generates DTS from PTS automatically
            // - Muxer (av_interleaved_write_frame) validates monotonicity and DTS <= PTS
            // - No manual intervention needed
            return true; // Packet ready
        }
    }

    return true; // Successfully processed, but no packet ready yet
}

void init_audio_pts_after_seek(const InputContext &in, OutputContext &out, int64_t global_baseline_pts_us)
{
    if (!out.audioConfig.enabled || !out.aenc)
    {
        return;
    }

    // Initialize audio PTS tracking to align with video timeline
    // Both video and audio use the same baseline (global_baseline_us)
    // Audio should start from 0 (relative to baseline), just like video
    if (in.seek_offset_us > 0 && global_baseline_pts_us != AV_NOPTS_VALUE)
    {
        // Start from 0 to match video (video: first_frame - baseline, audio: start at 0)
        out.next_audio_pts = 0;
        LOG_DEBUG("Audio PTS initialized to 0 (baseline at %.3fs)", global_baseline_pts_us / 1000000.0);
    }
    else
    {
        out.next_audio_pts = 0;
    }

    // Reset audio baseline
    out.a_start_pts = AV_NOPTS_VALUE;
}