#include "ffmpeg_utils.h"
#include "logger.h"
#include "audio_config.h"
#include "config_parser.h"
#include "stream_mapper.h"
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

        // Apply -seek_timestamp behavior (FFmpeg compatibility)
        // When -seek_timestamp is disabled (default), add stream start_time to seek position
        // When -seek_timestamp is enabled, seek to absolute timestamp without adjustment
        if (!options->seekTimestamp && in.fmt->start_time != AV_NOPTS_VALUE)
        {
            seek_target += in.fmt->start_time;
            LOG_DEBUG("seek_timestamp disabled: adjusted seek target by start_time (%lld us)", in.fmt->start_time);
        }

        in.seek_offset_us = seek_target;

        int seek_flags = AVSEEK_FLAG_BACKWARD;

        // Apply -seek2any to allow seeking to non-keyframes at demuxer level
        // AVSEEK_FLAG_ANY allows seeking to any frame type (B/P frames, not just I-frames)
        // Warning: May produce garbled output until next keyframe is decoded
        if (options->seek2any)
        {
            seek_flags |= AVSEEK_FLAG_ANY;
            LOG_DEBUG("seek2any enabled: AVSEEK_FLAG_BACKWARD | AVSEEK_FLAG_ANY (can seek to non-keyframes)");
        }

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

    // Find primary ("best") audio stream for info/logging
    // Note: Decoders are created later based on stream mapping decisions
    in.primary_audio_stream = av_find_best_stream(in.fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (in.primary_audio_stream >= 0)
    {
        AVStream *primary_ast = in.fmt->streams[in.primary_audio_stream];
        LOG_INFO("Primary audio stream: %d (%s, %d Hz, %d channels)",
                 in.primary_audio_stream,
                 avcodec_get_name(primary_ast->codecpar->codec_id),
                 primary_ast->codecpar->sample_rate,
                 primary_ast->codecpar->ch_layout.nb_channels);
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

    // Clean up all audio decoders
    for (auto &pair : in.audio_decoders)
    {
        if (pair.second)
        {
            avcodec_free_context(&pair.second);
        }
    }
    in.audio_decoders.clear();

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
    in.primary_audio_stream = -1;
    in.seek_offset_us = 0;
}

// Open multiple inputs (for multi-input support)
bool open_inputs(const std::vector<std::string> &inPaths, std::vector<InputContext> &inputs, const InputOpenOptions *options)
{
    inputs.clear();
    inputs.resize(inPaths.size());

    for (size_t i = 0; i < inPaths.size(); i++)
    {
        LOG_INFO("Opening input %d: %s", (int)i, inPaths[i].c_str());
        if (!open_input(inPaths[i].c_str(), inputs[i], options))
        {
            LOG_ERROR("Failed to open input %d: %s", (int)i, inPaths[i].c_str());
            // Clean up previously opened inputs
            for (size_t j = 0; j < i; j++)
            {
                close_input(inputs[j]);
            }
            inputs.clear();
            return false;
        }
    }

    LOG_INFO("Successfully opened %d input(s)", (int)inPaths.size());
    return true;
}

// Close multiple inputs
void close_inputs(std::vector<InputContext> &inputs)
{
    for (InputContext &in : inputs)
    {
        close_input(in);
    }
    inputs.clear();
}

bool open_output(const char *outPath, const InputContext &in, OutputContext &out, const std::vector<std::string> &streamMaps, const std::string &outputFormatName)
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
    else if (!outputFormatName.empty())
    {
        // Use explicitly specified format from -f flag
        effectiveFormat = outputFormatName.c_str();
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
        LOG_DEBUG("HLS muxer enabled, setting up options");
        AVDictionary *muxOpts = nullptr;
        if (out.hlsOptions.hlsTime > 0)
        {
            std::string value = std::to_string(out.hlsOptions.hlsTime);
            av_dict_set(&muxOpts, "hls_time", value.c_str(), 0);
            LOG_DEBUG("Set hls_time = %s", value.c_str());
        }
        if (!out.hlsOptions.segmentFilename.empty())
        {
            av_dict_set(&muxOpts, "hls_segment_filename", out.hlsOptions.segmentFilename.c_str(), 0);
            LOG_DEBUG("Set hls_segment_filename = %s", out.hlsOptions.segmentFilename.c_str());
        }
        if (!out.hlsOptions.segmentType.empty())
        {
            av_dict_set(&muxOpts, "hls_segment_type", out.hlsOptions.segmentType.c_str(), 0);
            LOG_DEBUG("Set hls_segment_type = %s", out.hlsOptions.segmentType.c_str());
        }
        if (!out.hlsOptions.initFilename.empty())
        {
            av_dict_set(&muxOpts, "hls_fmp4_init_filename", out.hlsOptions.initFilename.c_str(), 0);
            LOG_DEBUG("Set hls_fmp4_init_filename = %s", out.hlsOptions.initFilename.c_str());
        }
        if (out.hlsOptions.startNumber >= 0)
        {
            std::string value = std::to_string(out.hlsOptions.startNumber);
            av_dict_set(&muxOpts, "start_number", value.c_str(), 0);
            LOG_DEBUG("Set start_number = %s", value.c_str());
        }
        if (!out.hlsOptions.playlistType.empty())
        {
            av_dict_set(&muxOpts, "hls_playlist_type", out.hlsOptions.playlistType.c_str(), 0);
            LOG_DEBUG("Set hls_playlist_type = %s", out.hlsOptions.playlistType.c_str());
        }
        if (out.hlsOptions.listSize >= 0)
        {
            std::string value = std::to_string(out.hlsOptions.listSize);
            av_dict_set(&muxOpts, "hls_list_size", value.c_str(), 0);
            LOG_DEBUG("Set hls_list_size = %s", value.c_str());
        }
        if (out.hlsOptions.maxDelay >= 0)
        {
            std::string value = std::to_string(out.hlsOptions.maxDelay);
            av_dict_set(&muxOpts, "max_delay", value.c_str(), 0);
            LOG_DEBUG("Set max_delay = %s", value.c_str());
        }

        // Set hls_flags: user-specified flags take precedence over automatic flags
        std::string hlsFlags;
        if (!out.hlsOptions.customFlags.empty())
        {
            // User explicitly specified hls_flags via -hls_flags option
            hlsFlags = out.hlsOptions.customFlags;
            LOG_DEBUG("Using user-specified hls_flags = %s", hlsFlags.c_str());
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
            LOG_DEBUG("Set hls_flags = %s", hlsFlags.c_str());
        }

        // Apply user-specified segment options (e.g., movflags=+frag_discont for fMP4)
        // This is required for proper hls.js playback with fMP4 segments
        if (!out.hlsOptions.segmentOptions.empty())
        {
            av_dict_set(&muxOpts, "hls_segment_options", out.hlsOptions.segmentOptions.c_str(), 0);
            LOG_INFO("HLS: Applied segment options: %s (passed to individual segment muxers)", out.hlsOptions.segmentOptions.c_str());
        }
        else if (out.hlsOptions.segmentType == "fmp4")
        {
            LOG_INFO("HLS: No custom segment options specified. Using default movflags (+frag_discont+frag_keyframe) from main muxer.");
        }

        // Debug: Print all HLS options that will be passed to the muxer
        LOG_DEBUG("Final HLS muxer options:");
        AVDictionaryEntry *entry = nullptr;
        while ((entry = av_dict_get(muxOpts, "", entry, AV_DICT_IGNORE_SUFFIX)))
        {
            LOG_DEBUG("  %s = %s", entry->key, entry->value);
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

    // Mark audio streams for processing if re-encoding is enabled
    if (out.audioConfig.enabled && !out.audioConfig.codec.empty() && out.audioConfig.codec != "copy")
    {
        if (out.audioConfig.applyToAllAudioStreams)
        {
            // -codec:a applies to ALL audio streams
            int audio_count = 0;
            for (unsigned int i = 0; i < in.fmt->nb_streams; ++i)
            {
                if (in.fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO &&
                    out.stream_decisions[i] != StreamMapDecision::EXCLUDE)
                {
                    out.stream_decisions[i] = StreamMapDecision::PROCESS_AUDIO;
                    audio_count++;
                }
            }
            LOG_DEBUG("Marking %d audio streams for re-encoding (-codec:a)", audio_count);
        }
        else if (in.primary_audio_stream >= 0)
        {
            // -codec:a:0 applies only to the first/best audio stream
            out.stream_decisions[in.primary_audio_stream] = StreamMapDecision::PROCESS_AUDIO;
            LOG_DEBUG("Marking input audio stream %d for re-encoding (-codec:a:0, other audio streams will be copied)", in.primary_audio_stream);
        }
    }

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

    // Clean up all audio encoders
    for (auto &pair : out.audio_encoders)
    {
        AudioEncoderContext &ctx = pair.second;

        if (ctx.encoder)
        {
            avcodec_free_context(&ctx.encoder);
        }

        if (ctx.filter_graph)
        {
            avfilter_graph_free(&ctx.filter_graph);
            ctx.buffersrc = nullptr;
            ctx.buffersink = nullptr;
        }

        if (ctx.resampler)
        {
            swr_free(&ctx.resampler);
        }

        if (ctx.fifo)
        {
            av_audio_fifo_free(ctx.fifo);
        }
    }
    out.audio_encoders.clear();

    if (out.fmt)
    {
        if (!(out.fmt->oformat->flags & AVFMT_NOFILE) && out.fmt->pb)
            avio_closep(&out.fmt->pb);
        avformat_free_context(out.fmt);
        out.fmt = nullptr;
    }

    out.vstream = nullptr;
    out.stream_decisions.clear();
    out.input_to_output_map.clear();
}

// Apply metadata and chapter mapping settings (Jellyfin compatibility)
void apply_metadata_chapter_settings(OutputContext &out, const PipelineConfig &cfg, const InputContext &in)
{
    if (!out.fmt)
    {
        LOG_WARN("apply_metadata_chapter_settings: output format context not initialized");
        return;
    }

    // Handle -map_metadata flag
    if (cfg.hasMapMetadata)
    {
        if (cfg.mapMetadata == -1)
        {
            // Jellyfin case: explicitly disable metadata copying
            if (out.fmt->metadata)
            {
                av_dict_free(&out.fmt->metadata);
                out.fmt->metadata = nullptr;
            }
            LOG_DEBUG("Metadata copying disabled via -map_metadata -1 (Jellyfin mode)");
        }
        // Note: Positive indices would require multi-input support
        // For now, Jellyfin only uses -1, so we only implement disable
    }

    // Handle -map_chapters flag
    if (cfg.hasMapChapters)
    {
        if (cfg.mapChapters == -1)
        {
            // Jellyfin case: explicitly disable chapter copying
            if (out.fmt->chapters)
            {
                for (unsigned i = 0; i < out.fmt->nb_chapters; i++)
                {
                    if (out.fmt->chapters[i])
                    {
                        if (out.fmt->chapters[i]->metadata)
                        {
                            av_dict_free(&out.fmt->chapters[i]->metadata);
                        }
                        av_free(out.fmt->chapters[i]);
                    }
                }
                av_free(out.fmt->chapters);
                out.fmt->chapters = nullptr;
                out.fmt->nb_chapters = 0;
            }
            LOG_DEBUG("Chapter copying disabled via -map_chapters -1 (Jellyfin mode)");
        }
        // Note: Positive indices would require multi-input support
    }
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
// Helper: Check if a stream matches a StreamMapSpec (now uses unified stream_mapper.cpp implementation)
static bool stream_matches_spec_simple(const AVStream *stream, int stream_index, const StreamMapSpec &spec)
{
    // Check metadata filter
    if (spec.has_metadata_filter)
    {
        AVDictionaryEntry *tag = av_dict_get(stream->metadata, spec.metadata_key.c_str(), nullptr, 0);
        if (!tag || spec.metadata_value != tag->value)
        {
            return false;
        }
    }

    // Check specific stream index
    if (spec.stream_index >= 0 && spec.stream_index != stream_index)
    {
        return false;
    }

    // Check media type filter
    if (spec.stream_type != AVMEDIA_TYPE_UNKNOWN)
    {
        if (stream->codecpar->codec_type != spec.stream_type)
        {
            return false;
        }
    }

    return true;
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

    // Process all -map directives in order (now using unified stream_mapper.cpp parser)
    for (const auto &mapping : mappings)
    {
        if (mapping.empty())
            continue;

        // Parse using unified parser from stream_mapper.cpp
        StreamMapSpec spec;
        if (!parse_map_spec(mapping, spec))
        {
            LOG_WARN("Failed to parse -map argument: %s", mapping.c_str());
            continue;
        }

        // Only support input file 0 for now
        if (spec.input_file_index != 0)
        {
            LOG_WARN("Only input file 0 is supported: %s", mapping.c_str());
            continue;
        }

        // Apply the mapping
        int match_count = 0; // Track how many streams matched (for stream_type_index)

        for (unsigned int i = 0; i < nb_streams; ++i)
        {
            AVStream *stream = in.fmt->streams[i];

            // Check if this stream matches the spec (without index filtering)
            bool matches = stream_matches_spec_simple(stream, i, spec);

            if (matches)
            {
                // If stream_type_index is specified (e.g., "m:language:eng:0" or "a:1"),
                // only select the Nth matching stream (0-indexed)
                bool should_include_this_stream = true;
                if (spec.stream_type_index >= 0)
                {
                    // Only include this stream if it's the Nth match
                    if (match_count == spec.stream_type_index)
                    {
                        should_include_this_stream = true;
                        match_count++;
                    }
                    else
                    {
                        should_include_this_stream = false;
                        match_count++;
                    }
                }

                if (should_include_this_stream)
                {
                    if (spec.is_negative)
                    {
                        // Exclusion mapping (e.g., "-map -0:s")
                        out.stream_decisions[i] = StreamMapDecision::EXCLUDE;
                    }
                    else
                    {
                        // Inclusion mapping - decide between COPY and PROCESS based on stream type and config
                        AVMediaType codec_type = stream->codecpar->codec_type;
                        if (codec_type == AVMEDIA_TYPE_AUDIO && audio_needs_processing)
                        {
                            // Check if this audio stream should be re-encoded
                            bool should_process = out.audioConfig.applyToAllAudioStreams ||
                                                (in.primary_audio_stream >= 0 && (int)i == in.primary_audio_stream);
                            out.stream_decisions[i] = should_process ? StreamMapDecision::PROCESS_AUDIO : StreamMapDecision::COPY;
                        }
                        else
                        {
                            out.stream_decisions[i] = StreamMapDecision::COPY;
                        }
                    }

                    // If stream_type_index was specified and we found it, stop looking
                    if (spec.stream_type_index >= 0)
                    {
                        break;
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

// Setup multiple audio encoders for all streams marked PROCESS_AUDIO
bool setup_audio_encoders(const InputContext &in, OutputContext &out)
{
    if (!out.audioConfig.enabled)
    {
        return true; // No audio processing needed
    }

    if (!in.fmt)
    {
        LOG_WARN("Invalid input format context for audio setup");
        return false;
    }

    // Find all audio streams marked for processing
    std::vector<int> streams_to_encode;
    for (size_t i = 0; i < out.stream_decisions.size(); ++i)
    {
        if (out.stream_decisions[i] == StreamMapDecision::PROCESS_AUDIO)
        {
            streams_to_encode.push_back(i);
        }
    }

    if (streams_to_encode.empty())
    {
        LOG_DEBUG("No audio streams marked for re-encoding");
        return true;
    }

    LOG_DEBUG("Setting up encoders for %d audio streams", (int)streams_to_encode.size());

    // Set up encoder for each audio stream
    for (int stream_idx : streams_to_encode)
    {
        AVStream *input_stream = in.fmt->streams[stream_idx];
        if (input_stream->codecpar->codec_type != AVMEDIA_TYPE_AUDIO)
        {
            LOG_WARN("Stream %d is not an audio stream", stream_idx);
            continue;
        }

        AudioEncoderContext &ctx = out.audio_encoders[stream_idx];
        ctx.input_stream_index = stream_idx;

        // Set up codec
        std::string codec = out.audioConfig.codec.empty() ? "aac" : out.audioConfig.codec;
        const AVCodec *audio_encoder = avcodec_find_encoder_by_name(codec.c_str());
        if (!audio_encoder)
        {
            audio_encoder = avcodec_find_encoder(AV_CODEC_ID_AAC);
            codec = "aac";
        }
        if (!audio_encoder)
        {
            LOG_WARN("Failed to find audio encoder for stream %d", stream_idx);
            continue;
        }

        ctx.encoder = avcodec_alloc_context3(audio_encoder);
        if (!ctx.encoder)
        {
            LOG_WARN("Failed to allocate encoder for stream %d", stream_idx);
            continue;
        }

        // Configure encoder
        ctx.encoder->codec_id = audio_encoder->id;
        ctx.encoder->codec_type = AVMEDIA_TYPE_AUDIO;
        ctx.encoder->sample_rate = (out.audioConfig.sampleRate > 0) ? out.audioConfig.sampleRate : input_stream->codecpar->sample_rate;

        // Channel layout
        if (out.audioConfig.channels > 0)
        {
            av_channel_layout_default(&ctx.encoder->ch_layout, out.audioConfig.channels);
        }
        else
        {
            av_channel_layout_copy(&ctx.encoder->ch_layout, &input_stream->codecpar->ch_layout);
        }

        ctx.encoder->bit_rate = (out.audioConfig.bitrate > 0) ? out.audioConfig.bitrate : 128000;
        ctx.encoder->time_base = {1, ctx.encoder->sample_rate};

        // Sample format
        if (audio_encoder->sample_fmts)
        {
            ctx.encoder->sample_fmt = audio_encoder->sample_fmts[0];
        }
        else
        {
            ctx.encoder->sample_fmt = AV_SAMPLE_FMT_FLTP;
        }

        // Frame size
        if (audio_encoder->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)
        {
            ctx.encoder->frame_size = 0;
        }
        else
        {
            ctx.encoder->frame_size = 1024;
        }

        // Create output stream
        ctx.output_stream = avformat_new_stream(out.fmt, nullptr);
        if (!ctx.output_stream)
        {
            LOG_WARN("Failed to create output stream for audio %d", stream_idx);
            avcodec_free_context(&ctx.encoder);
            continue;
        }

        // Open encoder
        int ret = avcodec_open2(ctx.encoder, audio_encoder, nullptr);
        if (ret < 0)
        {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, sizeof(errbuf), ret);
            LOG_WARN("Failed to open encoder for stream %d: %s", stream_idx, errbuf);
            avcodec_free_context(&ctx.encoder);
            continue;
        }

        // Copy encoder parameters to stream
        avcodec_parameters_from_context(ctx.output_stream->codecpar, ctx.encoder);
        ctx.output_stream->time_base = {1, ctx.encoder->sample_rate};

        // Setup resampler
        ctx.resampler = swr_alloc();
        if (ctx.resampler)
        {
            av_opt_set_chlayout(ctx.resampler, "in_chlayout", &input_stream->codecpar->ch_layout, 0);
            av_opt_set_int(ctx.resampler, "in_sample_rate", input_stream->codecpar->sample_rate, 0);
            av_opt_set_sample_fmt(ctx.resampler, "in_sample_fmt", (AVSampleFormat)input_stream->codecpar->format, 0);

            av_opt_set_chlayout(ctx.resampler, "out_chlayout", &ctx.encoder->ch_layout, 0);
            av_opt_set_int(ctx.resampler, "out_sample_rate", ctx.encoder->sample_rate, 0);
            av_opt_set_sample_fmt(ctx.resampler, "out_sample_fmt", ctx.encoder->sample_fmt, 0);

            if (swr_init(ctx.resampler) < 0)
            {
                swr_free(&ctx.resampler);
            }
        }

        // Create FIFO (use safe effective frame size for variable-frame-size encoders)
        int effective_frame_size = (ctx.encoder->frame_size > 0) ? ctx.encoder->frame_size : 1024;
        ctx.fifo = av_audio_fifo_alloc(ctx.encoder->sample_fmt, ctx.encoder->ch_layout.nb_channels, effective_frame_size * 2);

        LOG_DEBUG("Audio encoder %d setup: input stream %d -> output stream %d (%s, %d Hz, %d channels)",
                  (int)out.audio_encoders.size(), stream_idx, ctx.output_stream->index,
                  codec.c_str(), ctx.encoder->sample_rate, ctx.encoder->ch_layout.nb_channels);
    }

    return !out.audio_encoders.empty();
}

// Setup audio decoders for all streams marked for re-encoding
bool setup_audio_decoders(InputContext &in, const OutputContext &out)
{
    if (!out.audioConfig.enabled)
    {
        return true; // No audio processing needed
    }

    // Find all streams marked for audio processing
    std::vector<int> streams_to_decode;
    for (size_t i = 0; i < out.stream_decisions.size(); ++i)
    {
        if (out.stream_decisions[i] == StreamMapDecision::PROCESS_AUDIO)
        {
            streams_to_decode.push_back(i);
        }
    }

    if (streams_to_decode.empty())
    {
        LOG_DEBUG("No audio streams marked for re-encoding");
        return true;
    }

    LOG_DEBUG("Setting up decoders for %zu audio streams", streams_to_decode.size());

    // Create decoder for each stream
    int decoders_created = 0;
    for (int stream_idx : streams_to_decode)
    {
        if (stream_idx < 0 || stream_idx >= (int)in.fmt->nb_streams)
        {
            LOG_WARN("Invalid stream index %d", stream_idx);
            continue;
        }

        AVStream *stream = in.fmt->streams[stream_idx];

        // Find decoder
        const AVCodec *decoder = avcodec_find_decoder(stream->codecpar->codec_id);
        if (!decoder)
        {
            LOG_WARN("No decoder found for stream %d codec %s",
                     stream_idx, avcodec_get_name(stream->codecpar->codec_id));
            continue;
        }

        // Allocate decoder context
        AVCodecContext *dec_ctx = avcodec_alloc_context3(decoder);
        if (!dec_ctx)
        {
            LOG_WARN("Failed to allocate decoder context for stream %d", stream_idx);
            continue;
        }

        // Copy parameters
        int ret = avcodec_parameters_to_context(dec_ctx, stream->codecpar);
        if (ret < 0)
        {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, sizeof(errbuf), ret);
            avcodec_free_context(&dec_ctx);
            LOG_WARN("Failed to copy decoder parameters for stream %d: %s", stream_idx, errbuf);
            continue;
        }

        dec_ctx->pkt_timebase = stream->time_base;

        // Open decoder
        ret = avcodec_open2(dec_ctx, decoder, nullptr);
        if (ret < 0)
        {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, sizeof(errbuf), ret);
            avcodec_free_context(&dec_ctx);
            LOG_WARN("Failed to open decoder for stream %d: %s", stream_idx, errbuf);
            continue;
        }

        // Store in map
        in.audio_decoders[stream_idx] = dec_ctx;
        decoders_created++;

        LOG_DEBUG("Audio decoder setup: stream %d (%s, %d Hz, %d channels)",
                  stream_idx,
                  avcodec_get_name(stream->codecpar->codec_id),
                  stream->codecpar->sample_rate,
                  stream->codecpar->ch_layout.nb_channels);
    }

    if (decoders_created == 0)
    {
        LOG_WARN("Failed to create any audio decoders");
        return false;
    }

    LOG_INFO("Created %d audio decoder(s) for re-encoding", decoders_created);
    return true;
}

bool process_audio_frame_multi(AVFrame *input_frame, int input_stream_index, OutputContext &out)
{
    if (!input_frame || input_stream_index < 0)
    {
        return false;
    }

    // Find the encoder context for this input stream
    auto it = out.audio_encoders.find(input_stream_index);
    if (it == out.audio_encoders.end())
    {
        LOG_WARN("No encoder found for input stream %d", input_stream_index);
        return false;
    }

    AudioEncoderContext &enc_ctx = it->second;
    if (!enc_ctx.encoder || !enc_ctx.output_stream)
    {
        LOG_WARN("Encoder context incomplete for input stream %d", input_stream_index);
        return false;
    }

    int ret;
    AVFrame *processed_frame = input_frame;

    // Handle first audio frame to establish baseline
    if (enc_ctx.start_pts == AV_NOPTS_VALUE && input_frame->pts != AV_NOPTS_VALUE)
    {
        enc_ctx.start_pts = input_frame->pts;

        // COPYTS mode: Preserve input timestamps (per-stream independence)
        if (out.audioConfig.copyts)
        {
            // Rescale input PTS to encoder timebase while preserving original value
            int64_t rescaled_pts = av_rescale_q(input_frame->pts,
                                                input_frame->time_base,
                                                enc_ctx.encoder->time_base);

            enc_ctx.accumulated_samples = rescaled_pts;
            LOG_DEBUG("COPYTS: Initialized audio stream %d PTS: %lld (input) -> %lld (output timebase), preserving original timestamps",
                      input_stream_index, input_frame->pts, rescaled_pts);
        }
        else if (out.hlsOptions.enabled && !out.audioConfig.copyts)
        {
            // For HLS, force audio to start at 0
            enc_ctx.accumulated_samples = 0;
            LOG_DEBUG("HLS: Initialized audio stream %d PTS to 0", input_stream_index);
        }
    }

    // Apply audio filter if configured for this stream
    if (enc_ctx.filter_graph && enc_ctx.buffersrc && enc_ctx.buffersink)
    {
        ret = av_buffersrc_add_frame_flags(enc_ctx.buffersrc, input_frame, AV_BUFFERSRC_FLAG_KEEP_REF);
        if (ret < 0)
        {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, sizeof(errbuf), ret);
            LOG_WARN("Failed to send frame to audio filter (stream %d): %s", input_stream_index, errbuf);
            return false;
        }

        AVFrame *filtered_frame = av_frame_alloc();
        if (!filtered_frame)
        {
            LOG_WARN("Failed to allocate filtered frame");
            return false;
        }

        ret = av_buffersink_get_frame_flags(enc_ctx.buffersink, filtered_frame, 0);
        if (ret >= 0)
        {
            processed_frame = filtered_frame;
        }
        else if (ret == AVERROR(EAGAIN))
        {
            av_frame_free(&filtered_frame);
            return true;
        }
        else if (ret == AVERROR_EOF)
        {
            av_frame_free(&filtered_frame);
            return false;
        }
        else
        {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, sizeof(errbuf), ret);
            LOG_WARN("Failed to get frame from audio filter (stream %d): %s", input_stream_index, errbuf);
            av_frame_free(&filtered_frame);
            return false;
        }
    }

    // Resample frame to encoder format if resampler is available
    AVFrame *resampled_frame = nullptr;
    if (enc_ctx.resampler)
    {
        resampled_frame = av_frame_alloc();
        if (!resampled_frame)
        {
            if (processed_frame != input_frame)
                av_frame_free(&processed_frame);
            return false;
        }

        resampled_frame->format = enc_ctx.encoder->sample_fmt;
        av_channel_layout_copy(&resampled_frame->ch_layout, &enc_ctx.encoder->ch_layout);
        resampled_frame->sample_rate = enc_ctx.encoder->sample_rate;

        int max_out_samples = swr_get_out_samples(enc_ctx.resampler, processed_frame->nb_samples);
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

        int out_samples = swr_convert(enc_ctx.resampler,
                                      resampled_frame->data, max_out_samples,
                                      (const uint8_t **)processed_frame->data, processed_frame->nb_samples);

        if (out_samples < 0)
        {
            LOG_WARN("Failed to resample audio (stream %d)", input_stream_index);
            av_frame_free(&resampled_frame);
            if (processed_frame != input_frame)
                av_frame_free(&processed_frame);
            return false;
        }

        resampled_frame->nb_samples = out_samples;
        if (processed_frame != input_frame)
            av_frame_free(&processed_frame);
        processed_frame = resampled_frame;
    }

    // Validate format if no resampler (format must match encoder)
    if (!enc_ctx.resampler)
    {
        if (processed_frame->format != enc_ctx.encoder->sample_fmt ||
            processed_frame->sample_rate != enc_ctx.encoder->sample_rate ||
            av_channel_layout_compare(&processed_frame->ch_layout, &enc_ctx.encoder->ch_layout) != 0)
        {
            LOG_WARN("Audio format mismatch without resampler (stream %d): skipping frame", input_stream_index);
            if (processed_frame != input_frame)
                av_frame_free(&processed_frame);
            return false;
        }
    }

    // Add samples to FIFO
    if (!enc_ctx.fifo)
    {
        LOG_WARN("FIFO not initialized for stream %d", input_stream_index);
        if (processed_frame != input_frame)
            av_frame_free(&processed_frame);
        return false;
    }

    // Cache nb_samples before freeing (avoid use-after-free)
    int input_nb_samples = processed_frame->nb_samples;
    ret = av_audio_fifo_write(enc_ctx.fifo, (void **)processed_frame->data, processed_frame->nb_samples);
    if (processed_frame != input_frame)
        av_frame_free(&processed_frame);

    if (ret < input_nb_samples)
    {
        LOG_WARN("Failed to write all samples to FIFO (stream %d): wrote %d/%d", input_stream_index, ret, input_nb_samples);
        return false;
    }

    // Encode frames while we have enough samples (use safe effective frame size)
    int effective_frame_size = (enc_ctx.encoder->frame_size > 0) ? enc_ctx.encoder->frame_size : 1024;
    while (av_audio_fifo_size(enc_ctx.fifo) >= effective_frame_size)
    {
        AVFrame *encoder_frame = av_frame_alloc();
        if (!encoder_frame)
        {
            LOG_WARN("Failed to allocate encoder frame");
            return false;
        }

        encoder_frame->nb_samples = effective_frame_size;
        encoder_frame->format = enc_ctx.encoder->sample_fmt;
        av_channel_layout_copy(&encoder_frame->ch_layout, &enc_ctx.encoder->ch_layout);
        encoder_frame->sample_rate = enc_ctx.encoder->sample_rate;

        ret = av_frame_get_buffer(encoder_frame, 0);
        if (ret < 0)
        {
            LOG_WARN("Failed to allocate encoder frame buffer");
            av_frame_free(&encoder_frame);
            return false;
        }

        ret = av_audio_fifo_read(enc_ctx.fifo, (void **)encoder_frame->data, effective_frame_size);
        if (ret < effective_frame_size)
        {
            LOG_WARN("Failed to read enough samples from FIFO (stream %d): read %d/%d", input_stream_index, ret, effective_frame_size);
            av_frame_free(&encoder_frame);
            return false;
        }

        // Set proper timestamp using accumulated sample count
        encoder_frame->pts = enc_ctx.accumulated_samples;

        // Advance sample counter to prevent duplicate timestamps
        int64_t frame_pts = enc_ctx.accumulated_samples;
        enc_ctx.accumulated_samples += enc_ctx.encoder->frame_size;

        // Encode frame
        ret = avcodec_send_frame(enc_ctx.encoder, encoder_frame);
        av_frame_free(&encoder_frame);

        if (ret < 0)
        {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, sizeof(errbuf), ret);
            LOG_WARN("Failed to send frame to audio encoder (stream %d): %s", input_stream_index, errbuf);
            return false;
        }

        // Get encoded packets and write them
        while (true)
        {
            AVPacket *output_packet = av_packet_alloc();
            if (!output_packet)
            {
                LOG_WARN("Failed to allocate output packet");
                return false;
            }

            ret = avcodec_receive_packet(enc_ctx.encoder, output_packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            {
                av_packet_free(&output_packet);
                break;
            }
            if (ret < 0)
            {
                char errbuf[AV_ERROR_MAX_STRING_SIZE];
                av_make_error_string(errbuf, sizeof(errbuf), ret);
                LOG_WARN("Failed to receive encoded audio packet (stream %d): %s", input_stream_index, errbuf);
                av_packet_free(&output_packet);
                return false;
            }

            // Packet is ready for writing
            output_packet->stream_index = enc_ctx.output_stream->index;

            // Ensure timestamps are set properly
            if (output_packet->pts == AV_NOPTS_VALUE)
            {
                output_packet->pts = frame_pts;
            }

            // Rescale timestamps from encoder to stream timebase
            // Note: FFmpeg automatically sets dts=pts for audio in avcodec_receive_packet()
            av_packet_rescale_ts(output_packet, enc_ctx.encoder->time_base, enc_ctx.output_stream->time_base);

            // Write packet to muxer
            ret = av_interleaved_write_frame(out.fmt, output_packet);
            av_packet_free(&output_packet);

            if (ret < 0)
            {
                char errbuf[AV_ERROR_MAX_STRING_SIZE];
                av_make_error_string(errbuf, sizeof(errbuf), ret);
                LOG_WARN("Failed to write audio packet (stream %d): %s", input_stream_index, errbuf);
                return false;
            }
        }
    }

    return true;
}