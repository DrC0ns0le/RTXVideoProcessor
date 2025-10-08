#include "input_config.h"
#include "logger.h"
#include "audio_config.h"

#include <cstring>

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/rational.h>
}

// Check if input is a network URL
bool is_network_input(const char* input)
{
    if (!input) return false;
    return (std::strncmp(input, "http://", 7) == 0 ||
            std::strncmp(input, "https://", 8) == 0 ||
            std::strncmp(input, "rtmp://", 7) == 0 ||
            std::strncmp(input, "rtsp://", 7) == 0 ||
            std::strncmp(input, "tcp://", 6) == 0 ||
            std::strncmp(input, "udp://", 6) == 0);
}

// Configure input HDR detection and reopen with P010 if needed
bool configure_input_hdr_detection(PipelineConfig& cfg, InputContext& in)
{
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
            cfg.targetBitrateMultiplier = cfg.targetBitrateMultiplier * 0.8;
        }

        // Reopen input with P010 preference for HDR content
        close_input(in);
        InputOpenOptions inputOpts;
        inputOpts.fflags = cfg.fflags;
        inputOpts.preferP010ForHDR = true;
        inputOpts.seekTime = cfg.seekTime;
        inputOpts.noAccurateSeek = cfg.noAccurateSeek;
        inputOpts.seek2any = cfg.seek2any;
        inputOpts.seekTimestamp = cfg.seekTimestamp;
        inputOpts.enableErrorConcealment = !cfg.ffCompatible;
        inputOpts.flushOnSeek = false;
        open_input(cfg.inputPath, in, &inputOpts);
        LOG_INFO("Configured decoder for P010 output to preserve full 10-bit HDR pipeline");
    }

    return inputIsHDR;
}

// Auto-disable VSR for high-resolution inputs
void configure_vsr_auto_disable(PipelineConfig& cfg, const InputContext& in)
{
    if (cfg.rtxCfg.enableVSR) {
        bool ge1080p = (in.vdec->width > 2560 && in.vdec->height > 1440) ||
                       (in.vdec->width > 1440 && in.vdec->height > 2560);
        if (ge1080p) {
            LOG_INFO("Input resolution is %dx%d (>=1080p). Disabling VSR.", in.vdec->width, in.vdec->height);
            cfg.rtxCfg.enableVSR = false;
            cfg.targetBitrateMultiplier = cfg.targetBitrateMultiplier / 2.0;
        }
    }
}

// Configure audio processing
void configure_audio_processing(PipelineConfig& cfg, InputContext& in, OutputContext& out)
{
    if (cfg.ffCompatible) {
        LOG_DEBUG("Compatibility mode enabled, configuring audio...\n");

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
            // Stream mappings are now applied in open_output(), no need to call here

            // Skip encoder setup for copy mode (audio will be copied directly)
            if (out.audioConfig.codec == "copy") {
                LOG_DEBUG("Audio copy mode enabled, skipping encoder setup\n");
            } else {
                LOG_DEBUG("Setting up audio encoder...\n");
                if (!setup_audio_encoder(in, out)) {
                    LOG_WARN("Failed to setup audio encoder, disabling audio processing");
                    out.audioConfig.enabled = false;
                } else {
                    LOG_DEBUG("Audio encoder setup complete\n");
                    LOG_DEBUG("Setting up audio filter...\n");
                    if (!setup_audio_filter(in, out)) {
                        LOG_WARN("Failed to setup audio filter, continuing without filtering");
                    } else {
                        LOG_DEBUG("Audio filter setup complete\n");
                    }
                }
            }
        }
    }
    LOG_DEBUG("Audio configuration complete, proceeding...\n");
}

// Setup progress tracking
int64_t setup_progress_tracking(const InputContext& in, const AVRational& fr)
{
    int64_t total_frames = 0;
    if (in.vst->nb_frames > 0) {
        total_frames = in.vst->nb_frames;
    } else {
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
