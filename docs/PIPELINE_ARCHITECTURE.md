# RTXVideoProcessor Pipeline Architecture

Complete end-to-end documentation of the video processing pipeline with code logic paths, data flow, and decision points.

---

## Table of Contents

1. [Overview](#overview)
2. [Terminology Clarification](#terminology-clarification)
3. [Pipeline Phases](#pipeline-phases)
4. [Phase 0: Mode Selection](#phase-0-mode-selection)
5. [Phase 1: Initialization](#phase-1-initialization)
6. [Phase 2: Input Configuration](#phase-2-input-configuration)
7. [Phase 3: Output Configuration](#phase-3-output-configuration)
8. [Phase 4: Processing Loop](#phase-4-processing-loop)
9. [Phase 5: Cleanup](#phase-5-cleanup)
10. [Data Flow Diagrams](#data-flow-diagrams)
11. [Error Handling](#error-handling)

---

## Overview

RTXVideoProcessor operates in **three distinct modes**:

1. **FFmpeg Passthrough Mode**: Delegates to real FFmpeg binary (for unsupported operations)
2. **Default Mode**: Simplified RTX enhancement mode for regular users (applies RTX processing without FFmpeg complexity)
3. **FFmpeg-Compatible Mode**: Drop-in FFmpeg replacement with RTX processing + strict FFmpeg behavior

### Core Capabilities
- **Hardware decoding**: NVDEC (CUDA)
- **GPU processing**: RTX Video Super Resolution, TrueHDR (Default & FFmpeg-Compatible modes)
- **Hardware encoding**: NVENC (HEVC)
- **Audio processing**: Re-encoding or passthrough
- **Stream mapping**: Full FFmpeg `-map` syntax support
- **HLS output**: Segmented streaming with fMP4 or MPEGTS

### Architecture Philosophy

1. **FFmpeg Drop-in Replacement**: Can be renamed to `ffmpeg.exe` and used as direct replacement
2. **Single Source of Truth**: Each decision is made once and stored
3. **Separation of Concerns**: Config → Setup → Process → Cleanup
4. **Smart Mode Selection**: Automatically delegates to real FFmpeg for unsupported operations
5. **Error Resilience**: Graceful degradation with detailed logging

---

## Terminology Clarification

This document uses two levels of "modes" that can be confusing:

### Level 1: Operating Modes (Top-level)
**What it controls**: How the entire application runs

1. **FFmpeg Passthrough Mode**: Completely delegates to real `ffmpeg.exe` binary
   - Trigger: Binary named "ffmpeg.exe" AND (unsupported format OR video exclusion `-map -0:v?`)
   - Supported formats: `.mp4`, `.mkv`, `.m3u8` (input/output), HLS muxer, MP4-to-pipe
   - No RTX processing, no internal logic - pure delegation

2. **Simple Mode (Default)**: Simplified RTX enhancement mode for regular users
   - **Auto-detected**: No `-i` flag present (expects positional args)
   - Syntax: `RTXVideoProcessor.exe input.mp4 output.mp4 [options]`
   - Simplified syntax without FFmpeg's complex flags
   - RTX processing enabled by default (use `--no-vsr` or `--no-thdr` to disable)
   - Auto-fixes timestamps, enables error concealment

3. **FFmpeg-Compatible Mode**: Drop-in FFmpeg replacement with strict behavior
   - **Auto-detected**: `-i` flag present in arguments
   - Typical syntax: `RTXVideoProcessor.exe -i input.mp4 [options] output.mp4` (FFmpeg-style)
   - Full FFmpeg syntax support with RTX processing capability
   - RTX processing enabled by default (use `--no-vsr` or `--no-thdr` to disable)
   - No auto-corrections, strict FFmpeg compliance

### RTX Processing Control

**What it controls**: Whether RTX features (VSR, TrueHDR) are applied to video frames

RTX processing is **independent of operating mode** and controlled by command-line flags:

- **`--no-vsr`**: Disables Video Super Resolution (enabled by default)
- **`--no-thdr`**: Disables TrueHDR tone mapping (enabled by default)
- **Auto-disable**: VSR automatically disables for inputs >1440p (>2560x1440 or >1440x2560) (src/input_config.cpp:66-80)
- **Auto-disable**: THDR automatically disables for HDR inputs to preserve HDR metadata (src/input_config.cpp:27-64)

**Both Simple Mode and FFmpeg-Compatible Mode support RTX processing**. The mode only affects argument parsing syntax, not RTX capabilities.

### Quick Reference

| Scenario | Operating Mode | RTX Processing |
|----------|---------------|----------------|
| `ffmpeg.exe -i input.avi -o output.mov` (unsupported formats) | FFmpeg Passthrough | N/A (delegated to real ffmpeg) |
| `ffmpeg.exe -i input.mp4 -map -0:v? -o output.mp4` (video exclusion) | FFmpeg Passthrough | N/A (delegated to real ffmpeg) |
| `RTXVideoProcessor.exe input.mp4 output.mp4` | Simple Mode | VSR + THDR enabled (default) |
| `RTXVideoProcessor.exe input.mp4 output.mp4 --no-vsr` | Simple Mode | THDR only |
| `RTXVideoProcessor.exe -i input.mp4 -o output.mp4` | FFmpeg-Compatible | VSR + THDR enabled (default) |
| `RTXVideoProcessor.exe -i input.mp4 --no-thdr -o output.mp4` | FFmpeg-Compatible | VSR only |

---

## Pipeline Phases

```
                        ENTRY POINT: main()
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODE SELECTION (passthrough_required)               │
│  Check: Binary name, input/output formats, special features     │
└───────┬─────────────────────────────────────────┬───────────────┘
        │                                         │
   PASSTHROUGH?                                  NO
        │                                         │
       YES                                        ▼
        │                      ┌─────────────────────────────────┐
        ▼                      │     PHASE 1: INITIALIZATION     │
┌──────────────┐               │  Parse CLI → Validate Config    │
│   FFmpeg     │               └────────────┬────────────────────┘
│  Passthrough │                            │
│    Mode      │                            ▼
│              │               ┌─────────────────────────────────┐
│ Delegate to  │               │  PHASE 2: INPUT CONFIGURATION   │
│ real ffmpeg  │               │  Open Input → Setup Decoder     │
│   binary     │               └────────────┬────────────────────┘
└──────────────┘                            │
                                            ▼
                               ┌─────────────────────────────────┐
                               │ PHASE 3: OUTPUT CONFIGURATION   │
                               │ Map Streams → Setup Encoder     │
                               └────────────┬────────────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────────────┐
                               │   PHASE 4: PROCESSING LOOP      │
                               │ Demux → Decode → Process →      │
                               │      Encode → Mux               │
                               └────────────┬────────────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────────────┐
                               │      PHASE 5: CLEANUP           │
                               │ Flush → Write Trailer → Free    │
                               └─────────────────────────────────┘
```

---

## Mode Selection (Phase 0)

### The Three Operating Modes

RTXVideoProcessor intelligently selects one of three modes based on how it's invoked and what operations are requested:

#### Mode 1: FFmpeg Passthrough Mode 🔄

**When activated**: Automatically when operation isn't supported or binary is named `ffmpeg.exe`

**Function**: `passthrough_required()` (src/ffmpeg_passthrough.cpp:161-213)

**Trigger Conditions**:
```cpp
// Activated when binary is named "ffmpeg" or "ffmpeg.exe" AND:
1. Input/output formats not supported (.mp4, .mkv, .m3u8) AND
   No special features requested (HLS, MP4-to-pipe)

2. OR video exclusion requested (-map -0:v?)
   (Can't process without video stream)
```

**What happens**:
- Finds real `ffmpeg` binary on system PATH
- Forwards ALL arguments unchanged
- Uses CreateProcess (Windows) or fork/exec (Unix)
- Inherits stdin/stdout/stderr for pipe support
- Returns ffmpeg's exit code

**Use case**: Drop-in replacement strategy
```bash
# Rename RTXVideoProcessor.exe → ffmpeg.exe
# Unsupported operations automatically delegate to real FFmpeg
ffmpeg.exe -i input.wav -codec:a libmp3lame output.mp3  # Audio-only → Passthrough
ffmpeg.exe -i input.mp4 -map -0:v output.mp4            # No video → Passthrough
```

**Code**: src/ffmpeg_passthrough.cpp:215-357

---

#### Mode 2: Simple Mode (Default) ⚙️

**When activated**: No `-i` flag present (expects positional arguments)

**Syntax**: `RTXVideoProcessor.exe input.mp4 output.mp4 [options]`

**Characteristics**:
- ✅ **Simplified syntax**: No need for `-i` or `-o` flags
- ✅ **RTX Processing**: Video Super Resolution + TrueHDR enabled by default
- ✅ **Auto-corrections**: Timestamp monotonicity fixes, error concealment
- ✅ **Enhanced compatibility**: Auto HLS discontinuity marking
- ✅ **User-friendly**: More forgiving, aims for successful output

**Behavior Differences from FFmpeg**:
| Feature | Simple Mode | FFmpeg |
|---------|-------------|--------|
| Syntax | `input output [opts]` | `-i input [opts] output` |
| Timestamp violations | Auto-fix | Report error |
| Decoder errors | Show corrupt frames | May drop frames |
| HLS discontinuity | Auto-mark on seek | Manual only |
| Processing | RTX enhancements | Standard |

**Use case**: Quick video upscaling and enhancement without FFmpeg complexity
```bash
RTXVideoProcessor.exe 1080p.mp4 4k.mp4  # Simple upscale with RTX
```

---

#### Mode 3: FFmpeg-Compatible Mode 🎯

**When activated**: `-i` flag present in arguments (auto-detected)

**Syntax**: `RTXVideoProcessor.exe -i input.mp4 [options] output.mp4` (FFmpeg-style)

**Purpose**: **Drop-in FFmpeg replacement WITH RTX processing**

**Characteristics**:
- ✅ **FFmpeg syntax**: Uses `-i`, `-map`, `-codec`, etc.
- ✅ **RTX Processing**: Still applies VSR + TrueHDR (NOT passthrough!)
- ✅ **Strict FFmpeg behavior**: No auto-corrections, strict error handling
- ✅ **Full `-map` support**: Identical stream mapping behavior
- ✅ **Compatible flags**: All FFmpeg timestamp/seeking flags work

**Behavior Differences**:
| Feature | FFmpeg-Compatible Mode | Simple Mode |
|---------|------------------------|-------------|
| Syntax | `-i input [opts] output` | `input output [opts]` |
| RTX Processing | ✅ YES | ✅ YES |
| Timestamp violations | Report only (like FFmpeg) | Auto-fix |
| Error concealment | Disabled (like FFmpeg) | Enabled |
| HLS discontinuity | Manual (like FFmpeg) | Auto-mark |
| Monotonicity | Detect (like FFmpeg) | Enforce |

**Key insight**: This mode lets you use RTXVideoProcessor as a drop-in ffmpeg replacement while getting RTX acceleration benefits AND maintaining strict FFmpeg compatibility!

**Use case**: Replace FFmpeg in existing scripts while adding RTX processing
```bash
# Existing FFmpeg command:
# ffmpeg -i input.mp4 -codec:v hevc -o output.mp4

# Just replace binary name (auto-detects FFmpeg mode from syntax):
RTXVideoProcessor.exe -i input.mp4 -codec:v hevc -o output.mp4
# Result: Same behavior + RTX processing!
```

---

### Mode Selection Logic

**Entry Point**: main.cpp:880-883

```cpp
if (passthrough_required(argc, argv)) {
    return run_ffmpeg_passthrough(argc, argv);
}
// Otherwise: Continue to Default or FFmpeg-Compatible mode
```

**Decision Tree**:
```
main()
  │
  ├─ Is binary named "ffmpeg(.exe)"?
  │    │
  │   YES → passthrough_required() evaluates:
  │           │
  │           ├─ Unsupported formats + no special features? → PASSTHROUGH MODE
  │           ├─ Video exclusion (-map -0:v?)? → PASSTHROUGH MODE
  │           └─ Else (supported operation) → Continue ↓
  │
  └─ Parse config
       │
       ├─ -i flag present? → FFMPEG-COMPATIBLE MODE
       └─ Else → DEFAULT MODE (LEGACY)
```

**passthrough_required() Logic** (src/ffmpeg_passthrough.cpp:161-213):
```cpp
// Step 1: Check binary name
bool binary_is_ffmpeg = endsWith(argv[0], "ffmpeg") ||
                        endsWith(argv[0], "ffmpeg.exe");
if (!binary_is_ffmpeg) return false;  // Not invoked as ffmpeg

// Step 2: Check supported formats
bool input_supported = endsWith(argv[1], ".mp4") || endsWith(argv[1], ".mkv");
bool output_supported = endsWith(argv[2], ".mp4") ||
                        endsWith(argv[2], ".mkv") ||
                        endsWith(argv[2], ".m3u8");

// Step 3: Check special features we support
bool requests_hls = (has "-f hls");
bool requests_mp4_to_pipe = (has "-f mp4" AND output is pipe);
bool requests_video_exclusion = (has "-map -0:v?");

// Decision
if (requests_video_exclusion) return true;  // Can't process without video
if (!input_supported && !output_supported &&
    !requests_hls && !requests_mp4_to_pipe) {
    return true;  // Not our wheelhouse
}
return false;  // We can handle this
```

---

### Configuration Flow by Mode

#### Passthrough Mode
```
main() → passthrough_required() returns true
  ↓
run_ffmpeg_passthrough()
  ↓
Find real ffmpeg binary on PATH
  ↓
CreateProcess/fork+exec with original arguments
  ↓
Wait for process, return exit code
  ↓
[END - RTXVideoProcessor code never runs]
```

#### Default Mode
```
main() → passthrough_required() returns false
  ↓
parse_arguments() → cfg.ffCompatible = false (default)
  ↓
[Standard pipeline with legacy enhancements]
  ├─ ts_config.enforce_monotonicity = true
  ├─ inputOpts.enableErrorConcealment = true
  └─ hlsOpts.autoDiscontinuity = true
```

#### FFmpeg-Compatible Mode
```
main() → passthrough_required() returns false
  ↓
parse_arguments() → cfg.ffCompatible = true
  ↓
[Standard pipeline with strict FFmpeg behavior]
  ├─ ts_config.enforce_monotonicity = false
  ├─ inputOpts.enableErrorConcealment = false
  └─ hlsOpts.autoDiscontinuity = false
```

---

### Mode Comparison Matrix

| Aspect | Passthrough | Default Mode | FFmpeg-Compatible |
|--------|-------------|------------------|-------------------|
| **Delegates to FFmpeg** | ✅ YES | ❌ NO | ❌ NO |
| **RTX Processing** | ❌ NO | ✅ YES | ✅ YES |
| **Custom Processing** | ❌ NO | ✅ YES | ✅ YES |
| **Auto-fix timestamps** | - | ✅ YES | ❌ NO |
| **Error concealment** | - | ✅ YES | ❌ NO |
| **HLS auto-discontinuity** | - | ✅ YES | ❌ NO |
| **Stream mapping** | - | ✅ FFmpeg-compatible | ✅ FFmpeg-compatible |
| **Use Case** | Unsupported ops | Video enhancement | FFmpeg replacement |

---

## Phase 1: Initialization

### Entry Point: `main()` (src/main.cpp:873)

```cpp
int main(int argc, char **argv)
```

### Step 1.1: Parse Command-Line Arguments

**Function**: `parseArguments()` (src/config_parser.cpp)

```
main.cpp:875-880
    ↓
parseArguments(argc, argv)
    ↓
Returns: Config object with all settings
```

**Key Configuration Fields**:
```cpp
struct Config {
    // Input/Output
    const char* inputPath;
    const char* outputPath;

    // Processing modes
    bool cpuOnly;
    bool defaultMode;
    bool ffCompatible;

    // Stream mapping
    std::vector<std::string> streamMaps;  // -map arguments

    // Audio
    std::string audioCodec;
    int audioChannels;
    int audioBitrate;

    // Seeking
    std::string seekTime;
    bool seek2any;
    bool seekTimestamp;

    // Timestamp options
    bool copyts;
    bool startAtZero;
    AvoidNegativeTs avoidNegativeTs;

    // Output format
    std::string outputFormat;
    HLS options (hlsTime, segmentType, etc.)
};
```

**Decision Points**:
- If `cpuOnly`: Disable GPU processing paths
- If `defaultMode`: Use passthrough mode (FFmpeg-only, no RTX processing)
- If `ffCompatible`: Disable all legacy enhancements

**Code Path**:
```
config_parser.cpp:
  parseArguments() [handles all CLI flags]
    ├─ Basic flags: -i, -o, -format
    ├─ Processing: -cpu, --no-vsr, --no-thdr
    ├─ Stream mapping: -map, -codec:a
    ├─ Seeking: -ss, -seek2any, -seek_timestamp
    ├─ Timestamp: -copyts, -start_at_zero, -avoid_negative_ts
    └─ HLS: -hls_time, -hls_segment_type, etc.
```

### Step 1.2: Validate Configuration

**Location**: main.cpp:882-895

```cpp
if (!cfg.inputPath || !cfg.outputPath) {
    fprintf(stderr, "Error: Both input and output paths are required\n");
    return 1;
}
```

**Validations**:
1. Input/output paths exist
2. Mutually exclusive flags (e.g., `-default` + RTX processing)
3. Required dependencies (e.g., HLS requires segment type)

### Step 1.3: Initialize Libraries

**Location**: main.cpp:256-260

```cpp
// Initialize logger
Logger::instance().setVerbose(cfg.verbose || cfg.debug);
Logger::instance().setDebug(cfg.debug);

// No av_register_all() needed in FFmpeg 4+
LOG_VERBOSE("Starting video processing pipeline");

// Note: CUDA initialization happens later during RTX processor setup (main.cpp:445-448)
// This allows for graceful fallback to CPU if GPU initialization fails
```

**Initialized Components**:
- Logger system
- FFmpeg libraries (automatic in FFmpeg 4+)

**CUDA Initialization** (deferred to RTX processor setup):
- Location: main.cpp:445-448 via `initialize_rtx_processor()`
- Allows graceful CPU fallback without aborting early

---

## Phase 2: Input Configuration

### Entry Point: `configure_input()` or direct `open_input()`

### Step 2.1: Open Input File

**Function**: `open_input()` (src/ffmpeg_utils.cpp:28-216)

```cpp
bool open_input(const char *inPath, InputContext &in,
                const InputOpenOptions *options)
```

**Code Path**:
```
main.cpp:258 → open_input()
    ↓
ffmpeg_utils.cpp:28
    │
    ├─ Step 2.2.1: Open Format Context
    │   avformat_open_input(&in.fmt, inPath, ...) [line 63]
    │   avformat_find_stream_info(in.fmt, ...) [line 67]
    │
    ├─ Step 2.2.2: Handle Seeking (if -ss specified)
    │   av_parse_time(&seek_target, options->seekTime) [line 74]
    │   in.seek_offset_us = seek_target [line 81]
    │
    │   Compose seek flags: [line 84-98]
    │   ├─ Base: AVSEEK_FLAG_BACKWARD
    │   ├─ If -seek2any: add AVSEEK_FLAG_ANY
    │   └─ If -seek_timestamp: add AVSEEK_FLAG_FRAME
    │
    │   avformat_seek_file(in.fmt, ..., seek_flags) [line 100]
    │
    ├─ Step 2.2.3: Find Video Stream
    │   av_find_best_stream(in.fmt, AVMEDIA_TYPE_VIDEO) [line 115]
    │   in.vstream = vstream [line 119]
    │   in.vst = in.fmt->streams[vstream] [line 120]
    │
    ├─ Step 2.2.4: Setup Video Decoder
    │   avcodec_find_decoder(in.vst->codecpar->codec_id) [line 122]
    │   in.vdec = avcodec_alloc_context3(decoder) [line 126]
    │   avcodec_parameters_to_context(in.vdec, in.vst->codecpar) [line 130]
    │
    │   Enable error concealment (if !ffCompatible): [line 136-138]
    │   ├─ in.vdec->flags2 |= AV_CODEC_FLAG2_SHOW_ALL
    │   └─ in.vdec->flags |= AV_CODEC_FLAG_OUTPUT_CORRUPT
    │
    │   Try CUDA hardware decoding: [line 141-167]
    │   ├─ av_hwdevice_ctx_create(&in.hw_device_ctx, AV_HWDEVICE_TYPE_CUDA)
    │   ├─ If preferP010ForHDR: setup P010 HW frames context
    │   └─ in.vdec->hw_device_ctx = av_buffer_ref(in.hw_device_ctx)
    │
    │   avcodec_open2(in.vdec, decoder) [line 173]
    │   └─ If fails: Fallback to software decoding [line 177-183]
    │
    └─ Step 2.2.5: Find Audio Stream (optional)
        av_find_best_stream(in.fmt, AVMEDIA_TYPE_AUDIO) [line 187]
        If found:
        ├─ avcodec_find_decoder(in.ast->codecpar->codec_id) [line 193]
        ├─ in.adec = avcodec_alloc_context3(audio_decoder) [line 196]
        └─ avcodec_open2(in.adec, audio_decoder) [line 202]
```

**Data Structures Populated**:
```cpp
struct InputContext {
    AVFormatContext *fmt;        // Demuxer
    int vstream;                 // Video stream index
    AVStream *vst;               // Video stream
    AVCodecContext *vdec;        // Video decoder
    int astream;                 // Audio stream index
    AVStream *ast;               // Audio stream
    AVCodecContext *adec;        // Audio decoder
    AVBufferRef *hw_device_ctx;  // CUDA device
    int64_t seek_offset_us;      // Seek offset for sync
};
```

### Step 2.3: HDR Detection

**Function**: `configure_input_hdr_detection()` (src/input_config.cpp:14-35)

```cpp
bool configure_input_hdr_detection(const Config &cfg, const InputContext &in)
```

**Logic**:
```
Check video stream color properties:
    ├─ color_primaries == AVCOL_PRI_BT2020
    ├─ color_trc == AVCOL_TRC_SMPTE2084 (PQ) or AVCOL_TRC_ARIB_STD_B67 (HLG)
    └─ color_space == AVCOL_SPC_BT2020_NCL
         │
         └─ If all match: HDR detected
              ├─ LOG_INFO("HDR content detected")
              └─ Return true
```

---

## Phase 3: Output Configuration

### Step 3.1: Configure Output Context

**Function**: `configure_output()` (src/output_config.cpp:9-189)

```cpp
bool configure_output(const Config &cfg, const InputContext &in,
                      OutputContext &out)
```

**Code Path**:
```
output_config.cpp:9 → configure_output()
    │
    ├─ Step 3.1.1: Setup HLS Options (if format == "hls")
    │   [lines 72-176]
    │   ├─ hlsOpts.enabled = true
    │   ├─ hlsOpts.overwrite = cfg.overwrite
    │   ├─ hlsOpts.autoDiscontinuity = !cfg.ffCompatible
    │   ├─ Parse hls_time, segment_type, init_filename
    │   └─ Generate segment filename pattern
    │
    ├─ Step 3.1.2: Setup Timestamp Configuration
    │   [lines 178-181]
    │   ├─ out.avoidNegativeTs = parsed from -avoid_negative_ts
    │   └─ out.startAtZero = cfg.startAtZero
    │
    └─ Step 3.1.3: Configure Audio Processing
        [lines 183-187]
        configure_audio_processing(cfg, in, out)
```

### Step 3.2: Open Output File

**Function**: `open_output()` (src/ffmpeg_utils.cpp:251-562)

```cpp
bool open_output(const char *outPath, const InputContext &in,
                 OutputContext &out,
                 const std::vector<std::string> &streamMaps)
```

**Code Path**:
```
main.cpp:278 → open_output()
    ↓
ffmpeg_utils.cpp:251
    │
    ├─ Step 3.2.1: Allocate Output Context
    │   [lines 278-283]
    │   effectiveFormat = hlsOptions.enabled ? "hls" : (isPipe ? "mp4" : nullptr)
    │   avformat_alloc_output_context2(&out.fmt, ..., effectiveFormat, outPath)
    │
    ├─ Step 3.2.2: Configure HLS Muxer Options (if enabled)
    │   [lines 285-384]
    │   Set AVDictionary options:
    │   ├─ hls_time
    │   ├─ hls_segment_filename
    │   ├─ hls_segment_type (fmp4 or mpegts)
    │   ├─ hls_fmp4_init_filename
    │   ├─ start_number
    │   ├─ hls_playlist_type
    │   ├─ hls_list_size
    │   ├─ max_delay
    │   │
    │   Compute hls_flags based on mode: [lines 338-362]
    │   ├─ If !autoDiscontinuity (FFmpeg mode):
    │   │   └─ Minimal flags, let FFmpeg handle defaults
    │   └─ Else (Legacy mode):
    │       ├─ fmp4: "+append_list"
    │       └─ mpegts: "split_by_time+append_list"
    │   │
    │   If seeking + autoDiscontinuity: [lines 366-374]
    │   └─ Add "+discont_start" flag
    │
    ├─ Step 3.2.3: Setup Video Encoder
    │   [lines 391-405]
    │   ├─ avcodec_find_encoder_by_name("hevc_nvenc")
    │   ├─ out.venc = avcodec_alloc_context3(encoder)
    │   ├─ Set codec_id, codec_type, time_base
    │   └─ Encoder params configured later in main loop setup
    │
    ├─ Step 3.2.4: Decide Stream Mappings ⭐ CRITICAL
    │   [lines 407-422]
    │
    │   If streamMaps provided:
    │       apply_stream_mappings(streamMaps, in, out)
    │           ↓
    │       decide_stream_mappings() [SINGLE SOURCE OF TRUTH]
    │           │
    │           ├─ Initialize all as EXCLUDE [line 811]
    │           │
    │           ├─ Determine audio processing mode [lines 804-806]
    │           │   audio_needs_processing = enabled && codec != "copy"
    │           │
    │           ├─ Check for explicit inclusions [lines 809-816]
    │           │   has_explicit_inclusions = any non-exclusion -map
    │           │
    │           ├─ If NO explicit inclusions: [lines 819-823]
    │           │   └─ Default: include ALL streams as COPY
    │           │
    │           └─ Process each -map directive [lines 826-859]
    │               For each mapping:
    │               ├─ parse_stream_mapping() → {exclude, stream_index, media_type}
    │               ├─ Match against input streams
    │               └─ Set decision:
    │                   ├─ If exclude: EXCLUDE
    │                   ├─ If audio + needs_processing: PROCESS_AUDIO
    │                   └─ Else: COPY
    │
    │   Else (no -map):
    │       Default: include all as COPY [lines 414-415]
    │
    │   Mark video stream for processing [line 422]
    │   out.stream_decisions[in.vstream] = PROCESS_VIDEO
    │
    ├─ Step 3.2.5: Create Output Streams
    │   [lines 419-429, 436-514]
    │
    │   Create video stream (always): [lines 425-429]
    │   ├─ out.vstream = avformat_new_stream(out.fmt, nullptr)
    │   ├─ out.vstream->time_base = out.venc->time_base
    │   └─ out.input_to_output_map[in.vstream] = out.vstream->index
    │
    │   For each input stream: [lines 437-514]
    │   ├─ Skip if video (already handled) [lines 440-441]
    │   ├─ Skip if EXCLUDE [lines 444-445]
    │   │
    │   ├─ Apply output-specific filters:
    │   │   ├─ Drop subtitles for pipe output [lines 452-459]
    │   │   ├─ Drop non-WebVTT subtitles for HLS [lines 462-470]
    │   │   └─ Drop unsupported codecs [lines 473-480]
    │   │
    │   ├─ If PROCESS_AUDIO: [lines 483-490]
    │   │   └─ Skip (stream created later in setup_audio_encoder)
    │   │
    │   └─ If COPY: [lines 492-513]
    │       ├─ ost = avformat_new_stream(out.fmt, nullptr)
    │       ├─ avcodec_parameters_copy(ost->codecpar, ist->codecpar)
    │       ├─ Set time_base (audio: {1, sample_rate}, else: input time_base)
    │       ├─ Track first audio stream: out.astream = ost
    │       └─ out.input_to_output_map[i] = ost->index
    │
    └─ Step 3.2.6: Open Output File
        [lines 528-558]
        If not AVFMT_NOFILE:
        ├─ Set AVIO flags (write, truncate for HLS)
        ├─ Handle pipe output (set binary mode on Windows)
        └─ avio_open2(&out.fmt->pb, outPath, ...)
```

### Step 3.3: Configure Audio Processing

**Function**: `configure_audio_processing()` (src/input_config.cpp:65-111)

```cpp
void configure_audio_processing(const Config &cfg, const InputContext &in,
                                OutputContext &out)
```

**Code Path**:
```
input_config.cpp:65
    │
    ├─ Build AudioParameters from Config [lines 77-87]
    │   audioParams.codec = cfg.audioCodec
    │   audioParams.channels = cfg.audioChannels
    │   audioParams.bitrate = cfg.audioBitrate
    │   audioParams.sampleRate = cfg.audioSampleRate
    │   audioParams.filter = cfg.audioFilter
    │   audioParams.streamMaps = cfg.streamMaps
    │
    ├─ Configure from parameters [line 89]
    │   configure_audio_from_params(audioParams, out)
    │       ↓
    │   ffmpeg_utils.cpp:713
    │   ├─ Set out.audioConfig fields [lines 716-720]
    │   ├─ Determine if enabled [lines 724-733]
    │   │   enabled = hasAudioParams || has_audio_in_mappings()
    │   └─ LOG_INFO if enabled [lines 735-737]
    │
    └─ If enabled and codec != "copy": [lines 96-112]
        ├─ setup_audio_encoder(in, out) [line 100]
        │       ↓
        │   ffmpeg_utils.cpp:888
        │   ├─ Find best audio stream [line 101]
        │   ├─ Find/create audio encoder [lines 114-126]
        │   ├─ Configure encoder context [lines 134-174]
        │   │   ├─ Sample rate (use config or input)
        │   │   ├─ Channel layout (use config or copy from input)
        │   │   ├─ Bit rate (use config or default 128k)
        │   │   └─ Sample format (from encoder or FLTP)
        │   ├─ Create output stream if needed [lines 177-183]
        │   ├─ Open encoder [line 186]
        │   ├─ Copy parameters to stream [lines 195-199]
        │   ├─ Setup resampler (SwrContext) [lines 209-229]
        │   └─ Create audio FIFO buffer [lines 232-237]
        │
        └─ setup_audio_filter(in, out) [line 105]
                ↓
            ffmpeg_utils.cpp:242 (if filter specified)
            ├─ Create filter graph [line 267]
            ├─ Create abuffer source [lines 282-293]
            ├─ Create abuffersink [lines 296-301]
            ├─ Parse filter string [line 318]
            └─ Configure graph [line 325]
```

### Step 3.4: Initialize Timestamp Manager

**Location**: main.cpp:435-471

```cpp
TimestampManager::Config ts_config;
ts_config.mode = cfg.copyts ? TimestampManager::Mode::COPYTS
                             : TimestampManager::Mode::NORMAL;
ts_config.input_seek_us = in.seek_offset_us;
ts_config.avoid_negative_ts = out.avoidNegativeTs;
ts_config.start_at_zero = out.startAtZero;
ts_config.enforce_monotonicity = !cfg.ffCompatible;
ts_config.expected_frame_rate = av_guess_frame_rate(in.fmt, in.vst, nullptr);

TimestampManager ts_manager(ts_config);
```

**Timestamp Modes**:
```
NORMAL mode:
    ├─ Establishes baseline from first frame
    ├─ Outputs zero-based timestamps
    └─ Accounts for seek offset

COPYTS mode:
    ├─ Preserves original timestamps
    ├─ Optionally shifts with -start_at_zero
    └─ Applies output_ts_offset if specified
```

---

## Phase 4: Processing Loop

### Main Loop Structure (main.cpp:495-855)

```cpp
while (true) {
    // Step 4.1: Demux packet
    // Step 4.2: Decode frame
    // Step 4.3: Process video (RTX or passthrough)
    // Step 4.4: Encode and write
    // Step 4.5: Handle audio
}
```

### Step 4.1: Demultiplexing

**Location**: main.cpp:499-510

```cpp
PacketPtr pkt(av_packet_alloc(), av_packet_free_single);
int ret = av_read_frame(in.fmt, pkt.get());

if (ret == AVERROR_EOF) {
    // End of file - flush encoders
    break;
}
ff_check(ret, "read frame");
```

**Packet Router** [lines 513-528]:
```
Determine packet destination:
    │
    ├─ Is video stream? [line 513]
    │   └─ Process in video path
    │
    ├─ Is audio stream AND mapped? [line 682]
    │   ├─ Decision == PROCESS_AUDIO?
    │   │   └─ Process through encoder/filter
    │   └─ Decision == COPY?
    │       └─ Copy packet directly to output
    │
    └─ Is other stream AND mapped? [line 719]
        └─ Copy packet directly to output (subtitles, data, etc.)
```

### Step 4.2: Video Decoding

**Location**: main.cpp:515-570

```cpp
// Send packet to decoder
ret = avcodec_send_packet(in.vdec, pkt.get());
if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_make_error_string(errbuf, sizeof(errbuf), ret);
    LOG_WARN("Decoder error: %s (continuing)", errbuf);
    continue;
}

// Receive decoded frames
while (true) {
    FramePtr frame(av_frame_alloc(), av_frame_free_single);
    ret = avcodec_receive_frame(in.vdec, frame.get());

    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        break;

    ff_check(ret, "receive frame");

    // Frame ready for processing
    processVideoFrame(frame.get());
}
```

**Frame Flow**:
```
avcodec_receive_frame()
    ↓
Check for output seeking [lines 573-577]
    └─ ts_manager.shouldDropFrameForOutputSeek()
        └─ If before target: drop frame
            ↓
Establish global baseline [lines 579-583]
    └─ ts_manager.establishGlobalBaseline()
        └─ Sets baseline from first frame PTS
            ↓
[Proceeds to Step 4.3]
```

### Step 4.3: Video Processing

**Unified Processing Architecture using IProcessor Abstraction**:

**Location**: main.cpp:456-469 (processor initialization), main.cpp:669-675 (processing)

The pipeline uses a unified `IProcessor` interface that abstracts away GPU vs CPU processing details:

```cpp
// Processor selection (main.cpp:456-469)
std::unique_ptr<IProcessor> processor;
if (use_cuda_path) {
    // GPU path: CUDA frames stay on GPU through entire pipeline
    processor = std::make_unique<GpuProcessor>(rtx, cuda_pool,
                                               in.vdec->colorspace, outputHDR);
} else {
    // CPU path: Software processing with RTX
    auto cpuProc = std::make_unique<CpuProcessor>(rtx, in.vdec->width,
                                                  in.vdec->height, dstW, dstH);
    RTXProcessConfig cpuConfig = cfg.rtxCfg;
    cpuConfig.enableTHDR = outputHDR;
    cpuProc->setConfig(cpuConfig);
    processor = std::move(cpuProc);
}

// Unified processing call (main.cpp:671)
AVFrame *outFrame = nullptr;
if (!processor->process(decframe, outFrame)) {
    throw std::runtime_error("Processor failed to produce output frame");
}
```

**Processing Paths**:

#### GPU Path (`GpuProcessor`)
```cpp
GpuProcessor::process():
    ├─ Frame already on GPU (AV_PIX_FMT_CUDA)
    ├─ Apply RTX Video Super Resolution (if enabled)
    ├─ Apply TrueHDR tone mapping (if enabled)
    ├─ Convert colorspace (BT.709/BT.2020 as needed)
    ├─ Pack to P010LE/NV12 for encoder
    └─ Return CUDA frame (stays on GPU for NVENC)
```

#### CPU Path (`CpuProcessor`)
```cpp
CpuProcessor::process():
    ├─ Transfer from GPU to CPU (if needed)
    ├─ Convert to RGBA for RTX processing
    ├─ Apply RTX Video Super Resolution (if enabled)
    ├─ Apply TrueHDR tone mapping (if enabled)
    ├─ Convert to P010LE/NV12 for software encoder
    └─ Return CPU frame
```

**Key Insight**: Both paths apply RTX processing. The difference is where processing occurs (GPU vs CPU), not whether it occurs.

#### CUDA Colorspace Conversion Kernels

**Location**: src/cuda_kernels.cu, src/cuda_kernels.h

The pipeline uses custom CUDA kernels to bridge the gap between RTX SDK output formats (ABGR10/BGRA8) and FFmpeg encoder input formats (P010/NV12). These kernels handle colorspace conversion and format transformation entirely on the GPU, avoiding expensive GPU→CPU→GPU transfers.

**Implemented Kernels**:

1. **NV12 → BGRA8** (cuda_kernels.cu:286-422)
   - Converts 8-bit YUV 4:2:0 to 8-bit RGBA
   - Used for: GPU path when preparing decoded NV12 frames for RTX processing
   - Colorspace: Configurable BT.709 or BT.2020 YUV→RGB matrix
   - Range: Limited range YUV → Full range RGB

2. **ABGR10 → P010** (cuda_kernels.cu:13-105)
   - Converts 10-bit packed RGB (X2BGR10LE) to 10-bit YUV 4:2:0 (P010)
   - Used for: GPU path when converting RTX THDR output to encoder input
   - Colorspace: Configurable BT.709 or BT.2020 RGB→YUV matrix
   - Range: Full range RGB → Limited range YUV (Y: 64-940, UV: 64-960 in 10-bit)
   - Optimization: Processes 2×2 pixel blocks for chroma subsampling

3. **BGRA8 → P010** (cuda_kernels.cu:107-189)
   - Converts 8-bit RGBA to 10-bit YUV 4:2:0 (P010)
   - Used for: GPU path when upconverting 8-bit RTX output to 10-bit HDR encoder
   - Colorspace: Configurable BT.709 or BT.2020
   - Range: Full range RGB → Limited range YUV

4. **BGRA8 → NV12** (cuda_kernels.cu:191-266)
   - Converts 8-bit RGBA to 8-bit YUV 4:2:0 (NV12)
   - Used for: GPU path when preparing 8-bit RTX output for SDR encoder
   - Colorspace: Configurable BT.709 or BT.2020
   - Range: Full range RGB → Limited range YUV (Y: 16-235, UV: 16-240)

5. **P010 → NV12** (cuda_kernels.cu:375-436)
   - Converts 10-bit YUV 4:2:0 to 8-bit YUV 4:2:0
   - Used for: GPU path when downsampling HDR content to SDR
   - Method: Simple bit-shift (takes upper 8 bits of 10-bit data)

6. **P010 → X2BGR10LE** (cuda_kernels.cu:326-448)
   - Converts 10-bit YUV 4:2:0 (P010) to 10-bit packed RGB
   - Used for: GPU path when preparing P010 decoded frames for RTX processing
   - Colorspace: Configurable BT.709 or BT.2020 YUV→RGB matrix
   - Range: Limited range YUV → Full range RGB

**Technical Details**:

- **Grid Configuration**: Kernels use 16×16 or 32×16 thread blocks
- **Memory Access**: Optimized pitched memory access for proper alignment
- **Chroma Subsampling**: RGB→YUV kernels process 2×2 pixel blocks to compute averaged chroma values
- **Limited Range Encoding**: Proper compliance with ITU-R BT.709/2020 limited range specifications
  - 10-bit Y: 64-940 (876 levels)
  - 10-bit UV: 64-960 (896 levels), centered at 512
  - 8-bit Y: 16-235 (219 levels)
  - 8-bit UV: 16-240 (224 levels), centered at 128

**Why Custom Kernels?**

The RTX Video SDK outputs in ABGR10 (for THDR) or BGRA8 (for VSR-only) formats, which are not directly compatible with NVENC's expected P010/NV12 inputs. FFmpeg's `libswscale` cannot handle these conversions on GPU, so custom CUDA kernels are essential to:

1. Maintain the entire pipeline on GPU (avoid expensive transfers)
2. Support both BT.709 and BT.2020 colorspaces
3. Properly handle limited range YUV encoding
4. Optimize chroma subsampling for 4:2:0 formats

### Step 4.4: Video Encoding and Writing

**Location**: main.cpp:614-667

```cpp
// Derive timestamps for encoder
TimestampManager::TimestampPair ts =
    ts_manager.deriveVideoTimestamps(
        decframe,
        in.vst->time_base,
        out.venc->time_base
    );

decframe->pts = ts.pts;
// Note: DTS handled by encoder

// Send to encoder
encode_and_write(out.venc, out.vstream, out.fmt, out,
                 decframe, opkt, "encode video");
```

**encode_and_write()** (main.cpp:91-176):
```cpp
avcodec_send_frame(enc, frame)
    ↓
while (avcodec_receive_packet(enc, opkt) == 0) {
    ├─ Set stream_index
    ├─ Rescale timestamps to stream timebase
    ├─ Enforce DTS monotonicity [lines 136-141]
    │   └─ If DTS <= last_video_dts: DTS = last_video_dts + 1
    ├─ Validate PTS >= DTS [lines 144-147]
    │   └─ If PTS < DTS: PTS = DTS
    ├─ Write packet
    │   av_interleaved_write_frame(ofmt, opkt)
    └─ av_packet_unref(opkt)
}
```

### Step 4.5: Audio Processing

**Two paths based on stream decision**:

#### Path A: Audio Re-encoding (PROCESS_AUDIO)

**Location**: main.cpp:686-707

```cpp
// Decode audio packet
ret = avcodec_send_packet(in.adec, pkt.get());
while (avcodec_receive_frame(in.adec, aframe.get()) == 0) {

    // Process through filter and encoder
    process_audio_frame(aframe.get(), out, opkt.get())
        ↓
    ffmpeg_utils.cpp:345
        ├─ Apply audio filter (if configured) [lines 363-399]
        │   ├─ av_buffersrc_add_frame()
        │   └─ av_buffersink_get_frame()
        │
        ├─ Resample to encoder format [lines 402-446]
        │   └─ swr_convert() using SwrContext
        │
        ├─ Buffer in audio FIFO [lines 449-456]
        │   └─ av_audio_fifo_write()
        │
        └─ Encode fixed-size frames [lines 462-545]
            └─ While FIFO has enough samples:
                ├─ Read encoder->frame_size samples
                ├─ Set PTS = next_audio_pts
                ├─ avcodec_send_frame()
                ├─ avcodec_receive_packet()
                ├─ Rescale timestamps
                ├─ Enforce DTS monotonicity
                └─ Return packet for writing

    // Write audio packet
    if (process_audio_frame returned packet) {
        av_interleaved_write_frame(out.fmt, opkt.get());
    }
}
```

#### Path B: Audio Copy (COPY)

**Location**: main.cpp:712-717

```cpp
// Adjust timestamps for copied stream
ts_manager.adjustPacketTimestamps(pkt.get(), in.ast->time_base, in.astream);

// Remap stream index
int out_stream_idx = out.input_to_output_map[pkt->stream_index];
pkt->stream_index = out_stream_idx;

// Write directly
av_interleaved_write_frame(out.fmt, pkt.get());
```

### Step 4.6: Other Stream Copy

**Location**: main.cpp:719-741

```cpp
// For subtitles, data streams, etc.
int out_stream_idx = out.input_to_output_map[pkt->stream_index];

if (out_stream_idx >= 0) {
    // Adjust timestamps
    ts_manager.adjustPacketTimestamps(pkt.get(),
                                      in.fmt->streams[pkt->stream_index]->time_base,
                                      pkt->stream_index);

    // Remap and write
    pkt->stream_index = out_stream_idx;
    av_interleaved_write_frame(out.fmt, pkt.get());
}
```

---

## Phase 5: Cleanup

### Step 5.1: Flush Encoders

**Location**: main.cpp:743-839

```cpp
// Flush video encoder
LOG_DEBUG("Flushing video encoder");
encode_and_write(out.venc, out.vstream, out.fmt, out,
                 nullptr, opkt, "flush video");

// Flush audio encoder (if processing audio)
if (out.aenc) {
    LOG_DEBUG("Flushing audio encoder");

    // Process remaining samples in FIFO
    while (out.audio_fifo && av_audio_fifo_size(out.audio_fifo) > 0) {
        // Create partial frame with remaining samples
        // ... encode and write
    }

    // Flush encoder
    avcodec_send_frame(out.aenc, nullptr);
    while (avcodec_receive_packet(out.aenc, opkt.get()) == 0) {
        // Write final packets
    }
}
```

### Step 5.2: Write Trailer

**Location**: main.cpp:841-845

```cpp
LOG_DEBUG("Writing trailer");
ret = av_write_trailer(out.fmt);
if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_make_error_string(errbuf, sizeof(errbuf), ret);
    LOG_WARN("Error writing trailer: %s", errbuf);
}
```

**What av_write_trailer() does**:
- Finalizes output file format
- For HLS: Writes playlist footer (`#EXT-X-ENDLIST` for VOD)
- For MP4: Updates moov atom with duration/seeking data
- Flushes any buffered data

### Step 5.3: Free Resources

**Location**: main.cpp:847-854

```cpp
// Print statistics
ts_manager.dumpState();
auto stats = ts_manager.getStats();
LOG_INFO("Processing complete:");
LOG_INFO("  Dropped frames: %d", stats.dropped_frames);
LOG_INFO("  Discontinuities: %d", stats.discontinuities);
LOG_INFO("  Monotonicity fixes: %d", stats.monotonicity_violations);

// Cleanup (RAII handles most)
close_output(out);  // Frees encoder, filter, resampler
close_input(in);    // Frees decoder, demuxer
// CUDA resources freed by destructors
```

**close_output()** (src/ffmpeg_utils.cpp:564-613):
```cpp
├─ Free video encoder: avcodec_free_context(&out.venc)
├─ Free audio encoder: avcodec_free_context(&out.aenc)
├─ Free filter graph: avfilter_graph_free(&out.filter_graph)
├─ Free resampler: swr_free(&out.swr_ctx)
├─ Free audio FIFO: av_audio_fifo_free(out.audio_fifo)
├─ Close output file: avio_closep(&out.fmt->pb)
├─ Free format context: avformat_free_context(out.fmt)
└─ Clear tracking: stream_decisions, input_to_output_map
```

**close_input()** (src/ffmpeg_utils.cpp:218-249):
```cpp
├─ Free video decoder: avcodec_free_context(&in.vdec)
├─ Free audio decoder: avcodec_free_context(&in.adec)
├─ Release CUDA device: av_buffer_unref(&in.hw_device_ctx)
├─ Close input file: avformat_close_input(&in.fmt)
└─ Reset tracking: vstream, astream, seek_offset_us
```

---

## Data Flow Diagrams

### Video Processing Flow

```
┌──────────────┐
│ Input File   │
└──────┬───────┘
       │ av_read_frame()
       ▼
┌──────────────────────────┐
│ AVPacket (compressed)    │
└──────┬───────────────────┘
       │ avcodec_send_packet()
       ▼
┌──────────────────────────┐
│ Video Decoder (NVDEC)    │
└──────┬───────────────────┘
       │ avcodec_receive_frame()
       ▼
┌──────────────────────────┐
│ AVFrame (YUV, GPU/CPU)   │
└──────┬───────────────────┘
       │
       ├─ GPU Path ────────────┐
       │ (use_cuda_path)        ▼
       │              ┌────────────────────┐
       │              │ GpuProcessor       │
       │              │ - VSR (GPU)        │
       │              │ - THDR (GPU)       │
       │              │ - Colorspace (GPU) │
       │              └────────┬───────────┘
       │                       │
       └─ CPU Path ────────────┤
         (!use_cuda_path)      ▼
                    ┌──────────────────────┐
                    │ CpuProcessor         │
                    │ - VSR (CPU+RTX)      │
                    │ - THDR (CPU+RTX)     │
                    │ - Colorspace (CPU)   │
                    └──────┬───────────────┘
                           │
                           ▼
                    ┌──────────────────────┐
                    │ AVFrame (processed)  │
                    └──────┬───────────────┘
                           │ TimestampManager.deriveTimestamps()
                           ▼
                    ┌──────────────────────┐
                    │ AVFrame (with PTS)   │
                    └──────┬───────────────┘
                           │ avcodec_send_frame()
                           ▼
                    ┌──────────────────────┐
                    │ Video Encoder (NVENC)│
                    └──────┬───────────────┘
                           │ avcodec_receive_packet()
                           ▼
                    ┌──────────────────────┐
                    │ AVPacket (HEVC)      │
                    └──────┬───────────────┘
                           │ av_interleaved_write_frame()
                           ▼
                    ┌──────────────────────┐
                    │ Output File          │
                    │ (MP4/HLS)            │
                    └──────────────────────┘
```

### Audio Processing Flow

```
                    ┌──────────────┐
                    │ AVPacket     │
                    │ (audio)      │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
       PROCESS_AUDIO                 COPY
              │                         │
              ▼                         ▼
    ┌─────────────────┐      ┌──────────────────┐
    │ Audio Decoder   │      │ Adjust Timestamps│
    └────────┬────────┘      └────────┬─────────┘
             │                        │
             ▼                        │
    ┌─────────────────┐              │
    │ AVFrame (PCM)   │              │
    └────────┬────────┘              │
             │                        │
             ├─ Audio Filter          │
             │  (optional)            │
             ▼                        │
    ┌─────────────────┐              │
    │ Resampler       │              │
    │ (SwrContext)    │              │
    └────────┬────────┘              │
             │                        │
             ▼                        │
    ┌─────────────────┐              │
    │ Audio FIFO      │              │
    │ (buffering)     │              │
    └────────┬────────┘              │
             │                        │
             ▼                        │
    ┌─────────────────┐              │
    │ Audio Encoder   │              │
    │ (AAC/etc)       │              │
    └────────┬────────┘              │
             │                        │
             └────────────┬───────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ AVPacket (audio)│
                 └────────┬────────┘
                          │ av_interleaved_write_frame()
                          ▼
                 ┌─────────────────┐
                 │ Output File     │
                 └─────────────────┘
```

### Stream Mapping Decision Flow

```
                  ┌────────────────────┐
                  │ -map arguments     │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ decide_stream_     │
                  │ mappings()         │
                  └─────────┬──────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌──────────────┐
│ Parse each    │   │ Check audio   │   │ Has explicit │
│ -map argument │   │ codec != copy │   │ inclusions?  │
└───────┬───────┘   └───────┬───────┘   └──────┬───────┘
        │                   │                   │
        │                   │         ┌─────────┴────────┐
        │                   │         │                  │
        │                   │        YES                NO
        │                   │         │                  │
        │                   │         │      ┌───────────▼──────┐
        │                   │         │      │ Default: include │
        │                   │         │      │ ALL as COPY      │
        │                   │         │      └───────────┬──────┘
        │                   │         │                  │
        └───────────────────┴─────────┴──────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │ For each input stream:      │
              │                             │
              │ ┌─────────────────────────┐ │
              │ │ Exclusion (-map -0:s)?  │ │
              │ └────┬──────────────────┬─┘ │
              │     YES               NO    │
              │      │                 │    │
              │      ▼                 ▼    │
              │ ┌─────────┐      ┌─────────┐│
              │ │ EXCLUDE │      │ Audio + ││
              │ └─────────┘      │process? ││
              │                  └────┬────┘│
              │                      YES│NO  │
              │                       │  │   │
              │                       ▼  ▼   │
              │              ┌──────────────┐│
              │              │PROCESS_AUDIO ││
              │              │or COPY       ││
              │              └──────────────┘│
              └─────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │ StreamMapDecision for each  │
              │ input stream stored in:     │
              │ out.stream_decisions[]      │
              └─────────────────────────────┘
```

---

## Error Handling

### Error Concealment Strategy

**Location**: FFmpeg decoder flags (ffmpeg_utils.cpp:136-138)

```cpp
// Enable error concealment (if !ffCompatible)
in.vdec->flags2 |= AV_CODEC_FLAG2_SHOW_ALL;
in.vdec->flags |= AV_CODEC_FLAG_OUTPUT_CORRUPT;
```

**Behavior**:
- **Legacy mode**: Shows frames with artifacts instead of dropping
- **FFmpeg mode**: Strict - may drop frames with errors
- **Use case**: Seeking to non-keyframes (`-seek2any 1`)

### Decoder Error Handling (main.cpp:517-527)

```cpp
ret = avcodec_send_packet(in.vdec, pkt.get());
if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_make_error_string(errbuf, sizeof(errbuf), ret);
    LOG_WARN("Decoder error: %s (continuing)", errbuf);
    continue;  // Skip packet, continue processing
}
```

**Strategy**: Log and continue (graceful degradation)

### Timestamp Violations (timestamp_manager.h:362-375)

```cpp
if (!cfg_.enforce_monotonicity) {
    // FFmpeg mode: Detect and log
    LOG_WARN("PTS monotonicity violation (not fixing)");
} else {
    // Legacy mode: Auto-fix
    ts.pts = last_video_pts_ + frame_duration;
}
```

### Muxer Errors (main.cpp:151-169)

```cpp
ret = av_interleaved_write_frame(ofmt, opkt);
if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_make_error_string(errbuf, sizeof(errbuf), ret);
    LOG_ERROR("Failed to write packet: %s", errbuf);

    // Attempt recovery
    if (ret == AVERROR(EINVAL)) {
        // Timestamp issue - logged elsewhere
    } else {
        // Fatal error - abort
        throw std::runtime_error("Muxer error");
    }
}
```

---

## Performance Optimizations

### 1. CUDA Hardware Acceleration

```
┌─────────────┐         ┌──────────────┐
│  GPU Memory │ ◄────── │ NVDEC Decode │
└──────┬──────┘         └──────────────┘
       │
       │ (No CPU transfer)
       ▼
┌─────────────┐
│ RTX Process │
│ (GPU-only)  │
└──────┬──────┘
       │
       │ (No CPU transfer)
       ▼
┌─────────────┐         ┌──────────────┐
│  GPU Memory │ ──────► │ NVENC Encode │
└─────────────┘         └──────────────┘
```

**Benefit**: Zero-copy pipeline when fully GPU-accelerated

### 2. Frame Pool (GPU mode)

**File**: src/frame_pool.h

**Location**: Initialized in output_config.cpp:318

The CUDA frame pool pre-allocates and reuses encoder frames to avoid repeated GPU memory allocation overhead during encoding.

**Configuration**:
- **Pool Size**: 64 frames (output_config.cpp:318: `fctx->initial_pool_size = 64`)
- **Strategy**: Round-robin allocation with `acquire()` method
- **Frame Format**: AV_PIX_FMT_CUDA with sw_format P010LE (HDR) or NV12 (SDR)

**Implementation** (frame_pool.h:65-78):
```cpp
AVFrame *acquire() {
    FramePtr &slot = m_frames[m_index];
    m_index = (m_index + 1) % m_frames.size();  // Round-robin
    av_frame_unref(slot.get());                 // Clear previous data
    av_hwframe_get_buffer(m_hw_frames_ctx, slot.get(), 0);  // Get fresh buffer
    return slot.get();
}
```

**Rationale for 64 frames**:
- Provides sufficient buffering for B-frames (max_b_frames can be up to 4)
- Accommodates encoder lookahead and reordering
- Balances GPU memory usage vs allocation frequency
- Typical B-frame pyramid: I-frame + 4 B-frames + lookahead = ~10-15 frames active
- 64 frames provides 4x safety margin for high-throughput scenarios

### 3. Interleaved Writing

```cpp
av_interleaved_write_frame()  // Instead of av_write_frame()
```

**Benefit**: Muxer handles stream interleaving and buffering automatically

### 4. Audio FIFO Buffering

**Purpose**: Batch fixed-size frames for encoder

```
Input: Variable-size decoded frames
  ↓
FIFO Buffer
  ↓
Output: Fixed-size encoder frames (e.g., 1024 samples for AAC)
```

### 5. NVENC Encoder Optimizations

**Location**: src/output_config.cpp:176-329

The pipeline enables several NVENC-specific optimizations that are not explicitly documented in user-facing docs:

#### Temporal Adaptive Quantization (Temporal AQ)

**Code**: output_config.cpp:289
```cpp
av_opt_set(out.venc->priv_data, "temporal-aq", "1", 0);
```

**What it does**:
- Analyzes temporal (across frames) motion and complexity
- Dynamically adjusts quantization parameters based on temporal redundancy
- Allocates more bits to complex/high-motion areas, fewer to static regions
- **Result**: Better perceptual quality at same bitrate, especially for motion

**Always enabled**: This optimization is applied to all encodes (both HDR and SDR)

#### GOP Alignment for HLS (output_config.cpp:216-238)

For HLS outputs, GOP size is automatically aligned to segment duration:

```cpp
if (hls_enabled && out.hlsOptions.hlsTime > 0) {
    gop_duration_sec = out.hlsOptions.hlsTime;
    gop_size_frames = (int)round((double)gop_duration_sec * fr.num / fr.den);

    // For 23.976fps with 3-sec segments: 3 * 24000 / 1001 = 72 frames
}
```

Combined with forced IDR and strict GOP:
```cpp
av_opt_set(out.venc->priv_data, "forced-idr", "1", 0);
av_opt_set(out.venc->priv_data, "strict_gop", "1", 0);
```

**Benefit**: Ensures every HLS segment starts with an I-frame for seamless seeking

#### Encoder Timebase Strategy (output_config.cpp:186-203)

**HLS mode**: Uses framerate-based timebase (1/fps) for muxer compatibility
```cpp
// Already set in ffmpeg_utils.cpp as 1/fps
LOG_INFO("HLS: Using framerate-based encoder timebase %d/%d",
         out.venc->time_base.num, out.venc->time_base.den);
```

**Non-HLS mode**: Uses input stream timebase for `-copyts` compatibility
```cpp
out.venc->time_base = in.vst->time_base;
LOG_DEBUG("Non-HLS: Using input stream timebase %d/%d for encoder",
          out.venc->time_base.num, out.venc->time_base.den);
```

**Why this matters**: FFmpeg's HLS muxer expects specific timebase formats to generate correct segment durations

### 6. TimestampManager Performance Optimizations

**Location**: src/timestamp_manager.h

The timestamp manager uses several performance optimization techniques:

#### Function Pointer Strategy (timestamp_manager.h:22)

**Technique**: Pre-computes mode-dependent behavior flags to avoid branches in hot path

```cpp
// Constructor pre-computes const flags (line 58)
const bool needs_baseline_wait_ = (config.input_seek_us > 0 && config.mode == Mode::NORMAL);
const int64_t frame_duration_us_ = av_rescale_q(1, av_inv_q(config.expected_frame_rate), {1, AV_TIME_BASE});
```

**Benefit**: Mode checks become simple const bool comparisons instead of enum switches

#### Pre-computed Frame Duration (timestamp_manager.h:58, 246)

Instead of computing frame duration on every monotonicity fix:
```cpp
// One-time calculation in constructor
frame_duration_us_ = av_rescale_q(1, av_inv_q(config.expected_frame_rate), {1, AV_TIME_BASE});

// Fast lookup during processing (line 445)
int64_t frame_duration = av_rescale_q(1, av_inv_q(cfg_.expected_frame_rate), out_tb);
```

**Benefit**: Eliminates repeated av_rescale_q() calls in tight encode loop

#### Elimination of Redundant Timebase Conversions (timestamp_manager.h:19)

For copied streams in NORMAL mode with seeking:
```cpp
// Convert to microseconds once, adjust, convert back once (lines 381-394)
int64_t pts_us = av_rescale_q(pkt->pts, in_tb, {1, AV_TIME_BASE});
int64_t adjusted_us = pts_us - global_baseline_us_;
pkt->pts = av_rescale_q(adjusted_us, {1, AV_TIME_BASE}, in_tb);
```

Instead of multiple rescale operations per timestamp field

### 7. CpuProcessor Dynamic Configuration

**Location**: src/processor.h:169-194

The `CpuProcessor` class supports runtime THDR configuration via `setConfig()` method:

**Use case**: Allows changing HDR output settings after processor initialization

```cpp
// Initial setup (main.cpp:518-522)
auto cpuProc = std::make_unique<CpuProcessor>(rtx, in.vdec->width,
                                              in.vdec->height, dstW, dstH);
RTXProcessConfig cpuConfig = cfg.rtxCfg;
cpuConfig.enableTHDR = outputHDR;
cpuProc->setConfig(cpuConfig);
```

**What setConfig() does** (processor.h:169-194):
1. Rebuilds SwsContext for colorspace conversion based on new THDR setting
2. Reallocates output frame buffer in correct format (P010LE for HDR, NV12 for SDR)
3. Updates colorspace details (BT.2020 vs BT.709)

**Current usage**: Called once during initialization. Could theoretically support mid-stream HDR toggling if needed for adaptive streaming scenarios.

**Note**: GpuProcessor does not have `setConfig()` - its HDR mode is determined at construction and immutable.

---

## Key Files Reference

| Component | Primary File | Supporting Files |
|-----------|-------------|------------------|
| **Main Pipeline** | src/main.cpp | - |
| **Config Parsing** | src/config_parser.cpp | src/config_parser.h |
| **Input Setup** | src/input_config.cpp | src/ffmpeg_utils.cpp (open_input) |
| **Output Setup** | src/output_config.cpp | src/ffmpeg_utils.cpp (open_output) |
| **Stream Mapping** | src/ffmpeg_utils.cpp | decide_stream_mappings() |
| **Timestamp Manager** | src/timestamp_manager.h | - |
| **Audio Processing** | src/ffmpeg_utils.cpp | process_audio_frame(), setup_audio_encoder() |
| **RTX Processing** | src/rtx_processor.cpp | src/processor.h |
| **Frame Pool** | src/frame_pool.h | - |
| **Logger** | src/logger.h | - |
| **Pipeline Types** | src/pipeline_types.h | InputContext, OutputContext, HlsMuxOptions |

---

## Debugging Tips

### Enable Verbose Logging

```bash
RTXVideoProcessor.exe -i input.mp4 -o output.mp4 -loglevel verbose
```

### Check Stream Decisions

Look for this log output from `decide_stream_mappings()`:
```
[DEBUG] Stream mapping decisions:
[DEBUG]   Stream 0 (video): PROCESS_VIDEO
[DEBUG]   Stream 1 (audio): PROCESS_AUDIO
[DEBUG]   Stream 2 (subtitle): EXCLUDE
```

### Verify Timestamp Manager State

At the end of processing:
```
[DEBUG] === TimestampManager State ===
[DEBUG]   Mode: NORMAL
[DEBUG]   Dropped frames: 0
[DEBUG]   Monotonicity violations: 0
```

### Check HLS Options

```
[DEBUG] Final HLS muxer options:
[DEBUG]   hls_time = 4
[DEBUG]   hls_segment_type = fmp4
[DEBUG]   hls_flags = +append_list
```

---

## Version History

- **v1.0** (2025-01-09): Initial complete pipeline documentation
  - Full end-to-end code paths
  - Phase-by-phase breakdown
  - Data flow diagrams
  - Processing modes explained

---

## See Also

- [FFMPEG_COMPATIBILITY.md](./FFMPEG_COMPATIBILITY.md) - FFmpeg vs Legacy behavior differences
- [README.md](../README.md) - User-facing documentation
- [src/pipeline_types.h](../src/pipeline_types.h) - Core data structures
