# RTXVideoProcessor: Concepts and Glossary

A comprehensive reference guide to video processing concepts, FFmpeg terminology, and RTXVideoProcessor-specific features.

---

## Table of Contents

1. [Core Video Processing Concepts](#core-video-processing-concepts)
2. [FFmpeg Library Architecture](#ffmpeg-library-architecture)
3. [Timestamp and Synchronization](#timestamp-and-synchronization)
4. [Encoding and Compression](#encoding-and-compression)
5. [Streaming and Output Formats](#streaming-and-output-formats)
6. [RTX Video Processing](#rtx-video-processing)
7. [Glossary](#glossary)

---

## Core Video Processing Concepts

### Container vs Codec

**Container** (Format)
- A file format that packages video, audio, subtitles, and metadata together
- Examples: MP4, MKV, AVI, HLS (.m3u8)
- Think of it as a "box" that holds multiple streams
- RTXVideoProcessor supports: MP4, MKV, HLS

**Codec** (Encoder/Decoder)
- Algorithm for compressing and decompressing video/audio data
- Video codecs: H.264/AVC, H.265/HEVC, VP9, AV1
- Audio codecs: AAC, MP3, Opus, FLAC
- RTXVideoProcessor uses: HEVC (H.265) for video, AAC for audio

**Example**: An MP4 file (container) might contain HEVC video (codec) + AAC audio (codec) + WebVTT subtitles

### Demuxing and Muxing

**Demuxing** (De-multiplexing)
- Process of extracting individual streams from a container
- Reads container file and separates video, audio, subtitle packets
- Performed by `av_read_frame()` in FFmpeg
- RTXVideoProcessor demuxer: Reads input file and routes packets to decoders

**Muxing** (Multiplexing)
- Process of combining multiple streams into a single container
- Takes encoded video/audio packets and writes them to output file
- Performed by `av_interleaved_write_frame()` in FFmpeg
- RTXVideoProcessor muxer: Writes processed packets to output file

**Pipeline Flow**:
```
Input File → Demux → [Video Stream, Audio Stream, Subtitle Stream]
                           ↓             ↓              ↓
                        Decode        Decode         Copy
                           ↓             ↓              ↓
                        Process       Process         Copy
                           ↓             ↓              ↓
                        Encode        Encode          Copy
                           ↓             ↓              ↓
[Video Packets, Audio Packets, Subtitle Packets] → Mux → Output File
```

### Decoding and Encoding

**Decoding**
- Converting compressed video/audio data into raw uncompressed frames
- Input: Compressed packet (e.g., HEVC bitstream)
- Output: Raw frame (e.g., YUV 4:2:0 pixel data)
- RTXVideoProcessor uses NVDEC (NVIDIA hardware decoder) when available

**Encoding**
- Converting raw frames into compressed data
- Input: Raw frame (e.g., RGB or YUV data)
- Output: Compressed packet (e.g., HEVC bitstream)
- RTXVideoProcessor uses NVENC (NVIDIA hardware encoder)

**Why decode then re-encode?**
- To apply processing (RTX upscaling, tone mapping)
- To change codec (H.264 → HEVC)
- To change resolution or quality
- Note: Re-encoding always causes some quality loss (generation loss)

### Pixel Formats and Color Spaces

**Pixel Format**
- How pixel color data is stored in memory
- Common formats:
  - **RGB/RGBA**: Red, Green, Blue (+ Alpha) - 8 bits per channel
  - **YUV/YCbCr**: Luminance (Y) + Chrominance (U, V) - separates brightness from color
  - **NV12**: YUV 4:2:0 format (8-bit) - most common for video
  - **P010LE**: YUV 4:2:0 format (10-bit) - used for HDR
  - **X2BGR10LE**: 10-bit RGB packed format - used by RTX THDR output

**Color Space**
- Mathematical model for representing colors
- **BT.709** (Rec. 709): Standard for HD video (1080p)
- **BT.2020** (Rec. 2020): Standard for UHD/HDR video (4K+)
- **sRGB**: Standard for computer displays
- RTXVideoProcessor handles both BT.709 (SDR) and BT.2020 (HDR)

**Chroma Subsampling**
- Technique to reduce color information while preserving brightness
- Human eyes are more sensitive to brightness than color
- **4:4:4**: Full color resolution (no subsampling)
- **4:2:2**: Half horizontal color resolution
- **4:2:0**: Half horizontal + half vertical color resolution (most efficient)
- Most video uses 4:2:0 (NV12, P010)

### HDR vs SDR

**SDR (Standard Dynamic Range)**
- Traditional video with limited brightness range
- Typically 8-bit color depth
- Brightness: ~0.1 - 100 nits
- Color space: BT.709
- Most common format for streaming and broadcast

**HDR (High Dynamic Range)**
- Extended brightness and color range
- Typically 10-bit or 12-bit color depth
- Brightness: up to 1000-10,000 nits
- Color space: BT.2020
- Transfer functions: PQ (Perceptual Quantizer), HLG (Hybrid Log-Gamma)
- Used for premium content (4K Blu-ray, streaming services)

**RTXVideoProcessor HDR Handling**:
- Detects HDR input by checking color primaries, transfer function, color space
- Preserves HDR when input is HDR and THDR is disabled
- Converts SDR to HDR when THDR is enabled
- Auto-disables THDR for HDR inputs to preserve metadata

---

## FFmpeg Library Architecture

### Library Components

**libavformat**
- Handles container formats (muxing/demuxing)
- Reads and writes MP4, MKV, HLS, MPEG-TS, etc.
- Provides `AVFormatContext` for file I/O
- Key functions: `avformat_open_input()`, `av_read_frame()`, `av_write_frame()`

**libavcodec**
- Implements encoders and decoders
- Supports H.264, HEVC, AAC, MP3, etc.
- Provides `AVCodecContext` and `AVCodec`
- Key functions: `avcodec_send_frame()`, `avcodec_receive_packet()`, `avcodec_send_packet()`, `avcodec_receive_frame()`

**libavutil**
- Core utilities and data structures
- Memory management, mathematics, logging
- Provides `AVFrame`, `AVPacket`, `AVRational`, `AVDictionary`
- Key functions: `av_frame_alloc()`, `av_packet_alloc()`, `av_rescale_q()`

**libswscale**
- Video scaling and color space conversion
- Converts between pixel formats (RGB ↔ YUV)
- Resizes video frames (bilinear, bicubic, lanczos)
- Provides `SwsContext`
- Key function: `sws_scale()`

**libswresample**
- Audio resampling and format conversion
- Changes sample rate (48kHz → 44.1kHz)
- Converts channel layouts (stereo → 5.1)
- Provides `SwrContext`
- Key function: `swr_convert()`

**libavfilter**
- Audio and video filtering framework
- Applies effects, overlays, transitions
- Graph-based processing pipeline
- RTXVideoProcessor uses for audio filters

### Core Data Structures

**AVFormatContext**
```c
struct AVFormatContext {
    AVInputFormat *iformat;     // Input format (demuxer)
    AVOutputFormat *oformat;    // Output format (muxer)
    AVStream **streams;         // Array of streams
    unsigned int nb_streams;    // Number of streams
    AVIOContext *pb;            // I/O context
    int64_t duration;           // File duration (AV_TIME_BASE units)
    AVDictionary *metadata;     // File metadata
    int64_t output_ts_offset;   // Output timestamp offset
};
```

**Purpose**: Represents an opened media file (input or output)

**Usage in RTXVideoProcessor**:
```cpp
AVFormatContext *fmt_ctx = nullptr;
avformat_open_input(&fmt_ctx, "input.mp4", nullptr, nullptr);
avformat_find_stream_info(fmt_ctx, nullptr);
// Process streams...
avformat_close_input(&fmt_ctx);
```

**AVCodecContext**
```c
struct AVCodecContext {
    enum AVMediaType codec_type; // AVMEDIA_TYPE_VIDEO or AUDIO
    enum AVCodecID codec_id;     // AV_CODEC_ID_HEVC, etc.
    int width, height;           // Video dimensions
    enum AVPixelFormat pix_fmt;  // Pixel format
    AVRational time_base;        // Timebase for PTS/DTS
    int bit_rate;                // Target bitrate
    int gop_size;                // GOP length in frames
    int max_b_frames;            // Max B-frames
    AVDictionary *priv_data;     // Codec-specific options
};
```

**Purpose**: Encoder or decoder configuration

**Usage in RTXVideoProcessor**:
```cpp
AVCodecContext *enc_ctx = avcodec_alloc_context3(codec);
enc_ctx->width = 3840;
enc_ctx->height = 2160;
enc_ctx->pix_fmt = AV_PIX_FMT_P010LE;
enc_ctx->time_base = (AVRational){1, 60};
avcodec_open2(enc_ctx, codec, &opts);
```

**AVStream**
```c
struct AVStream {
    int index;                   // Stream index in file
    int id;                      // Stream ID
    AVCodecParameters *codecpar; // Codec parameters
    AVRational time_base;        // Stream timebase
    int64_t start_time;          // First PTS
    int64_t duration;            // Stream duration
    AVDictionary *metadata;      // Stream metadata
    AVRational avg_frame_rate;   // Average framerate
};
```

**Purpose**: Represents a single audio/video/subtitle stream in a file

**Usage in RTXVideoProcessor**:
```cpp
AVStream *video_stream = fmt_ctx->streams[video_index];
AVRational fps = av_guess_frame_rate(fmt_ctx, video_stream, nullptr);
```

**AVFrame**
```c
struct AVFrame {
    uint8_t *data[AV_NUM_DATA_POINTERS]; // Pointers to pixel/sample data
    int linesize[AV_NUM_DATA_POINTERS];  // Row stride (bytes per line)
    int width, height;           // Frame dimensions
    int format;                  // AVPixelFormat or AVSampleFormat
    int64_t pts;                 // Presentation timestamp
    int64_t best_effort_timestamp; // Fallback PTS
    AVBufferRef *buf[AV_NUM_DATA_POINTERS]; // Reference-counted buffers
    AVBufferRef *hw_frames_ctx;  // Hardware frames context
};
```

**Purpose**: Holds raw (uncompressed) audio or video data

**Usage in RTXVideoProcessor**:
```cpp
AVFrame *frame = av_frame_alloc();
avcodec_receive_frame(decoder, frame);  // Decode into frame
// Process frame...
avcodec_send_frame(encoder, frame);     // Encode from frame
av_frame_free(&frame);
```

**AVPacket**
```c
struct AVPacket {
    AVBufferRef *buf;            // Reference-counted buffer
    int64_t pts;                 // Presentation timestamp
    int64_t dts;                 // Decode timestamp
    uint8_t *data;               // Compressed data
    int size;                    // Data size in bytes
    int stream_index;            // Which stream this belongs to
    int flags;                   // Keyframe, etc.
    int64_t duration;            // Duration in timebase units
};
```

**Purpose**: Holds compressed audio or video data

**Usage in RTXVideoProcessor**:
```cpp
AVPacket *pkt = av_packet_alloc();
av_read_frame(fmt_ctx, pkt);      // Read from file
avcodec_send_packet(decoder, pkt); // Send to decoder
av_packet_unref(pkt);             // Release data
av_packet_free(&pkt);             // Free packet
```

**AVDictionary**
```c
typedef struct AVDictionary {
    // Internal implementation
    // Key-value pairs for options/metadata
} AVDictionary;
```

**Purpose**: Key-value store for options and metadata

**Usage in RTXVideoProcessor**:
```cpp
AVDictionary *opts = nullptr;
av_dict_set(&opts, "preset", "p7", 0);
av_dict_set(&opts, "tune", "hq", 0);
av_dict_set(&opts, "rc", "constqp", 0);
avcodec_open2(encoder, codec, &opts);
av_dict_free(&opts);
```

**AVRational**
```c
typedef struct AVRational {
    int num;  // Numerator
    int den;  // Denominator
} AVRational;
```

**Purpose**: Represents fractions (timebases, aspect ratios, frame rates)

**Usage in RTXVideoProcessor**:
```cpp
AVRational fps = {24000, 1001};  // 23.976 fps
AVRational timebase = {1, 90000}; // MPEG-TS timebase
double fps_float = av_q2d(fps);   // Convert to double
int64_t rescaled = av_rescale_q(pts, in_tb, out_tb);
```

### Memory Management

**Reference Counting**

FFmpeg uses reference counting to manage memory safely:

```cpp
// Allocate frame
AVFrame *frame = av_frame_alloc();  // Refcount = 1

// Share frame (increase refcount)
AVFrame *frame_ref = av_frame_clone(frame);  // Refcount = 2

// Release references
av_frame_free(&frame_ref);  // Refcount = 1
av_frame_free(&frame);      // Refcount = 0, memory freed
```

**Why Reference Counting?**
- Avoids unnecessary copies (efficient)
- Safe sharing across threads
- Automatic memory cleanup when last reference released

**AVBufferRef**

Core of FFmpeg's reference counting system:

```cpp
// Frame data is backed by AVBufferRef
AVFrame *frame = av_frame_alloc();
av_frame_get_buffer(frame, 32);  // Allocates AVBufferRef internally

// Each data plane can have its own buffer
AVBufferRef *buf = frame->buf[0];
int refcount = av_buffer_get_ref_count(buf);
```

**Frame Ownership**

Best practices for frame management:

```cpp
// Make frame writable before modifying
if (av_frame_make_writable(frame) < 0) {
    // Failed - frame is shared, couldn't make copy
}

// Transfer ownership (zero-copy)
AVFrame *dst = av_frame_alloc();
av_frame_move_ref(dst, src);  // src is now empty
```

**RAII in C++**

RTXVideoProcessor uses RAII wrappers:

```cpp
// Smart pointer for automatic cleanup
using FramePtr = std::unique_ptr<AVFrame, decltype(&av_frame_free_single)>;
FramePtr frame(av_frame_alloc(), av_frame_free_single);

// Automatic cleanup when frame goes out of scope
// No manual av_frame_free() needed
```

### Hardware Acceleration

**AVHWDeviceContext**

Represents a hardware device (GPU):

```cpp
AVBufferRef *hw_device_ctx = nullptr;
av_hwdevice_ctx_create(&hw_device_ctx,
                       AV_HWDEVICE_TYPE_CUDA,
                       nullptr,  // Device name (nullptr = default)
                       nullptr,  // Options
                       0);       // Flags

// Attach to decoder
decoder->hw_device_ctx = av_buffer_ref(hw_device_ctx);

// Cleanup
av_buffer_unref(&hw_device_ctx);
```

**AVHWFramesContext**

Manages a pool of hardware frames:

```cpp
// Create frames context
AVBufferRef *hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
AVHWFramesContext *hw_frames_ctx = (AVHWFramesContext*)hw_frames_ref->data;

// Configure
hw_frames_ctx->format = AV_PIX_FMT_CUDA;
hw_frames_ctx->sw_format = AV_PIX_FMT_P010LE;  // Underlying format
hw_frames_ctx->width = 3840;
hw_frames_ctx->height = 2160;
hw_frames_ctx->initial_pool_size = 64;  // Pre-allocate 64 frames

// Initialize
av_hwframe_ctx_init(hw_frames_ref);

// Use with encoder
encoder->hw_frames_ctx = av_buffer_ref(hw_frames_ref);
```

**Hardware Pixel Formats**

Special pixel formats for hardware acceleration:

- **AV_PIX_FMT_CUDA**: NVIDIA CUDA surfaces (GPU memory)
- **AV_PIX_FMT_D3D11**: Direct3D 11 textures
- **AV_PIX_FMT_VAAPI**: Intel VA-API surfaces
- **AV_PIX_FMT_VIDEOTOOLBOX**: Apple VideoToolbox surfaces

**Getting Hardware Frames**

```cpp
// Allocate frame on GPU
AVFrame *hw_frame = av_frame_alloc();
av_hwframe_get_buffer(hw_frames_ref, hw_frame, 0);

// Decode directly to GPU
avcodec_receive_frame(decoder, hw_frame);
// hw_frame->format == AV_PIX_FMT_CUDA
// hw_frame->data[0] contains GPU pointer (not CPU-accessible!)
```

**Transfer Between CPU and GPU**

```cpp
// GPU → CPU
AVFrame *sw_frame = av_frame_alloc();
av_hwframe_transfer_data(sw_frame, hw_frame, 0);
// sw_frame now has CPU-accessible pixel data

// CPU → GPU
AVFrame *hw_frame2 = av_frame_alloc();
av_hwframe_get_buffer(hw_frames_ref, hw_frame2, 0);
av_hwframe_transfer_data(hw_frame2, sw_frame, 0);
```

### Common FFmpeg Patterns

**Open Input File**

```cpp
AVFormatContext *fmt_ctx = nullptr;
int ret = avformat_open_input(&fmt_ctx, "input.mp4", nullptr, nullptr);
if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_make_error_string(errbuf, sizeof(errbuf), ret);
    fprintf(stderr, "Could not open input: %s\n", errbuf);
    return -1;
}

avformat_find_stream_info(fmt_ctx, nullptr);
```

**Find Best Stream**

```cpp
int video_idx = av_find_best_stream(fmt_ctx,
                                    AVMEDIA_TYPE_VIDEO,
                                    -1,  // Wanted stream number (-1 = auto)
                                    -1,  // Related stream (-1 = none)
                                    nullptr,  // Decoder out
                                    0);  // Flags
```

**Open Decoder**

```cpp
AVStream *stream = fmt_ctx->streams[video_idx];
const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
AVCodecContext *dec_ctx = avcodec_alloc_context3(codec);
avcodec_parameters_to_context(dec_ctx, stream->codecpar);
avcodec_open2(dec_ctx, codec, nullptr);
```

**Decode Loop**

```cpp
AVPacket *pkt = av_packet_alloc();
AVFrame *frame = av_frame_alloc();

while (av_read_frame(fmt_ctx, pkt) >= 0) {
    if (pkt->stream_index == video_idx) {
        avcodec_send_packet(dec_ctx, pkt);

        while (avcodec_receive_frame(dec_ctx, frame) == 0) {
            // Process frame
            printf("Frame PTS: %lld\n", frame->pts);
        }
    }
    av_packet_unref(pkt);
}

// Flush decoder
avcodec_send_packet(dec_ctx, nullptr);
while (avcodec_receive_frame(dec_ctx, frame) == 0) {
    // Process remaining frames
}
```

**Encode Loop**

```cpp
AVCodecContext *enc_ctx = /* configured encoder */;
AVFrame *frame = /* processed frame */;
AVPacket *pkt = av_packet_alloc();

avcodec_send_frame(enc_ctx, frame);

while (avcodec_receive_packet(enc_ctx, pkt) == 0) {
    // Rescale timestamps
    av_packet_rescale_ts(pkt, enc_ctx->time_base, stream->time_base);
    pkt->stream_index = stream->index;

    // Write to output
    av_interleaved_write_frame(out_fmt_ctx, pkt);
    av_packet_unref(pkt);
}
```

**Error Handling**

```cpp
int ret = some_ffmpeg_function();
if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_make_error_string(errbuf, sizeof(errbuf), ret);
    LOG_ERROR("Operation failed: %s", errbuf);
    return ret;
}
```

---

## Timestamp and Synchronization

### Presentation Timestamp (PTS)

**Definition**
- Time when a frame/sample should be presented to the user
- Measured in **timebase** units (not seconds)
- Used to synchronize audio and video

**Example**:
```
Frame PTS: 90000
Timebase: 1/30000
Real time: 90000 * (1/30000) = 3.0 seconds
```

**PTS in RTXVideoProcessor**:
- Derived by TimestampManager
- Two modes: NORMAL (zero-based) and COPYTS (preserve original)
- Used by encoder to determine frame ordering

### Decode Timestamp (DTS)

**Definition**
- Time when a frame should be decoded (not presented)
- Always <= PTS for the same frame
- Required for video codecs with B-frames (bidirectional prediction)

**Why DTS ≠ PTS?**
- B-frames reference both past and future frames
- Must decode in different order than presentation order
- Example (decode order): I, P, B, B, P
- Example (display order): I, B, B, P, P

**RTXVideoProcessor Handling**:
- **Encoder generates DTS automatically** based on frame reordering
- Pipeline only sets PTS on input frames
- Muxer validates DTS <= PTS constraint

### Timebase

**Definition**
- Fractional unit used to represent timestamps
- Format: `numerator / denominator` (e.g., 1/30000)
- Different streams can have different timebases

**Common Timebases**:
- **1/90000**: MPEG-TS standard
- **1/30000**: Common for 30 fps video
- **1/48000**: Audio at 48 kHz sample rate
- **1/AV_TIME_BASE** (1/1000000): Microseconds

**Timebase Conversion**:
```cpp
// Convert PTS from input timebase to output timebase
int64_t out_pts = av_rescale_q(in_pts, in_timebase, out_timebase);
```

**RTXVideoProcessor Usage**:
- Input streams: Use original timebase from input file
- Encoder: Uses framerate-based timebase (1/fps) for HLS, input timebase for others
- Muxer: Converts all timestamps to stream timebases

### A/V Sync (Audio-Video Synchronization)

**Concept**
- Ensuring audio and video play in perfect sync
- Achieved by matching PTS values across streams
- Drift tolerance: typically ±20ms before noticeable

**Challenges**:
- Different timebases for audio vs video
- Variable frame rates
- Seeking operations
- Network streaming delays

**RTXVideoProcessor Approach**:
- Uses timestamp manager for consistent PTS derivation
- Relies on FFmpeg's muxer for A/V interleaving
- COPYTS mode preserves original sync
- NORMAL mode establishes new baseline from first frame

### Monotonicity

**Definition**
- Requirement that timestamps always increase
- PTS(frame N+1) > PTS(frame N)
- DTS(packet N+1) > DTS(packet N)
- Violation causes muxer errors

**Why Required?**
- Players expect chronological order
- Seeking depends on sorted timestamps
- Buffering and interleaving require predictable order

**RTXVideoProcessor Handling**:
- Encoder ensures PTS/DTS monotonicity internally
- Muxer (`av_interleaved_write_frame`) validates and may reject violations
- Simple Mode: Would auto-fix violations (legacy behavior)
- FFmpeg-Compatible Mode: Reports violations without auto-fix

---

## Encoding and Compression

### Quantization Parameter (QP)

**Definition**
- Controls compression level in video encoding
- Lower QP = higher quality, larger file size
- Higher QP = lower quality, smaller file size
- Range: 0 (lossless) to 51 (maximum compression)

**QP Values**:
- **0-18**: Visually lossless to near-lossless
- **19-23**: High quality (recommended for archival)
- **24-28**: Good quality (streaming/broadcast)
- **29-35**: Medium quality (web video)
- **36-51**: Low quality (not recommended)

**RTXVideoProcessor Default**: QP 21 (high quality, constant quality mode)

### Rate Control Modes

**Constant QP (CQP)**
- Maintains fixed quality level
- File size varies based on content complexity
- Best for: Archival, predictable quality
- RTXVideoProcessor default mode

**Constant Bitrate (CBR)**
- Maintains fixed bitrate
- Quality varies based on content complexity
- Best for: Live streaming, bandwidth-limited scenarios

**Variable Bitrate (VBR)**
- Adjusts bitrate to maintain target quality
- More efficient than CBR
- Best for: File-based delivery, VOD

**Average Bitrate (ABR)**
- Targets average bitrate over time
- Allows short-term bitrate spikes
- Best for: Streaming with buffering

### GOP (Group of Pictures)

**Definition**
- Sequence of frames between two I-frames
- Structure determines compression efficiency and seek accuracy

**Frame Types**:
- **I-frame (Intra)**: Complete frame, no dependencies (keyframe)
- **P-frame (Predicted)**: References previous frames
- **B-frame (Bidirectional)**: References both past and future frames

**GOP Structure Example**:
```
I B B P B B P B B I
↑                   ↑
|---- GOP length ----|
```

**GOP Length**:
- Shorter GOP: More I-frames, easier seeking, larger file size
- Longer GOP: Fewer I-frames, harder seeking, smaller file size
- RTXVideoProcessor: 3 seconds × framerate (e.g., 72 frames for 24 fps)

**Why 3 seconds for HLS?**
- HLS segment duration is typically 4-6 seconds
- Each segment should start with an I-frame
- GOP = segment duration ensures clean segment boundaries

### B-Frames

**Definition**
- Frames that reference both previous and future frames
- Highest compression efficiency
- Increases encoding complexity and latency

**Benefits**:
- Better compression (smaller file size)
- Smoother motion prediction

**Drawbacks**:
- Increases encoding latency (need future frames)
- Complicates seeking (need to decode multiple frames)
- DTS ≠ PTS (reordering required)

**RTXVideoProcessor**: Uses 2 B-frames by default (balanced approach)

### NVENC (NVIDIA Encoder)

**What is NVENC?**
- Hardware video encoder built into NVIDIA GPUs
- Dedicated silicon for encoding (doesn't use CUDA cores)
- Much faster than software encoding (x264, x265)
- Quality comparable to medium-to-fast software presets

**Presets**:
- **p1**: Fastest, lowest quality
- **p4**: Balanced (old default)
- **p7**: Slower, highest quality (RTXVideoProcessor default)

**Profiles**:
- **Main**: 8-bit, most compatible
- **Main10**: 10-bit, HDR support (RTXVideoProcessor uses this)
- **Main12**: 12-bit (rarely used)

**Tuning**:
- **hq**: High quality (RTXVideoProcessor default)
- **ll**: Low latency (for streaming)
- **ull**: Ultra-low latency (for game streaming)

### NVDEC (NVIDIA Decoder)

**What is NVDEC?**
- Hardware video decoder built into NVIDIA GPUs
- Offloads CPU decoding to GPU
- Supports H.264, HEVC, VP9, AV1

**Benefits**:
- Frees CPU for other tasks
- Lower power consumption
- Enables GPU-only pipeline (no CPU↔GPU transfers)

**RTXVideoProcessor Usage**:
- Tries NVDEC first
- Falls back to software decoding if unavailable
- Decoded frames stay on GPU for RTX processing

---

## Streaming and Output Formats

### HLS (HTTP Live Streaming)

**What is HLS?**
- Adaptive bitrate streaming protocol developed by Apple
- Breaks video into small segments (typically 4-10 seconds)
- Playlist file (.m3u8) lists segment files
- Widely supported (iOS, Android, web browsers)

**Components**:
- **Playlist (.m3u8)**: Text file listing segments
- **Segments (.ts or .m4s)**: Video/audio chunks
- **Init file (.mp4)**: Initialization segment for fMP4

**Segment Types**:
- **MPEG-TS (.ts)**: Traditional format, widely compatible
- **fMP4 (.m4s)**: Fragmented MP4, better for modern players

**RTXVideoProcessor HLS Support**:
- Auto-detects .m3u8 output
- Supports both MPEG-TS and fMP4 segments
- Configurable segment duration (default: 4 seconds)
- GOP aligned to segment boundaries

### fMP4 (Fragmented MP4)

**Definition**
- MP4 container split into small fragments
- Each fragment can be decoded independently
- Designed for streaming

**Structure**:
```
Init Segment (header) → Fragment 1 → Fragment 2 → Fragment 3 → ...
```

**Benefits over regular MP4**:
- Enables live streaming (no need to finish file first)
- Supports seeking without complete file download
- Better error recovery

**RTXVideoProcessor Usage**:
- Default for HLS output (hls_segment_type=fmp4)
- Requires `movflags=+frag_discont` for hls.js compatibility
- Uses `baseMediaDecodeTime` for timestamp continuity

### Playlist Types

**VOD (Video on Demand)**
- Complete, static playlist
- Contains #EXT-X-ENDLIST tag
- All segments are available upfront
- RTXVideoProcessor default for file-based HLS

**Event**
- Growing playlist
- No #EXT-X-ENDLIST (until stream ends)
- New segments appended over time
- For live events with DVR

**Live**
- Sliding window playlist
- Old segments removed as new ones added
- For continuous live streams
- Not typically used with RTXVideoProcessor

---

## RTX Video Processing

### RTX Video Super Resolution (VSR)

**What is VSR?**
- AI-powered upscaling using NVIDIA RTX GPUs
- Part of NVIDIA RTX Video SDK
- Uses dedicated Tensor Cores for AI inference

**How it works**:
- Trained on millions of video frames
- Reconstructs high-frequency details lost in compression
- Different from simple bicubic/lanczos scaling

**Quality Levels**:
- **1**: Fastest, lowest quality improvement
- **2**: Balanced
- **3**: Better quality
- **4**: Best quality (RTXVideoProcessor default)

**RTXVideoProcessor Behavior**:
- Automatically upscales by 2× (1080p → 4K)
- Auto-disables for inputs >1440p (already high resolution)
- Can be disabled with `--no-vsr` or `RTX_NO_VSR=1`

**Performance**:
- Much faster than software upscaling (ESRGAN, Waifu2x)
- Real-time capable on RTX 20-series and newer
- Uses GPU hardware, minimal CPU usage

### RTX TrueHDR (THDR)

**What is THDR?**
- AI-powered SDR to HDR conversion
- Part of NVIDIA RTX Video SDK
- Creates HDR metadata and expands dynamic range

**How it works**:
- Analyzes scene brightness and color
- Maps SDR luminance (0-100 nits) to HDR (0-1000+ nits)
- Adjusts saturation and contrast for HDR displays
- Generates HDR10 metadata (MaxCLL, MaxFALL)

**Parameters**:
- **Contrast**: Intensity of highlights (0-200, default 115)
- **Saturation**: Color intensity (0-200, default 75)
- **Middle Gray**: Midpoint mapping (0-100, default 30)
- **Max Luminance**: Peak brightness in nits (default 1000)

**RTXVideoProcessor Behavior**:
- Auto-disables for HDR inputs (preserves original HDR)
- Outputs 10-bit HEVC Main10 profile
- Can be disabled with `--no-thdr` or `RTX_NO_THDR=1`

**Output Format**:
- Pixel format: P010LE (10-bit YUV 4:2:0)
- Color space: BT.2020
- Transfer function: PQ (SMPTE ST 2084)

### GPU Pipeline

**Zero-Copy Pipeline**:
```
Input File
    ↓
NVDEC Decode (GPU) → CUDA Frame (GPU memory)
    ↓
RTX VSR (GPU) → Upscaled Frame (GPU)
    ↓
RTX THDR (GPU) → HDR Frame (GPU)
    ↓
Color Conversion (CUDA kernels) → P010/NV12 (GPU)
    ↓
NVENC Encode (GPU) → Compressed Packet
    ↓
Muxer → Output File
```

**Benefits**:
- No CPU↔GPU memory transfers (except final packets)
- Maximum throughput
- Low latency
- Minimal CPU usage

**CPU Fallback**:
- Used when NVDEC fails or for unsupported codecs
- Frames transferred to CPU, processed, then uploaded to GPU
- Slower but more compatible

---

## Glossary

### A

**AAC (Advanced Audio Codec)**
- Lossy audio codec, successor to MP3
- Standard for video containers (MP4, HLS)
- RTXVideoProcessor default audio codec

**Adaptive Bitrate Streaming (ABR)**
- Streaming technique with multiple quality levels
- Player switches based on bandwidth
- HLS and DASH are ABR protocols

**av_interleaved_write_frame()**
- FFmpeg function to write packets to output
- Handles buffering and stream interleaving
- Ensures proper A/V sync in output file

**AVCodecContext**
- FFmpeg structure representing encoder/decoder configuration
- Contains settings like bitrate, framerate, pixel format

**AVFormatContext**
- FFmpeg structure representing input/output file
- Contains streams, metadata, muxer/demuxer

**AVFrame**
- FFmpeg structure representing raw audio/video frame
- Contains pixel data, sample data, timestamps

**AVPacket**
- FFmpeg structure representing compressed data
- Contains encoded video/audio bitstream
- Includes PTS, DTS, duration

**AVSEEK_FLAG_BACKWARD**
- Seek to nearest keyframe before target
- Default FFmpeg seeking behavior

**AVSEEK_FLAG_ANY**
- Allow seeking to non-keyframes (faster but less accurate)
- Enabled by `-seek2any` or `-noaccurate_seek`

**av_rescale_q()**
- FFmpeg function to convert timestamps between timebases
- Essential for timestamp management

**AVBufferRef**
- Reference-counted buffer for memory management
- Core of FFmpeg's zero-copy system
- Used internally by AVFrame and AVPacket

**AVHWDeviceContext**
- FFmpeg structure representing hardware device (GPU)
- Created with `av_hwdevice_ctx_create()`
- Attached to decoder/encoder for hardware acceleration

**AVHWFramesContext**
- Pool of hardware-allocated frames
- Manages GPU memory for frames
- Contains pixel format, dimensions, pool size

**av_find_best_stream()**
- Finds best stream of specified type
- Automatically selects default video/audio track
- Returns stream index

**av_frame_make_writable()**
- Ensures frame can be modified
- Creates copy if frame is shared (refcount > 1)
- Required before editing frame data

**av_hwframe_transfer_data()**
- Transfers frame data between CPU and GPU
- GPU→CPU or CPU→GPU
- Used when mixing hardware and software processing

**av_interleaved_write_frame()**
- Writes packet to output file
- Handles buffering and stream interleaving
- Ensures correct A/V sync in output

**av_packet_rescale_ts()**
- Converts packet timestamps between timebases
- Updates PTS, DTS, and duration
- Used when moving packets between streams

**av_read_frame()**
- Reads next packet from input file
- Returns compressed data (AVPacket)
- Demuxes container into individual streams

**avcodec_parameters_to_context()**
- Copies codec parameters to decoder/encoder context
- Used when initializing decoder from stream
- Essential for proper codec configuration

**avcodec_receive_frame()**
- Retrieves decoded frame from decoder
- Non-blocking (returns EAGAIN if not ready)
- Part of modern send/receive API

**avcodec_receive_packet()**
- Retrieves encoded packet from encoder
- Non-blocking (returns EAGAIN if not ready)
- Part of modern send/receive API

**avcodec_send_frame()**
- Sends raw frame to encoder
- Non-blocking (buffers internally)
- Part of modern send/receive API

**avcodec_send_packet()**
- Sends compressed packet to decoder
- Non-blocking (buffers internally)
- Part of modern send/receive API

**avformat_find_stream_info()**
- Analyzes file to detect stream parameters
- Reads several frames to determine codec, duration, etc.
- Required after `avformat_open_input()`

**avformat_open_input()**
- Opens input file and reads header
- Allocates AVFormatContext
- First step in processing a file

**avformat_write_header()**
- Writes output file header
- Must be called before writing packets
- Finalizes stream parameters

**av_write_trailer()**
- Writes output file trailer
- Finalizes and closes output file
- Flushes remaining buffered data

### B

**Baseline**
- Reference timestamp for relative PTS calculation
- NORMAL mode: PTS of first decoded frame
- COPYTS mode: Optional offset with `-start_at_zero`

**Bitrate**
- Amount of data per second in video/audio
- Measured in kbps (kilobits per second) or Mbps (megabits per second)
- Higher bitrate = better quality, larger file

**BT.601, BT.709, BT.2020**
- ITU standards for color space and gamma
- BT.601: SD video (480p/576p)
- BT.709: HD video (720p/1080p)
- BT.2020: UHD/HDR video (4K+)

### C

**Chroma**
- Color information in video (U and V components in YUV)
- Separate from luma (brightness)

**Codec**
- Compressor/Decompressor algorithm
- Examples: H.264, HEVC, AAC, Opus

**Container**
- File format wrapping multiple streams
- Examples: MP4, MKV, AVI, WebM

**COPYTS Mode**
- Timestamp mode that preserves original timestamps
- Activated with `-copyts` flag
- Useful for maintaining sync when cutting video

**CUDA**
- NVIDIA parallel computing platform
- Used for custom kernels (color conversion, scaling)
- Not the same as NVENC/NVDEC (dedicated hardware)

### D

**Demuxer**
- Component that reads container and extracts streams
- Opposite of muxer

**DTS (Decode Timestamp)**
- When a frame should be decoded
- Different from PTS for codecs with B-frames

**Duration**
- Length of media in time units
- Can be specified with `-t` option to limit output

### E

**Encoder**
- Converts raw frames to compressed bitstream
- RTXVideoProcessor uses NVENC (hardware encoder)

**EOF (End of File)**
- Signal that input stream has ended
- Triggers encoder flushing

### F

**FFmpeg**
- Open-source multimedia framework
- Provides libavcodec, libavformat, libavutil
- RTXVideoProcessor uses FFmpeg libraries

**FIFO (First In, First Out)**
- Buffer for audio samples
- Ensures fixed-size frames for AAC encoder

**Flushing**
- Sending nullptr to encoder/decoder to retrieve buffered frames
- Required at end of stream to get remaining output
- Example: `avcodec_send_frame(encoder, nullptr)`

**Framerate**
- Frames per second (fps)
- Common: 23.976, 24, 25, 29.97, 30, 50, 59.94, 60

**Fragmented MP4 (fMP4)**
- MP4 split into independently decodable fragments
- Used for HLS streaming

### G

**GOP (Group of Pictures)**
- Sequence of frames between I-frames
- Affects compression efficiency and seeking

### H

**Hardware Acceleration**
- Using dedicated GPU hardware (NVENC/NVDEC)
- Much faster than CPU (software) encoding/decoding

**HEVC (High Efficiency Video Coding)**
- Also known as H.265
- Successor to H.264, 50% better compression
- RTXVideoProcessor output codec

**HLS (HTTP Live Streaming)**
- Apple's adaptive streaming protocol
- Uses .m3u8 playlists and segment files

**HLG (Hybrid Log-Gamma)**
- HDR transfer function
- Backward compatible with SDR displays

### I

**I-frame (Intra-frame)**
- Complete frame, no dependencies
- Also called keyframe
- Required for seeking

**Interleaving**
- Mixing video and audio packets in chronological order
- Ensures smooth playback

### K

**Keyframe**
- See I-frame
- Essential for seeking and HLS segments

### L

**Latency**
- Delay between input and output
- B-frames increase latency (need future frames)

**libavcodec**
- FFmpeg library for encoders and decoders
- Supports 100+ codecs
- Core of video/audio processing

**libavformat**
- FFmpeg library for container formats
- Handles muxing and demuxing
- Supports MP4, MKV, HLS, etc.

**libavutil**
- FFmpeg utility library
- Provides AVFrame, AVPacket, memory functions
- Foundation for other FFmpeg libraries

**libswresample**
- FFmpeg audio resampling library
- Converts sample rate and channel layout
- Provides SwrContext

**libswscale**
- FFmpeg video scaling library
- Converts pixel formats and resizes frames
- Provides SwsContext

**Linesize**
- Bytes per row in a frame
- May be larger than width × bytes_per_pixel (padding)
- Stored in AVFrame->linesize[]

**Lossless**
- Compression with no quality loss
- Examples: FLAC (audio), FFV1 (video)
- Much larger file sizes

**Lossy**
- Compression with some quality loss
- Examples: AAC, MP3 (audio), H.264, HEVC (video)
- Smaller file sizes

**Luma**
- Brightness information in video (Y component in YUV)

### M

**Main10 Profile**
- HEVC profile supporting 10-bit color
- Required for HDR
- RTXVideoProcessor uses Main10

**Metadata**
- Information about media (title, duration, codec)
- Stored in container format

**Monotonicity**
- Requirement that timestamps always increase
- Violated timestamps cause muxer errors

**MPEG-TS (MPEG Transport Stream)**
- Container format for HLS segments
- Alternative to fMP4

**Muxer**
- Component that combines streams into container
- Opposite of demuxer

### N

**NV12**
- YUV 4:2:0 pixel format, 8-bit
- Most common format for video encoding
- Used for SDR output

**NVDEC**
- NVIDIA hardware video decoder
- Part of GPU, separate from CUDA cores

**NVENC**
- NVIDIA hardware video encoder
- Part of GPU, separate from CUDA cores

### P

**P010LE**
- YUV 4:2:0 pixel format, 10-bit, little-endian
- Used for HDR video
- RTXVideoProcessor HDR output format

**Packet**
- Compressed audio/video data unit
- Contains PTS, DTS, duration, data

**P-frame (Predicted frame)**
- Frame referencing previous frames
- Better compression than I-frames

**Pixel Format**
- How pixel data is arranged in memory
- Examples: RGB, YUV, NV12, P010LE

**Playlist**
- M3U8 file listing HLS segments
- Types: VOD, Event, Live

**PQ (Perceptual Quantizer)**
- HDR transfer function (SMPTE ST 2084)
- Maps luminance for HDR displays
- Used in HDR10

**Preset**
- Encoder speed/quality tradeoff setting
- NVENC presets: p1 (fast) to p7 (slow, best quality)

**PTS (Presentation Timestamp)**
- When a frame/sample should be presented
- Key to A/V synchronization

### Q

**QP (Quantization Parameter)**
- Encoding quality setting (0-51)
- Lower = better quality, larger file
- RTXVideoProcessor default: 21

### R

**RAII (Resource Acquisition Is Initialization)**
- C++ pattern for automatic resource management
- Resources freed when object goes out of scope
- RTXVideoProcessor uses smart pointers for FFmpeg objects

**Rate Control**
- How encoder manages bitrate/quality
- Modes: CQP, CBR, VBR, ABR

**Reference Counting**
- Memory management technique tracking number of references
- FFmpeg uses for AVFrame, AVPacket data
- Automatically frees memory when refcount reaches 0

**Rescaling**
- Converting timestamps between timebases
- Also refers to resizing video resolution

**RTX Video SDK**
- NVIDIA SDK for AI video processing
- Includes VSR and THDR

### S

**Sample**
- Single audio data point
- Audio is a sequence of samples

**SDR (Standard Dynamic Range)**
- Traditional video (8-bit, limited brightness)
- Contrast to HDR

**Seeking**
- Jumping to specific time in video
- Requires keyframes for accuracy

**Segment**
- Individual file in HLS stream
- Typically 4-10 seconds long

**Stream**
- Individual audio, video, or subtitle track
- Container can have multiple streams

**Stream Mapping**
- Selecting which input streams go to output
- Configured with `-map` flag

**SwrContext**
- FFmpeg audio resampler context
- Converts sample rate, channel layout, format

**SwsContext**
- FFmpeg video scaler context
- Converts pixel format, resolution, color space

### T

**Tensor Cores**
- AI acceleration hardware in NVIDIA RTX GPUs
- Used by RTX VSR for super resolution

**Timebase**
- Fractional unit for timestamps
- Format: numerator/denominator (e.g., 1/30000)

**Timestamp**
- Time value attached to frames/packets
- Types: PTS, DTS

**Tone Mapping**
- Converting between SDR and HDR
- RTX THDR performs SDR→HDR

**Transfer Function**
- Gamma curve for luminance encoding
- Examples: sRGB, PQ, HLG

### U

**Upscaling**
- Increasing video resolution
- RTX VSR: AI-powered upscaling (2×)

### V

**VOD (Video on Demand)**
- Pre-recorded content (not live)
- HLS playlist type for completed videos

**VSR (Video Super Resolution)**
- NVIDIA RTX AI upscaling feature

### Y

**YUV**
- Color space separating brightness (Y) and color (UV)
- More efficient for compression than RGB
- Used in most video codecs

### Z

**Zero-Copy Pipeline**
- Processing without unnecessary memory copies
- GPU pipeline: data stays on GPU throughout

---

## Additional Resources

- [PIPELINE_ARCHITECTURE.md](./PIPELINE_ARCHITECTURE.md) - Detailed pipeline documentation
- [FFMPEG_COMPATIBILITY.md](./FFMPEG_COMPATIBILITY.md) - FFmpeg compatibility notes
- [README.md](../README.md) - User guide and setup instructions
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html) - Official FFmpeg docs
- [NVIDIA RTX Video SDK](https://developer.nvidia.com/rtx/video) - RTX Video documentation

---

**Document Version**: 1.0 (2025-01-21)
