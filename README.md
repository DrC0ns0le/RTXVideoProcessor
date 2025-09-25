# RTXVideoProcessor

A Windows 11 C++17 CLI tool that processes H.264 MP4 input using NVIDIA RTX Video SDK (VSR + TrueHDR) and outputs HDR HEVC (Main10) MP4. Audio and subtitles are preserved.

Pipeline:
- Demux/Decode: FFmpeg with CUDA hwaccel (NVDEC)
- Process: RTX VSR (2x, quality 4) then TrueHDR (default params)
- Encode: NVENC HEVC Main10 VBR preset P4, target bitrate = 2x input bitrate
- Mux: MP4 with HDR10 metadata (BT.2020 + PQ + Mastering + CLL)

Usage:
```
RTXVideoProcessor.exe input.mp4 output.mp4
```

## Prerequisites
- Windows 11 x64, NVIDIA RTX 4070 (or newer recommended)
- CUDA Toolkit 12.8 installed
- NVIDIA RTX Video SDK (headers + lib)
  - Set environment variable `NV_RTX_VIDEO_SDK` to the SDK root (contains `include/` and `lib/`)
- NVIDIA Video Codec SDK (NVENC/NVDEC)
  - Set environment variable `NV_VIDEO_CODEC_SDK` to the SDK root (contains `include/`, `Interface/`, and `Lib/x64/` or `lib/x64/`)
- FFmpeg prebuilt at `C:\ffmpeg` with the following layout:
  - `C:\ffmpeg\include` (headers)
  - `C:\ffmpeg\lib` (import libs: avcodec.lib, avformat.lib, avutil.lib, swscale.lib, swresample.lib, avfilter.lib)
  - `C:\ffmpeg\bin` (runtime DLLs)

## Build
Using CMake (MSVC 2022):
```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j
```
The resulting binary will be at:
```
build/Release/RTXVideoProcessor.exe
```

Copy FFmpeg runtime DLLs next to the executable if they are not on your PATH:
```
copy C:\ffmpeg\bin\*.dll build\Release\
```

## Notes
- Defaults: VSR enabled (2x, quality 4), TrueHDR enabled (Contrast=100, Saturation=100, MiddleGray=50, MaxLuminance=650).
- Output color: BT.2020 + PQ (HDR10 signaling). Mastering metadata and CLL are added to the video stream.
- Framerate is preserved; bitrate target is doubled from input container bitrate when available (fallback 25 Mbps).
- Audio and subtitle streams are copied (passthrough) without re-encoding.

## Known considerations
- Color channel ordering for the intermediate 10‑bit packed RGBA surface can vary by driver/SDK. If colors appear swapped (e.g., blue/red), we can flip channel order in the pack/unpack stage. The current implementation assumes a packed 10‑bit ABGR layout compatible with FFmpeg `AV_PIX_FMT_X2BGR10LE`.
- If your input decoder does not negotiate CUDA hwaccel, FFmpeg will fall back to software decode automatically.
- Ensure your NVIDIA driver is up to date for TrueHDR support on CUDA arrays with UNORM 10:10:10:2.
