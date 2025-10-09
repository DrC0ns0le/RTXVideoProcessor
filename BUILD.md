# Build Instructions

## Development (Fast, Dynamic FFmpeg)
```bash
cmake -B build
cmake --build build --config Release
```
Needs FFmpeg DLLs at runtime.

## Production (Static FFmpeg for Distribution)
```bash
cmake -B build-static -DUSE_STATIC_FFMPEG=ON -DCMAKE_TOOLCHAIN_FILE="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows-static

cmake --build build-static --config Release
```
Self-contained binary. First build takes ~30min (vcpkg compiles FFmpeg).

## Deployment
- **Development**: exe + 2 RTX DLLs + 7 FFmpeg DLLs
- **Production**: exe + 2 RTX DLLs only
