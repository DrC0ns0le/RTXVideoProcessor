cmake_minimum_required(VERSION 3.24)
project(RTXVideoProcessor LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target recent NVIDIA architectures (adjust if needed). 89 = Ada (RTX 40 series)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Use static MSVC runtime (/MT for Release, /MTd for Debug)
# Requires CMake >= 3.15; this project requires 3.24 at top, so it's safe.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Help IDEs (VS Code, CLion) with accurate IntelliSense by exporting compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ensure Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "FFmpeg libs:\n  avformat=${FFMPEG_AVFORMAT_LIB}\n  avcodec=${FFMPEG_AVCODEC_LIB}\n  avfilter=${FFMPEG_AVFILTER_LIB}\n  avutil=${FFMPEG_AVUTIL_LIB}\n  swscale=${FFMPEG_SWSCALE_LIB}\n  swresample=${FFMPEG_SWRESAMPLE_LIB}")

# CUDA (Toolkit 12.x)
find_package(CUDAToolkit REQUIRED)

# Find packages
find_package(PkgConfig)

# FFmpeg - Try to find using pkg-config first, then manual search
if(PKG_CONFIG_FOUND)
    pkg_check_modules(FFMPEG QUIET
        libavformat>=60.0
        libavcodec>=60.0 
        libavutil>=58.0
        libswscale>=7.0
    )
endif()

# Required FFmpeg libraries
set(FFMPEG_LIBS
  avformat
  avcodec
  avfilter
  avutil
  swscale
  swresample
)

# FFmpeg from C:\\ffmpeg
set(FFMPEG_ROOT "C:/ffmpeg" CACHE PATH "Path to FFmpeg prebuilt root containing include/ and lib/")
set(FFMPEG_INCLUDE_DIR "${FFMPEG_ROOT}/include")
set(FFMPEG_LIB_DIR "${FFMPEG_ROOT}/lib")

# Resolve FFmpeg import libraries explicitly to avoid linker "cannot open input file" issues
find_library(FFMPEG_AVFORMAT_LIB  NAMES avformat avformat.lib avformat-61 avformat-61.lib avformat-60 avformat-60.lib PATHS ${FFMPEG_LIB_DIR} NO_DEFAULT_PATH)
find_library(FFMPEG_AVCODEC_LIB   NAMES avcodec  avcodec.lib  avcodec-61  avcodec-61.lib  avcodec-60  avcodec-60.lib  PATHS ${FFMPEG_LIB_DIR} NO_DEFAULT_PATH)
find_library(FFMPEG_AVFILTER_LIB  NAMES avfilter avfilter.lib avfilter-10 avfilter-10.lib avfilter-9  avfilter-9.lib  PATHS ${FFMPEG_LIB_DIR} NO_DEFAULT_PATH)
find_library(FFMPEG_AVUTIL_LIB    NAMES avutil   avutil.lib   avutil-59   avutil-59.lib   avutil-58   avutil-58.lib   PATHS ${FFMPEG_LIB_DIR} NO_DEFAULT_PATH)
find_library(FFMPEG_SWSCALE_LIB   NAMES swscale  swscale.lib  swscale-8   swscale-8.lib   swscale-7   swscale-7.lib   PATHS ${FFMPEG_LIB_DIR} NO_DEFAULT_PATH)
find_library(FFMPEG_SWRESAMPLE_LIB NAMES swresample swresample.lib swresample-5 swresample-5.lib swresample-4 swresample-4.lib PATHS ${FFMPEG_LIB_DIR} NO_DEFAULT_PATH)

if(NOT FFMPEG_AVFORMAT_LIB OR NOT FFMPEG_AVCODEC_LIB OR NOT FFMPEG_AVFILTER_LIB OR NOT FFMPEG_AVUTIL_LIB OR NOT FFMPEG_SWSCALE_LIB OR NOT FFMPEG_SWRESAMPLE_LIB)
  message(FATAL_ERROR "FFmpeg import libraries not found in ${FFMPEG_LIB_DIR}.\n"
    "Expected files like avformat.lib, avcodec.lib, avutil.lib, swscale.lib, swresample.lib.\n"
    "Set FFMPEG_ROOT to a directory with include/ and lib/ from an FFmpeg development build (MSVC). Example: -DFFMPEG_ROOT=C:/ffmpeg")
endif()
set(FFMPEG_LIB_DIR "${FFMPEG_ROOT}/lib")

# RTX Video SDK via env var NV_RTX_VIDEO_SDK
if(NOT DEFINED ENV{NV_RTX_VIDEO_SDK})
  message(FATAL_ERROR "Environment variable NV_RTX_VIDEO_SDK not set. Point it to the RTX Video SDK root (contains include/ lib/)")
endif()
set(RTX_SDK_ROOT $ENV{NV_RTX_VIDEO_SDK})
set(RTX_SDK_INCLUDE "${RTX_SDK_ROOT}/include")
set(RTX_SDK_LIB_DIR "${RTX_SDK_ROOT}/lib")
set(RTX_VIDEO_API_INCLUDE_DIR "${RTX_SDK_ROOT}/samples/RTX_Video_API")
# Correct bin path for redistributable feature DLLs (TrueHDR/VSR)
set(RTX_VIDEO_SDK_BIN "${RTX_SDK_ROOT}/bin/Windows/x64/rel")

# NVIDIA Video Codec SDK via env var NV_VIDEO_CODEC_SDK
if(NOT DEFINED ENV{NV_VIDEO_CODEC_SDK})
  message(FATAL_ERROR "Environment variable NV_VIDEO_CODEC_SDK not set. Point it to the Video Codec SDK root (contains Interface/ Lib/ include/ lib/)")
endif()
set(NV_CODEC_SDK_ROOT $ENV{NV_VIDEO_CODEC_SDK})
# Newer SDKs place headers in include/ and Interface/
set(NV_CODEC_SDK_INCLUDE "${NV_CODEC_SDK_ROOT}/include")
if(EXISTS "${NV_CODEC_SDK_ROOT}/Interface")
  list(APPEND NV_CODEC_SDK_INCLUDE "${NV_CODEC_SDK_ROOT}/Interface")
endif()
# Lib directory (Windows x64)
if(EXISTS "${NV_CODEC_SDK_ROOT}/Lib/x64")
  set(NV_CODEC_SDK_LIB_DIR "${NV_CODEC_SDK_ROOT}/Lib/x64")
elseif(EXISTS "${NV_CODEC_SDK_ROOT}/lib/x64")
  set(NV_CODEC_SDK_LIB_DIR "${NV_CODEC_SDK_ROOT}/lib/x64")
else()
  set(NV_CODEC_SDK_LIB_DIR "${NV_CODEC_SDK_ROOT}/lib")
endif()

# Build options
add_definitions(-DWIN32_LEAN_AND_MEAN)
add_definitions(-DNOMINMAX)

# Sources
file(GLOB RVP_SOURCES
  src/*.cpp
  src/cuda_kernels.cu
  ${RTX_VIDEO_API_INCLUDE_DIR}/rtx_video_api_cuda_impl.cpp
)
file(GLOB RVP_HEADERS
  src/*.h
  src/*.hpp
)

add_executable(RTXVideoProcessor ${RVP_SOURCES} ${RVP_HEADERS})

# Ensure CUDA source files build cleanly with NVCC on the final exe (so device link runs)
set_target_properties(RTXVideoProcessor PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_PROPAGATE_HOST_FLAGS OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_RUNTIME_LIBRARY Static
  # Ensure console subsystem for proper stdin/stdout/stderr handling
  WIN32_EXECUTABLE FALSE
)

# Include directories
target_include_directories(RTXVideoProcessor PRIVATE
  ${FFMPEG_INCLUDE_DIR}
  ${RTX_SDK_INCLUDE}
  ${NV_CODEC_SDK_INCLUDE}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${RTX_VIDEO_API_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  # Fallback for IDE IntelliSense or non-CMake consumers: use CUDA_PATH if defined
  $ENV{CUDA_PATH}/include
)

# Link directories
link_directories(${FFMPEG_LIB_DIR} ${RTX_SDK_LIB_DIR} ${NV_CODEC_SDK_LIB_DIR})

# Link libraries
# FFmpeg (link by absolute paths for reliability)
target_link_libraries(RTXVideoProcessor PRIVATE
  ${FFMPEG_AVFORMAT_LIB}
  ${FFMPEG_AVCODEC_LIB}
  ${FFMPEG_AVFILTER_LIB}
  ${FFMPEG_AVUTIL_LIB}
  ${FFMPEG_SWSCALE_LIB}
  ${FFMPEG_SWRESAMPLE_LIB}
)

# CUDA core libs
target_link_libraries(RTXVideoProcessor PRIVATE CUDA::cuda_driver CUDA::cudart_static)

# NVCodec common (some SDKs provide these; keep optional)
# Commonly needed: nvcuvid for NVDEC interop (shipped in NVIDIA drivers), and nvencodeapi for NVENC
find_library(NVCUVID_LIB NAMES nvcuvid PATHS ${NV_CODEC_SDK_LIB_DIR} ENV LIB PATH_SUFFIXES . REQUIRED)
find_library(NVENCODEAPI_LIB NAMES nvencodeapi PATHS ${NV_CODEC_SDK_LIB_DIR} ENV LIB PATH_SUFFIXES . REQUIRED)
if(NOT NVCUVID_LIB OR NOT NVENCODEAPI_LIB)
  message(FATAL_ERROR "Could not find nvcuvid or nvencodeapi in ${NV_CODEC_SDK_LIB_DIR}")
endif()

# RTX Video SDK provided static/dynamic lib name (varies). Try nvsdk_ngx_s.
find_library(RTX_VIDEO_LIB NAMES nvsdk_ngx_s PATHS ${RTX_SDK_LIB_DIR} REQUIRED)

# Link SDK libs
target_link_libraries(RTXVideoProcessor PRIVATE ${NVCUVID_LIB} ${NVENCODEAPI_LIB} ${RTX_VIDEO_LIB})

# NGX SDK.
find_library(NVSDK_NGX_LIB_RELEASE NAMES nvsdk_ngx_s PATHS ${RTX_SDK_LIB_DIR} NO_DEFAULT_PATH)
if(NVSDK_NGX_LIB_RELEASE OR NVSDK_NGX_LIB_DEBUG)
  message(STATUS "Linking NGX: ${NVSDK_NGX_LIB_RELEASE} ${NVSDK_NGX_LIB_DEBUG}")
  target_link_libraries(RTXVideoProcessor PRIVATE ${NVSDK_NGX_LIB_RELEASE})
endif()

# Windows system libs
if(MSVC)
  target_link_libraries(RTXVideoProcessor PRIVATE ws2_32.lib bcrypt.lib Secur32.lib)
  # Apply MSVC options only to C++ compilation units, not to CUDA. Passing /EHsc or /MP to NVCC causes build failures.
  target_compile_options(RTXVideoProcessor PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc /MP>
  )
  # Explicitly set console subsystem to ensure proper stdin/stdout/stderr and prevent new window
  set_target_properties(RTXVideoProcessor PROPERTIES
    LINK_FLAGS "/SUBSYSTEM:CONSOLE"
  )
endif()

# Copy RTX Video SDK DLLs to output directory
set(RTX_DLLS
    "${RTX_VIDEO_SDK_BIN}/nvngx_truehdr.dll"
    "${RTX_VIDEO_SDK_BIN}/nvngx_vsr.dll"
)

# Copy NGX feature DLLs next to the executable if they exist. These come from the RTX Video SDK redistributables.
foreach(dll ${RTX_DLLS})
    if(EXISTS ${dll})
        add_custom_command(TARGET RTXVideoProcessor POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                "$<TARGET_FILE_DIR:RTXVideoProcessor>/"
        )
    else()
        message(WARNING "RTX DLL not found: ${dll}")
    endif()
endforeach()

# Runtime paths and post build copy for FFmpeg DLLs (user to provide in C:/ffmpeg/bin)
add_custom_command(TARGET RTXVideoProcessor POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Copy FFmpeg runtime DLLs from ${FFMPEG_ROOT}/bin next to the exe if needed."
)

# Install
install(TARGETS RTXVideoProcessor RUNTIME DESTINATION bin)
