cmake_minimum_required(VERSION 3.24)
project(RTXVideoProcessor LANGUAGES C CXX CUDA)

# Modern CMake policies
cmake_policy(SET CMP0144 NEW)

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

# CUDA (Toolkit 12.x)
find_package(CUDAToolkit REQUIRED)

# FFmpeg - Prioritize vcpkg, fallback to manual C:/ffmpeg
find_package(PkgConfig)
set(FFMPEG_FOUND FALSE)

# Try vcpkg/CMake package first
find_package(FFMPEG QUIET COMPONENTS avformat avcodec avfilter avutil swscale swresample)
if(NOT FFMPEG_FOUND)
  # Try individual find_package for vcpkg FFmpeg
  find_package(ffmpeg QUIET CONFIG)
  if(ffmpeg_FOUND)
    set(FFMPEG_FOUND TRUE)
    message(STATUS "Found FFmpeg via vcpkg")
  endif()
endif()

# Fallback to pkg-config
if(NOT FFMPEG_FOUND AND PKG_CONFIG_FOUND)
  pkg_check_modules(FFMPEG QUIET
    libavformat>=60.0
    libavcodec>=60.0
    libavfilter>=9.0
    libavutil>=58.0
    libswscale>=7.0
    libswresample>=4.0
  )
  if(FFMPEG_FOUND)
    message(STATUS "Found FFmpeg via pkg-config")
  endif()
endif()

# Fallback to manual C:/ffmpeg
if(NOT FFMPEG_FOUND)
  message(STATUS "FFmpeg not found via vcpkg/pkg-config, trying manual path")
  set(FFMPEG_ROOT "C:/ffmpeg" CACHE PATH "Path to FFmpeg prebuilt root containing include/ and lib/")
  set(FFMPEG_INCLUDE_DIR "${FFMPEG_ROOT}/include")
  set(FFMPEG_LIB_DIR "${FFMPEG_ROOT}/lib")

  # Resolve FFmpeg import libraries explicitly
  foreach(lib avformat avcodec avfilter avutil swscale swresample)
    string(TOUPPER ${lib} lib_upper)
    find_library(FFMPEG_${lib_upper}_LIB
      NAMES ${lib} ${lib}.lib ${lib}-61 ${lib}-61.lib ${lib}-60 ${lib}-60.lib
            ${lib}-10 ${lib}-10.lib ${lib}-9 ${lib}-9.lib ${lib}-59 ${lib}-59.lib
            ${lib}-58 ${lib}-58.lib ${lib}-8 ${lib}-8.lib ${lib}-7 ${lib}-7.lib
            ${lib}-5 ${lib}-5.lib ${lib}-4 ${lib}-4.lib
      PATHS ${FFMPEG_LIB_DIR} NO_DEFAULT_PATH
    )
  endforeach()

  if(NOT FFMPEG_AVFORMAT_LIB OR NOT FFMPEG_AVCODEC_LIB OR NOT FFMPEG_AVFILTER_LIB OR NOT FFMPEG_AVUTIL_LIB OR NOT FFMPEG_SWSCALE_LIB OR NOT FFMPEG_SWRESAMPLE_LIB)
    message(FATAL_ERROR "FFmpeg libraries not found.\n"
      "Install via vcpkg: vcpkg install ffmpeg:x64-windows\n"
      "Or set FFMPEG_ROOT to FFmpeg directory with include/ and lib/. Example: -DFFMPEG_ROOT=C:/ffmpeg")
  endif()

  message(STATUS "Found FFmpeg at ${FFMPEG_ROOT}")
endif()

# RTX Video SDK via env var NV_RTX_VIDEO_SDK
if(NOT DEFINED ENV{NV_RTX_VIDEO_SDK})
  message(FATAL_ERROR "Environment variable NV_RTX_VIDEO_SDK not set. Point it to the RTX Video SDK root (contains include/ lib/)")
endif()
set(RTX_SDK_ROOT $ENV{NV_RTX_VIDEO_SDK})
set(RTX_SDK_INCLUDE "${RTX_SDK_ROOT}/include")
set(RTX_VIDEO_API_INCLUDE_DIR "${RTX_SDK_ROOT}/samples/RTX_Video_API")
set(RTX_VIDEO_SDK_BIN "${RTX_SDK_ROOT}/bin/Windows/x64/rel")

# Lib directory (Windows x64)
if(EXISTS "${RTX_SDK_ROOT}/lib/Windows/x64")
  set(RTX_SDK_LIB_DIR "${RTX_SDK_ROOT}/lib/Windows/x64")
elseif(EXISTS "${RTX_SDK_ROOT}/lib/x64")
  set(RTX_SDK_LIB_DIR "${RTX_SDK_ROOT}/lib/x64")
else()
  set(RTX_SDK_LIB_DIR "${RTX_SDK_ROOT}/lib")
endif()

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

# Compiler definitions
add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)

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
  WIN32_EXECUTABLE FALSE
)

# Include directories
target_include_directories(RTXVideoProcessor PRIVATE
  ${RTX_SDK_INCLUDE}
  ${NV_CODEC_SDK_INCLUDE}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${RTX_VIDEO_API_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# FFmpeg includes (vcpkg handles automatically, manual needs explicit path)
if(NOT ffmpeg_FOUND AND FFMPEG_INCLUDE_DIR)
  target_include_directories(RTXVideoProcessor PRIVATE ${FFMPEG_INCLUDE_DIR})
endif()


# Link libraries
# FFmpeg (vcpkg provides targets, manual uses absolute paths)
if(ffmpeg_FOUND)
  target_link_libraries(RTXVideoProcessor PRIVATE
    ffmpeg::avformat
    ffmpeg::avcodec
    ffmpeg::avfilter
    ffmpeg::avutil
    ffmpeg::swscale
    ffmpeg::swresample
  )
else()
  target_link_libraries(RTXVideoProcessor PRIVATE
    ${FFMPEG_AVFORMAT_LIB}
    ${FFMPEG_AVCODEC_LIB}
    ${FFMPEG_AVFILTER_LIB}
    ${FFMPEG_AVUTIL_LIB}
    ${FFMPEG_SWSCALE_LIB}
    ${FFMPEG_SWRESAMPLE_LIB}
  )
endif()

# CUDA core libs
target_link_libraries(RTXVideoProcessor PRIVATE CUDA::cuda_driver CUDA::cudart_static)

# NVIDIA Video Codec SDK libs
find_library(NVCUVID_LIB NAMES nvcuvid PATHS ${NV_CODEC_SDK_LIB_DIR} ENV LIB PATH_SUFFIXES .)
find_library(NVENCODEAPI_LIB NAMES nvencodeapi PATHS ${NV_CODEC_SDK_LIB_DIR} ENV LIB PATH_SUFFIXES .)
if(NOT NVCUVID_LIB OR NOT NVENCODEAPI_LIB)
  message(FATAL_ERROR "Could not find nvcuvid or nvencodeapi in ${NV_CODEC_SDK_LIB_DIR}")
endif()
target_link_libraries(RTXVideoProcessor PRIVATE ${NVCUVID_LIB} ${NVENCODEAPI_LIB})

# RTX Video SDK NGX lib
find_library(NVSDK_NGX_LIB NAMES nvsdk_ngx_s PATHS ${RTX_SDK_LIB_DIR} NO_DEFAULT_PATH)
if(NOT NVSDK_NGX_LIB)
  message(FATAL_ERROR "Could not find nvsdk_ngx_s in ${RTX_SDK_LIB_DIR}")
endif()
message(STATUS "Linking NGX: ${NVSDK_NGX_LIB}")
target_link_libraries(RTXVideoProcessor PRIVATE ${NVSDK_NGX_LIB})

# Windows-specific configuration
if(MSVC)
  target_link_libraries(RTXVideoProcessor PRIVATE ws2_32 bcrypt Secur32)
  target_compile_options(RTXVideoProcessor PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/EHsc /MP>)
  set_target_properties(RTXVideoProcessor PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
endif()

# Post-build: Copy RTX Video SDK DLLs
foreach(dll_name nvngx_truehdr.dll nvngx_vsr.dll)
  set(dll_path "${RTX_VIDEO_SDK_BIN}/${dll_name}")
  if(EXISTS ${dll_path})
    add_custom_command(TARGET RTXVideoProcessor POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll_path}" "$<TARGET_FILE_DIR:RTXVideoProcessor>/"
      COMMENT "Copying ${dll_name}"
    )
  else()
    message(WARNING "RTX DLL not found: ${dll_path}")
  endif()
endforeach()

# Post-build: Reminder for FFmpeg DLLs
if(NOT ffmpeg_FOUND AND FFMPEG_ROOT)
  add_custom_command(TARGET RTXVideoProcessor POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Note: Copy FFmpeg runtime DLLs from ${FFMPEG_ROOT}/bin if needed."
  )
endif()

# Install
install(TARGETS RTXVideoProcessor RUNTIME DESTINATION bin)
