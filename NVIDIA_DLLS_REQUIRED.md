RTXVideoProcessor requires NVIDIA RTX Video SDK runtime DLLs
============================================================

Place the following DLLs in the same directory as RTXVideoProcessor.exe:

  [ ] nvngx_vsr.dll
  [ ] nvngx_truehdr.dll

These DLLs are proprietary NVIDIA software and cannot be redistributed with this project.

How to Obtain:
--------------
1. Visit: https://developer.nvidia.com/rtx-video-sdk
2. Register/sign in with NVIDIA Developer account (free)
3. Download the RTX Video SDK for Windows
4. Extract and navigate to: RTX_Video_SDK/bin/Windows/x64/rel/
5. Copy nvngx_vsr.dll and nvngx_truehdr.dll to this directory

For more information, see the README.md file.
