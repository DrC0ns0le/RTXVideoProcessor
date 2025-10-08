#pragma once

// Check if passthrough to original ffmpeg is required based on command-line arguments
bool passthrough_required(int argc, char **argv);

// Execute ffmpeg passthrough by launching the original ffmpeg binary
int run_ffmpeg_passthrough(int argc, char **argv);
