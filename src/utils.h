#pragma once

#include <string>

// String utility functions

// Check if a string ends with a suffix
bool endsWith(const std::string &str, const std::string &suffix);

// Convert string to lowercase (returns a copy)
std::string lowercase_copy(const std::string &s);

// Check if output path is a pipe (-, pipe:, pipe:1, etc.)
bool is_pipe_output(const char *path);
