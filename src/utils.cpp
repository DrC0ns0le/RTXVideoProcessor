#include "utils.h"
#include <algorithm>
#include <cctype>
#include <cstring>

// Check if a string ends with a suffix
bool endsWith(const std::string &str, const std::string &suffix)
{
    if (suffix.size() > str.size())
        return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Convert string to lowercase (returns a copy)
std::string lowercase_copy(const std::string &s)
{
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });
    return result;
}

// Check if output path is a pipe (-, pipe:, pipe:1, etc.)
bool is_pipe_output(const char *path)
{
    if (!path)
        return false;
    return (std::strcmp(path, "-") == 0) ||
           (std::strcmp(path, "pipe:") == 0) ||
           (std::strcmp(path, "pipe:1") == 0) ||
           (std::strncmp(path, "pipe:", 5) == 0);
}
