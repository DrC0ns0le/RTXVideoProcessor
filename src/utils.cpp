#include "utils.h"
#include <algorithm>
#include <cctype>

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
