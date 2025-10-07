#include "ffmpeg_passthrough.h"

#include <string>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cerrno>
#include <filesystem>
#include <system_error>
#include <algorithm>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

// Helper function to check if a string ends with a suffix
static bool endsWith(const std::string &str, const std::string &suffix)
{
    if (suffix.size() > str.size())
        return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

#ifdef _WIN32
// Find ffmpeg.exe on the system PATH
static std::string find_ffmpeg_on_path()
{
    const char *path_env = std::getenv("PATH");
    if (!path_env)
        return {};

    const std::string exe_name = "ffmpeg.exe";
    std::string path_str(path_env);
    size_t start = 0;
    while (start <= path_str.size())
    {
        size_t end = path_str.find(';', start);
        std::string dir = path_str.substr(start, end == std::string::npos ? std::string::npos : end - start);

        // Clean up directory path
        dir.erase(std::remove(dir.begin(), dir.end(), '"'), dir.end());
        auto first = dir.find_first_not_of(" \t");
        if (first == std::string::npos)
        {
            dir.clear();
        }
        else
        {
            auto last = dir.find_last_not_of(" \t");
            dir = dir.substr(first, last - first + 1);
        }

        // Check if ffmpeg.exe exists in this directory
        if (!dir.empty() && dir != "." && dir != ".\\" && dir != "./")
        {
            std::filesystem::path candidate = std::filesystem::path(dir) / exe_name;
            std::error_code ec;
            if (std::filesystem::exists(candidate, ec) && !std::filesystem::is_directory(candidate, ec))
            {
                return candidate.string();
            }
        }

        if (end == std::string::npos)
            break;
        start = end + 1;
    }

    return {};
}

// Quote a Windows command-line argument properly
static std::string quote_windows_arg(const std::string &arg)
{
    if (arg.empty())
        return "\"\"";

    bool needs_quotes = arg.find_first_of(" \t\"") != std::string::npos;
    if (!needs_quotes)
        return arg;

    std::string result;
    result.reserve(arg.size() + 2);
    result.push_back('"');

    size_t backslash_count = 0;
    for (char ch : arg)
    {
        if (ch == '\\')
        {
            ++backslash_count;
        }
        else if (ch == '"')
        {
            result.append(backslash_count * 2 + 1, '\\');
            result.push_back('"');
            backslash_count = 0;
        }
        else
        {
            if (backslash_count > 0)
            {
                result.append(backslash_count, '\\');
                backslash_count = 0;
            }
            result.push_back(ch);
        }
    }

    if (backslash_count > 0)
    {
        result.append(backslash_count * 2, '\\');
    }

    result.push_back('"');
    return result;
}

// Format Windows error code to human-readable string
static std::string format_windows_error(DWORD error_code)
{
    if (error_code == 0)
        return "";

    LPSTR buffer = nullptr;
    DWORD size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        error_code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buffer),
        0,
        nullptr);

    std::string message;
    if (size != 0 && buffer)
    {
        message.assign(buffer, size);
        while (!message.empty() && (message.back() == '\r' || message.back() == '\n'))
            message.pop_back();
    }
    else
    {
        message = "Unknown error";
    }

    if (buffer)
        LocalFree(buffer);

    return message;
}
#endif

bool passthrough_required(int argc, char **argv)
{
    if (argc <= 0)
        return false;

    // Only passthrough if invoked as "ffmpeg" or "ffmpeg.exe"
    bool binary_is_ffmpeg = endsWith(argv[0], "ffmpeg") || endsWith(argv[0], "ffmpeg.exe");
    if (!binary_is_ffmpeg)
        return false;

    // Check for supported input/output formats
    bool input_looks_supported = (argc > 1) && (endsWith(argv[1], ".mp4") || endsWith(argv[1], ".mkv"));
    bool output_looks_supported = (argc > 2) && (endsWith(argv[2], ".mp4") || endsWith(argv[2], ".mkv") || endsWith(argv[2], ".m3u8"));

    // Check for special features that we support
    bool requests_hls_muxer = false;
    bool requests_mp4_to_pipe = false;
    bool requests_video_exclusion = false;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-f") == 0 && i + 1 < argc)
        {
            if (std::strcmp(argv[i + 1], "hls") == 0)
            {
                requests_hls_muxer = true;
            }
            else if (std::strcmp(argv[i + 1], "mp4") == 0)
            {
                // Check if output is pipe
                for (int j = i + 2; j < argc; ++j)
                {
                    if (std::strcmp(argv[j], "-") == 0 ||
                        std::strcmp(argv[j], "pipe:") == 0 ||
                        std::strcmp(argv[j], "pipe:1") == 0)
                    {
                        requests_mp4_to_pipe = true;
                        break;
                    }
                }
            }
        }
        else if (std::strcmp(argv[i], "-map") == 0 && i + 1 < argc && std::strcmp(argv[i + 1], "-0:v?") == 0)
        {
            requests_video_exclusion = true;
        }
    }

    // Passthrough is needed if:
    // 1. Neither input nor output formats are supported AND no special features are requested
    // 2. OR video exclusion is requested (we can't process without video)
    return (!input_looks_supported && !output_looks_supported && !requests_hls_muxer && !requests_mp4_to_pipe) || requests_video_exclusion;
}

int run_ffmpeg_passthrough(int argc, char **argv)
{
    fprintf(stderr, "Running ffmpeg passthrough\n");

    std::string ffmpeg_binary;
#ifdef _WIN32
    bool use_absolute_ffmpeg = false;
    if (auto resolved = find_ffmpeg_on_path(); !resolved.empty())
    {
        ffmpeg_binary = std::move(resolved);
        use_absolute_ffmpeg = true;
    }
    else
    {
        ffmpeg_binary = "ffmpeg";
    }
#else
    ffmpeg_binary = "ffmpeg";
#endif

    // Build argument list with ffmpeg binary as first argument
    std::vector<std::string> forwarded_args_storage;
    forwarded_args_storage.reserve(static_cast<size_t>(argc));
    forwarded_args_storage.emplace_back(ffmpeg_binary);
    for (int i = 1; i < argc; ++i)
    {
        forwarded_args_storage.emplace_back(argv[i] ? argv[i] : "");
    }

#ifdef _WIN32
    // Windows implementation using CreateProcess
    std::string command_line;
    command_line.reserve(256);
    for (size_t i = 0; i < forwarded_args_storage.size(); ++i)
    {
        if (i > 0)
            command_line.push_back(' ');
        command_line.append(quote_windows_arg(forwarded_args_storage[i]));
    }

    std::vector<char> command_line_buffer(command_line.begin(), command_line.end());
    command_line_buffer.push_back('\0');

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);

    // Inherit stdin/stdout/stderr for pipe support
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

    const char *application_name = use_absolute_ffmpeg ? forwarded_args_storage[0].c_str() : nullptr;

    BOOL success = CreateProcessA(
        application_name,
        command_line_buffer.data(),
        nullptr,
        nullptr,
        TRUE,  // bInheritHandles = TRUE to inherit stdout/stderr/stdin
        0,
        nullptr,
        nullptr,
        &si,
        &pi);

    if (!success)
    {
        DWORD err = GetLastError();
        std::string err_msg = format_windows_error(err);
        fprintf(stderr, "Failed to launch ffmpeg passthrough binary '%s': %s (0x%08lX)\n",
                forwarded_args_storage[0].c_str(), err_msg.c_str(), static_cast<unsigned long>(err));
        return err ? static_cast<int>(err) : 1;
    }

    DWORD wait_result = WaitForSingleObject(pi.hProcess, INFINITE);
    if (wait_result == WAIT_FAILED)
    {
        DWORD err = GetLastError();
        std::string err_msg = format_windows_error(err);
        fprintf(stderr, "Failed waiting for ffmpeg passthrough process: %s (0x%08lX)\n",
                err_msg.c_str(), static_cast<unsigned long>(err));
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return err ? static_cast<int>(err) : 1;
    }

    DWORD exit_code = 0;
    if (!GetExitCodeProcess(pi.hProcess, &exit_code))
    {
        DWORD err = GetLastError();
        std::string err_msg = format_windows_error(err);
        fprintf(stderr, "Failed to retrieve ffmpeg passthrough exit code: %s (0x%08lX)\n",
                err_msg.c_str(), static_cast<unsigned long>(err));
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return err ? static_cast<int>(err) : 1;
    }

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    return static_cast<int>(exit_code);
#else
    // Unix/Linux implementation using fork/exec
    std::vector<char *> passthrough_argv;
    passthrough_argv.reserve(forwarded_args_storage.size() + 1);
    for (auto &stored : forwarded_args_storage)
    {
        passthrough_argv.push_back(const_cast<char *>(stored.c_str()));
    }
    passthrough_argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0)
    {
        fprintf(stderr, "Failed to fork for ffmpeg passthrough: %s\n", std::strerror(errno));
        return errno ? errno : 1;
    }
    if (pid == 0)
    {
        execvp(ffmpeg_binary.c_str(), passthrough_argv.data());
        fprintf(stderr, "Failed to exec ffmpeg passthrough binary '%s': %s\n", ffmpeg_binary.c_str(), std::strerror(errno));
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0)
    {
        fprintf(stderr, "Failed to wait for ffmpeg passthrough process: %s\n", std::strerror(errno));
        return errno ? errno : 1;
    }

    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    if (WIFSIGNALED(status))
        return 128 + WTERMSIG(status);
    return 1;
#endif
}
