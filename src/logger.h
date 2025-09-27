#pragma once

#include <cstdarg>
#include <mutex>
#include <string>

enum class LogLevel
{
    Error,
    Warn,
    Info,
    Verbose,
    Debug
};

class Logger
{
public:
    static Logger &instance();

    void setVerbose(bool enabled);
    void setDebug(bool enabled);

    bool verboseEnabled() const;
    bool debugEnabled() const;

    void log(LogLevel level, const char *fmt, ...) noexcept;

private:
    Logger() = default;
    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

    bool shouldLog(LogLevel level) const;
    const char *prefix(LogLevel level) const;

    mutable std::mutex m_mutex;
    bool m_verbose = false;
    bool m_debug = false;
};

#define LOG_ERROR(...) Logger::instance().log(LogLevel::Error, __VA_ARGS__)
#define LOG_WARN(...) Logger::instance().log(LogLevel::Warn, __VA_ARGS__)
#define LOG_INFO(...) Logger::instance().log(LogLevel::Info, __VA_ARGS__)
#define LOG_VERBOSE(...) Logger::instance().log(LogLevel::Verbose, __VA_ARGS__)
#define LOG_DEBUG(...) Logger::instance().log(LogLevel::Debug, __VA_ARGS__)
