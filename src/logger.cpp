#include "logger.h"

#include <cstdarg>
#include <cstdio>

Logger &Logger::instance()
{
    static Logger s_instance;
    return s_instance;
}

void Logger::setVerbose(bool enabled)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_verbose = enabled;
}

void Logger::setDebug(bool enabled)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_debug = enabled;
}

bool Logger::verboseEnabled() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_verbose;
}

bool Logger::debugEnabled() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_debug;
}

bool Logger::shouldLog(LogLevel level) const
{
    switch (level)
    {
    case LogLevel::Error:
        return true;
    case LogLevel::Warn:
        return true;
    case LogLevel::Info:
        return true;
    case LogLevel::Verbose:
        return m_verbose;
    case LogLevel::Debug:
        return m_debug;
    default:
        return false;
    }
}

const char *Logger::prefix(LogLevel level) const
{
    switch (level)
    {
    case LogLevel::Error:
        return "[ERROR] ";
    case LogLevel::Warn:
        return "[WARN] ";
    case LogLevel::Info:
        return "[INFO] ";
    case LogLevel::Verbose:
        return "[VERBOSE] ";
    case LogLevel::Debug:
        return "[DEBUG] ";
    default:
        return "";
    }
}

void Logger::log(LogLevel level, const char *fmt, ...) noexcept
{
    if (!fmt)
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    if (!shouldLog(level))
        return;

    std::fputs(prefix(level), stderr);

    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    std::fputc('\n', stderr);
    std::fflush(stderr);
}
