#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/packet.h>
}

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <memory>
#include "logger.h"

/**
 * Async packet demuxer with bounded queue
 * Prevents blocking I/O from stalling the processing pipeline
 *
 * Features:
 * - Background thread for av_read_frame
 * - Bounded queue with backpressure (prevents memory exhaustion)
 * - Graceful shutdown on EOF or error
 * - Thread-safe packet delivery
 */
class AsyncDemuxer {
public:
    struct Config {
        size_t max_queue_size = 60;  // ~2 seconds at 30fps
        bool enable_stats = true;
    };

    explicit AsyncDemuxer(AVFormatContext* fmt_ctx, const Config& config = Config())
        : fmt_ctx_(fmt_ctx)
        , config_(config)
        , running_(false)
        , eof_(false)
        , error_code_(0)
    {}

    ~AsyncDemuxer() {
        stop();
    }

    // Non-copyable
    AsyncDemuxer(const AsyncDemuxer&) = delete;
    AsyncDemuxer& operator=(const AsyncDemuxer&) = delete;

    /**
     * Start background demuxing thread
     */
    bool start() {
        if (running_.exchange(true)) {
            LOG_WARN("AsyncDemuxer already running");
            return false;
        }

        eof_ = false;
        error_code_ = 0;

        demux_thread_ = std::thread([this]() {
            this->demuxLoop();
        });

        LOG_DEBUG("AsyncDemuxer started (queue_size=%zu)", config_.max_queue_size);
        return true;
    }

    /**
     * Stop demuxing and join thread
     */
    void stop() {
        if (!running_.exchange(false)) {
            return;
        }

        // Wake up demuxer if waiting
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cv_not_full_.notify_all();
        }

        if (demux_thread_.joinable()) {
            demux_thread_.join();
        }

        // Cleanup remaining packets
        std::lock_guard<std::mutex> lock(mutex_);
        while (!packet_queue_.empty()) {
            AVPacket* pkt = packet_queue_.front();
            packet_queue_.pop();
            av_packet_free(&pkt);
        }

        if (config_.enable_stats) {
            LOG_DEBUG("AsyncDemuxer stopped - Stats: reads=%zu, queue_full_waits=%zu, max_queue=%zu",
                     stats_.total_reads, stats_.queue_full_waits, stats_.max_queue_depth);
        }
    }

    /**
     * Get next packet (blocking if queue empty)
     * Returns nullptr on EOF or error (check isEOF/getError)
     */
    AVPacket* getPacket() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait for packet or termination
        cv_not_empty_.wait(lock, [this]() {
            return !packet_queue_.empty() || eof_ || error_code_ != 0;
        });

        if (!packet_queue_.empty()) {
            AVPacket* pkt = packet_queue_.front();
            packet_queue_.pop();

            // Update stats
            if (packet_queue_.size() < stats_.min_queue_depth) {
                stats_.min_queue_depth = packet_queue_.size();
            }

            // Wake up demuxer if it was waiting
            cv_not_full_.notify_one();

            return pkt;
        }

        // Queue empty and EOF or error
        return nullptr;
    }

    /**
     * Try to get packet without blocking
     * Returns nullptr if queue empty (not EOF)
     */
    AVPacket* tryGetPacket() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!packet_queue_.empty()) {
            AVPacket* pkt = packet_queue_.front();
            packet_queue_.pop();
            cv_not_full_.notify_one();
            return pkt;
        }

        return nullptr;
    }

    bool isEOF() const { return eof_; }
    int getError() const { return error_code_; }
    size_t getQueueSize() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return packet_queue_.size();
    }

    struct Stats {
        size_t total_reads = 0;
        size_t queue_full_waits = 0;
        size_t max_queue_depth = 0;
        size_t min_queue_depth = SIZE_MAX;
    };

    Stats getStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

private:
    void demuxLoop() {
        LOG_DEBUG("AsyncDemuxer thread started");

        while (running_) {
            AVPacket* pkt = av_packet_alloc();
            if (!pkt) {
                LOG_ERROR("Failed to allocate packet in demux thread");
                error_code_ = AVERROR(ENOMEM);
                break;
            }

            // Read frame (may block on network I/O)
            int ret = av_read_frame(fmt_ctx_, pkt);

            if (ret == AVERROR_EOF) {
                av_packet_free(&pkt);
                LOG_DEBUG("AsyncDemuxer reached EOF");
                eof_ = true;
                cv_not_empty_.notify_all();
                break;
            }

            if (ret < 0) {
                av_packet_free(&pkt);
                char errbuf[AV_ERROR_MAX_STRING_SIZE];
                av_make_error_string(errbuf, sizeof(errbuf), ret);
                LOG_ERROR("AsyncDemuxer read error: %s", errbuf);
                error_code_ = ret;
                cv_not_empty_.notify_all();
                break;
            }

            stats_.total_reads++;

            // Wait for queue space (backpressure)
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_not_full_.wait(lock, [this]() {
                    return packet_queue_.size() < config_.max_queue_size || !running_;
                });

                if (!running_) {
                    av_packet_free(&pkt);
                    break;
                }

                if (packet_queue_.size() >= config_.max_queue_size) {
                    stats_.queue_full_waits++;
                }

                packet_queue_.push(pkt);

                // Update stats
                if (packet_queue_.size() > stats_.max_queue_depth) {
                    stats_.max_queue_depth = packet_queue_.size();
                }
            }

            cv_not_empty_.notify_one();
        }

        LOG_DEBUG("AsyncDemuxer thread exiting");
    }

    AVFormatContext* fmt_ctx_;
    Config config_;

    std::thread demux_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> eof_;
    std::atomic<int> error_code_;

    mutable std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;

    std::queue<AVPacket*> packet_queue_;
    mutable Stats stats_;
};
