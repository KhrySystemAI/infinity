#pragma once

#ifndef __INCLUDE_CINFINITY_WRITE_PREF_LOCK_HPP__
#define __INCLUDE_CINFINITY_WRITE_PREF_LOCK_HPP__

#include "config.hpp"

namespace cinfinity {
    class WritePrefLock {
    public:
        void lock_shared() {
            std::unique_lock<std::mutex> lk(mutex_);
            cond_.wait(lk, [&] {
                return num_writers_waiting_ == 0 && !writer_active_;
            });
            ++num_readers_active_;
        }

        void unlock_shared() {
            std::unique_lock<std::mutex> lk(mutex_);
            --num_readers_active_;
            if (num_readers_active_ == 0) {
                cond_.notify_all(); // wake up writers waiting
            }
        }

        void lock() {
            std::unique_lock<std::mutex> lk(mutex_);
            ++num_writers_waiting_;
            cond_.wait(lk, [&] {
                return num_readers_active_ == 0 && !writer_active_;
            });
            --num_writers_waiting_;
            writer_active_ = true;
        }

        void unlock() {
            std::unique_lock<std::mutex> lk(mutex_);
            writer_active_ = false;
            cond_.notify_all(); // wake readers or other writers
        }

    private:
        std::mutex mutex_;
        std::condition_variable cond_;
        int num_readers_active_ = 0;
        int num_writers_waiting_ = 0;
        bool writer_active_ = false;
    }; // class WritePrefLock
} // namespace cinfinity

#endif // __INCLUDE_CINFINITY_WRITE_PREF_LOCK_HPP__