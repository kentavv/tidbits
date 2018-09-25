#ifndef _PROFILE_H_
#define _PROFILE_H_

#include <chrono>
#include <stdio.h>

struct Profile {
    explicit Profile(const char *str, int rank_ = -1) : name_(str), rank_(rank_), paused_(true), split_time_(0)  {
        restart();
    }

    Profile(const Profile &) = delete;
    Profile operator=(const Profile &) = delete;

    ~Profile() {
        if (!paused_) report();
    }

    void restart() {
        split_time_ = std::chrono::duration<double>::zero();
        proceed();
    }

    void proceed() {
        update_split_time();
        paused_ = false;
    }

    void stop() {
        update_split_time();
        paused_ = true;
    }

    void setRank(int rank) {
        rank_ = rank;
    }

    void report() {
        if (rank_ >= 0) {
            printf("<%d> ", rank_);
        }
        printf("duration(%s): %f s\n", name_, elapsed());
    }

    double elapsed() {
        update_split_time();
        return split_time_.count();
    }

protected:
    const char *name_;
    int rank_;
    bool paused_;

    std::chrono::time_point<std::chrono::high_resolution_clock> lap_start_;
    std::chrono::duration<double> split_time_;

    void update_split_time() {
        auto now = std::chrono::high_resolution_clock::now();
        if (!paused_) {
            split_time_ += (now - lap_start_);
        }
        lap_start_ = now;
    }
};

#endif
