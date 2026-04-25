#pragma once

#include <chrono>
#include <map>
#include <string>

#include <shark/types/u128.hpp>

namespace shark {
    namespace utils {
        struct TimerStat
        {
            u64 accumulated_time;
            u64 accumulated_comm;
            u64 accumulated_rounds;
            u64 start_time;
            u64 start_comm;
            u64 start_rounds;
        };

        extern std::map<std::string, TimerStat> timers;

        void start_timer(const std::string& name);
        void stop_timer(const std::string& name);
        void print_timer(const std::string& name);
        void print_all_timers(const std::string& name = "");
        bool get_timer_stat(const std::string& name, TimerStat& stat);
    }
}
