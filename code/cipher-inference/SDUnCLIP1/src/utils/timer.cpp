#include <iostream>
#include <mutex>

#include <shark/utils/timer.hpp>
#include <shark/protocols/common.hpp>

namespace shark {
    namespace utils {
        std::map<std::string, TimerStat> timers;
        std::mutex timersMutex;

        void start_timer(const std::string& name)
        {
            std::lock_guard<std::mutex> lock(timersMutex);
            timers[name].start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            if (shark::protocols::peer)
            {
                timers[name].start_comm = shark::protocols::peer->bytesReceived() + shark::protocols::peer->bytesSent();
                timers[name].start_rounds = shark::protocols::peer->roundsReceived() + shark::protocols::peer->roundsSent();
            }
            else
            {
                timers[name].start_comm = 0;
                timers[name].start_rounds = 0;
            }
        }

        void stop_timer(const std::string& name)
        {
            std::lock_guard<std::mutex> lock(timersMutex);
            u64 end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            timers[name].accumulated_time += end - timers[name].start_time;
            if (shark::protocols::peer)
            {
                timers[name].accumulated_comm += shark::protocols::peer->bytesReceived() + shark::protocols::peer->bytesSent() - timers[name].start_comm;
                timers[name].accumulated_rounds += shark::protocols::peer->roundsReceived() + shark::protocols::peer->roundsSent() - timers[name].start_rounds;
            }
            else
            {
                timers[name].accumulated_comm = 0;
                timers[name].accumulated_rounds = 0;
            }
        }

        void print_timer(const std::string& name)
        {
            std::lock_guard<std::mutex> lock(timersMutex);
            std::cout << name << ": " << timers[name].accumulated_time << " ms, "
                      << (timers[name].accumulated_comm / 1024.0) << " KB, "
                      << timers[name].accumulated_rounds << " rounds" << std::endl;
        }

        void print_all_timers(const std::string& prefix)
        {
            std::lock_guard<std::mutex> lock(timersMutex);
            for (auto& timer : timers)
            {
                if (prefix.empty() || timer.first.find(prefix) == 0)
                {
                    std::cout << timer.first << ": " << timer.second.accumulated_time << " ms, "
                              << (timer.second.accumulated_comm / 1024.0) << " KB, "
                              << timer.second.accumulated_rounds << " rounds" << std::endl;
                }
            }
        }

        bool get_timer_stat(const std::string& name, TimerStat& stat)
        {
            std::lock_guard<std::mutex> lock(timersMutex);
            auto it = timers.find(name);
            if (it == timers.end())
            {
                stat = TimerStat{};
                return false;
            }
            stat = it->second;
            return true;
        }

        // void __attribute__((destructor)) destruct_timers()
        // {
        //     print_all_timers();
        // }
    }
}
