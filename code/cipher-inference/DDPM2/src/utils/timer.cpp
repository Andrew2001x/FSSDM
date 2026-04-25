#include <iostream>
#include <mutex>
#include <vector>

#include <shark/utils/timer.hpp>
#include <shark/protocols/common.hpp>

namespace shark {
    namespace utils {
        std::map<std::string, TimerStat> timers;
        std::mutex timersMutex;

        namespace {
            struct ActiveTimerFrame {
                u64 start_time = 0;
                u64 start_comm = 0;
                u64 start_rounds = 0;
            };

            std::map<std::string, std::vector<ActiveTimerFrame>> activeTimerFrames;

            static inline u64 current_time_ms()
            {
                return std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            }

            static inline u64 current_comm_bytes()
            {
                if (!shark::protocols::peer) return 0;
                return shark::protocols::peer->bytesReceived() + shark::protocols::peer->bytesSent();
            }

            static inline u64 current_rounds()
            {
                if (!shark::protocols::peer) return 0;
                return shark::protocols::peer->roundsReceived() + shark::protocols::peer->roundsSent();
            }
        }

        void start_timer(const std::string& name)
        {
            std::lock_guard<std::mutex> lock(timersMutex);
            ActiveTimerFrame frame;
            frame.start_time = current_time_ms();
            frame.start_comm = current_comm_bytes();
            frame.start_rounds = current_rounds();
            activeTimerFrames[name].push_back(frame);
            timers[name].start_time = frame.start_time;
            timers[name].start_comm = frame.start_comm;
            timers[name].start_rounds = frame.start_rounds;
        }

        void stop_timer(const std::string& name)
        {
            std::lock_guard<std::mutex> lock(timersMutex);
            auto frames_it = activeTimerFrames.find(name);
            if (frames_it == activeTimerFrames.end() || frames_it->second.empty())
            {
                return;
            }

            const ActiveTimerFrame frame = frames_it->second.back();
            frames_it->second.pop_back();
            const u64 end = current_time_ms();
            const u64 end_comm = current_comm_bytes();
            const u64 end_rounds = current_rounds();

            timers[name].accumulated_time += end - frame.start_time;
            timers[name].accumulated_comm += end_comm - frame.start_comm;
            timers[name].accumulated_rounds += end_rounds - frame.start_rounds;

            if (frames_it->second.empty())
            {
                timers[name].start_time = 0;
                timers[name].start_comm = 0;
                timers[name].start_rounds = 0;
                activeTimerFrames.erase(frames_it);
            }
            else
            {
                const ActiveTimerFrame &parent = frames_it->second.back();
                timers[name].start_time = parent.start_time;
                timers[name].start_comm = parent.start_comm;
                timers[name].start_rounds = parent.start_rounds;
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
