#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/common.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/conv.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/ars.hpp>
#include <shark/protocols/drelu.hpp>
#include <shark/protocols/select.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/comm.hpp>
#include <shark/utils/timer.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "para.hpp"

using namespace shark;
using namespace shark::protocols;

static int f = 24;

struct AuthShare {
    span<u128> share;
    span<u128> tag;
};

struct OpProfileStat {
    const char *name = "";
    u64 calls = 0;
    u64 comm_bytes = 0;
    u64 rounds = 0;
    double time_ms = 0.0;
};

enum OpProfileIndex : size_t {
    OP_INPUT = 0,
    OP_TIME_EMBEDDING_EXP = 1,
    OP_TIME_EMBEDDING_OPEN = 2,
    OP_SOFTMAX_EXP = 3,
    OP_SOFTMAX = 4,
    OP_LINEAR = 5,
    OP_CONV = 6,
    OP_SILU = 7,
    OP_GROUPNORM = 8,
    OP_LAYERNORM = 9,
    OP_GELU = 10,
    OP_SCHEDULER = 11,
    OP_RECONSTRUCT = 12,
    OP_PROFILE_COUNT = 13
};

static std::array<OpProfileStat, OP_PROFILE_COUNT> g_op_stats{{
    {"input", 0, 0, 0, 0.0},
    {"time_embedding_exp", 0, 0, 0, 0.0},
    {"time_embedding_open", 0, 0, 0, 0.0},
    {"softmax_exp", 0, 0, 0, 0.0},
    {"softmax", 0, 0, 0, 0.0},
    {"linear", 0, 0, 0, 0.0},
    {"conv", 0, 0, 0, 0.0},
    {"silu", 0, 0, 0, 0.0},
    {"groupnorm", 0, 0, 0, 0.0},
    {"layernorm", 0, 0, 0, 0.0},
    {"gelu", 0, 0, 0, 0.0},
    {"scheduler", 0, 0, 0, 0.0},
    {"final_reveal", 0, 0, 0, 0.0},
}};

struct EvalProfileStat {
    bool active = false;
    u64 comm0 = 0;
    u64 rounds0 = 0;
    u64 key_read_ms0 = 0;
    u64 comm_bytes = 0;
    u64 rounds = 0;
    double time_ms = 0.0;
    std::chrono::steady_clock::time_point t0{};
};

struct OpProfileFrame {
    size_t idx = 0;
    u64 comm0 = 0;
    u64 rounds0 = 0;
    u64 key_read_ms0 = 0;
    u64 child_comm = 0;
    u64 child_rounds = 0;
    u64 child_key_read_ms = 0;
    double child_time_ms = 0.0;
    std::chrono::steady_clock::time_point t0{};
};

static EvalProfileStat g_total_eval_profile{};
static thread_local std::vector<OpProfileFrame> g_op_profile_stack;

static inline u64 current_comm_bytes() {
    if (peer == nullptr) return 0;
    return peer->bytesReceived() + peer->bytesSent();
}

static inline u64 current_rounds() {
    if (peer == nullptr) return 0;
    return peer->roundsReceived() + peer->roundsSent();
}

static inline u64 current_key_read_ms() {
    shark::utils::TimerStat stat{};
    if (!shark::utils::get_timer_stat("dealer_read_local", stat)) return 0;
    return stat.accumulated_time;
}

struct OpProfileScope {
    size_t idx = 0;
    bool active = false;

    explicit OpProfileScope(size_t idx_in)
        : idx(idx_in), active(party != DEALER) {
        if (!active) return;
        g_op_profile_stack.push_back(OpProfileFrame{
            idx,
            current_comm_bytes(),
            current_rounds(),
            current_key_read_ms(),
            0,
            0,
            0,
            0.0,
            std::chrono::steady_clock::now(),
        });
    }

    ~OpProfileScope() {
        if (!active) return;
        const auto t1 = std::chrono::steady_clock::now();
        auto frame = g_op_profile_stack.back();
        g_op_profile_stack.pop_back();

        const u64 total_comm = current_comm_bytes() - frame.comm0;
        const u64 total_rounds = current_rounds() - frame.rounds0;
        const u64 total_key_read_ms = current_key_read_ms() - frame.key_read_ms0;
        const double total_time_ms = std::chrono::duration<double, std::milli>(t1 - frame.t0).count();
        const u64 direct_key_read_ms =
            (total_key_read_ms >= frame.child_key_read_ms) ? (total_key_read_ms - frame.child_key_read_ms) : 0;
        double direct_time_ms = total_time_ms - frame.child_time_ms - static_cast<double>(direct_key_read_ms);
        if (direct_time_ms < 0.0 && direct_time_ms > -1e-6) direct_time_ms = 0.0;
        if (direct_time_ms < 0.0) direct_time_ms = 0.0;

        auto &stat = g_op_stats[idx];
        stat.calls += 1;
        stat.comm_bytes += total_comm - frame.child_comm;
        stat.rounds += total_rounds - frame.child_rounds;
        stat.time_ms += direct_time_ms;

        if (!g_op_profile_stack.empty()) {
            auto &parent = g_op_profile_stack.back();
            parent.child_comm += total_comm;
            parent.child_rounds += total_rounds;
            parent.child_key_read_ms += total_key_read_ms;
            parent.child_time_ms += total_time_ms;
        }
    }
};

static void start_total_eval_profile() {
    if (party == DEALER) return;
    g_total_eval_profile.active = true;
    g_total_eval_profile.comm0 = current_comm_bytes();
    g_total_eval_profile.rounds0 = current_rounds();
    g_total_eval_profile.key_read_ms0 = current_key_read_ms();
    g_total_eval_profile.comm_bytes = 0;
    g_total_eval_profile.rounds = 0;
    g_total_eval_profile.time_ms = 0.0;
    g_total_eval_profile.t0 = std::chrono::steady_clock::now();
}

static void stop_total_eval_profile() {
    if (!g_total_eval_profile.active) return;
    const auto t1 = std::chrono::steady_clock::now();
    g_total_eval_profile.comm_bytes = current_comm_bytes() - g_total_eval_profile.comm0;
    g_total_eval_profile.rounds = current_rounds() - g_total_eval_profile.rounds0;
    const double raw_time_ms =
        std::chrono::duration<double, std::milli>(t1 - g_total_eval_profile.t0).count();
    const u64 total_key_read_ms = current_key_read_ms() - g_total_eval_profile.key_read_ms0;
    g_total_eval_profile.time_ms = raw_time_ms - static_cast<double>(total_key_read_ms);
    if (g_total_eval_profile.time_ms < 0.0 && g_total_eval_profile.time_ms > -1e-6) {
        g_total_eval_profile.time_ms = 0.0;
    }
    if (g_total_eval_profile.time_ms < 0.0) {
        g_total_eval_profile.time_ms = 0.0;
    }
    g_total_eval_profile.active = false;
}

struct AvgOpProfileStat {
    double comm_kb = 0.0;
    double comm_mb = 0.0;
    double rounds = 0.0;
    double time_ms = 0.0;
};

static AvgOpProfileStat avg_op_profile(size_t idx) {
    AvgOpProfileStat out;
    const auto &stat = g_op_stats[idx];
    if (stat.calls == 0) return out;
    const double inv_calls = 1.0 / (double)stat.calls;
    out.comm_kb = ((double)stat.comm_bytes / 1024.0) * inv_calls;
    out.comm_mb = ((double)stat.comm_bytes / (1024.0 * 1024.0)) * inv_calls;
    out.rounds = (double)stat.rounds * inv_calls;
    out.time_ms = stat.time_ms * inv_calls;
    return out;
}

static void print_avg_op_profile_line(size_t idx) {
    const auto avg = avg_op_profile(idx);
    const auto &stat = g_op_stats[idx];
    const auto old_flags = std::cout.flags();
    const auto old_prec = std::cout.precision();
    std::cout << std::fixed << std::setprecision(3)
              << g_op_stats[idx].name << ": " << avg.time_ms << " ms, "
              << avg.comm_kb << " KB, "
              << avg.rounds << " rounds, "
              << "calls=" << stat.calls << std::endl;
    std::cout.flags(old_flags);
    std::cout.precision(old_prec);
}

static void print_profile_timers() {
    using shark::utils::print_timer;
    std::cout << "[PROFILE] time(ms), comm(KB) per leaf op" << std::endl;
    std::cout << "total_eval: " << g_total_eval_profile.time_ms << " ms, "
              << (g_total_eval_profile.comm_bytes / 1024.0) << " KB, "
              << g_total_eval_profile.rounds << " rounds" << std::endl;
    print_timer("dealer_read_local");
    print_timer("input");
    print_timer("auth_reconstruct_internal");
    print_timer("final_reveal");
    print_avg_op_profile_line(OP_INPUT);
    print_avg_op_profile_line(OP_TIME_EMBEDDING_EXP);
    print_avg_op_profile_line(OP_TIME_EMBEDDING_OPEN);
    print_avg_op_profile_line(OP_SOFTMAX_EXP);
    print_avg_op_profile_line(OP_SOFTMAX);
    print_avg_op_profile_line(OP_LINEAR);
    print_avg_op_profile_line(OP_CONV);
    print_avg_op_profile_line(OP_SILU);
    print_avg_op_profile_line(OP_GROUPNORM);
    print_avg_op_profile_line(OP_LAYERNORM);
    print_avg_op_profile_line(OP_GELU);
    print_avg_op_profile_line(OP_SCHEDULER);
    print_avg_op_profile_line(OP_RECONSTRUCT);
}

static void print_profile_components_table() {
    if (party == DEALER) return;
    const auto old_flags = std::cout.flags();
    const auto old_prec = std::cout.precision();
    std::cout << std::fixed << std::setprecision(3);

    struct TimerRow {
        const char *component;
        const char *timer_name;
    };

    const TimerRow timer_rows[] = {
        {"dealer_read_local", "dealer_read_local"},
        {"input", "input"},
        {"auth_reconstruct_internal", "auth_reconstruct_internal"},
        {"final_reveal", "final_reveal"},
        {"end-to-end", "total_eval"},
    };

    double accounted_time_ms = 0.0;
    u64 accounted_comm_bytes = 0;
    u64 accounted_rounds = 0;

    std::cout << "[PROFILE_TABLE] component,total_time_ms,total_comm_mb,total_rounds" << std::endl;
    for (size_t idx = 0; idx < OP_PROFILE_COUNT; ++idx) {
        const auto &stat = g_op_stats[idx];
        std::cout << "[PROFILE_TABLE] " << g_op_stats[idx].name
                  << "," << stat.time_ms
                  << "," << ((double)stat.comm_bytes / (1024.0 * 1024.0))
                  << "," << stat.rounds
                  << std::endl;
        accounted_time_ms += stat.time_ms;
        accounted_comm_bytes += stat.comm_bytes;
        accounted_rounds += stat.rounds;
    }

    double other_time_ms = g_total_eval_profile.time_ms - accounted_time_ms;
    if (other_time_ms < 0.0 && other_time_ms > -1e-6) other_time_ms = 0.0;
    u64 other_comm_bytes = (g_total_eval_profile.comm_bytes >= accounted_comm_bytes)
        ? (g_total_eval_profile.comm_bytes - accounted_comm_bytes) : 0;
    u64 other_rounds = (g_total_eval_profile.rounds >= accounted_rounds)
        ? (g_total_eval_profile.rounds - accounted_rounds) : 0;

    std::cout << "[PROFILE_TABLE] other"
              << "," << other_time_ms
              << "," << ((double)other_comm_bytes / (1024.0 * 1024.0))
              << "," << other_rounds
              << std::endl;
    std::cout << "[PROFILE_TABLE] leaf_accounted"
              << "," << (accounted_time_ms + other_time_ms)
              << "," << ((double)(accounted_comm_bytes + other_comm_bytes) / (1024.0 * 1024.0))
              << "," << (accounted_rounds + other_rounds)
              << std::endl;

    std::cout << "[PROFILE_OP_TABLE] component,calls,total_time_ms,total_comm_mb,total_rounds,avg_time_ms,avg_comm_mb,avg_rounds" << std::endl;
    for (size_t idx = 0; idx < OP_PROFILE_COUNT; ++idx) {
        const auto avg = avg_op_profile(idx);
        const auto &stat = g_op_stats[idx];
        std::cout << "[PROFILE_OP_TABLE] " << g_op_stats[idx].name
                  << "," << stat.calls
                  << "," << stat.time_ms
                  << "," << ((double)stat.comm_bytes / (1024.0 * 1024.0))
                  << "," << stat.rounds
                  << "," << avg.time_ms
                  << "," << avg.comm_mb
                  << "," << avg.rounds
                  << std::endl;
    }

    std::cout << "[PROFILE_TIMER_TABLE] component,total_time_ms,total_comm_mb,total_rounds" << std::endl;
    for (const auto &row : timer_rows) {
        if (std::string(row.timer_name) == "total_eval") {
            std::cout << "[PROFILE_TIMER_TABLE] " << row.component
                      << "," << g_total_eval_profile.time_ms
                      << "," << ((double)g_total_eval_profile.comm_bytes / (1024.0 * 1024.0))
                      << "," << g_total_eval_profile.rounds
                      << std::endl;
        } else {
            shark::utils::TimerStat stat{};
            shark::utils::get_timer_stat(row.timer_name, stat);
            std::cout << "[PROFILE_TIMER_TABLE] " << row.component
                      << "," << stat.accumulated_time
                      << "," << (double)stat.accumulated_comm / (1024.0 * 1024.0)
                      << "," << stat.accumulated_rounds
                      << std::endl;
        }
    }

    if (peer) {
        u64 total_comm = peer->bytesReceived() + peer->bytesSent();
        u64 total_rounds = peer->roundsReceived() + peer->roundsSent();
        std::cout << "[PROFILE_TIMER_TABLE] network_total"
                  << ",-"
                  << "," << (double)total_comm / (1024.0 * 1024.0)
                  << "," << total_rounds
                  << std::endl;
    }

    std::cout << "[PROFILE_RECONCILE] leaf_accounted_time_ms=" << (accounted_time_ms + other_time_ms)
              << " total_eval_time_ms=" << g_total_eval_profile.time_ms
              << std::endl;
    std::cout << "[PROFILE_RECONCILE] leaf_accounted_comm_mb="
              << ((double)(accounted_comm_bytes + other_comm_bytes) / (1024.0 * 1024.0))
              << " total_eval_comm_mb=" << ((double)g_total_eval_profile.comm_bytes / (1024.0 * 1024.0))
              << std::endl;
    std::cout << "[PROFILE_RECONCILE] leaf_accounted_rounds="
              << (accounted_rounds + other_rounds)
              << " total_eval_rounds=" << g_total_eval_profile.rounds
              << std::endl;
    std::cout.flags(old_flags);
    std::cout.precision(old_prec);
}

static void print_legacy_profile_lines() {
    if (party == DEALER) return;

    auto print_ms_kb = [](const char *alias, const char *timer_name) {
        if (std::string(timer_name) == "total_eval") {
            std::cout << alias << ": " << g_total_eval_profile.time_ms << " ms, "
                      << (g_total_eval_profile.comm_bytes / 1024.0) << " KB" << std::endl;
        } else {
            shark::utils::TimerStat stat{};
            shark::utils::get_timer_stat(timer_name, stat);
            std::cout << alias << ": " << stat.accumulated_time << " ms, "
                      << (stat.accumulated_comm / 1024.0) << " KB" << std::endl;
        }
    };

    print_ms_kb("ddpm", "total_eval");
    print_ms_kb("dealer_read_local", "dealer_read_local");
    print_ms_kb("input", "input");
    print_ms_kb("auth_reconstruct_internal", "auth_reconstruct_internal");
    print_ms_kb("final_reveal", "final_reveal");
}

struct KeygenProgressState {
    bool enabled = false;
    u64 total = 0;
    u64 current = 0;
    u64 denoise_step_index = 0;
    u64 denoise_step_total = 0;
    int denoise_t = -1;
    std::chrono::steady_clock::time_point start_time{};
};

static KeygenProgressState g_keygen_progress;

static const std::vector<std::string> &keygen_model_components() {
    static const std::vector<std::string> kComponents = {
        "share_model.time1", "share_model.time2",
        "share_model.conv_in", "share_model.conv_out",
        "share_model.down0_r0", "share_model.down0_r1", "share_model.down0_down",
        "share_model.down1_r0", "share_model.down1_r1", "share_model.down1_down",
        "share_model.down2_r0", "share_model.down2_r1",
        "share_model.mid_r0", "share_model.mid_r1",
        "share_model.up0_r0", "share_model.up0_r1", "share_model.up0_r2", "share_model.up0_up",
        "share_model.up1_r0", "share_model.up1_r1", "share_model.up1_r2", "share_model.up1_up",
        "share_model.up2_r0", "share_model.up2_r1", "share_model.up2_r2",
    };
    return kComponents;
}

static const std::vector<std::string> &keygen_step_components() {
    static const std::vector<std::string> kComponents = {
        "time_embedding",
        "unet.conv_in",
        "unet.down0_r0", "unet.down0_r1", "unet.down0_down",
        "unet.down1_r0", "unet.down1_r1", "unet.down1_down",
        "unet.down2_r0", "unet.down2_r1",
        "unet.mid_r0", "unet.mid_r1",
        "unet.up0_r0", "unet.up0_r1", "unet.up0_r2", "unet.up0_up",
        "unet.up1_r0", "unet.up1_r1", "unet.up1_r2", "unet.up1_up",
        "unet.up2_r0", "unet.up2_r1", "unet.up2_r2",
        "unet.out",
        "scheduler_noise",
        "scheduler_step",
    };
    return kComponents;
}

static std::string join_component_names(const std::vector<std::string> &components) {
    std::string out;
    for (size_t i = 0; i < components.size(); ++i) {
        if (i != 0) out += ", ";
        out += components[i];
    }
    return out;
}

static bool keygen_progress_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = std::getenv("SHARK_KEYGEN_PROGRESS");
        if (env == nullptr) {
            enabled = 1;
        } else if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' ||
                   env[0] == 'f' || env[0] == 'F') {
            enabled = 0;
        } else {
            enabled = 1;
        }
    }
    return enabled == 1;
}

static void keygen_progress_begin(u64 denoise_steps) {
    if (party != DEALER || !keygen_progress_enabled()) return;
    const auto &model_components = keygen_model_components();
    const auto &step_components = keygen_step_components();
    g_keygen_progress.enabled = true;
    g_keygen_progress.current = 0;
    g_keygen_progress.denoise_step_index = 0;
    g_keygen_progress.denoise_step_total = 0;
    g_keygen_progress.denoise_t = -1;
    g_keygen_progress.start_time = std::chrono::steady_clock::now();
    g_keygen_progress.total = (u64)model_components.size() + 2 + denoise_steps * (u64)step_components.size();

    std::cout << "[KEYGEN] total_steps=" << g_keygen_progress.total
              << " = model_share(" << model_components.size()
              << ") + inputs(2) + denoise_steps(" << denoise_steps
              << " x " << step_components.size() << ")\n";
    std::cout << "[KEYGEN] model_share components: "
              << join_component_names(model_components) << "\n";
    std::cout << "[KEYGEN] input components: input.cond_image, input.init_noise\n";
    std::cout << "[KEYGEN] per_step components: "
              << join_component_names(step_components) << "\n";
    std::cout.flush();
}

static void keygen_progress_set_step(u64 step_index, u64 step_total, int timestep) {
    if (!g_keygen_progress.enabled) return;
    g_keygen_progress.denoise_step_index = step_index;
    g_keygen_progress.denoise_step_total = step_total;
    g_keygen_progress.denoise_t = timestep;
}

static void keygen_progress_clear_step() {
    if (!g_keygen_progress.enabled) return;
    g_keygen_progress.denoise_step_index = 0;
    g_keygen_progress.denoise_step_total = 0;
    g_keygen_progress.denoise_t = -1;
}

static void keygen_progress_tick(const std::string &component) {
    if (!g_keygen_progress.enabled) return;
    ++g_keygen_progress.current;
    const u64 elapsed_ms = (u64)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - g_keygen_progress.start_time).count();
    std::cout << "[KEYGEN] global=" << g_keygen_progress.current
              << "/" << g_keygen_progress.total;
    if (g_keygen_progress.denoise_step_total != 0) {
        std::cout << " step=" << g_keygen_progress.denoise_step_index
                  << "/" << g_keygen_progress.denoise_step_total
                  << " t=" << g_keygen_progress.denoise_t;
    }
    std::cout << " elapsed_ms=" << elapsed_ms
              << " component=" << component << "\n";
    std::cout.flush();
}

static void keygen_progress_end() {
    if (!g_keygen_progress.enabled) return;
    const u64 elapsed_ms = (u64)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - g_keygen_progress.start_time).count();
    std::cout << "[KEYGEN] completed " << g_keygen_progress.current
              << "/" << g_keygen_progress.total << " steps\n";
    std::cout << "[KEYGEN] total_elapsed_ms=" << elapsed_ms << "\n";
    std::cout.flush();
    g_keygen_progress = KeygenProgressState{};
}

struct LinearWeights;
struct ConvWeights;
struct ConvTransposeWeights;
struct ResnetBlockWeights;
struct SelfAttnWeights;
struct DDPMWeights;
static void zero_plain(span<u64> &x);
static void init_attn(SelfAttnWeights &a, u64 dim);
static AuthShare conv_apply(u64 B, u64 H, u64 W, const AuthShare &x, const ConvWeights &w);
static AuthShare conv_transpose2x_apply(u64 B, u64 H, u64 W, const AuthShare &x, const ConvTransposeWeights &w);
static AuthShare auth_matmul_secret(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w);
static AuthShare auth_conv_secret(u64 k, u64 padding, u64 stride, u64 inC, u64 outC, u64 H, u64 W,
                                  const AuthShare &x, const AuthShare &w);
static AuthShare auth_sub(const AuthShare &a, const AuthShare &b);
static void clip_batch_check(const char *label);

static AuthShare auth_alloc(u64 size) {
    return AuthShare{span<u128>(size), span<u128>(size)};
}

static AuthShare auth_clone(const AuthShare &x) {
    AuthShare out = auth_alloc(x.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < x.share.size(); ++i) {
        out.share[i] = x.share[i];
        out.tag[i] = x.tag[i];
    }
    return out;
}

static AuthShare auth_from_public_raw(u64 size, u64 val) {
    AuthShare out = auth_alloc(size);
    const u128 lifted = u128(val);
    const u128 tag_val = mac_mul_u128(lifted);
    #pragma omp parallel for
    for (u64 i = 0; i < size; ++i) {
        out.share[i] = (party == SERVER || party == DEALER) ? lifted : u128(0);
        out.tag[i] = tag_val;
    }
    return out;
}

static AuthShare auth_from_public_const(u64 size, double v, int fp_bits) {
    int64_t q = (int64_t)std::llround(v * (double)(1ULL << fp_bits));
    return auth_from_public_raw(size, (u64)q);
}

static AuthShare auth_mul_const(const AuthShare &a, u64 c) {
    AuthShare out = auth_alloc(a.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < a.share.size(); ++i) {
        auth_local_mul_public_u128(a.share[i], a.tag[i], u128(c), out.share[i], out.tag[i]);
    }
    return out;
}

static AuthShare auth_from_public_vec(const std::vector<u64> &vals) {
    AuthShare out = auth_alloc((u64)vals.size());
    #pragma omp parallel for
    for (u64 i = 0; i < vals.size(); ++i) {
        out.share[i] = (party == SERVER || party == DEALER) ? u128(vals[i]) : u128(0);
        out.tag[i] = mac_mul_u128(u128(vals[i]));
    }
    return out;
}

static AuthShare profiled_time_embedding_from_public_vec(const std::vector<u64> &vals) {
    OpProfileScope profile(OP_TIME_EMBEDDING_OPEN);
    return auth_from_public_vec(vals);
}

static span<u64> auth_open_authenticated(const AuthShare &x) {
    span<u64> out(x.share.size());
    if (party == DEALER) {
        #pragma omp parallel for
        for (u64 i = 0; i < x.share.size(); ++i) out[i] = getLow(x.share[i]);
        return out;
    }
    clip_batch_check("auth_open_authenticated:before_open");
    auto tmp = auth_clone(x);
    auto opened = authenticated_reconstruct_full(tmp.share, tmp.tag);
    clip_batch_check("auth_open_authenticated:after_open");
    #pragma omp parallel for
    for (u64 i = 0; i < opened.size(); ++i) out[i] = getLow(opened[i]);
    return out;
}

static span<u64> profiled_auth_open_authenticated(const AuthShare &x, size_t profile_idx) {
    OpProfileScope profile(profile_idx);
    return auth_open_authenticated(x);
}

static span<u64> auth_reveal_to_owner_authenticated(const AuthShare &x, int owner) {
    always_assert(owner == SERVER || owner == CLIENT);
    const u64 size = x.share.size();

    if (party == DEALER) {
        span<u64> mask(size);
        randomize(mask);
        send_authenticated_ashare(mask);
        if (owner == SERVER) {
            server->send_array(mask);
        } else {
            client->send_array(mask);
        }
        return span<u64>(0);
    }

    auto [mask_share, mask_tag] = recv_authenticated_ashare(size);
    AuthShare mask_auth = auth_alloc(size);
    #pragma omp parallel for
    for (u64 i = 0; i < size; ++i) {
        mask_auth.share[i] = u128(mask_share[i]);
        mask_auth.tag[i] = mask_tag[i];
    }

    auto masked_auth = auth_sub(x, mask_auth);
    auto masked_plain = auth_open_authenticated(masked_auth);
    if (party != owner) {
        return span<u64>(0);
    }

    auto mask = dealer->recv_array<u64>(size);
    span<u64> out(size);
    #pragma omp parallel for
    for (u64 i = 0; i < size; ++i) {
        out[i] = masked_plain[i] + mask[i];
    }
    return out;
}

static span<u64> profiled_auth_reveal_to_owner_authenticated(const AuthShare &x, int owner, size_t profile_idx) {
    if (party == DEALER) {
        return auth_reveal_to_owner_authenticated(x, owner);
    }
    OpProfileScope profile(profile_idx);
    return auth_reveal_to_owner_authenticated(x, owner);
}

static AuthShare authenticated_input_from_owner(const span<u64> &x, int owner) {
    always_assert(owner == SERVER || owner == CLIENT);

    const u64 size = x.size();
    AuthShare out = auth_alloc(size);

    if (party == DEALER) {
        span<u64> r_clear(size);
        randomize(r_clear);
        send_authenticated_ashare(r_clear);
        if (owner == SERVER) {
            server->send_array(r_clear);
        } else {
            client->send_array(r_clear);
        }
        #pragma omp parallel for
        for (u64 i = 0; i < size; ++i) {
            out.share[i] = u128(r_clear[i]);
            out.tag[i] = mac_mul_u64(r_clear[i]);
        }
        return out;
    }

    auto [r_share, r_tag] = recv_authenticated_ashare(size);
    span<u64> d(size);
    #pragma omp parallel for
    for (u64 i = 0; i < size; ++i) d[i] = u64(0);

    if (party == owner) {
        auto r_clear = dealer->recv_array<u64>(size);
        #pragma omp parallel for
        for (u64 i = 0; i < size; ++i) {
            d[i] = x[i] - r_clear[i];
        }
        peer->send_array(d);
    } else {
        peer->recv_array(d);
    }

    #pragma omp parallel for
    for (u64 i = 0; i < size; ++i) {
        auth_local_add_public_u128(u128(r_share[i]), r_tag[i], u128(d[i]),
                                   party == SERVER, out.share[i], out.tag[i]);
    }
    return out;
}

static AuthShare profiled_authenticated_input_from_owner(const span<u64> &x, int owner) {
    OpProfileScope profile(OP_INPUT);
    return authenticated_input_from_owner(x, owner);
}

static AuthShare auth_add(const AuthShare &a, const AuthShare &b) {
    always_assert(a.share.size() == b.share.size());
    AuthShare out = auth_alloc(a.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < a.share.size(); ++i) {
        out.share[i] = a.share[i] + b.share[i];
        out.tag[i] = a.tag[i] + b.tag[i];
    }
    return out;
}

static AuthShare auth_sub(const AuthShare &a, const AuthShare &b) {
    always_assert(a.share.size() == b.share.size());
    AuthShare out = auth_alloc(a.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < a.share.size(); ++i) {
        out.share[i] = a.share[i] - b.share[i];
        out.tag[i] = a.tag[i] - b.tag[i];
    }
    return out;
}

static AuthShare auth_neg(const AuthShare &a) {
    AuthShare out = auth_alloc(a.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < a.share.size(); ++i) {
        out.share[i] = u128(0) - a.share[i];
        out.tag[i] = u128(0) - a.tag[i];
    }
    return out;
}

static AuthShare auth_mul(const AuthShare &a, const AuthShare &b) {
    auto tmp = mul::call_share_secret_full(a.share, a.tag, b.share, b.tag);
    return AuthShare{std::move(tmp.share), std::move(tmp.tag)};
}

static AuthShare auth_matmul_secret(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w) {
    auto tmp = matmul::call_share_secret_full(M, K, N, x.share, x.tag, w.share, w.tag);
    return AuthShare{std::move(tmp.share), std::move(tmp.tag)};
}

static AuthShare auth_conv_secret(u64 k, u64 padding, u64 stride, u64 inC, u64 outC, u64 H, u64 W,
                                  const AuthShare &x, const AuthShare &w) {
    auto tmp = conv::call_share_secret_full(k, padding, stride, inC, outC, H, W, x.share, x.tag, w.share, w.tag);
    return AuthShare{std::move(tmp.share), std::move(tmp.tag)};
}

static AuthShare auth_shift(const AuthShare &x, u64 shift) {
    auto tmp = ars::call_share_secret_full(x.share, x.tag, (int)shift);
    return AuthShare{std::move(tmp.share), std::move(tmp.tag)};
}

static AuthShare make_public_const(u64 size, double v, int fp_bits) {
    return auth_from_public_const(size, v, fp_bits);
}

static AuthShare make_public_raw(u64 size, u64 val) {
    return auth_from_public_raw(size, val);
}

static AuthShare neg_span(const AuthShare &x) {
    return auth_neg(x);
}

static AuthShare ss_ars(const AuthShare &x, u64 shift) {
    return auth_shift(x, shift);
}

static AuthShare mul_noopen(const AuthShare &x, const AuthShare &y) {
    return auth_mul(x, y);
}

static AuthShare matmul_secret_noopen(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w) {
    return auth_matmul_secret(M, K, N, x, w);
}

static AuthShare conv_secret_noopen(u64 k, u64 padding, u64 stride, u64 inC, u64 outC, u64 H, u64 W,
                                    const AuthShare &x, const AuthShare &w) {
    return auth_conv_secret(k, padding, stride, inC, outC, H, W, x, w);
}

static AuthShare cmp_ge_zero_noopen(const AuthShare &x) {
    u64 n = x.share.size();
    AuthShare out = auth_alloc(n);
    const u128 bias = u128(1) << 63;

    if (party == DEALER) {
        span<u128> r(n);
        span<u128> alpha(n);
        #pragma omp parallel for
        for (u64 i = 0; i < n; ++i) {
            r[i] = u128(rand<u64>());
            alpha[i] = u128(getLow(r[i] + bias));
        }
        send_authenticated_ashare_full(r);
        send_dcfring(alpha, 64);
        for (u64 i = 0; i < n; ++i) {
            u64 bit = ((int64_t)getLow(x.share[i]) >= 0) ? 1ull : 0ull;
            out.share[i] = u128(bit);
            out.tag[i] = mac_mul_u128(u128(bit));
        }
        return out;
    }

    auto [r_share, r_tag] = recv_authenticated_ashare_full(n);
    AuthShare masked_auth = auth_alloc(n);
    #pragma omp parallel for
    for (u64 i = 0; i < n; ++i) {
        masked_auth.share[i] = x.share[i] + r_share[i] + ((party == SERVER) ? bias : u128(0));
        masked_auth.tag[i] = x.tag[i] + r_tag[i] + mac_mul_u128(bias);
    }
    clip_batch_check("cmp_ge_zero_noopen:before_masked_open");
    auto masked_tmp = auth_clone(masked_auth);
    auto masked = authenticated_reconstruct_full(masked_tmp.share, masked_tmp.tag);
    clip_batch_check("cmp_ge_zero_noopen:after_masked_open");

    auto keys = recv_dcfring(n, 64);
    span<u128> lt_share(n);
    span<u128> lt_tag(n);
    for (u64 i = 0; i < n; ++i) {
        auto [s_share, s_tag] = crypto::dcfring_eval(party, keys[i], u128(getLow(masked[i])), false);
        lt_share[i] = s_share;
        lt_tag[i] = s_tag;
    }
    AuthShare lt{std::move(lt_share), std::move(lt_tag)};
    return auth_sub(make_public_raw(n, 1), lt);
}

static AuthShare select_noopen(const AuthShare &cond, const AuthShare &x) {
    return mul_noopen(cond, x);
}

#define ADD_CALL(...) auth_add(__VA_ARGS__)
#define MUL_CALL(...) mul_noopen(__VA_ARGS__)
#define LRS_CALL(...) ss_ars(__VA_ARGS__)
#define CMP_GE_ZERO_CALL(...) cmp_ge_zero_noopen(__VA_ARGS__)
#define SELECT_CALL(cond, ...) select_noopen(cond, __VA_ARGS__)

static u64 current_rng_seed = 0x123456789ULL;

static inline u64 rng_next(u64 &s) {
    s = s * 6364136223846793005ULL + 1;
    return s;
}

static inline double rng_u01(u64 &s) {
    const double inv = 1.0 / 9007199254740992.0;
    u64 x = rng_next(s) >> 11;
    return (x + 1.0) * inv;
}

static inline u64 qrand_normal(int fp_bits, double stddev) {
    const double two_pi = 6.2831853071795864769;
    double u1 = rng_u01(current_rng_seed);
    double u2 = rng_u01(current_rng_seed);
    double r = std::sqrt(-2.0 * std::log(u1));
    double z = r * std::cos(two_pi * u2);
    int64_t q = (int64_t)std::llround(z * stddev * (double)(1ULL << fp_bits));
    return (u64)q;
}

static inline u64 qrand_uniform_symmetric(int fp_bits, double bound) {
    double u = rng_u01(current_rng_seed);
    double x = (2.0 * u - 1.0) * bound;
    int64_t q = (int64_t)std::llround(x * (double)(1ULL << fp_bits));
    return (u64)q;
}

static inline u64 idx4(u64 b, u64 h, u64 w, u64 c, u64 H, u64 W, u64 C) {
    return ((b * H + h) * W + w) * C + c;
}

static inline u64 conv_out_dim(u64 in, u64 k, u64 stride, u64 padding) {
    always_assert(in + 2 * padding >= k);
    return (in - k + 2 * padding) / stride + 1;
}

static void clip_batch_check(const char *label) {
    if (party == DEALER) return;
    (void)label;
    shark::protocols::batch_check();
}

static bool write_jpg_from_fixed(const span<u64> &img, u64 H, u64 W, u64 C, int fp_bits, const char *path) {
    if (img.size() != H * W * C) return false;
    const int outC = (C == 1) ? 3 : (int)C;
    std::vector<unsigned char> buf((size_t)H * (size_t)W * (size_t)outC);
    const double scale = 1.0 / (double)(1ULL << fp_bits);
    for (u64 h = 0; h < H; ++h) {
        for (u64 widx = 0; widx < W; ++widx) {
            if (C == 1) {
                size_t src = ((size_t)h * W + widx) * C;
                double v = (double)(int64_t)img[src] * scale;
                v = (v + 1.0) * 0.5;
                v = std::max(0.0, std::min(1.0, v));
                unsigned char u = (unsigned char)std::llround(v * 255.0);
                size_t dst = ((size_t)h * W + widx) * outC;
                buf[dst + 0] = u;
                buf[dst + 1] = u;
                buf[dst + 2] = u;
            } else {
                for (u64 c = 0; c < C; ++c) {
                    size_t src = (((size_t)h * W + widx) * C) + c;
                    double v = (double)(int64_t)img[src] * scale;
                    v = (v + 1.0) * 0.5;
                    v = std::max(0.0, std::min(1.0, v));
                    unsigned char u = (unsigned char)std::llround(v * 255.0);
                    size_t dst = ((size_t)h * W + widx) * outC + c;
                    buf[dst] = u;
                }
            }
        }
    }
    return stbi_write_jpg(path, (int)W, (int)H, outC, buf.data(), 95) != 0;
}

static int find_arg(int argc, char **argv, const std::string &name) {
    for (int i = 1; i < argc; ++i) {
        if (name == argv[i]) return i;
    }
    return -1;
}

static bool has_flag(int argc, char **argv, const std::string &name) {
    return find_arg(argc, argv, name) >= 0;
}

static std::string get_arg_string(int argc, char **argv, const std::string &name, const std::string &def) {
    int idx = find_arg(argc, argv, name);
    if (idx >= 0 && idx + 1 < argc) return argv[idx + 1];
    return def;
}

static u64 get_arg_u64(int argc, char **argv, const std::string &name, u64 def) {
    int idx = find_arg(argc, argv, name);
    if (idx >= 0 && idx + 1 < argc) return (u64)std::strtoull(argv[idx + 1], nullptr, 10);
    return def;
}

static double get_arg_double(int argc, char **argv, const std::string &name, double def) {
    int idx = find_arg(argc, argv, name);
    if (idx >= 0 && idx + 1 < argc) return std::strtod(argv[idx + 1], nullptr);
    return def;
}

static std::string read_text_file(const char *path) {
    if (path == nullptr || path[0] == '\0') return {};
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static size_t json_find_key(const std::string &text, const char *key) {
    const std::string quoted = std::string("\"") + key + "\"";
    return text.find(quoted);
}

static bool json_has_key(const std::string &text, const char *key) {
    return json_find_key(text, key) != std::string::npos;
}

static size_t json_find_value_start(const std::string &text, const char *key) {
    size_t pos = json_find_key(text, key);
    if (pos == std::string::npos) return std::string::npos;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return std::string::npos;
    ++pos;
    while (pos < text.size() && std::isspace((unsigned char)text[pos])) {
        ++pos;
    }
    return pos;
}

static std::string json_extract_balanced_blob(const std::string &text, size_t start, char open_ch, char close_ch) {
    if (start == std::string::npos || start >= text.size() || text[start] != open_ch) return {};
    int depth = 0;
    for (size_t i = start; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch == open_ch) {
            ++depth;
        } else if (ch == close_ch) {
            --depth;
            if (depth == 0) {
                return text.substr(start, i - start + 1);
            }
        }
    }
    return {};
}

static std::string json_extract_object_blob(const std::string &text, const char *key) {
    const size_t start = json_find_value_start(text, key);
    return json_extract_balanced_blob(text, start, '{', '}');
}

static std::string json_extract_number_token(const std::string &text, const char *key) {
    size_t pos = json_find_value_start(text, key);
    if (pos == std::string::npos) return {};
    size_t end = pos;
    while (end < text.size()) {
        const char ch = text[end];
        if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E') {
            ++end;
        } else {
            break;
        }
    }
    if (end == pos) return {};
    return text.substr(pos, end - pos);
}

static std::string json_extract_string_like(const std::string &text, const char *key, const std::string &def) {
    size_t pos = json_find_value_start(text, key);
    if (pos == std::string::npos) return def;
    if (text[pos] == '"') {
        ++pos;
        size_t end = pos;
        while (end < text.size()) {
            if (text[end] == '"' && text[end - 1] != '\\') {
                return text.substr(pos, end - pos);
            }
            ++end;
        }
        return def;
    }
    size_t end = pos;
    while (end < text.size()) {
        const char ch = text[end];
        if (std::isspace((unsigned char)ch) || ch == ',' || ch == '}' || ch == ']') break;
        ++end;
    }
    if (end == pos) return def;
    return text.substr(pos, end - pos);
}

static u64 json_extract_u64(const std::string &text, const char *key, u64 def) {
    const std::string token = json_extract_number_token(text, key);
    if (token.empty()) return def;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(token.c_str(), &end, 10);
    if (end == token.c_str()) return def;
    return (u64)parsed;
}

static int json_extract_int(const std::string &text, const char *key, int def) {
    const std::string token = json_extract_number_token(text, key);
    if (token.empty()) return def;
    char *end = nullptr;
    const long parsed = std::strtol(token.c_str(), &end, 10);
    if (end == token.c_str()) return def;
    return (int)parsed;
}

static double json_extract_double(const std::string &text, const char *key, double def) {
    const std::string token = json_extract_number_token(text, key);
    if (token.empty()) return def;
    char *end = nullptr;
    const double parsed = std::strtod(token.c_str(), &end);
    if (end == token.c_str()) return def;
    return parsed;
}

static bool json_extract_bool(const std::string &text, const char *key, bool def) {
    const std::string token = json_extract_string_like(text, key, "");
    if (token.empty()) return def;
    if (token == "true" || token == "1") return true;
    if (token == "false" || token == "0") return false;
    return def;
}

static u64 fingerprint_text_blob(const std::string &text) {
    u64 acc = 1469598103934665603ULL;
    for (unsigned char ch : text) {
        acc ^= (u64)ch;
        acc *= 1099511628211ULL;
    }
    return acc;
}

struct DdpmBenchConfig {
    bool loaded = false;
    std::string config_path{};
    u64 config_fingerprint = 0;
    std::string selected_launch{};

    int launch_party = -1;
    std::string launch_ip = "127.0.0.1";
    int launch_port = 42069;
    bool seed_set = false;
    u64 seed = 0;

    std::string shark_keybuf_io_mb{};
    std::string omp_proc_bind{};
    std::string omp_places{};
    std::string omp_num_threads{};
    std::string omp_wait_policy{};
    std::string gomp_cpu_affinity{};

    u64 image_hw = (u64)ddpm_para::kModelConfig.image_hw;
    u64 base_ch = (u64)ddpm_para::kModelConfig.base_ch;
    u64 mid_ch = (u64)ddpm_para::kModelConfig.mid_ch;
    u64 temb_dim = (u64)ddpm_para::kModelConfig.temb_dim;
    u64 norm_groups = (u64)ddpm_para::kModelConfig.norm_groups;

    u64 model_input_channels = (u64)ddpm_para::kIoConfig.model_input_channels;
    u64 sample_channels = (u64)ddpm_para::kIoConfig.sample_channels;
    u64 cond_channels = (u64)ddpm_para::kIoConfig.cond_channels;
    u64 output_select_channel = (u64)ddpm_para::kIoConfig.output_select_channel;
    u64 saved_image_channels = (u64)ddpm_para::kIoConfig.saved_image_channels;
    double noise_stddev = ddpm_para::kIoConfig.noise_stddev;

    u64 num_timesteps = (u64)ddpm_para::kSchedulerConfig.num_train_timesteps;
    std::string beta_schedule = ddpm_para::kSchedulerConfig.beta_schedule;
    std::string prediction_type = ddpm_para::kSchedulerConfig.prediction_type;
    bool clip_sample = ddpm_para::kSchedulerConfig.clip_sample;
    double clip_sample_range = ddpm_para::kSchedulerConfig.clip_sample_range;
    bool thresholding = ddpm_para::kSchedulerConfig.thresholding;
    double dynamic_thresholding_ratio = ddpm_para::kSchedulerConfig.dynamic_thresholding_ratio;
    double sample_max_value = ddpm_para::kSchedulerConfig.sample_max_value;
    std::string variance_type = ddpm_para::kSchedulerConfig.variance_type;
    std::string timestep_spacing = ddpm_para::kSchedulerConfig.timestep_spacing;

    u64 steps = (u64)ddpm_para::kRuntimeConfig.steps;
    double strength = ddpm_para::kRuntimeConfig.strength;
    std::string image_path = ddpm_para::kRuntimeConfig.image_path;
    std::string out_path = ddpm_para::kRuntimeConfig.out_path;
    std::string cond_out_path = ddpm_para::kRuntimeConfig.cond_out_path;
};

static int party_from_launch_name(const std::string &name, int def = -1) {
    if (name == "server") return SERVER;
    if (name == "client") return CLIENT;
    if (name == "dealer") return DEALER;
    if (name == "emul") return EMUL;
    return def;
}

static std::string launch_name_from_party_value(int party_value) {
    switch (party_value) {
        case SERVER: return "server";
        case CLIENT: return "client";
        case DEALER: return "dealer";
        case EMUL: return "emul";
        default: return {};
    }
}

static int parse_party_arg_if_present(int argc, char **argv) {
    if (argc <= 1) return -1;
    const std::string arg = argv[1];
    if (arg.empty() || arg[0] == '-') return -1;
    char *end = nullptr;
    const long parsed = std::strtol(arg.c_str(), &end, 10);
    if (end == arg.c_str() || *end != '\0') return -1;
    return (int)parsed;
}

static void apply_env_object(DdpmBenchConfig &cfg, const std::string &env_obj) {
    if (env_obj.empty()) return;
    const std::string keybuf = json_extract_string_like(env_obj, "shark_keybuf_io_mb", "");
    if (!keybuf.empty()) cfg.shark_keybuf_io_mb = keybuf;
    const std::string proc_bind = json_extract_string_like(env_obj, "omp_proc_bind", "");
    if (!proc_bind.empty()) cfg.omp_proc_bind = proc_bind;
    const std::string places = json_extract_string_like(env_obj, "omp_places", "");
    if (!places.empty()) cfg.omp_places = places;
    const std::string threads = json_extract_string_like(env_obj, "omp_num_threads", "");
    if (!threads.empty()) cfg.omp_num_threads = threads;
    const std::string wait_policy = json_extract_string_like(env_obj, "omp_wait_policy", "");
    if (!wait_policy.empty()) cfg.omp_wait_policy = wait_policy;
    const std::string affinity = json_extract_string_like(env_obj, "gomp_cpu_affinity", "");
    if (!affinity.empty()) cfg.gomp_cpu_affinity = affinity;
}

static void apply_launch_object(DdpmBenchConfig &cfg, const std::string &launch_obj) {
    if (launch_obj.empty()) return;
    const std::string role = json_extract_string_like(launch_obj, "role", "");
    if (!role.empty()) {
        cfg.launch_party = party_from_launch_name(role, cfg.launch_party);
    } else if (json_has_key(launch_obj, "party")) {
        cfg.launch_party = json_extract_int(launch_obj, "party", cfg.launch_party);
    }
    cfg.launch_ip = json_extract_string_like(launch_obj, "ip", cfg.launch_ip);
    cfg.launch_port = json_extract_int(launch_obj, "port", cfg.launch_port);
    if (json_has_key(launch_obj, "seed")) {
        cfg.seed = json_extract_u64(launch_obj, "seed", cfg.seed);
        cfg.seed_set = true;
    }
    apply_env_object(cfg, json_extract_object_blob(launch_obj, "env"));
}

static DdpmBenchConfig load_ddpm_bench_config(int argc, char **argv) {
    DdpmBenchConfig cfg;
    std::string path = get_arg_string(argc, argv, "--config", "");
    if (path.empty()) {
        const char *env_path = std::getenv("DDPM_BENCH_CONFIG_JSON");
        if (env_path != nullptr && env_path[0] != '\0') {
            path = env_path;
        }
    }
    if (path.empty()) return cfg;

    const std::string json = read_text_file(path.c_str());
    always_assert(!json.empty());

    cfg.loaded = true;
    cfg.config_path = path;
    cfg.config_fingerprint = fingerprint_text_blob(json);

    const std::string model = json_extract_object_blob(json, "model");
    if (!model.empty()) {
        if (json_has_key(model, "channels")) {
            const u64 channels = json_extract_u64(model, "channels", cfg.base_ch);
            cfg.base_ch = channels;
            cfg.mid_ch = (u64)ddpm_para::mid_channels_from_channels(channels);
            cfg.temb_dim = (u64)ddpm_para::temb_from_channels(channels);
        }
        cfg.image_hw = json_extract_u64(model, "image_hw", cfg.image_hw);
        cfg.base_ch = json_extract_u64(model, "base_ch", cfg.base_ch);
        cfg.mid_ch = json_extract_u64(model, "mid_ch", cfg.mid_ch);
        cfg.temb_dim = json_extract_u64(model, "temb_dim", cfg.temb_dim);
        cfg.norm_groups = json_extract_u64(model, "norm_groups", cfg.norm_groups);
    }

    const std::string io = json_extract_object_blob(json, "io");
    if (!io.empty()) {
        cfg.model_input_channels = json_extract_u64(io, "model_input_channels", cfg.model_input_channels);
        cfg.sample_channels = json_extract_u64(io, "sample_channels", cfg.sample_channels);
        cfg.cond_channels = json_extract_u64(io, "cond_channels", cfg.cond_channels);
        cfg.output_select_channel = json_extract_u64(io, "output_select_channel", cfg.output_select_channel);
        cfg.saved_image_channels = json_extract_u64(io, "saved_image_channels", cfg.saved_image_channels);
        cfg.noise_stddev = json_extract_double(io, "noise_stddev", cfg.noise_stddev);
    }

    const std::string scheduler = json_extract_object_blob(json, "scheduler");
    if (!scheduler.empty()) {
        cfg.num_timesteps = json_extract_u64(scheduler, "num_train_timesteps", cfg.num_timesteps);
        cfg.beta_schedule = json_extract_string_like(scheduler, "beta_schedule", cfg.beta_schedule);
        cfg.prediction_type = json_extract_string_like(scheduler, "prediction_type", cfg.prediction_type);
        cfg.clip_sample = json_extract_bool(scheduler, "clip_sample", cfg.clip_sample);
        cfg.clip_sample_range = json_extract_double(scheduler, "clip_sample_range", cfg.clip_sample_range);
        cfg.thresholding = json_extract_bool(scheduler, "thresholding", cfg.thresholding);
        cfg.dynamic_thresholding_ratio =
            json_extract_double(scheduler, "dynamic_thresholding_ratio", cfg.dynamic_thresholding_ratio);
        cfg.sample_max_value = json_extract_double(scheduler, "sample_max_value", cfg.sample_max_value);
        cfg.variance_type = json_extract_string_like(scheduler, "variance_type", cfg.variance_type);
        cfg.timestep_spacing =
            json_extract_string_like(scheduler, "ddim_discr_method",
                                     json_extract_string_like(scheduler, "timestep_spacing", cfg.timestep_spacing));
    }

    const std::string runtime = json_extract_object_blob(json, "runtime");
    if (!runtime.empty()) {
        cfg.steps = json_extract_u64(runtime, "steps", cfg.steps);
        cfg.strength = json_extract_double(runtime, "strength", cfg.strength);
        cfg.image_path = json_extract_string_like(runtime, "image_path", cfg.image_path);
        cfg.out_path = json_extract_string_like(runtime, "out_path", cfg.out_path);
        cfg.cond_out_path = json_extract_string_like(runtime, "cond_out_path", cfg.cond_out_path);
    }

    apply_env_object(cfg, json_extract_object_blob(json, "env"));

    std::string launch_name = get_arg_string(argc, argv, "--launch", "");
    if (launch_name.empty()) {
        const char *env_launch = std::getenv("DDPM_LAUNCH_ROLE");
        if (env_launch != nullptr && env_launch[0] != '\0') {
            launch_name = env_launch;
        }
    }
    if (launch_name.empty()) {
        launch_name = launch_name_from_party_value(parse_party_arg_if_present(argc, argv));
    }

    const std::string launch_profiles = json_extract_object_blob(json, "launch_profiles");
    if (!launch_profiles.empty()) {
        always_assert(!launch_name.empty());
        const std::string launch_obj = json_extract_object_blob(launch_profiles, launch_name.c_str());
        always_assert(!launch_obj.empty());
        cfg.selected_launch = launch_name;
        apply_launch_object(cfg, launch_obj);
    } else {
        const std::string launch_obj = json_extract_object_blob(json, "launch");
        if (!launch_obj.empty()) {
            cfg.selected_launch = launch_name.empty() ? "launch" : launch_name;
            apply_launch_object(cfg, launch_obj);
        }
    }

    if (cfg.launch_party < 0) {
        cfg.launch_party = parse_party_arg_if_present(argc, argv);
    }
    return cfg;
}

static void set_env_string_if_nonempty(const char *name, const std::string &value) {
    if (name == nullptr || value.empty()) return;
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static void apply_ddpm_bench_env(const DdpmBenchConfig &cfg) {
    set_env_string_if_nonempty("SHARK_KEYBUF_IO_MB", cfg.shark_keybuf_io_mb);
    set_env_string_if_nonempty("OMP_PROC_BIND", cfg.omp_proc_bind);
    set_env_string_if_nonempty("OMP_PLACES", cfg.omp_places);
    set_env_string_if_nonempty("OMP_NUM_THREADS", cfg.omp_num_threads);
    set_env_string_if_nonempty("OMP_WAIT_POLICY", cfg.omp_wait_policy);
    set_env_string_if_nonempty("GOMP_CPU_AFFINITY", cfg.gomp_cpu_affinity);
}

static void init_from_ddpm_bench_config(const DdpmBenchConfig &cfg, int argc, char **argv) {
    if (!cfg.loaded || cfg.selected_launch.empty()) {
        init::from_args(argc, argv);
        return;
    }
    if (cfg.launch_party < 0) {
        init::from_args(argc, argv);
        return;
    }
    if (cfg.launch_party == EMUL) {
        party = EMUL;
        return;
    }
    if (cfg.launch_party == DEALER) {
        init::gen(0xdeadbeef);
        return;
    }
    always_assert(cfg.launch_party == SERVER || cfg.launch_party == CLIENT);
    init::eval(cfg.launch_party, cfg.launch_ip, cfg.launch_port, false);
}

static AuthShare broadcast_rows(u64 rows, u64 cols, const AuthShare &x) {
    always_assert(x.share.size() == rows);
    AuthShare out = auth_alloc(rows * cols);
    #pragma omp parallel for collapse(2)
    for (u64 i = 0; i < rows; ++i) {
        for (u64 j = 0; j < cols; ++j) {
            u64 idx = i * cols + j;
            out.share[idx] = x.share[i];
            out.tag[idx] = x.tag[i];
        }
    }
    return out;
}

static AuthShare broadcast_row_vector(u64 rows, u64 cols, const AuthShare &x) {
    always_assert(x.share.size() == cols);
    AuthShare out = auth_alloc(rows * cols);
    #pragma omp parallel for collapse(2)
    for (u64 i = 0; i < rows; ++i) {
        for (u64 j = 0; j < cols; ++j) {
            u64 idx = i * cols + j;
            out.share[idx] = x.share[j];
            out.tag[idx] = x.tag[j];
        }
    }
    return out;
}

static AuthShare broadcast_channels_nhwc(u64 B, u64 H, u64 W, u64 C, const AuthShare &x) {
    always_assert(x.share.size() == C);
    AuthShare out = auth_alloc(B * H * W * C);
    #pragma omp parallel for collapse(4)
    for (u64 n = 0; n < B; ++n) {
        for (u64 h = 0; h < H; ++h) {
            for (u64 widx = 0; widx < W; ++widx) {
                for (u64 c = 0; c < C; ++c) {
                    u64 idx = idx4(n, h, widx, c, H, W, C);
                    out.share[idx] = x.share[c];
                    out.tag[idx] = x.tag[c];
                }
            }
        }
    }
    return out;
}

static AuthShare broadcast_batch_vector_nhwc(u64 B, u64 H, u64 W, u64 C, const AuthShare &x) {
    always_assert(x.share.size() == B * C);
    AuthShare out = auth_alloc(B * H * W * C);
    #pragma omp parallel for collapse(4)
    for (u64 n = 0; n < B; ++n) {
        for (u64 h = 0; h < H; ++h) {
            for (u64 widx = 0; widx < W; ++widx) {
                for (u64 c = 0; c < C; ++c) {
                    u64 idx = idx4(n, h, widx, c, H, W, C);
                    u64 src = n * C + c;
                    out.share[idx] = x.share[src];
                    out.tag[idx] = x.tag[src];
                }
            }
        }
    }
    return out;
}

static AuthShare cmp_ge_public(const AuthShare &x, double v) {
    return CMP_GE_ZERO_CALL(ADD_CALL(x, make_public_const(x.share.size(), -v, f)));
}

static AuthShare cmp_gt_public(const AuthShare &x, double v) {
    const double eps = 1.0 / (double)(1ULL << f);
    return cmp_ge_public(x, v + eps);
}

static AuthShare cmp_le_public(const AuthShare &x, double v) {
    return auth_sub(make_public_raw(x.share.size(), 1), cmp_gt_public(x, v));
}

static void update_if(AuthShare &cur, const AuthShare &cond, const AuthShare &next) {
    auto delta = ADD_CALL(next, neg_span(cur));
    cur = ADD_CALL(cur, SELECT_CALL(cond, delta));
}

static constexpr int kNormPolyOrder = 3;
static constexpr int kNormNewtonIters = 2;
static constexpr double kNormVarFloor = 1e-12;
static constexpr double kNormEps = 1e-5;

struct SecureNormApproxConfig {
    bool approx_mode = true;
    int poly_order = kNormPolyOrder;
    int nr_iters = kNormNewtonIters;
    bool use_exact_fallback = false;
    double eps = kNormEps;
    double min_denominator = 1e-4;
};

static SecureNormApproxConfig make_secure_norm_approx_config(double eps = kNormEps) {
    SecureNormApproxConfig cfg;
    cfg.eps = eps;
    return cfg;
}

static double rsqrt_scale_from_shift(int shift) {
    if (shift >= 0) {
        int half = shift / 2;
        bool odd = (shift & 1) != 0;
        double scale = std::ldexp(1.0, half);
        if (odd) scale *= std::sqrt(2.0);
        return scale;
    }

    int pos_shift = -shift;
    int half = pos_shift / 2;
    bool odd = (pos_shift & 1) != 0;
    double scale = std::ldexp(1.0, -half);
    if (odd) scale /= std::sqrt(2.0);
    return scale;
}

static AuthShare chexp_unprofiled(const AuthShare &x);

static double time_embedding_exp_approx(double x) {
    const double u = 0.5 * x;
    const double c0 = 1.0;
    const double c1 = 0.9736789799002139;
    const double c2 = 0.4349844528710021;
    const double c3 = 0.10637438059443326;
    const double c4 = 0.013503439864618789;
    const double c5 = 0.000682062067913879;
    const double u2 = u * u;
    const double u4 = u2 * u2;
    const double a0 = c0 + c1 * u;
    const double a1 = c2 + c3 * u;
    const double a2 = c4 + c5 * u;
    const double y = a0 + a1 * u2 + a2 * u4;
    return y * y;
}

static std::vector<u64> make_timestep_embedding_plain(u64 timestep, u64 dim) {
    std::vector<u64> out(dim, 0);
    if (dim == 0) return out;
    const u64 half = dim / 2;
    always_assert(dim >= 2);
    always_assert(half > 0);
    const double log_max = std::log(10000.0);
    for (u64 i = 0; i < half; ++i) {
        const double freq = time_embedding_exp_approx(-log_max * (double)i / (double)half);
        const double arg = (double)timestep * freq;
        out[i] = (u64)(int64_t)std::llround(std::sin(arg) * (double)(1ULL << f));
        out[i + half] = (u64)(int64_t)std::llround(std::cos(arg) * (double)(1ULL << f));
    }
    if (dim & 1) out.back() = 0;
    return out;
}

static std::vector<int> build_timesteps(u64 num_train_timesteps, u64 steps,
                                        const std::string &timestep_spacing) {
    always_assert(num_train_timesteps > 0);
    always_assert(steps > 0);
    always_assert(steps <= num_train_timesteps);
    std::vector<int> out;
    out.reserve(steps);

    if (timestep_spacing == "uniform") {
        const u64 c = num_train_timesteps / steps;
        always_assert(c >= 1);
        for (u64 t = 0; t < num_train_timesteps && out.size() < steps; t += c) {
            const int shifted = (int)t + 1;
            always_assert(shifted >= 1);
            always_assert((u64)shifted <= num_train_timesteps);
            out.push_back(shifted);
        }
        std::reverse(out.begin(), out.end());
    } else if (timestep_spacing == "quad") {
        const double upper = std::sqrt((double)num_train_timesteps * 0.8);
        if (steps == 1) {
            out.push_back(1);
        } else {
            std::vector<int> forward;
            forward.reserve(steps);
            for (u64 i = 0; i < steps; ++i) {
                const double frac = (double)i / (double)(steps - 1);
                const double raw = upper * frac;
                const int t = (int)(raw * raw);
                forward.push_back(t + 1);
            }
            out.assign(forward.rbegin(), forward.rend());
        }
    } else {
        always_assert(false && "unsupported timestep_spacing; expected uniform or quad");
    }
    return out;
}

static std::vector<double> build_betas_linear_cipherdm(u64 num_timesteps) {
    std::vector<double> betas(num_timesteps, 0.0);
    const double low = 1e-4 * 1000.0 / (double)num_timesteps;
    const double high = 2e-2 * 1000.0 / (double)num_timesteps;
    if (num_timesteps == 1) {
        betas[0] = low;
        return betas;
    }
    for (u64 i = 0; i < num_timesteps; ++i) {
        const double frac = (double)i / (double)(num_timesteps - 1);
        betas[i] = low + (high - low) * frac;
    }
    return betas;
}

static std::vector<double> build_betas_cosine_cipherdm(u64 num_timesteps) {
    auto alpha_bar = [num_timesteps](double t) {
        const double x = (t / (double)num_timesteps + 0.008) / 1.008;
        return std::pow(std::cos(x * 3.14159265358979323846 * 0.5), 2.0);
    };

    std::vector<double> alphas(num_timesteps + 1, 0.0);
    const double f0 = alpha_bar(0.0);
    for (u64 t = 0; t <= num_timesteps; ++t) {
        alphas[t] = alpha_bar((double)t) / f0;
    }

    std::vector<double> betas(num_timesteps, 0.0);
    for (u64 t = 1; t <= num_timesteps; ++t) {
        betas[t - 1] = std::min(0.999, 1.0 - alphas[t] / alphas[t - 1]);
    }
    return betas;
}

static std::vector<double> build_alphas_cumprod_from_betas(const std::vector<double> &betas) {
    std::vector<double> out(betas.size(), 0.0);
    double acc = 1.0;
    for (u64 i = 0; i < (u64)betas.size(); ++i) {
        acc *= (1.0 - betas[i]);
        out[i] = acc;
    }
    return out;
}

static std::vector<double> build_alphas_cumprod(const std::string &schedule, u64 num_timesteps) {
    if (schedule == "linear") {
        return build_alphas_cumprod_from_betas(build_betas_linear_cipherdm(num_timesteps));
    }
    if (schedule == "cosine") {
        return build_alphas_cumprod_from_betas(build_betas_cosine_cipherdm(num_timesteps));
    }
    always_assert(false && "unsupported beta_schedule; expected linear or cosine");
    return {};
}

static AuthShare mul_qf(const AuthShare &a, const AuthShare &b) {
    return LRS_CALL(MUL_CALL(a, b), f);
}

static AuthShare scale_public(const AuthShare &x, double v) {
    int64_t q = (int64_t)std::llround(v * (double)(1ULL << f));
    auto prod = auth_mul_const(x, (u64)q);
    clip_batch_check("ddpm:scale_public_after_mul_const");
    auto out = LRS_CALL(prod, f);
    clip_batch_check("ddpm:scale_public_after_shift");
    return out;
}

static AuthShare ss_reciprocal(const AuthShare &x) {
    shark::utils::start_timer("reciprocal");

    const u64 n = x.share.size();
    const double alpha = 1.0 / 256.0;
    const double eps = 1.0 / (double)(1ULL << f);

    auto one = make_public_const(n, 1.0, f);
    auto two = make_public_const(n, 2.0, f);
    auto alpha_const = make_public_const(n, alpha, f);
    auto eps_const = make_public_const(n, eps, f);

    auto ge_eps = cmp_ge_public(x, eps);
    AuthShare x_safe = auth_clone(eps_const);
    update_if(x_safe, ge_eps, x);

    auto t = mul_qf(x_safe, alpha_const);
    AuthShare r = auth_clone(one);
    for (int iter = 0; iter < 2; ++iter) {
        auto tr = mul_qf(t, r);
        auto corr = ADD_CALL(two, neg_span(tr));
        r = mul_qf(r, corr);
    }

    auto out = mul_qf(r, alpha_const);
    shark::utils::stop_timer("reciprocal");
    return out;
}

static AuthShare silu_apply(const AuthShare &x) {
    OpProfileScope profile(OP_SILU);
    shark::utils::start_timer("silu");
    auto ge_zero = cmp_ge_public(x, 0.0);
    auto x_neg = neg_span(x);
    AuthShare ax = auth_clone(x_neg);
    update_if(ax, ge_zero, x);

    auto ax2 = mul_qf(ax, ax);
    auto ax3 = mul_qf(ax2, ax);
    auto ax4 = mul_qf(ax2, ax2);
    auto ax5 = mul_qf(ax4, ax);

    auto outer0 = make_public_const(x.share.size(), 5.90010486e-01, f);
    auto outer1 = make_public_const(x.share.size(), 4.21319936e-01, f);
    auto outer2 = make_public_const(x.share.size(), 1.26610092e-01, f);
    auto outer3 = make_public_const(x.share.size(), 1.97985686e-02, f);
    auto outer4 = make_public_const(x.share.size(), 1.59538912e-03, f);
    auto outer5 = make_public_const(x.share.size(), 5.25625056e-05, f);

    auto inner0 = make_public_const(x.share.size(), 4.99957471e-01, f);
    auto inner1 = make_public_const(x.share.size(), 2.49155471e-01, f);
    auto inner2 = make_public_const(x.share.size(), -4.01655846e-03, f);
    auto inner3 = make_public_const(x.share.size(), -2.84974728e-02, f);
    auto inner4 = make_public_const(x.share.size(), -6.76484741e-03, f);
    auto inner5 = make_public_const(x.share.size(), -4.34163661e-04, f);

    auto p_outer_abs = outer0;
    p_outer_abs = ADD_CALL(p_outer_abs, neg_span(mul_qf(outer1, ax)));
    p_outer_abs = ADD_CALL(p_outer_abs, mul_qf(outer2, ax2));
    p_outer_abs = ADD_CALL(p_outer_abs, neg_span(mul_qf(outer3, ax3)));
    p_outer_abs = ADD_CALL(p_outer_abs, mul_qf(outer4, ax4));
    p_outer_abs = ADD_CALL(p_outer_abs, neg_span(mul_qf(outer5, ax5)));

    auto p_inner_abs = inner0;
    p_inner_abs = ADD_CALL(p_inner_abs, neg_span(mul_qf(inner1, ax)));
    p_inner_abs = ADD_CALL(p_inner_abs, mul_qf(inner2, ax2));
    p_inner_abs = ADD_CALL(p_inner_abs, neg_span(mul_qf(inner3, ax3)));
    p_inner_abs = ADD_CALL(p_inner_abs, mul_qf(inner4, ax4));
    p_inner_abs = ADD_CALL(p_inner_abs, neg_span(mul_qf(inner5, ax5)));

    auto one = make_public_const(x.share.size(), 1.0, f);
    auto zero = make_public_const(x.share.size(), 0.0, f);
    auto ge_abs_2_2 = cmp_ge_public(ax, 2.2);
    auto p_abs = auth_clone(p_inner_abs);
    update_if(p_abs, ge_abs_2_2, p_outer_abs);

    auto gate = auth_clone(p_abs);
    auto gate_pos = ADD_CALL(one, neg_span(p_abs));
    update_if(gate, ge_zero, gate_pos);
    update_if(gate, cmp_ge_public(x, 7.0), one);
    update_if(gate, cmp_gt_public(x_neg, 7.0), zero);

    auto gate_nonneg = cmp_ge_public(gate, 0.0);
    AuthShare gate_clamped = auth_clone(zero);
    update_if(gate_clamped, gate_nonneg, gate);
    update_if(gate_clamped, cmp_ge_public(gate_clamped, 1.0), one);

    auto ret = mul_qf(x, gate_clamped);
    shark::utils::stop_timer("silu");
    return ret;
}

static AuthShare transpose_auth_matrix(const AuthShare &x, u64 rows, u64 cols) {
    always_assert(x.share.size() == rows * cols);
    AuthShare out = auth_alloc(rows * cols);
    #pragma omp parallel for collapse(2)
    for (u64 i = 0; i < rows; ++i) {
        for (u64 j = 0; j < cols; ++j) {
            out.share[j * rows + i] = x.share[i * cols + j];
            out.tag[j * rows + i] = x.tag[i * cols + j];
        }
    }
    return out;
}

static AuthShare chexp_unprofiled(const AuthShare &x) {
    shark::utils::start_timer("exp");
    u64 n = x.share.size();
    auto u = scale_public(x, 0.5);
    auto u2 = mul_qf(u, u);
    auto u4 = mul_qf(u2, u2);

    auto c0 = make_public_const(n, 1.0, f);
    auto c1 = make_public_const(n, 0.9736789799002139, f);
    auto c2 = make_public_const(n, 0.4349844528710021, f);
    auto c3 = make_public_const(n, 0.10637438059443326, f);
    auto c4 = make_public_const(n, 0.013503439864618789, f);
    auto c5 = make_public_const(n, 0.000682062067913879, f);

    auto a0 = ADD_CALL(c0, mul_qf(c1, u));
    auto a1 = ADD_CALL(c2, mul_qf(c3, u));
    auto a2 = ADD_CALL(c4, mul_qf(c5, u));
    auto y = ADD_CALL(a0, mul_qf(a1, u2));
    y = ADD_CALL(y, mul_qf(a2, u4));
    auto ex = mul_qf(y, y);
    shark::utils::stop_timer("exp");
    return ex;
}

static AuthShare chexp_softmax(const AuthShare &x) {
    OpProfileScope profile(OP_SOFTMAX_EXP);
    return chexp_unprofiled(x);
}

static AuthShare rowmax_share(u64 rows, u64 cols, const AuthShare &x) {
    always_assert(x.share.size() == rows * cols);
    AuthShare cur = auth_clone(x);
    u64 curr_cols = cols;
    while (curr_cols > 1) {
        u64 pair_cols = curr_cols / 2;
        u64 next_cols = pair_cols + (curr_cols & 1);
        AuthShare next = auth_alloc(rows * next_cols);

        for (u64 r = 0; r < rows; ++r) {
            for (u64 j = 0; j < pair_cols; ++j) {
                u64 lidx = r * curr_cols + 2 * j;
                u64 ridx = lidx + 1;
                AuthShare left = auth_alloc(1);
                AuthShare right = auth_alloc(1);
                left.share[0] = cur.share[lidx];
                left.tag[0] = cur.tag[lidx];
                right.share[0] = cur.share[ridx];
                right.tag[0] = cur.tag[ridx];
                auto diff = ADD_CALL(left, neg_span(right));
                auto ge = CMP_GE_ZERO_CALL(diff);
                auto max_lr = ADD_CALL(right, SELECT_CALL(ge, diff));
                next.share[r * next_cols + j] = max_lr.share[0];
                next.tag[r * next_cols + j] = max_lr.tag[0];
            }
        }

        if (curr_cols & 1) {
            #pragma omp parallel for
            for (u64 r = 0; r < rows; ++r) {
                u64 src = r * curr_cols + curr_cols - 1;
                u64 dst = r * next_cols + next_cols - 1;
                next.share[dst] = cur.share[src];
                next.tag[dst] = cur.tag[src];
            }
        }

        cur = std::move(next);
        curr_cols = next_cols;
    }

    AuthShare out = auth_alloc(rows);
    #pragma omp parallel for
    for (u64 i = 0; i < rows; ++i) {
        out.share[i] = cur.share[i];
        out.tag[i] = cur.tag[i];
    }
    return out;
}

static AuthShare softmax_cheb(u64 rows, u64 cols, const AuthShare &x) {
    OpProfileScope profile(OP_SOFTMAX);
    shark::utils::start_timer("softmax");
    always_assert(x.share.size() == rows * cols);
    auto row_max = rowmax_share(rows, cols, x);
    AuthShare shifted = auth_alloc(rows * cols);
    #pragma omp parallel for collapse(2)
    for (u64 i = 0; i < rows; ++i) {
        for (u64 j = 0; j < cols; ++j) {
            u64 idx = i * cols + j;
            shifted.share[idx] = x.share[idx] - row_max.share[i];
            shifted.tag[idx] = x.tag[idx] - row_max.tag[i];
        }
    }

    auto exp_vals = chexp_softmax(shifted);

    AuthShare row_sum = auth_alloc(rows);
    #pragma omp parallel for
    for (u64 i = 0; i < rows; ++i) {
        u128 acc = 0;
        u128 acc_tag = 0;
        for (u64 j = 0; j < cols; ++j) {
            u64 idx = i * cols + j;
            acc += exp_vals.share[idx];
            acc_tag += exp_vals.tag[idx];
        }
        row_sum.share[i] = acc;
        row_sum.tag[i] = acc_tag;
    }

    auto inv_sum = ss_reciprocal(row_sum);
    auto out = mul_qf(exp_vals, broadcast_rows(rows, cols, inv_sum));
    shark::utils::stop_timer("softmax");
    return out;
}

static AuthShare linear_apply(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w, const AuthShare *b = nullptr) {
    OpProfileScope profile(OP_LINEAR);
    shark::utils::start_timer("linear");
    auto y = LRS_CALL(matmul_secret_noopen(M, K, N, x, w), f);
    if (b && b->share.size() == N) y = ADD_CALL(y, broadcast_row_vector(M, N, *b));
    shark::utils::stop_timer("linear");
    return y;
}

static AuthShare linear_matmul_apply(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w) {
    OpProfileScope profile(OP_LINEAR);
    shark::utils::start_timer("linear");
    auto y = LRS_CALL(matmul_secret_noopen(M, K, N, x, w), f);
    shark::utils::stop_timer("linear");
    return y;
}

static AuthShare linear_matmul_apply(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w, double post_scale) {
    OpProfileScope profile(OP_LINEAR);
    shark::utils::start_timer("linear");
    auto y = LRS_CALL(matmul_secret_noopen(M, K, N, x, w), f);
    y = scale_public(y, post_scale);
    shark::utils::stop_timer("linear");
    return y;
}

static AuthShare conv_apply(u64 B, u64 H, u64 W, u64 inC, u64 outC, u64 k, u64 stride, u64 padding,
                            const AuthShare &x, const AuthShare &w, const AuthShare *b = nullptr) {
    OpProfileScope profile(OP_CONV);
    auto y = LRS_CALL(conv_secret_noopen(k, padding, stride, inC, outC, H, W, x, w), f);
    if (b && b->share.size() == outC) {
        u64 outH = (H - k + 2 * padding) / stride + 1;
        u64 outW = (W - k + 2 * padding) / stride + 1;
        y = ADD_CALL(y, broadcast_channels_nhwc(B, outH, outW, outC, *b));
    }
    return y;
}

static AuthShare concat_channels(u64 B, u64 H, u64 W, u64 C1, const AuthShare &a, u64 C2, const AuthShare &b) {
    always_assert(a.share.size() == B * H * W * C1);
    always_assert(b.share.size() == B * H * W * C2);
    AuthShare out = auth_alloc(B * H * W * (C1 + C2));
    #pragma omp parallel for collapse(4)
    for (u64 n = 0; n < B; ++n) {
        for (u64 h = 0; h < H; ++h) {
            for (u64 widx = 0; widx < W; ++widx) {
                for (u64 c = 0; c < C1 + C2; ++c) {
                    u64 dst = idx4(n, h, widx, c, H, W, C1 + C2);
                    if (c < C1) {
                        u64 src = idx4(n, h, widx, c, H, W, C1);
                        out.share[dst] = a.share[src];
                        out.tag[dst] = a.tag[src];
                    } else {
                        u64 src = idx4(n, h, widx, c - C1, H, W, C2);
                        out.share[dst] = b.share[src];
                        out.tag[dst] = b.tag[src];
                    }
                }
            }
        }
    }
    return out;
}

static AuthShare select_channels(u64 B, u64 H, u64 W, u64 C, const AuthShare &x, u64 channel_start, u64 channel_count) {
    always_assert(x.share.size() == B * H * W * C);
    always_assert(channel_start + channel_count <= C);
    AuthShare out = auth_alloc(B * H * W * channel_count);
    #pragma omp parallel for collapse(4)
    for (u64 n = 0; n < B; ++n) {
        for (u64 h = 0; h < H; ++h) {
            for (u64 widx = 0; widx < W; ++widx) {
                for (u64 c = 0; c < channel_count; ++c) {
                    u64 src = idx4(n, h, widx, channel_start + c, H, W, C);
                    u64 dst = idx4(n, h, widx, c, H, W, channel_count);
                    out.share[dst] = x.share[src];
                    out.tag[dst] = x.tag[src];
                }
            }
        }
    }
    return out;
}

static AuthShare select_channel(u64 B, u64 H, u64 W, u64 C, const AuthShare &x, u64 channel_idx) {
    always_assert(x.share.size() == B * H * W * C);
    always_assert(channel_idx < C);
    AuthShare out = auth_alloc(B * H * W);
    #pragma omp parallel for collapse(3)
    for (u64 n = 0; n < B; ++n) {
        for (u64 h = 0; h < H; ++h) {
            for (u64 widx = 0; widx < W; ++widx) {
                u64 src = idx4(n, h, widx, channel_idx, H, W, C);
                u64 dst = n * H * W + h * W + widx;
                out.share[dst] = x.share[src];
                out.tag[dst] = x.tag[src];
            }
        }
    }
    return out;
}

static AuthShare clamp_max_public(const AuthShare &x, double max_value, int fp_bits) {
    auto max_const = make_public_const(x.share.size(), max_value, fp_bits);
    auto diff = ADD_CALL(x, neg_span(max_const));
    auto ge_max = CMP_GE_ZERO_CALL(diff);
    auto delta = ADD_CALL(max_const, neg_span(x));
    return ADD_CALL(x, SELECT_CALL(ge_max, delta));
}

static AuthShare clamp_min_public(const AuthShare &x, double min_value, int fp_bits) {
    auto min_const = make_public_const(x.share.size(), min_value, fp_bits);
    auto diff = ADD_CALL(x, neg_span(min_const));
    auto ge_min = CMP_GE_ZERO_CALL(diff);
    auto one = make_public_raw(x.share.size(), 1);
    auto lt_min = ADD_CALL(one, neg_span(ge_min));
    auto delta = ADD_CALL(min_const, neg_span(x));
    return ADD_CALL(x, SELECT_CALL(lt_min, delta));
}

static AuthShare clip_sample_public(const AuthShare &x, double clip_value, int fp_bits) {
    return clamp_min_public(clamp_max_public(x, clip_value, fp_bits), -clip_value, fp_bits);
}

static AuthShare rsqrt_seed_interval_1_2(const AuthShare &u, int fp_bits, int poly_order = kNormPolyOrder) {
    auto one = make_public_const(u.share.size(), 1.0, fp_bits);
    auto half = make_public_const(u.share.size(), 0.5, fp_bits);
    auto three_eighths = make_public_const(u.share.size(), 0.375, fp_bits);
    auto five_sixteenths = make_public_const(u.share.size(), 0.3125, fp_bits);

    auto z = ADD_CALL(u, neg_span(one));
    auto z2 = mul_qf(z, z);

    auto out = ADD_CALL(one, neg_span(mul_qf(half, z)));
    out = ADD_CALL(out, mul_qf(three_eighths, z2));

    if (poly_order >= 3) {
        auto z3 = mul_qf(z2, z);
        out = ADD_CALL(out, neg_span(mul_qf(five_sixteenths, z3)));
    }
    return out;
}

static AuthShare rsqrt_newton_refine(const AuthShare &u, AuthShare r, int fp_bits,
                                     int nr_iters = kNormNewtonIters) {
    auto half = make_public_const(u.share.size(), 0.5, fp_bits);
    auto three = make_public_const(u.share.size(), 3.0, fp_bits);
    for (int iter = 0; iter < nr_iters; ++iter) {
        auto r2 = mul_qf(r, r);
        auto ur2 = mul_qf(u, r2);
        auto corr = ADD_CALL(three, neg_span(ur2));
        r = mul_qf(mul_qf(half, r), corr);
    }
    return r;
}

static AuthShare approx_inv_std_from_var(const AuthShare &var, int fp_bits,
                                         const SecureNormApproxConfig &cfg,
                                         double max_inv_std) {
    const u64 n = var.share.size();
    const double min_var = std::max(kNormVarFloor, std::ldexp(1.0, -fp_bits));

    auto var_clamped = clamp_min_public(var, min_var, fp_bits);
    (void)cfg.approx_mode;
    (void)cfg.use_exact_fallback;
    (void)cfg.min_denominator;

    AuthShare u = auth_mul_const(var_clamped, (u64)(1ULL << fp_bits));
    auto scale = make_public_const(n, rsqrt_scale_from_shift(fp_bits), fp_bits);

    for (int j = 1; j <= 48; ++j) {
        u64 threshold_val = (1ULL << j) - 1ULL;
        auto threshold = make_public_raw(n, threshold_val);
        auto diff = ADD_CALL(var_clamped, neg_span(threshold));
        auto ge_j = CMP_GE_ZERO_CALL(diff);

        int shift_j = fp_bits - j;
        AuthShare u_j;
        if (shift_j >= 0) {
            u_j = auth_mul_const(var_clamped, (u64)(1ULL << shift_j));
        } else {
            u_j = LRS_CALL(var_clamped, (u64)(-shift_j));
        }

        auto scale_j = make_public_const(n, rsqrt_scale_from_shift(shift_j), fp_bits);
        auto u_delta = ADD_CALL(u_j, neg_span(u));
        u = ADD_CALL(u, SELECT_CALL(ge_j, u_delta));

        auto scale_delta = ADD_CALL(scale_j, neg_span(scale));
        scale = ADD_CALL(scale, SELECT_CALL(ge_j, scale_delta));
    }

    auto r = rsqrt_seed_interval_1_2(u, fp_bits, cfg.poly_order);
    r = rsqrt_newton_refine(u, r, fp_bits, cfg.nr_iters);
    auto inv_std = mul_qf(scale, r);
    return clamp_max_public(inv_std, max_inv_std, fp_bits);
}

static u64 choose_groupnorm_groups(u64 C, u64 preferred_groups = 2) {
    if (preferred_groups > 0 && C >= preferred_groups && (C % preferred_groups) == 0) {
        return preferred_groups;
    }
    u64 max_groups = std::min<u64>(32, C);
    for (u64 groups = max_groups; groups > 1; --groups) {
        if ((C % groups) == 0) return groups;
    }
    return 1;
}

static AuthShare groupnorm_apply(u64 B, u64 H, u64 W, u64 C, const AuthShare &x, u64 preferred_groups = 2) {
    OpProfileScope profile(OP_GROUPNORM);
    shark::utils::start_timer("groupnorm");
    always_assert(x.share.size() == B * H * W * C);
    const auto cfg = make_secure_norm_approx_config();
    const double max_inv_std = 1.0 / std::sqrt(cfg.eps);
    u64 groups = choose_groupnorm_groups(C, preferred_groups);
    u64 cg = C / groups;
    u64 elems_per_group = H * W * cg;

    AuthShare group_sum = auth_alloc(B * groups);
    #pragma omp parallel for collapse(2)
    for (u64 n = 0; n < B; ++n) {
        for (u64 g = 0; g < groups; ++g) {
            u128 acc_share = 0;
            u128 acc_tag = 0;
            for (u64 h = 0; h < H; ++h) {
                for (u64 widx = 0; widx < W; ++widx) {
                    for (u64 kc = 0; kc < cg; ++kc) {
                        u64 idx = idx4(n, h, widx, g * cg + kc, H, W, C);
                        acc_share += x.share[idx];
                        acc_tag += x.tag[idx];
                    }
                }
            }
            u64 out_idx = n * groups + g;
            group_sum.share[out_idx] = acc_share;
            group_sum.tag[out_idx] = acc_tag;
        }
    }

    auto inv_n = make_public_const(B * groups, 1.0 / (double)elems_per_group, f);
    auto mean = mul_qf(group_sum, inv_n);

    AuthShare centered = auth_alloc(B * H * W * C);
    #pragma omp parallel for collapse(5)
    for (u64 n = 0; n < B; ++n) {
        for (u64 g = 0; g < groups; ++g) {
            for (u64 h = 0; h < H; ++h) {
                for (u64 widx = 0; widx < W; ++widx) {
                    for (u64 kc = 0; kc < cg; ++kc) {
                        u64 idx = idx4(n, h, widx, g * cg + kc, H, W, C);
                        u64 mean_idx = n * groups + g;
                        centered.share[idx] = x.share[idx] - mean.share[mean_idx];
                        centered.tag[idx] = x.tag[idx] - mean.tag[mean_idx];
                    }
                }
            }
        }
    }

    auto sqr = mul_qf(centered, centered);
    AuthShare var_sum = auth_alloc(B * groups);
    #pragma omp parallel for collapse(2)
    for (u64 n = 0; n < B; ++n) {
        for (u64 g = 0; g < groups; ++g) {
            u128 acc_share = 0;
            u128 acc_tag = 0;
            for (u64 h = 0; h < H; ++h) {
                for (u64 widx = 0; widx < W; ++widx) {
                    for (u64 kc = 0; kc < cg; ++kc) {
                        u64 idx = idx4(n, h, widx, g * cg + kc, H, W, C);
                        acc_share += sqr.share[idx];
                        acc_tag += sqr.tag[idx];
                    }
                }
            }
            u64 out_idx = n * groups + g;
            var_sum.share[out_idx] = acc_share;
            var_sum.tag[out_idx] = acc_tag;
        }
    }

    auto var = mul_qf(var_sum, inv_n);
    auto var_eps = ADD_CALL(var, make_public_const(B * groups, cfg.eps, f));
    auto inv_std = approx_inv_std_from_var(var_eps, f, cfg, max_inv_std);

    AuthShare inv_std_b = auth_alloc(B * H * W * C);
    #pragma omp parallel for collapse(5)
    for (u64 n = 0; n < B; ++n) {
        for (u64 g = 0; g < groups; ++g) {
            for (u64 h = 0; h < H; ++h) {
                for (u64 widx = 0; widx < W; ++widx) {
                    for (u64 kc = 0; kc < cg; ++kc) {
                        u64 idx = idx4(n, h, widx, g * cg + kc, H, W, C);
                        u64 stat_idx = n * groups + g;
                        inv_std_b.share[idx] = inv_std.share[stat_idx];
                        inv_std_b.tag[idx] = inv_std.tag[stat_idx];
                    }
                }
            }
        }
    }
    auto out = mul_qf(centered, inv_std_b);
    shark::utils::stop_timer("groupnorm");
    return out;
}

struct LinearWeights {
    u64 in_dim = 0;
    u64 out_dim = 0;
    span<u64> w;
    AuthShare w_auth;
    span<u64> b_plain;
    AuthShare b;
};

struct ConvWeights {
    u64 k = 0;
    u64 inC = 0;
    u64 outC = 0;
    u64 stride = 1;
    u64 padding = 0;
    span<u64> w;
    AuthShare w_auth;
    span<u64> b_plain;
    AuthShare b;
};

struct ConvTransposeWeights {
    u64 k = 0;
    u64 inC = 0;
    u64 outC = 0;
    u64 stride = 1;
    u64 padding = 0;
    u64 output_padding = 0;
    span<u64> w;
    AuthShare w_auth;
    span<u64> b_plain;
    AuthShare b;
};

struct SelfAttnWeights {
    u64 dim = 0;
    ConvWeights to_qkv;
    ConvWeights to_out;
};

struct ResnetBlockWeights {
    u64 inC = 0;
    u64 outC = 0;
    u64 tembC = 0;
    ConvWeights conv1;
    ConvWeights conv2;
    LinearWeights temb_proj;
    ConvWeights shortcut;
    bool use_shortcut = false;
    bool use_attention = false;
    SelfAttnWeights attn;
};

struct DDPMWeights {
    u64 inC = (u64)ddpm_para::kIoConfig.model_input_channels;
    u64 outC = (u64)ddpm_para::kIoConfig.model_input_channels;
    u64 image_h = (u64)ddpm_para::kModelConfig.image_hw;
    u64 image_w = (u64)ddpm_para::kModelConfig.image_hw;
    u64 time_in = (u64)ddpm_para::kModelConfig.base_ch;
    u64 temb = (u64)ddpm_para::kModelConfig.temb_dim;
    u64 c0 = (u64)ddpm_para::kModelConfig.base_ch;
    u64 c1 = (u64)ddpm_para::kModelConfig.mid_ch;
    u64 norm_groups = (u64)ddpm_para::kModelConfig.norm_groups;

    LinearWeights time1;
    LinearWeights time2;

    ConvWeights conv_in;
    ConvWeights conv_out;

    ResnetBlockWeights down0_r0;
    ResnetBlockWeights down0_r1;
    ConvWeights down0_down;

    ResnetBlockWeights down1_r0;
    ResnetBlockWeights down1_r1;
    ConvWeights down1_down;

    ResnetBlockWeights down2_r0;
    ResnetBlockWeights down2_r1;

    ResnetBlockWeights mid_r0;
    ResnetBlockWeights mid_r1;

    ResnetBlockWeights up0_r0;
    ResnetBlockWeights up0_r1;
    ResnetBlockWeights up0_r2;
    ConvTransposeWeights up0_up;

    ResnetBlockWeights up1_r0;
    ResnetBlockWeights up1_r1;
    ResnetBlockWeights up1_r2;
    ConvTransposeWeights up1_up;

    ResnetBlockWeights up2_r0;
    ResnetBlockWeights up2_r1;
    ResnetBlockWeights up2_r2;
};

static AuthShare linear_apply(u64 M, const AuthShare &x, const LinearWeights &w) {
    always_assert(w.w_auth.share.size() == w.in_dim * w.out_dim);
    return linear_apply(M, w.in_dim, w.out_dim, x, w.w_auth, &w.b);
}

static inline double q_to_double(u64 x) {
    return (double)(int64_t)x / (double)(1ULL << f);
}

static inline u64 double_to_q(double x) {
    return (u64)(int64_t)std::llround(x * (double)(1ULL << f));
}

static inline double time_mlp_silu_plain_value(double xv) {
    const double ax = std::abs(xv);
    const double ax2 = ax * ax;
    const double ax3 = ax2 * ax;
    const double ax4 = ax2 * ax2;
    const double ax5 = ax4 * ax;

    const double p_outer_abs = 5.90010486e-01
        - 4.21319936e-01 * ax
        + 1.26610092e-01 * ax2
        - 1.97985686e-02 * ax3
        + 1.59538912e-03 * ax4
        - 5.25625056e-05 * ax5;
    const double p_inner_abs = 4.99957471e-01
        - 2.49155471e-01 * ax
        - 4.01655846e-03 * ax2
        + 2.84974728e-02 * ax3
        - 6.76484741e-03 * ax4
        + 4.34163661e-04 * ax5;

    double p_abs = (ax >= 2.2) ? p_outer_abs : p_inner_abs;
    double gate = (xv >= 0.0) ? (1.0 - p_abs) : p_abs;
    if (xv >= 7.0) gate = 1.0;
    if (xv < -7.0) gate = 0.0;
    if (gate < 0.0) gate = 0.0;
    if (gate > 1.0) gate = 1.0;
    return xv * gate;
}

static span<u64> plain_linear_apply(u64 M, const span<u64> &x, const LinearWeights &w) {
    always_assert(x.size() == M * w.in_dim);
    span<u64> out(M * w.out_dim);
    zero_plain(out);
    if (party != SERVER) return out;
    for (u64 m = 0; m < M; ++m) {
        for (u64 j = 0; j < w.out_dim; ++j) {
            double acc = q_to_double(w.b_plain[j]);
            for (u64 k = 0; k < w.in_dim; ++k) {
                acc += q_to_double(x[m * w.in_dim + k]) *
                       q_to_double(w.w[k * w.out_dim + j]);
            }
            out[m * w.out_dim + j] = double_to_q(acc);
        }
    }
    return out;
}

static AuthShare conv_apply(u64 B, u64 H, u64 W, const AuthShare &x, const ConvWeights &w) {
    always_assert(w.w_auth.share.size() == w.outC * w.k * w.k * w.inC);
    return conv_apply(B, H, W, w.inC, w.outC, w.k, w.stride, w.padding, x, w.w_auth, &w.b);
}

static void zero_plain(span<u64> &x) {
    #pragma omp parallel for
    for (u64 i = 0; i < x.size(); ++i) x[i] = 0;
}

static void init_linear(LinearWeights &w, u64 in_dim, u64 out_dim) {
    w.in_dim = in_dim;
    w.out_dim = out_dim;
    w.w = span<u64>(in_dim * out_dim);
    w.b_plain = span<u64>(out_dim);
    zero_plain(w.w);
    zero_plain(w.b_plain);
}

static void init_conv(ConvWeights &w, u64 k, u64 inC, u64 outC, u64 stride, u64 padding) {
    w.k = k;
    w.inC = inC;
    w.outC = outC;
    w.stride = stride;
    w.padding = padding;
    w.w = span<u64>(outC * k * k * inC);
    w.b_plain = span<u64>(outC);
    zero_plain(w.w);
    zero_plain(w.b_plain);
}

static void init_conv_transpose(ConvTransposeWeights &w, u64 k, u64 inC, u64 outC,
                                u64 stride, u64 padding, u64 output_padding) {
    w.k = k;
    w.inC = inC;
    w.outC = outC;
    w.stride = stride;
    w.padding = padding;
    w.output_padding = output_padding;
    w.w = span<u64>(k * k * outC * inC);
    w.b_plain = span<u64>(outC);
    zero_plain(w.w);
    zero_plain(w.b_plain);
}

static void init_resnet(ResnetBlockWeights &rb, u64 inC, u64 outC, u64 tembC, bool use_attention = false) {
    rb.inC = inC;
    rb.outC = outC;
    rb.tembC = tembC;
    init_conv(rb.conv1, 3, inC, outC, 1, 1);
    init_conv(rb.conv2, 3, outC, outC, 1, 1);
    init_linear(rb.temb_proj, tembC, outC);
    rb.use_shortcut = (inC != outC);
    if (rb.use_shortcut) init_conv(rb.shortcut, 1, inC, outC, 1, 0);
    rb.use_attention = use_attention;
    if (rb.use_attention) init_attn(rb.attn, outC);
}

static void init_attn(SelfAttnWeights &a, u64 dim) {
    a.dim = dim;
    init_conv(a.to_qkv, 1, dim, dim * 3, 1, 0);
    init_conv(a.to_out, 1, dim, dim, 1, 0);
}

static void fill_attn(SelfAttnWeights &a);
static void share_attn(SelfAttnWeights &a);

static void fill_linear(LinearWeights &w) {
    if (party != SERVER) return;
    double bound = std::sqrt(6.0 / (double)std::max<u64>(1, w.in_dim));
    for (u64 i = 0; i < w.w.size(); ++i) w.w[i] = qrand_uniform_symmetric(f, bound);
    for (u64 i = 0; i < w.b_plain.size(); ++i) w.b_plain[i] = qrand_uniform_symmetric(f, bound);
}

static void fill_conv(ConvWeights &w) {
    if (party != SERVER) return;
    double fan_in = (double)(w.k * w.k * w.inC);
    double bound = std::sqrt(6.0 / std::max(1.0, fan_in));
    for (u64 i = 0; i < w.w.size(); ++i) w.w[i] = qrand_uniform_symmetric(f, bound);
    for (u64 i = 0; i < w.b_plain.size(); ++i) w.b_plain[i] = qrand_uniform_symmetric(f, bound);
}

static void fill_conv_transpose(ConvTransposeWeights &w) {
    if (party != SERVER) return;
    double fan_in = (double)(w.k * w.k * w.outC);
    double bound = std::sqrt(6.0 / std::max(1.0, fan_in));
    for (u64 i = 0; i < w.w.size(); ++i) w.w[i] = qrand_uniform_symmetric(f, bound);
    for (u64 i = 0; i < w.b_plain.size(); ++i) w.b_plain[i] = qrand_uniform_symmetric(f, bound);
}

static void fill_resnet(ResnetBlockWeights &rb) {
    fill_conv(rb.conv1);
    fill_conv(rb.conv2);
    fill_linear(rb.temb_proj);
    if (rb.use_shortcut) fill_conv(rb.shortcut);
    if (rb.use_attention) fill_attn(rb.attn);
}

static void fill_attn(SelfAttnWeights &a) {
    fill_conv(a.to_qkv);
    fill_conv(a.to_out);
}

static void init_model(DDPMWeights &m, u64 inC, u64 image_hw, u64 c0, u64 c1, u64 temb, u64 norm_groups) {
    always_assert(image_hw >= 2);
    always_assert((image_hw % 2) == 0);
    always_assert(c0 >= 1);
    always_assert(c1 >= 1);
    always_assert(temb >= 1);

    m.inC = inC;
    m.outC = inC;
    m.image_h = image_hw;
    m.image_w = image_hw;
    m.time_in = c0;
    m.temb = temb;
    m.c0 = c0;
    m.c1 = c1;
    m.norm_groups = norm_groups;

    init_linear(m.time1, m.time_in, m.temb);
    init_linear(m.time2, m.temb, m.temb);

    init_conv(m.conv_in, 3, inC, m.c0, 1, 1);
    init_conv(m.conv_out, 3, m.c0, m.outC, 1, 1);

    init_resnet(m.down0_r0, m.c0, m.c0, m.temb, false);
    init_resnet(m.down0_r1, m.c0, m.c0, m.temb, false);
    init_conv(m.down0_down, 3, m.c0, m.c0, 2, 1);

    init_resnet(m.down1_r0, m.c0, m.c1, m.temb, true);
    init_resnet(m.down1_r1, m.c1, m.c1, m.temb, true);
    init_conv(m.down1_down, 3, m.c1, m.c1, 2, 1);

    init_resnet(m.down2_r0, m.c1, m.c1, m.temb, false);
    init_resnet(m.down2_r1, m.c1, m.c1, m.temb, false);

    init_resnet(m.mid_r0, m.c1, m.c1, m.temb, true);
    init_resnet(m.mid_r1, m.c1, m.c1, m.temb, false);

    init_resnet(m.up0_r0, m.c1 + m.c1, m.c1, m.temb, false);
    init_resnet(m.up0_r1, m.c1 + m.c1, m.c1, m.temb, false);
    init_resnet(m.up0_r2, m.c1 + m.c1, m.c1, m.temb, false);
    init_conv_transpose(m.up0_up, 3, m.c1, m.c1, 2, 1, 1);

    init_resnet(m.up1_r0, m.c1 + m.c1, m.c1, m.temb, true);
    init_resnet(m.up1_r1, m.c1 + m.c1, m.c1, m.temb, true);
    init_resnet(m.up1_r2, m.c1 + m.c0, m.c1, m.temb, true);
    init_conv_transpose(m.up1_up, 3, m.c1, m.c1, 2, 1, 1);

    init_resnet(m.up2_r0, m.c1 + m.c0, m.c0, m.temb, false);
    init_resnet(m.up2_r1, m.c0 + m.c0, m.c0, m.temb, false);
    init_resnet(m.up2_r2, m.c0 + m.c0, m.c0, m.temb, false);
}

static void fill_model(DDPMWeights &m) {
    fill_linear(m.time1);
    fill_linear(m.time2);
    fill_conv(m.conv_in);
    fill_conv(m.conv_out);
    fill_resnet(m.down0_r0);
    fill_resnet(m.down0_r1);
    fill_conv(m.down0_down);
    fill_resnet(m.down1_r0);
    fill_resnet(m.down1_r1);
    fill_conv(m.down1_down);
    fill_resnet(m.down2_r0);
    fill_resnet(m.down2_r1);
    fill_resnet(m.mid_r0);
    fill_resnet(m.mid_r1);
    fill_resnet(m.up0_r0);
    fill_resnet(m.up0_r1);
    fill_resnet(m.up0_r2);
    fill_conv_transpose(m.up0_up);
    fill_resnet(m.up1_r0);
    fill_resnet(m.up1_r1);
    fill_resnet(m.up1_r2);
    fill_conv_transpose(m.up1_up);
    fill_resnet(m.up2_r0);
    fill_resnet(m.up2_r1);
    fill_resnet(m.up2_r2);
}

static void share_linear(LinearWeights &w) {
    w.w_auth = authenticated_input_from_owner(w.w, SERVER);
    w.b = authenticated_input_from_owner(w.b_plain, SERVER);
}

static void share_conv(ConvWeights &w) {
    w.w_auth = authenticated_input_from_owner(w.w, SERVER);
    w.b = authenticated_input_from_owner(w.b_plain, SERVER);
}

static void share_conv_transpose(ConvTransposeWeights &w) {
    w.w_auth = authenticated_input_from_owner(w.w, SERVER);
    w.b = authenticated_input_from_owner(w.b_plain, SERVER);
}

static void share_resnet(ResnetBlockWeights &rb) {
    share_conv(rb.conv1);
    share_conv(rb.conv2);
    share_linear(rb.temb_proj);
    if (rb.use_shortcut) share_conv(rb.shortcut);
    if (rb.use_attention) share_attn(rb.attn);
}

static void share_attn(SelfAttnWeights &a) {
    share_conv(a.to_qkv);
    share_conv(a.to_out);
}

static void share_model(DDPMWeights &m) {
    keygen_progress_tick("share_model.time1");
    share_linear(m.time1);
    keygen_progress_tick("share_model.time2");
    share_linear(m.time2);
    keygen_progress_tick("share_model.conv_in");
    share_conv(m.conv_in);
    keygen_progress_tick("share_model.conv_out");
    share_conv(m.conv_out);
    keygen_progress_tick("share_model.down0_r0");
    share_resnet(m.down0_r0);
    keygen_progress_tick("share_model.down0_r1");
    share_resnet(m.down0_r1);
    keygen_progress_tick("share_model.down0_down");
    share_conv(m.down0_down);
    keygen_progress_tick("share_model.down1_r0");
    share_resnet(m.down1_r0);
    keygen_progress_tick("share_model.down1_r1");
    share_resnet(m.down1_r1);
    keygen_progress_tick("share_model.down1_down");
    share_conv(m.down1_down);
    keygen_progress_tick("share_model.down2_r0");
    share_resnet(m.down2_r0);
    keygen_progress_tick("share_model.down2_r1");
    share_resnet(m.down2_r1);
    keygen_progress_tick("share_model.mid_r0");
    share_resnet(m.mid_r0);
    keygen_progress_tick("share_model.mid_r1");
    share_resnet(m.mid_r1);
    keygen_progress_tick("share_model.up0_r0");
    share_resnet(m.up0_r0);
    keygen_progress_tick("share_model.up0_r1");
    share_resnet(m.up0_r1);
    keygen_progress_tick("share_model.up0_r2");
    share_resnet(m.up0_r2);
    keygen_progress_tick("share_model.up0_up");
    share_conv_transpose(m.up0_up);
    keygen_progress_tick("share_model.up1_r0");
    share_resnet(m.up1_r0);
    keygen_progress_tick("share_model.up1_r1");
    share_resnet(m.up1_r1);
    keygen_progress_tick("share_model.up1_r2");
    share_resnet(m.up1_r2);
    keygen_progress_tick("share_model.up1_up");
    share_conv_transpose(m.up1_up);
    keygen_progress_tick("share_model.up2_r0");
    share_resnet(m.up2_r0);
    keygen_progress_tick("share_model.up2_r1");
    share_resnet(m.up2_r1);
    keygen_progress_tick("share_model.up2_r2");
    share_resnet(m.up2_r2);
}

static AuthShare reshape_conv_transpose_input_for_matmul(
        u64 B, u64 H, u64 W, u64 inC, u64 outH, u64 outW,
        u64 k, u64 stride, u64 padding, const AuthShare &x) {
    always_assert(x.share.size() == B * H * W * inC);
    const u64 rows = k * k * inC;
    const u64 cols = B * outH * outW;
    AuthShare out = auth_alloc(rows * cols);
    #pragma omp parallel for collapse(6)
    for (u64 n = 0; n < B; ++n) {
        for (u64 oh = 0; oh < outH; ++oh) {
            for (u64 ow = 0; ow < outW; ++ow) {
                for (u64 kh = 0; kh < k; ++kh) {
                    for (u64 kw = 0; kw < k; ++kw) {
                        for (u64 ci = 0; ci < inC; ++ci) {
                            const u64 row = ((kh * k + kw) * inC) + ci;
                            const u64 col = n * outH * outW + oh * outW + ow;
                            const u64 dst = row * cols + col;
                            out.share[dst] = 0;
                            out.tag[dst] = 0;
                            if (oh + padding < kh || ow + padding < kw) continue;
                            const u64 num_h = oh + padding - kh;
                            const u64 num_w = ow + padding - kw;
                            if ((num_h % stride) != 0 || (num_w % stride) != 0) continue;
                            const u64 ih = num_h / stride;
                            const u64 iw = num_w / stride;
                            if (ih >= H || iw >= W) continue;
                            const u64 src = idx4(n, ih, iw, ci, H, W, inC);
                            out.share[dst] = x.share[src];
                            out.tag[dst] = x.tag[src];
                        }
                    }
                }
            }
        }
    }
    return out;
}

static AuthShare reshape_conv_transpose_weight_for_matmul(u64 k, u64 inC, u64 outC, const AuthShare &w) {
    always_assert(w.share.size() == k * k * outC * inC);
    const u64 cols = k * k * inC;
    AuthShare out = auth_alloc(outC * cols);
    #pragma omp parallel for collapse(4)
    for (u64 kh = 0; kh < k; ++kh) {
        for (u64 kw = 0; kw < k; ++kw) {
            for (u64 co = 0; co < outC; ++co) {
                for (u64 ci = 0; ci < inC; ++ci) {
                    const u64 src = (((kh * k + kw) * outC + co) * inC) + ci;
                    const u64 dst = co * cols + ((kh * k + kw) * inC + ci);
                    out.share[dst] = w.share[src];
                    out.tag[dst] = w.tag[src];
                }
            }
        }
    }
    return out;
}

static AuthShare reshape_conv_transpose_output_from_matmul(u64 B, u64 H, u64 W, u64 C, const AuthShare &x) {
    const u64 cols = B * H * W;
    always_assert(x.share.size() == C * cols);
    AuthShare out = auth_alloc(B * H * W * C);
    #pragma omp parallel for collapse(4)
    for (u64 n = 0; n < B; ++n) {
        for (u64 h = 0; h < H; ++h) {
            for (u64 widx = 0; widx < W; ++widx) {
                for (u64 c = 0; c < C; ++c) {
                    const u64 col = n * H * W + h * W + widx;
                    const u64 src = c * cols + col;
                    const u64 dst = idx4(n, h, widx, c, H, W, C);
                    out.share[dst] = x.share[src];
                    out.tag[dst] = x.tag[src];
                }
            }
        }
    }
    return out;
}

static AuthShare conv_transpose2x_apply(u64 B, u64 H, u64 W, const AuthShare &x, const ConvTransposeWeights &w) {
    always_assert(B * H * W > 0);
    always_assert(w.stride == 2);
    always_assert(w.output_padding == 1);
    always_assert(x.share.size() == B * H * W * w.inC);
    always_assert(w.w_auth.share.size() == w.k * w.k * w.outC * w.inC);
    const u64 outH = (H - 1) * w.stride - 2 * w.padding + w.k + w.output_padding;
    const u64 outW = (W - 1) * w.stride - 2 * w.padding + w.k + w.output_padding;
    const u64 K = w.k * w.k * w.inC;
    const u64 N = B * outH * outW;

    OpProfileScope profile(OP_CONV);
    auto x_cols = reshape_conv_transpose_input_for_matmul(
        B, H, W, w.inC, outH, outW, w.k, w.stride, w.padding, x);
    auto w_rows = reshape_conv_transpose_weight_for_matmul(w.k, w.inC, w.outC, w.w_auth);
    auto y = LRS_CALL(matmul_secret_noopen(w.outC, K, N, w_rows, x_cols), f);
    auto out = reshape_conv_transpose_output_from_matmul(B, outH, outW, w.outC, y);
    if (w.b.share.size() == w.outC) {
        out = ADD_CALL(out, broadcast_channels_nhwc(B, outH, outW, w.outC, w.b));
    }
    return out;
}

static AuthShare self_attn_rows(u64 rows, u64 dim, const AuthShare &q, const AuthShare &k, const AuthShare &v) {
    auto kt = transpose_auth_matrix(k, rows, dim);
    auto scores = linear_matmul_apply(rows, dim, rows, q, kt, 1.0 / std::sqrt((double)dim));
    auto probs = softmax_cheb(rows, rows, scores);
    auto ctx = linear_matmul_apply(rows, rows, dim, probs, v);
    return ctx;
}

static AuthShare self_attn_spatial(u64 B, u64 H, u64 W, u64 C, const AuthShare &x, const SelfAttnWeights &w) {
    always_assert(B == 1);
    auto qkv = conv_apply(B, H, W, x, w.to_qkv);
    // Under the current NHWC layout with B == 1, [B, H, W, C] and [H * W, C]
    // are layout-compatible views, so no data movement is needed here.
    auto q = select_channels(B, H, W, C * 3, qkv, 0, C);
    auto k = select_channels(B, H, W, C * 3, qkv, C, C);
    auto v = select_channels(B, H, W, C * 3, qkv, C * 2, C);
    auto out = self_attn_rows(H * W, C, q, k, v);
    return conv_apply(B, H, W, out, w.to_out);
}

static AuthShare resnet_forward(u64 B, u64 H, u64 W, const AuthShare &x, const AuthShare &temb,
                                const ResnetBlockWeights &rb, u64 norm_groups) {
    always_assert(x.share.size() == B * H * W * rb.inC);
    auto h = groupnorm_apply(B, H, W, rb.inC, x, norm_groups);
    h = silu_apply(h);
    h = conv_apply(B, H, W, h, rb.conv1);

    auto temb_h = silu_apply(temb);
    auto temb_proj = linear_apply(B, temb_h, rb.temb_proj);
    h = ADD_CALL(h, broadcast_batch_vector_nhwc(B, H, W, rb.outC, temb_proj));

    h = groupnorm_apply(B, H, W, rb.outC, h, norm_groups);
    h = silu_apply(h);
    h = conv_apply(B, H, W, h, rb.conv2);

    AuthShare shortcut = rb.use_shortcut
        ? conv_apply(B, H, W, x, rb.shortcut)
        : auth_clone(x);
    h = ADD_CALL(h, shortcut);
    if (rb.use_attention) {
        h = ADD_CALL(
            h,
            self_attn_spatial(B, H, W, rb.outC,
                              groupnorm_apply(B, H, W, rb.outC, h, norm_groups), rb.attn));
    }
    return h;
}

static AuthShare input_client_image(const std::string &path, u64 image_h, u64 image_w) {
    (void)path;
    std::vector<u64> vals;
    const size_t image_size = (size_t)image_h * (size_t)image_w;
    if (party == CLIENT) {
        vals.assign(image_size, 0);
        for (size_t i = 0; i < image_size; ++i) vals[i] = qrand_uniform_symmetric(f, 1.0);
    } else {
        vals.assign(image_size, 0);
    }
    span<u64> img((u64)vals.size());
    for (u64 i = 0; i < vals.size(); ++i) img[i] = vals[i];
    return profiled_authenticated_input_from_owner(img, CLIENT);
}

static AuthShare input_client_noise(u64 size, double stddev = 1.0) {
    span<u64> noise(size);
    zero_plain(noise);
    if (party == CLIENT && stddev > 0.0) {
        for (u64 i = 0; i < size; ++i) noise[i] = qrand_normal(f, stddev);
    }
    return profiled_authenticated_input_from_owner(noise, CLIENT);
}

static AuthShare build_time_embedding(const DDPMWeights &m, u64 timestep) {
    span<u64> t0_plain(m.time_in);
    zero_plain(t0_plain);
    if (party == SERVER) {
        auto t0_vec = make_timestep_embedding_plain(timestep, m.time_in);
        for (u64 i = 0; i < m.time_in; ++i) t0_plain[i] = t0_vec[i];
    }

    shark::utils::start_timer("linear1_mlp");
    auto t1_plain = plain_linear_apply(1, t0_plain, m.time1);
    shark::utils::stop_timer("linear1_mlp");

    shark::utils::start_timer("silu");
    if (party == SERVER) {
        for (u64 i = 0; i < t1_plain.size(); ++i) {
            t1_plain[i] = double_to_q(time_mlp_silu_plain_value(q_to_double(t1_plain[i])));
        }
    }
    shark::utils::stop_timer("silu");

    shark::utils::start_timer("linear3_mlp");
    auto temb_plain = plain_linear_apply(1, t1_plain, m.time2);
    shark::utils::stop_timer("linear3_mlp");
    return authenticated_input_from_owner(temb_plain, SERVER);
}

static AuthShare unet_forward(const DDPMWeights &m, const AuthShare &sample, const AuthShare &temb) {
    const u64 B = 1;
    u64 H = m.image_h, W = m.image_w;
    std::vector<AuthShare> skips;

    auto pop_skip = [&]() {
        always_assert(!skips.empty());
        AuthShare out = skips.back();
        skips.pop_back();
        return out;
    };

    keygen_progress_tick("unet.conv_in");
    auto h = conv_apply(B, H, W, sample, m.conv_in);
    skips.push_back(auth_clone(h));

    keygen_progress_tick("unet.down0_r0");
    h = resnet_forward(B, H, W, h, temb, m.down0_r0, m.norm_groups);
    skips.push_back(auth_clone(h));
    keygen_progress_tick("unet.down0_r1");
    h = resnet_forward(B, H, W, h, temb, m.down0_r1, m.norm_groups);
    skips.push_back(auth_clone(h));
    keygen_progress_tick("unet.down0_down");
    h = conv_apply(B, H, W, h, m.down0_down);
    H = conv_out_dim(H, m.down0_down.k, m.down0_down.stride, m.down0_down.padding);
    W = conv_out_dim(W, m.down0_down.k, m.down0_down.stride, m.down0_down.padding);
    skips.push_back(auth_clone(h));

    keygen_progress_tick("unet.down1_r0");
    h = resnet_forward(B, H, W, h, temb, m.down1_r0, m.norm_groups);
    skips.push_back(auth_clone(h));
    keygen_progress_tick("unet.down1_r1");
    h = resnet_forward(B, H, W, h, temb, m.down1_r1, m.norm_groups);
    skips.push_back(auth_clone(h));
    keygen_progress_tick("unet.down1_down");
    h = conv_apply(B, H, W, h, m.down1_down);
    H = conv_out_dim(H, m.down1_down.k, m.down1_down.stride, m.down1_down.padding);
    W = conv_out_dim(W, m.down1_down.k, m.down1_down.stride, m.down1_down.padding);
    skips.push_back(auth_clone(h));

    keygen_progress_tick("unet.down2_r0");
    h = resnet_forward(B, H, W, h, temb, m.down2_r0, m.norm_groups);
    skips.push_back(auth_clone(h));
    keygen_progress_tick("unet.down2_r1");
    h = resnet_forward(B, H, W, h, temb, m.down2_r1, m.norm_groups);
    skips.push_back(auth_clone(h));

    keygen_progress_tick("unet.mid_r0");
    h = resnet_forward(B, H, W, h, temb, m.mid_r0, m.norm_groups);
    keygen_progress_tick("unet.mid_r1");
    h = resnet_forward(B, H, W, h, temb, m.mid_r1, m.norm_groups);

    keygen_progress_tick("unet.up0_r0");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c1, h, m.c1, pop_skip()), temb, m.up0_r0, m.norm_groups);
    keygen_progress_tick("unet.up0_r1");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c1, h, m.c1, pop_skip()), temb, m.up0_r1, m.norm_groups);
    keygen_progress_tick("unet.up0_r2");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c1, h, m.c1, pop_skip()), temb, m.up0_r2, m.norm_groups);
    H *= 2;
    W *= 2;
    keygen_progress_tick("unet.up0_up");
    h = conv_transpose2x_apply(B, H / 2, W / 2, h, m.up0_up);

    keygen_progress_tick("unet.up1_r0");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c1, h, m.c1, pop_skip()), temb, m.up1_r0, m.norm_groups);
    keygen_progress_tick("unet.up1_r1");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c1, h, m.c1, pop_skip()), temb, m.up1_r1, m.norm_groups);
    keygen_progress_tick("unet.up1_r2");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c1, h, m.c0, pop_skip()), temb, m.up1_r2, m.norm_groups);
    H *= 2;
    W *= 2;
    keygen_progress_tick("unet.up1_up");
    h = conv_transpose2x_apply(B, H / 2, W / 2, h, m.up1_up);

    keygen_progress_tick("unet.up2_r0");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c1, h, m.c0, pop_skip()), temb, m.up2_r0, m.norm_groups);
    keygen_progress_tick("unet.up2_r1");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c0, h, m.c0, pop_skip()), temb, m.up2_r1, m.norm_groups);
    keygen_progress_tick("unet.up2_r2");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c0, h, m.c0, pop_skip()), temb, m.up2_r2, m.norm_groups);

    keygen_progress_tick("unet.out");
    h = silu_apply(groupnorm_apply(B, H, W, m.c0, h, m.norm_groups));
    return conv_apply(B, H, W, h, m.conv_out);
}

static AuthShare add_noise_ddpm(const AuthShare &x0, const AuthShare &noise, double alpha_t) {
    OpProfileScope profile(OP_SCHEDULER);
    auto sample_x0 = scale_public(x0, std::sqrt(alpha_t));
    clip_batch_check("ddpm:after_scale_x0");
    auto sample_noise = scale_public(noise, std::sqrt(std::max(0.0, 1.0 - alpha_t)));
    clip_batch_check("ddpm:after_scale_noise");
    auto sample = ADD_CALL(sample_x0, sample_noise);
    clip_batch_check("ddpm:after_add_noise");
    return sample;
}

static double scheduler_variance_value(double alpha_t, double alpha_prev,
                                       const std::string &variance_type) {
    const double beta_prod_t = std::max(1e-20, 1.0 - alpha_t);
    const double beta_prod_prev = std::max(0.0, 1.0 - alpha_prev);
    const double current_alpha_t = alpha_t / std::max(1e-20, alpha_prev);
    const double current_beta_t = std::max(0.0, 1.0 - current_alpha_t);
    double variance = current_beta_t * beta_prod_prev / beta_prod_t;
    variance = std::max(1e-20, variance);

    if (variance_type == "fixed_small" || variance_type == "fixed_small_log") {
        return variance;
    }
    if (variance_type == "fixed_large" || variance_type == "fixed_large_log") {
        return std::max(1e-20, current_beta_t);
    }
    always_assert(false && "unsupported variance_type");
    return variance;
}

static AuthShare scheduler_step(const AuthShare &sample, const AuthShare &model_output,
                                int timestep, double alpha_t, double alpha_prev,
                                const std::string &prediction_type,
                                bool clip_sample,
                                double clip_sample_range,
                                bool thresholding,
                                const std::string &variance_type,
                                const AuthShare &variance_noise) {
    OpProfileScope profile(OP_SCHEDULER);
    always_assert(!thresholding);

    const bool has_prev = alpha_prev >= 0.0;
    const double alpha_prod_t = std::max(1e-20, alpha_t);
    const double alpha_prod_prev = has_prev ? std::max(1e-20, alpha_prev) : 1.0;
    const double beta_prod_t = std::max(0.0, 1.0 - alpha_prod_t);
    const double beta_prod_prev = std::max(0.0, 1.0 - alpha_prod_prev);
    const double current_alpha_t = alpha_prod_t / alpha_prod_prev;
    const double current_beta_t = std::max(0.0, 1.0 - current_alpha_t);

    AuthShare pred_x0;
    if (prediction_type == "epsilon") {
        pred_x0 = ADD_CALL(scale_public(sample, 1.0 / std::sqrt(alpha_prod_t)),
                           scale_public(model_output, -std::sqrt(beta_prod_t) / std::sqrt(alpha_prod_t)));
    } else if (prediction_type == "sample") {
        pred_x0 = auth_clone(model_output);
    } else if (prediction_type == "v_prediction") {
        pred_x0 = ADD_CALL(scale_public(sample, std::sqrt(alpha_prod_t)),
                           scale_public(model_output, -std::sqrt(beta_prod_t)));
    } else {
        always_assert(false && "unsupported prediction_type");
    }

    if (clip_sample) {
        pred_x0 = clip_sample_public(pred_x0, clip_sample_range, f);
    }

    const double pred_original_sample_coeff =
        std::sqrt(alpha_prod_prev) * current_beta_t / std::max(1e-20, beta_prod_t);
    const double current_sample_coeff =
        std::sqrt(current_alpha_t) * beta_prod_prev / std::max(1e-20, beta_prod_t);
    auto prev_sample = ADD_CALL(scale_public(pred_x0, pred_original_sample_coeff),
                                scale_public(sample, current_sample_coeff));

    if (timestep > 0) {
        double variance = scheduler_variance_value(alpha_prod_t, alpha_prod_prev, variance_type);
        double stddev = std::sqrt(variance);
        prev_sample = ADD_CALL(prev_sample, scale_public(variance_noise, stddev));
    }
    return prev_sample;
}

int main(int argc, char **argv) {
    const DdpmBenchConfig bench_cfg = load_ddpm_bench_config(argc, argv);
    if (bench_cfg.loaded) {
        apply_ddpm_bench_env(bench_cfg);
        init_from_ddpm_bench_config(bench_cfg, argc, argv);
    } else {
        init::from_args(argc, argv);
    }
    if (party == DEALER && keygen_progress_enabled()) {
        std::cout.setf(std::ios::unitbuf);
    }
    mpspdz_32bit_compaison = false;

    if (has_flag(argc, argv, "--help")) {
        if (party == CLIENT) {
            std::cout
                << "Usage: benchmark-ddpm [--steps " << ddpm_para::kRuntimeConfig.steps
                << " | --eval_steps " << ddpm_para::kRuntimeConfig.steps << "]\n"
                << "                      [--num_timesteps " << ddpm_para::kSchedulerConfig.num_train_timesteps
                << "] [--beta_schedule linear|cosine]\n"
                << "                      [--timestep_spacing uniform|quad]\n"
                << "                      [--prediction_type epsilon|sample|v_prediction]\n"
                << "                      [--variance_type fixed_small|fixed_large]\n"
                << "                      [--strength " << ddpm_para::kRuntimeConfig.strength
                << "] [--image cond.txt] [--out " << ddpm_para::kRuntimeConfig.out_path << "]\n"
                << "                      [--cond_out " << ddpm_para::kRuntimeConfig.cond_out_path
                << "] [--seed 0]\n"
                << "                      [--config ddpm_bench_config.json] [--launch dealer|server|client|emul]\n"
                << "                      [--image_hw " << ddpm_para::kModelConfig.image_hw
                << "] [--channels " << ddpm_para::kModelConfig.base_ch
                << "] [--base_ch " << ddpm_para::kModelConfig.base_ch
                << "] [--mid_ch " << ddpm_para::kModelConfig.mid_ch
                << "] [--temb " << ddpm_para::kModelConfig.temb_dim << "]\n";
        }
        finalize::call();
        return 0;
    }

    u64 image_hw = bench_cfg.loaded ? bench_cfg.image_hw : (u64)ddpm_para::kModelConfig.image_hw;
    u64 base_ch = bench_cfg.loaded ? bench_cfg.base_ch : (u64)ddpm_para::kModelConfig.base_ch;
    u64 mid_ch = bench_cfg.loaded ? bench_cfg.mid_ch : (u64)ddpm_para::kModelConfig.mid_ch;
    u64 temb_dim = bench_cfg.loaded ? bench_cfg.temb_dim : (u64)ddpm_para::kModelConfig.temb_dim;
    u64 norm_groups = bench_cfg.loaded ? bench_cfg.norm_groups : (u64)ddpm_para::kModelConfig.norm_groups;
    u64 steps = bench_cfg.loaded ? bench_cfg.steps : (u64)ddpm_para::kRuntimeConfig.steps;
    u64 num_timesteps =
        bench_cfg.loaded ? bench_cfg.num_timesteps : (u64)ddpm_para::kSchedulerConfig.num_train_timesteps;
    std::string beta_schedule =
        bench_cfg.loaded ? bench_cfg.beta_schedule
                         : get_arg_string(argc, argv, "--beta_schedule", ddpm_para::kSchedulerConfig.beta_schedule);
    std::string timestep_spacing =
        bench_cfg.loaded ? bench_cfg.timestep_spacing
                         : get_arg_string(argc, argv, "--timestep_spacing", ddpm_para::kSchedulerConfig.timestep_spacing);
    std::string prediction_type =
        bench_cfg.loaded ? bench_cfg.prediction_type
                         : get_arg_string(argc, argv, "--prediction_type", ddpm_para::kSchedulerConfig.prediction_type);
    std::string variance_type =
        bench_cfg.loaded ? bench_cfg.variance_type
                         : get_arg_string(argc, argv, "--variance_type", ddpm_para::kSchedulerConfig.variance_type);
    bool clip_sample = bench_cfg.loaded ? bench_cfg.clip_sample : ddpm_para::kSchedulerConfig.clip_sample;
    double clip_sample_range =
        bench_cfg.loaded ? bench_cfg.clip_sample_range : ddpm_para::kSchedulerConfig.clip_sample_range;
    bool thresholding = bench_cfg.loaded ? bench_cfg.thresholding : ddpm_para::kSchedulerConfig.thresholding;
    double strength_in = bench_cfg.loaded ? bench_cfg.strength : ddpm_para::kRuntimeConfig.strength;
    std::string image_path =
        bench_cfg.loaded ? bench_cfg.image_path
                         : get_arg_string(argc, argv, "--image", ddpm_para::kRuntimeConfig.image_path);
    std::string out_path =
        bench_cfg.loaded ? bench_cfg.out_path
                         : get_arg_string(argc, argv, "--out", ddpm_para::kRuntimeConfig.out_path);
    std::string cond_out_path =
        bench_cfg.loaded ? bench_cfg.cond_out_path
                         : get_arg_string(argc, argv, "--cond_out", ddpm_para::kRuntimeConfig.cond_out_path);
    (void)cond_out_path;
    const u64 model_input_channels =
        bench_cfg.loaded ? bench_cfg.model_input_channels : (u64)ddpm_para::kIoConfig.model_input_channels;
    const u64 sample_channels =
        bench_cfg.loaded ? bench_cfg.sample_channels : (u64)ddpm_para::kIoConfig.sample_channels;
    const u64 cond_channels =
        bench_cfg.loaded ? bench_cfg.cond_channels : (u64)ddpm_para::kIoConfig.cond_channels;
    const u64 output_select_channel =
        bench_cfg.loaded ? bench_cfg.output_select_channel : (u64)ddpm_para::kIoConfig.output_select_channel;
    const u64 saved_image_channels =
        bench_cfg.loaded ? bench_cfg.saved_image_channels : (u64)ddpm_para::kIoConfig.saved_image_channels;
    const double noise_stddev =
        bench_cfg.loaded ? bench_cfg.noise_stddev : ddpm_para::kIoConfig.noise_stddev;

    if (!bench_cfg.loaded) {
        if (find_arg(argc, argv, "--image_hw") >= 0) image_hw = get_arg_u64(argc, argv, "--image_hw", image_hw);
        if (find_arg(argc, argv, "--channels") >= 0) {
            u64 channels = get_arg_u64(argc, argv, "--channels", base_ch);
            base_ch = channels;
            mid_ch = (u64)ddpm_para::mid_channels_from_channels(channels);
            temb_dim = (u64)ddpm_para::temb_from_channels(channels);
        }
        if (find_arg(argc, argv, "--base_ch") >= 0) base_ch = get_arg_u64(argc, argv, "--base_ch", base_ch);
        if (find_arg(argc, argv, "--mid_ch") >= 0) mid_ch = get_arg_u64(argc, argv, "--mid_ch", mid_ch);
        if (find_arg(argc, argv, "--temb") >= 0) temb_dim = get_arg_u64(argc, argv, "--temb", temb_dim);
        if (find_arg(argc, argv, "--steps") >= 0) steps = get_arg_u64(argc, argv, "--steps", steps);
        if (find_arg(argc, argv, "--eval_steps") >= 0) steps = get_arg_u64(argc, argv, "--eval_steps", steps);
        if (find_arg(argc, argv, "--num_timesteps") >= 0) {
            num_timesteps = get_arg_u64(argc, argv, "--num_timesteps", num_timesteps);
        }
        if (find_arg(argc, argv, "--strength") >= 0) {
            strength_in = get_arg_double(argc, argv, "--strength", strength_in);
        }
    }

    always_assert(image_hw >= 2);
    always_assert((image_hw % 2) == 0);
    always_assert(base_ch >= 1);
    always_assert(mid_ch >= 1);
    always_assert(norm_groups >= 1);
    always_assert(temb_dim >= 1);
    always_assert(steps >= 1);
    always_assert(num_timesteps >= 1);
    always_assert(!thresholding);
    always_assert(model_input_channels == sample_channels + cond_channels);
    always_assert(output_select_channel < model_input_channels);
    always_assert(sample_channels == 1);
    always_assert(cond_channels == 1);
    always_assert(saved_image_channels == 1);
    always_assert(noise_stddev >= 0.0);

    current_rng_seed =
        (bench_cfg.loaded && bench_cfg.seed_set) ? bench_cfg.seed : get_arg_u64(argc, argv, "--seed", 0);

    auto alphas_cumprod = build_alphas_cumprod(beta_schedule, num_timesteps);
    auto timesteps = build_timesteps(num_timesteps, steps, timestep_spacing);
    double strength = std::max(0.0, std::min(1.0, strength_in));
    u64 start_idx = (u64)std::llround(strength * (double)(steps - 1));
    if (start_idx >= steps) start_idx = steps - 1;
    u64 denoise_steps = (u64)timesteps.size() - start_idx;
    keygen_progress_begin(denoise_steps);

    DDPMWeights model;
    init_model(model, model_input_channels, image_hw, base_ch, mid_ch, temb_dim, norm_groups);
    fill_model(model);
    share_model(model);

    if (party != DEALER) {
        shark::utils::start_timer("total_eval");
        start_total_eval_profile();
    }

    if (party == CLIENT) {
        std::cout << "[ddpm] config image_hw=" << image_hw
                  << " base_ch=" << base_ch
                  << " mid_ch=" << mid_ch
                  << " temb=" << temb_dim
                  << " norm_groups=" << norm_groups
                  << " steps=" << steps
                  << " num_timesteps=" << num_timesteps
                  << " strength=" << strength
                  << " beta_schedule=" << beta_schedule
                  << " timestep_spacing=" << timestep_spacing
                  << " prediction_type=" << prediction_type
                  << " variance_type=" << variance_type
                  << " clip_sample=" << (clip_sample ? "true" : "false")
                  << " io=(" << model_input_channels << "," << sample_channels
                  << "," << cond_channels << "," << output_select_channel
                  << "," << saved_image_channels << "," << noise_stddev << ")";
        if (bench_cfg.loaded) {
            std::cout << " config=" << bench_cfg.config_path
                      << " config_fp=" << bench_cfg.config_fingerprint
                      << " launch=" << bench_cfg.selected_launch;
        }
        std::cout
                  << std::endl;
    }
    always_assert(std::getenv("SHARK_PUBLIC_ALPHA") == nullptr);
    always_assert(std::getenv("SHARK_PUBLIC_BITKEY") == nullptr);
    always_assert(std::getenv("SHARK_DEBUG_RINGKEY") == nullptr);
    always_assert(std::getenv("SHARK_DEBUG_BITKEY") == nullptr);
    always_assert(std::getenv("SHARK_INPUT_MASK_ZERO") == nullptr);
    always_assert(std::getenv("UNCLIP_DISABLE_BATCHCHECK") == nullptr);

    if (party != DEALER) shark::utils::start_timer("input");
    keygen_progress_tick("input.cond_image");
    AuthShare cond_img = input_client_image(image_path, image_hw, image_hw);
    if (party != DEALER) clip_batch_check("ddpm:after_cond_img");

    keygen_progress_tick("input.init_noise");
    AuthShare noise = input_client_noise(image_hw * image_hw, noise_stddev);
    if (party != DEALER) clip_batch_check("ddpm:after_noise");
    if (party != DEALER) shark::utils::stop_timer("input");
    const double input_alpha = alphas_cumprod[(u64)timesteps[start_idx]];
    AuthShare sample = add_noise_ddpm(cond_img, noise, input_alpha);
    if (party != DEALER) clip_batch_check("ddpm:after_input");

    for (u64 i = start_idx; i < timesteps.size(); ++i) {
        int t = timesteps[i];
        keygen_progress_set_step(i - start_idx + 1, denoise_steps, t);
        keygen_progress_tick("time_embedding");
        auto temb = build_time_embedding(model, (u64)t);
        if (party != DEALER) clip_batch_check("ddpm:after_temb");
        AuthShare model_in = concat_channels(1, image_hw, image_hw,
                                             sample_channels, sample,
                                             cond_channels, cond_img);
        auto eps_full = unet_forward(model, model_in, temb);
        if (party != DEALER) clip_batch_check("ddpm:after_unet");
        auto eps = select_channel(1, image_hw, image_hw, model.outC, eps_full,
                                  output_select_channel);
        double alpha_t = alphas_cumprod[(u64)t];
        int prev_t = (i + 1 < timesteps.size()) ? timesteps[i + 1] : -1;
        double alpha_prev = (prev_t >= 0) ? alphas_cumprod[(u64)prev_t] : -1.0;
        keygen_progress_tick("scheduler_noise");
        AuthShare variance_noise = (t > 0)
            ? input_client_noise(sample.share.size(), 1.0)
            : make_public_raw(sample.share.size(), 0);
        keygen_progress_tick("scheduler_step");
        sample = scheduler_step(sample, eps, t, alpha_t, alpha_prev,
                                prediction_type, clip_sample, clip_sample_range,
                                thresholding, variance_type, variance_noise);
        if (party != DEALER) clip_batch_check("ddpm:after_scheduler");
        keygen_progress_clear_step();

        if (party == CLIENT) {
            std::cout << "[ddpm] step " << (i - start_idx + 1) << "/" << (timesteps.size() - start_idx)
                      << " t=" << t << std::endl;
        }
    }

    if (party != DEALER) {
        clip_batch_check("ddpm:final");
        shark::utils::start_timer("final_reveal");
    }
    auto out_plain = profiled_auth_reveal_to_owner_authenticated(sample, CLIENT, OP_RECONSTRUCT);
    if (party != DEALER) {
        shark::utils::stop_timer("final_reveal");
        if (party == CLIENT) {
            bool ok = write_jpg_from_fixed(out_plain, image_hw, image_hw,
                                           (int)saved_image_channels,
                                           f, out_path.c_str());
            std::cout << (ok ? "[ddpm] wrote output: " : "[ddpm] failed output: ") << out_path << std::endl;
        }
        shark::utils::stop_timer("total_eval");
        stop_total_eval_profile();
        if (peer) {
            u64 comm = peer->bytesReceived() + peer->bytesSent();
            std::cout << "[PROFILE] total_comm: " << (comm / 1024.0) << " KB (" << comm << " bytes)" << std::endl;
        }
        print_profile_timers();
        print_profile_components_table();
        print_legacy_profile_lines();
    }

    keygen_progress_end();

    finalize::call();
    return 0;
}
