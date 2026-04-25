#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/common.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/conv.hpp>
#include <shark/protocols/add.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/ars.hpp>
#include <shark/protocols/lrs.hpp>
#include <shark/protocols/softmax.hpp>
#include <shark/protocols/drelu.hpp>
#include <shark/protocols/select.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/comm.hpp>
#include <shark/utils/timer.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../artifacts/stableunclip/stableunclip_spec_autogen.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace shark;
using namespace shark::protocols;

// `ScopedCommTrace` exists in some instrumented trees but not in the base
// communication utilities used by this benchmark. Keep a local no-op fallback
// so the benchmark still compiles when that tracing helper is unavailable.
struct ScopedCommTrace {
    explicit ScopedCommTrace(const char *) {}
};

// Plaintexts are encoded as 24-bit fixed-point payloads over Z_{2^64}.
// For AuthSS, we use a 64-bit MAC key and place authenticated shares, tags,
// and integrity checks in the extended ring Z_{2^128} = Z_{2^{64+64}}.
static int f = 24;

struct AuthShare {
    span<u128> share;
    span<u128> tag;
};

struct AuthTensor4D {
    AuthShare data;
    u64 B = 0;
    u64 H = 0;
    u64 W = 0;
    u64 C = 0;
};

static u64 mix_u64_fingerprint(u64 acc, u64 v);
static bool norm_use_ssrsqrt_enabled();
static bool norm_force_recip_div_enabled();
static double fixed_to_double_local(u64 x);
static u64 double_to_fixed_local(double x);
static void clip_batch_check(const char* label);
static void print_legacy_profile_lines();
static shark::utils::TimerStat make_manual_timer_delta(double time_ms, u64 comm_bytes, u64 rounds);

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
    {"reconstruct", 0, 0, 0, 0.0},
}};

struct EvalProfileStat {
    bool active = false;
    u64 comm0 = 0;
    u64 rounds0 = 0;
    u64 comm_bytes = 0;
    u64 rounds = 0;
    double time_ms = 0.0;
    std::chrono::steady_clock::time_point t0{};
};

struct OpProfileFrame {
    size_t idx = 0;
    u64 comm0 = 0;
    u64 rounds0 = 0;
    u64 child_comm = 0;
    u64 child_rounds = 0;
    double child_time_ms = 0.0;
    std::chrono::steady_clock::time_point t0{};
};

static EvalProfileStat g_total_eval_profile{};
static thread_local std::vector<OpProfileFrame> g_op_profile_stack;

struct EvalProfileTotals {
    u64 comm_bytes = 0;
    u64 rounds = 0;
    double time_ms = 0.0;
};

struct TimerProfileRow {
    const char *component = "";
    const char *timer_name = "";
};

static constexpr std::array<TimerProfileRow, 8> kProfileTimerRows{{
    {"input", "input"},
    {"text", "block_text"},
    {"vision", "block_vision"},
    {"unet_step", "block_unet_step"},
    {"vae", "block_vae"},
    {"superres", "block_superres"},
    {"reconstruct", "reconstruct"},
    {"end-to-end", "total_eval"},
}};

struct ProfileSnapshot {
    std::array<OpProfileStat, OP_PROFILE_COUNT> op_stats{};
    EvalProfileTotals total_eval{};
    std::array<shark::utils::TimerStat, kProfileTimerRows.size()> timers{};
    u64 peer_comm_bytes = 0;
    u64 peer_rounds = 0;
};

struct EstimatedProfileProjection {
    bool active = false;
    std::array<OpProfileStat, OP_PROFILE_COUNT> op_stats{};
    EvalProfileTotals total_eval{};
    std::array<shark::utils::TimerStat, kProfileTimerRows.size()> timers{};
    u64 network_total_comm_bytes = 0;
    u64 network_total_rounds = 0;
};

static EstimatedProfileProjection g_estimated_profile{};
static EstimatedProfileProjection g_estimated_profile_extras{};
static std::vector<std::string> g_estimated_profile_meta_lines;

static inline u64 current_comm_bytes() {
    if (peer == nullptr) return 0;
    return peer->bytesReceived() + peer->bytesSent();
}

static inline u64 current_rounds() {
    if (peer == nullptr) return 0;
    return peer->roundsReceived() + peer->roundsSent();
}

static bool minimal_terminal_output_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = std::getenv("SHARK_MINIMAL_TERMINAL");
        if (env == nullptr || env[0] == '\0') {
            // Default to the full benchmark report so comm / profile tables are
            // emitted without requiring extra env configuration.
            enabled = 0;
        } else if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' ||
                   env[0] == 'f' || env[0] == 'F') {
            enabled = 0;
        } else {
            enabled = 1;
        }
    }
    return enabled == 1;
}

static void print_minimal_step(u64 step_idx) {
    if (party == DEALER) return;
    std::cout << "step" << step_idx << std::endl;
}

static void print_minimal_final_cost() {
    if (party == DEALER) return;
    const auto old_flags = std::cout.flags();
    const auto old_prec = std::cout.precision();
    std::cout << std::fixed << std::setprecision(3)
              << "COST total_time_ms=" << g_total_eval_profile.time_ms
              << " total_comm_kb=" << ((double)current_comm_bytes() / 1024.0)
              << " total_rounds=" << current_rounds()
              << std::endl;
    std::cout.flags(old_flags);
    std::cout.precision(old_prec);
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
        const double total_time_ms = std::chrono::duration<double, std::milli>(t1 - frame.t0).count();

        auto &stat = g_op_stats[idx];
        stat.calls += 1;
        stat.comm_bytes += total_comm - frame.child_comm;
        stat.rounds += total_rounds - frame.child_rounds;
        stat.time_ms += total_time_ms - frame.child_time_ms;

        if (!g_op_profile_stack.empty()) {
            auto &parent = g_op_profile_stack.back();
            parent.child_comm += total_comm;
            parent.child_rounds += total_rounds;
            parent.child_time_ms += total_time_ms;
        }
    }
};

static void start_total_eval_profile() {
    if (party == DEALER) return;
    g_total_eval_profile.active = true;
    g_total_eval_profile.comm0 = current_comm_bytes();
    g_total_eval_profile.rounds0 = current_rounds();
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
    g_total_eval_profile.time_ms =
        std::chrono::duration<double, std::milli>(t1 - g_total_eval_profile.t0).count();
    g_total_eval_profile.active = false;
}

static inline u64 nonnegative_delta_u64(u64 after, u64 before) {
    return (after >= before) ? (after - before) : 0;
}

static inline double nonnegative_delta_double(double after, double before) {
    return (after >= before) ? (after - before) : 0.0;
}

static EvalProfileTotals current_total_eval_totals() {
    EvalProfileTotals out;
    if (party == DEALER) return out;
    if (g_total_eval_profile.active) {
        const auto t1 = std::chrono::steady_clock::now();
        out.comm_bytes = current_comm_bytes() - g_total_eval_profile.comm0;
        out.rounds = current_rounds() - g_total_eval_profile.rounds0;
        out.time_ms =
            std::chrono::duration<double, std::milli>(t1 - g_total_eval_profile.t0).count();
    } else {
        out.comm_bytes = g_total_eval_profile.comm_bytes;
        out.rounds = g_total_eval_profile.rounds;
        out.time_ms = g_total_eval_profile.time_ms;
    }
    return out;
}

static std::array<shark::utils::TimerStat, kProfileTimerRows.size()> capture_component_timer_stats() {
    std::array<shark::utils::TimerStat, kProfileTimerRows.size()> timers{};
    for (size_t i = 0; i < kProfileTimerRows.size(); ++i) {
        shark::utils::get_timer_stat(kProfileTimerRows[i].timer_name, timers[i]);
    }
    return timers;
}

static ProfileSnapshot capture_profile_snapshot() {
    ProfileSnapshot snapshot;
    snapshot.op_stats = g_op_stats;
    snapshot.total_eval = current_total_eval_totals();
    snapshot.timers = capture_component_timer_stats();
    snapshot.peer_comm_bytes = current_comm_bytes();
    snapshot.peer_rounds = current_rounds();
    return snapshot;
}

static OpProfileStat op_profile_delta(const OpProfileStat &after, const OpProfileStat &before) {
    OpProfileStat out = after;
    out.calls = nonnegative_delta_u64(after.calls, before.calls);
    out.comm_bytes = nonnegative_delta_u64(after.comm_bytes, before.comm_bytes);
    out.rounds = nonnegative_delta_u64(after.rounds, before.rounds);
    out.time_ms = nonnegative_delta_double(after.time_ms, before.time_ms);
    return out;
}

static EvalProfileTotals eval_profile_delta(const EvalProfileTotals &after,
                                            const EvalProfileTotals &before) {
    EvalProfileTotals out;
    out.comm_bytes = nonnegative_delta_u64(after.comm_bytes, before.comm_bytes);
    out.rounds = nonnegative_delta_u64(after.rounds, before.rounds);
    out.time_ms = nonnegative_delta_double(after.time_ms, before.time_ms);
    return out;
}

static shark::utils::TimerStat timer_stat_delta(const shark::utils::TimerStat &after,
                                                const shark::utils::TimerStat &before) {
    shark::utils::TimerStat out{};
    out.accumulated_time = nonnegative_delta_u64(after.accumulated_time, before.accumulated_time);
    out.accumulated_comm = nonnegative_delta_u64(after.accumulated_comm, before.accumulated_comm);
    out.accumulated_rounds = nonnegative_delta_u64(after.accumulated_rounds, before.accumulated_rounds);
    return out;
}

static void add_scaled_op_profile(OpProfileStat &dst, const OpProfileStat &delta, u64 scale) {
    dst.calls += delta.calls * scale;
    dst.comm_bytes += delta.comm_bytes * scale;
    dst.rounds += delta.rounds * scale;
    dst.time_ms += delta.time_ms * (double)scale;
}

static void add_scaled_eval_profile(EvalProfileTotals &dst, const EvalProfileTotals &delta, u64 scale) {
    dst.comm_bytes += delta.comm_bytes * scale;
    dst.rounds += delta.rounds * scale;
    dst.time_ms += delta.time_ms * (double)scale;
}

static void add_scaled_timer_stat(shark::utils::TimerStat &dst,
                                  const shark::utils::TimerStat &delta,
                                  u64 scale) {
    dst.accumulated_time += delta.accumulated_time * scale;
    dst.accumulated_comm += delta.accumulated_comm * scale;
    dst.accumulated_rounds += delta.accumulated_rounds * scale;
}

static void add_scaled_estimated_projection_delta(EstimatedProfileProjection &dst,
                                                  const EstimatedProfileProjection &after,
                                                  const EstimatedProfileProjection &before,
                                                  u64 scale) {
    for (size_t idx = 0; idx < OP_PROFILE_COUNT; ++idx) {
        const auto delta = op_profile_delta(after.op_stats[idx], before.op_stats[idx]);
        add_scaled_op_profile(dst.op_stats[idx], delta, scale);
    }

    const auto total_delta = eval_profile_delta(after.total_eval, before.total_eval);
    add_scaled_eval_profile(dst.total_eval, total_delta, scale);

    for (size_t i = 0; i < kProfileTimerRows.size(); ++i) {
        const auto delta = timer_stat_delta(after.timers[i], before.timers[i]);
        add_scaled_timer_stat(dst.timers[i], delta, scale);
    }

    dst.network_total_comm_bytes +=
        nonnegative_delta_u64(after.network_total_comm_bytes, before.network_total_comm_bytes) * scale;
    dst.network_total_rounds +=
        nonnegative_delta_u64(after.network_total_rounds, before.network_total_rounds) * scale;
}

static int profile_timer_row_index_by_component(const char *component) {
    for (size_t i = 0; i < kProfileTimerRows.size(); ++i) {
        if (std::strcmp(kProfileTimerRows[i].component, component) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static void reset_estimated_profile_projection() {
    g_estimated_profile = EstimatedProfileProjection{};
    g_estimated_profile_extras = EstimatedProfileProjection{};
    g_estimated_profile_meta_lines.clear();
}

static void accumulate_estimated_profile_repeat_segment(
    const char *segment_name,
    u64 total_count,
    u64 executed_count,
    u64 template_index,
    const ProfileSnapshot &template_before,
    const ProfileSnapshot &template_after,
    int manual_timer_row_idx = -1,
    const shark::utils::TimerStat *manual_timer_delta = nullptr,
    const EstimatedProfileProjection *nested_estimated_before = nullptr,
    const EstimatedProfileProjection *nested_estimated_after = nullptr) {
    if (party == DEALER) return;
    if (total_count <= executed_count) return;
    if (template_index < 1 || template_index > executed_count) return;

    const u64 skipped_count = total_count - executed_count;
    g_estimated_profile_extras.active = true;

    std::ostringstream meta;
    meta << "segment=" << segment_name
         << " total_count=" << total_count
         << " executed_count=" << executed_count
         << " template_index=" << template_index
         << " skipped_count=" << skipped_count;
    g_estimated_profile_meta_lines.push_back(meta.str());

    for (size_t idx = 0; idx < OP_PROFILE_COUNT; ++idx) {
        const auto delta = op_profile_delta(template_after.op_stats[idx], template_before.op_stats[idx]);
        add_scaled_op_profile(g_estimated_profile_extras.op_stats[idx], delta, skipped_count);
    }

    const auto total_delta = eval_profile_delta(template_after.total_eval, template_before.total_eval);
    add_scaled_eval_profile(g_estimated_profile_extras.total_eval, total_delta, skipped_count);

    for (size_t i = 0; i < kProfileTimerRows.size(); ++i) {
        const auto delta = timer_stat_delta(template_after.timers[i], template_before.timers[i]);
        add_scaled_timer_stat(g_estimated_profile_extras.timers[i], delta, skipped_count);
    }

    const int end_to_end_timer_row_idx = profile_timer_row_index_by_component("end-to-end");
    if (end_to_end_timer_row_idx >= 0) {
        const auto total_eval_timer_delta = make_manual_timer_delta(
            total_delta.time_ms,
            total_delta.comm_bytes,
            total_delta.rounds);
        add_scaled_timer_stat(g_estimated_profile_extras.timers[(size_t)end_to_end_timer_row_idx],
                              total_eval_timer_delta,
                              skipped_count);
    }

    if (manual_timer_row_idx >= 0 && manual_timer_delta != nullptr &&
        manual_timer_row_idx < (int)kProfileTimerRows.size()) {
        add_scaled_timer_stat(g_estimated_profile_extras.timers[(size_t)manual_timer_row_idx],
                              *manual_timer_delta, skipped_count);
    }

    g_estimated_profile_extras.network_total_comm_bytes +=
        nonnegative_delta_u64(template_after.peer_comm_bytes, template_before.peer_comm_bytes) * skipped_count;
    g_estimated_profile_extras.network_total_rounds +=
        nonnegative_delta_u64(template_after.peer_rounds, template_before.peer_rounds) * skipped_count;

    if (nested_estimated_before != nullptr && nested_estimated_after != nullptr) {
        add_scaled_estimated_projection_delta(
            g_estimated_profile_extras,
            *nested_estimated_after,
            *nested_estimated_before,
            skipped_count);
    }
}

static void finalize_estimated_profile_projection(const ProfileSnapshot &final_snapshot) {
    g_estimated_profile = EstimatedProfileProjection{};
    if (party == DEALER) return;
    if (!g_estimated_profile_extras.active) return;

    g_estimated_profile.active = true;
    g_estimated_profile.op_stats = final_snapshot.op_stats;
    g_estimated_profile.total_eval = final_snapshot.total_eval;
    g_estimated_profile.timers = final_snapshot.timers;
    g_estimated_profile.network_total_comm_bytes = final_snapshot.peer_comm_bytes;
    g_estimated_profile.network_total_rounds = final_snapshot.peer_rounds;

    for (size_t idx = 0; idx < OP_PROFILE_COUNT; ++idx) {
        add_scaled_op_profile(g_estimated_profile.op_stats[idx], g_estimated_profile_extras.op_stats[idx], 1);
    }
    add_scaled_eval_profile(g_estimated_profile.total_eval, g_estimated_profile_extras.total_eval, 1);
    for (size_t i = 0; i < kProfileTimerRows.size(); ++i) {
        add_scaled_timer_stat(g_estimated_profile.timers[i], g_estimated_profile_extras.timers[i], 1);
    }
    g_estimated_profile.network_total_comm_bytes += g_estimated_profile_extras.network_total_comm_bytes;
    g_estimated_profile.network_total_rounds += g_estimated_profile_extras.network_total_rounds;
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

static AvgOpProfileStat avg_op_profile_from_stat(const OpProfileStat &stat) {
    AvgOpProfileStat out;
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
              << stat.name << ": " << avg.time_ms << " ms, "
              << avg.comm_kb << " KB, "
              << avg.rounds << " rounds, "
              << "calls=" << stat.calls << std::endl;
    std::cout.flags(old_flags);
    std::cout.precision(old_prec);
}

static AuthShare auth_matmul_secret(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w);
static AuthShare auth_conv_secret(u64 k, u64 padding, u64 stride, u64 inC, u64 outC, u64 H, u64 W,
                                  const AuthShare &x, const AuthShare &w);

static AuthShare auth_view(const AuthShare &x, u64 offset, u64 size) {
    return AuthShare{
        span<u128>(x.share.data() + offset, size),
        span<u128>(x.tag.data() + offset, size),
    };
}

static void auth_copy_into(AuthShare &dst, u64 offset, const AuthShare &src) {
    always_assert(offset + src.share.size() <= dst.share.size());
    for (u64 i = 0; i < src.share.size(); ++i) {
        dst.share[offset + i] = src.share[i];
        dst.tag[offset + i] = src.tag[i];
    }
}

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
    u128 lifted = u128(val);
    u128 tag_val = mac_mul_u128(lifted);
    #pragma omp parallel for
    for (u64 i = 0; i < size; ++i) {
        if (party == SERVER || party == DEALER) {
            out.share[i] = lifted;
        } else {
            out.share[i] = 0;
        }
        out.tag[i] = tag_val;
    }
    return out;
}

static AuthShare auth_from_public_const(u64 size, double v, int fp) {
    int64_t q = (int64_t)std::llround(v * (double)(1ULL << fp));
    return auth_from_public_raw(size, (u64)q);
}

static AuthShare auth_from_public_span(const span<u64> &vals) {
    AuthShare out = auth_alloc(vals.size());
    #pragma omp parallel for
    for (u64 i = 0; i < vals.size(); ++i) {
        out.share[i] = (party == SERVER || party == DEALER) ? u128(vals[i]) : u128(0);
        out.tag[i] = mac_mul_u128(u128(vals[i]));
    }
    return out;
}

struct ClearInputSnapshot {
    const u64 *ptr = nullptr;
    u64 size = 0;
    int owner = -1;
    bool has_clear = false;
    span<u64> clear;
};

static std::vector<ClearInputSnapshot> &clear_input_snapshots() {
    static std::vector<ClearInputSnapshot> snapshots;
    return snapshots;
}

static void preserve_clear_input(const span<u64> &x, int owner) {
    if (x.size() == 0) return;
    auto &snapshots = clear_input_snapshots();
    for (auto &entry : snapshots) {
        if (entry.ptr == x.data() && entry.size == x.size()) {
            entry.owner = owner;
            if (party == owner) {
                if (entry.clear.size() != x.size()) entry.clear = span<u64>(x.size());
                entry.has_clear = true;
                #pragma omp parallel for
                for (u64 i = 0; i < x.size(); ++i) entry.clear[i] = x[i];
            }
            return;
        }
    }
    ClearInputSnapshot entry;
    entry.ptr = x.data();
    entry.size = x.size();
    entry.owner = owner;
    if (party == owner) {
        entry.clear = span<u64>(x.size());
        entry.has_clear = true;
        #pragma omp parallel for
        for (u64 i = 0; i < x.size(); ++i) entry.clear[i] = x[i];
    }
    snapshots.push_back(std::move(entry));
}

static const ClearInputSnapshot *find_clear_input_snapshot(const span<u64> &x) {
    auto &snapshots = clear_input_snapshots();
    for (auto &entry : snapshots) {
        if (entry.ptr == x.data() && entry.size == x.size()) return &entry;
    }
    return nullptr;
}

static void input_call_preserve_clear(span<u64> &x, int owner) {
    ScopedCommTrace trace(owner == SERVER ? "bench:input_call_preserve_clear:server"
                                          : "bench:input_call_preserve_clear:client");
    preserve_clear_input(x, owner);
    input::call(x, owner);
}

static void zero_plain(span<u64> &x) {
    #pragma omp parallel for
    for (u64 i = 0; i < x.size(); ++i) x[i] = 0;
}

static bool is_power_of_two_u64(u64 x) {
    return x != 0 && (x & (x - 1)) == 0;
}

static u64 log2_exact_u64(u64 x) {
    u64 shift = 0;
    while (x > 1) {
        x >>= 1;
        ++shift;
    }
    return shift;
}

// Securely open an authenticated arithmetic share.
// Call chain: auth_clone() copies the share/tag buffers, authenticated_reconstruct_full()
// exchanges peer shares and records MAC equations, and clip_batch_check() forces
// shark::protocols::batch_check() before and after the public reconstruction.
static span<u64> auth_open_authenticated(const AuthShare &x) {
    span<u64> out(x.share.size());
    if (party == DEALER) {
        #pragma omp parallel for
        for (u64 i = 0; i < x.share.size(); ++i) out[i] = getLow(x.share[i]);
        return out;
    }
    // Flush earlier pending MAC equations before this share becomes public.
    clip_batch_check("auth_open_authenticated:before_open");
    // Reconstruct from a scratch copy so the caller keeps its authenticated share for later use.
    auto tmp = auth_clone(x);
    // authenticated_reconstruct_full() exchanges arithmetic shares, reconstructs the clear value,
    // and appends the matching MAC equations to the batch-check buffer.
    auto opened = authenticated_reconstruct_full(tmp.share, tmp.tag);
    // Force the MAC equations generated by the reconstruction to be checked immediately.
    clip_batch_check("auth_open_authenticated:after_open");
    #pragma omp parallel for
    for (u64 i = 0; i < opened.size(); ++i) out[i] = getLow(opened[i]);
    return out;
}

static AuthShare auth_alloc(u64 size);
static AuthShare auth_add(const AuthShare &a, const AuthShare &b);

// Reveal an authenticated value only to CLIENT by publicly opening x + r while the clear
// one-time pad r is sent through the dealer key file to CLIENT alone.
static span<u64> auth_open_authenticated_to_client(const AuthShare &x) {
    span<u64> out(x.share.size());
    if (party == DEALER) {
        span<u64> mask_clear(x.share.size());
        randomize(mask_clear);
        send_authenticated_ashare(mask_clear);
        client->send_array(mask_clear);
        return out;
    }

    auto [mask_share_u64, mask_tag] = recv_authenticated_ashare(x.share.size());
    AuthShare mask_auth = auth_alloc(x.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < x.share.size(); ++i) {
        mask_auth.share[i] = u128(mask_share_u64[i]);
        mask_auth.tag[i] = mask_tag[i];
    }

    auto masked_open = auth_open_authenticated(auth_add(x, mask_auth));
    if (party != CLIENT) {
        return out;
    }

    auto mask_clear = dealer->recv_array<u64>(x.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < x.share.size(); ++i) {
        out[i] = masked_open[i] - mask_clear[i];
    }
    return out;
}

static span<u64> profiled_auth_open_authenticated_to_client(const AuthShare &x, size_t op_idx) {
    OpProfileScope profile(op_idx);
    return auth_open_authenticated_to_client(x);
}

static AuthShare authenticated_input_from_owner(const span<u64> &x, int owner);
static void owner_input_progress(const char *phase);

struct AuthenticatedInputCacheEntry {
    const u64 *ptr = nullptr;
    u64 size = 0;
    int owner = -1;
    AuthShare auth;
};

static std::vector<AuthenticatedInputCacheEntry> &authenticated_input_cache() {
    static std::vector<AuthenticatedInputCacheEntry> cache;
    return cache;
}

static void clear_authenticated_input_cache() {
    authenticated_input_cache().clear();
}

static AuthenticatedInputCacheEntry *find_authenticated_input_cache_entry(const u64 *ptr, u64 size, int owner) {
    auto &cache = authenticated_input_cache();
    for (auto &entry : cache) {
        if (entry.ptr == ptr && entry.size == size && entry.owner == owner) {
            return &entry;
        }
    }
    return nullptr;
}

static const AuthShare *find_authenticated_input_cache(const span<u64> &x, int owner) {
    auto *entry = find_authenticated_input_cache_entry(x.data(), x.size(), owner);
    return entry ? &entry->auth : nullptr;
}

// Lift a clear 64-bit vector into benchmark AuthShare form.
// The clear value is stored in the SERVER/DEALER share and each lane receives a MAC tag via
// mac_mul_u128(), so later secure operators can treat the result like any other AuthShare.
static AuthShare auth_from_plain_open(const span<u64> &x) {
    ScopedCommTrace trace("bench:auth_from_plain_open");
    AuthShare out = auth_alloc(x.size());
    #pragma omp parallel for
    for (u64 i = 0; i < x.size(); ++i) {
        out.share[i] = (party == SERVER || party == DEALER) ? u128(x[i]) : u128(0);
        out.tag[i] = mac_mul_u128(u128(x[i]));
    }
    return out;
}

// Authenticate a private input owned by SERVER or CLIENT.
// Call chain: the dealer samples a one-time mask with randomize(), exports authenticated mask
// shares through send_authenticated_ashare(), sends the clear mask to the owner, and both online
// parties fold the public delta back in with auth_local_add_public_u128().
static AuthShare authenticated_input_from_owner(const span<u64> &x, int owner) {
    ScopedCommTrace trace(owner == SERVER ? "bench:authenticated_input_from_owner:server"
                                          : "bench:authenticated_input_from_owner:client");
    always_assert(owner == SERVER || owner == CLIENT);

    const u64 size = x.size();
    AuthShare out = auth_alloc(size);
    owner_input_progress("enter");

    if (party == DEALER) {
        span<u64> r_clear(size);
        // Sample the additive one-time pad r that hides the owner's clear input x.
        randomize(r_clear);
        // send_authenticated_ashare() distributes authenticated shares/tags of r to online parties.
        send_authenticated_ashare(r_clear);
        if (owner == SERVER) {
            server->send_array(r_clear);
        } else {
            client->send_array(r_clear);
        }
        #pragma omp parallel for
        for (u64 i = 0; i < size; ++i) {
            // Dealer keeps the same authenticated mask locally so all parties refer to one AuthShare.
            out.share[i] = u128(r_clear[i]);
            out.tag[i] = mac_mul_u64(r_clear[i]);
        }
        owner_input_progress("dealer_done");
        return out;
    }

    owner_input_progress("before_recv_share");
    // recv_authenticated_ashare() yields this party's share/tag of the dealer mask r.
    auto [r_share, r_tag] = recv_authenticated_ashare(size);
    owner_input_progress("after_recv_share");
    span<u64> d(size);
    #pragma omp parallel for
    for (u64 i = 0; i < size; ++i) d[i] = u64(0);

    if (party == owner) {
        owner_input_progress("before_recv_mask");
        // The owner alone learns the clear mask and computes d = x - r.
        auto r_clear = dealer->recv_array<u64>(size);
        owner_input_progress("after_recv_mask");
        #pragma omp parallel for
        for (u64 i = 0; i < size; ++i) {
            d[i] = x[i] - r_clear[i];
        }
        owner_input_progress("before_peer_send");
        peer->send_array(d);
        owner_input_progress("after_peer_send");
    } else {
        // The non-owner learns only the masked delta d, never the owner's raw clear input x.
        owner_input_progress("before_peer_recv");
        peer->recv_array(d);
        owner_input_progress("after_peer_recv");
    }

    #pragma omp parallel for
    for (u64 i = 0; i < size; ++i) {
        // auth_local_add_public_u128() merges the public delta into the authenticated mask share/tag,
        // producing an authenticated sharing of x without another interactive MAC subprotocol.
        auth_local_add_public_u128(u128(r_share[i]), r_tag[i], u128(d[i]),
                                   party == SERVER, out.share[i], out.tag[i]);
    }
    owner_input_progress("done");
    return out;
}

static AuthShare profiled_authenticated_input_from_owner(const span<u64> &x, int owner, size_t op_idx) {
    OpProfileScope profile(op_idx);
    return authenticated_input_from_owner(x, owner);
}

static AuthShare auth_add(const AuthShare &a, const AuthShare &b) {
    AuthShare out = auth_alloc(a.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < a.share.size(); ++i) {
        out.share[i] = a.share[i] + b.share[i];
        out.tag[i] = a.tag[i] + b.tag[i];
    }
    return out;
}

static AuthShare auth_sub(const AuthShare &a, const AuthShare &b) {
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

static AuthShare auth_mul_const(const AuthShare &a, u64 c) {
    AuthShare out = auth_alloc(a.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < a.share.size(); ++i) {
        auth_local_mul_public_u128(a.share[i], a.tag[i], u128(c), out.share[i], out.tag[i]);
    }
    return out;
}

static AuthShare auth_mul(const AuthShare &a, const AuthShare &b) {
    auto tmp = mul::call_share_secret_full(a.share, a.tag, b.share, b.tag);
    AuthShare out{std::move(tmp.share), std::move(tmp.tag)};
    return out;
}

static AuthShare auth_matmul(u64 M, u64 K, u64 N, const AuthShare &x, const span<u64> &w) {
    return auth_matmul_secret(M, K, N, x, auth_from_plain_open(w));
}

static AuthShare auth_matmul_secret(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w) {
    ScopedCommTrace trace("bench:auth_matmul_secret");
    auto tmp = matmul::call_share_secret_full(M, K, N, x.share, x.tag, w.share, w.tag);
    AuthShare out{std::move(tmp.share), std::move(tmp.tag)};
    return out;
}

static AuthShare auth_conv(u64 k, u64 padding, u64 stride, u64 inC, u64 outC, u64 H, u64 W,
                           const AuthShare &x, const span<u64> &w) {
    return auth_conv_secret(k, padding, stride, inC, outC, H, W, x, auth_from_plain_open(w));
}

static AuthShare auth_conv_secret(u64 k, u64 padding, u64 stride, u64 inC, u64 outC, u64 H, u64 W,
                                  const AuthShare &x, const AuthShare &w) {
    ScopedCommTrace trace("bench:auth_conv_secret");
    auto tmp = conv::call_share_secret_full(k, padding, stride, inC, outC, H, W, x.share, x.tag, w.share, w.tag);
    AuthShare out{std::move(tmp.share), std::move(tmp.tag)};
    return out;
}

static AuthShare auth_shift(const AuthShare &x, u64 shift) {
    ScopedCommTrace trace("bench:auth_shift");
    auto tmp = ars::call_share_secret_full(x.share, x.tag, (int)shift);
    return AuthShare{std::move(tmp.share), std::move(tmp.tag)};
}

// Forward declarations for no-open helpers.
static AuthShare ss_ars(const AuthShare &x, u64 shift);
static AuthShare mul_noopen(const AuthShare &x, const AuthShare &y);
static AuthShare matmul_noopen(u64 M, u64 K, u64 N, const AuthShare &x, const span<u64> &w);
static AuthShare conv_noopen(u64 k, u64 padding, u64 stride, u64 inC, u64 outC, u64 H, u64 W,
                             const AuthShare &x, const span<u64> &w);
static AuthShare cmp_ge_zero_noopen(const AuthShare &x);
static AuthShare select_noopen(const AuthShare &cond, const AuthShare &x);

#define ADD_CALL(...) auth_add(__VA_ARGS__)
#define MUL_CALL(...) mul_noopen(__VA_ARGS__)
#define MATMUL_CALL(...) matmul_noopen(__VA_ARGS__)
#define CONV_CALL(...) conv_noopen(__VA_ARGS__)
#define LRS_CALL(...) ss_ars(__VA_ARGS__)
#define ARS_CALL(...) ss_ars(__VA_ARGS__)
#define CMP_GE_ZERO_CALL(...) cmp_ge_zero_noopen(__VA_ARGS__)
#define SELECT_CALL(cond, ...) select_noopen(cond, __VA_ARGS__)
// -----------------------------
// Deterministic RNG (copied pattern from clip.cpp)
// -----------------------------
static u64 current_rng_seed = 0x123456789ULL;

static inline u64 rng_next(u64 &s) {
    s = s * 6364136223846793005ULL + 1;
    return s;
}

static inline double rng_u01(u64 &s) {
    // Use 53 bits for a deterministic (0,1] double.
    const double inv = 1.0 / 9007199254740992.0; // 2^53
    u64 x = rng_next(s) >> 11;
    return (x + 1.0) * inv;
}

static inline std::pair<double, double> exact_sin_cos(double x) {
    return {std::sin(x), std::cos(x)};
}

static inline u64 qrand_normal(int fp, double stddev) {
    const double kTwoPi = 6.2831853071795864769;
    double u1 = rng_u01(current_rng_seed);
    double u2 = rng_u01(current_rng_seed);
    double r = std::sqrt(-2.0 * std::log(u1));
    double z = r * std::cos(kTwoPi * u2);
    double x = z * stddev;
    int64_t q = (int64_t)std::llround(x * (double)(1ULL << fp));
    return (u64)q;
}

static inline u64 qrand_uniform_symmetric(int fp, double bound) {
    if (bound <= 0.0) return 0;
    double u = rng_u01(current_rng_seed);
    double x = -bound + 2.0 * bound * u;
    int64_t q = (int64_t)std::llround(x * (double)(1ULL << fp));
    return (u64)q;
}

static double time_embedding_exp_approx(double x) {
    // Match modify1all's Chebyshev-style exp approximation used on [-14, 0].
    const double xt = x / 7.0 + 1.0;
    static constexpr double c0 = 0.14021878;
    static constexpr double c1 = 0.27541278;
    static constexpr double c2 = 0.22122865;
    static constexpr double c3 = 0.14934221;
    static constexpr double c4 = 0.09077360;
    static constexpr double c5 = 0.04369614;
    static constexpr double c6 = 0.02087868;
    static constexpr double c7 = 0.00996535;

    double t0 = 1.0;
    double t1 = xt;
    double ex = c0 + c1 * t1;
    const double two_xt = xt + xt;
    double t_nm2 = t0;
    double t_nm1 = t1;
    auto append_term = [&](double coeff) {
        const double next_t = two_xt * t_nm1 - t_nm2;
        ex += coeff * next_t;
        t_nm2 = t_nm1;
        t_nm1 = next_t;
    };
    append_term(c2);
    append_term(c3);
    append_term(c4);
    append_term(c5);
    append_term(c6);
    append_term(c7);
    return ex;
}

static span<u64> make_timestep_embedding(int timestep, u64 dim, int fp, bool flip_sin_to_cos = true) {
    OpProfileScope profile(OP_TIME_EMBEDDING_EXP);
    span<u64> out(dim);
    if (dim == 0) return out;
    const u64 half = dim / 2;
    always_assert(dim >= 2);
    always_assert(half > 0);
    std::vector<double> vals(dim, 0.0);
    // Match stableunclip.py Timesteps: exp(arange(half_dim) * -log(half_dim)/(half_dim-1)).
    double log_max = (half > 1) ? std::log((double)half) : 0.0;
    double denom = (half > 1) ? (half - 1) : 1;
    for (u64 i = 0; i < half; ++i) {
        double freq = time_embedding_exp_approx(-log_max * (double)i / denom);
        double arg = (double)timestep * freq;
        auto trig = exact_sin_cos(arg);
        double s = trig.first;
        double c = trig.second;
        if (flip_sin_to_cos) {
            vals[i] = c;
            vals[half + i] = s;
        } else {
            vals[i] = s;
            vals[half + i] = c;
        }
    }
    for (u64 i = 0; i < dim; ++i) {
        int64_t q = (int64_t)std::llround(vals[i] * (double)(1ULL << fp));
        out[i] = (u64)q;
    }
    return out;
}

static u64 prompt_token_hash(const std::string &token) {
    u64 h = 1469598103934665603ULL;
    for (unsigned char ch : token) {
        h ^= (u64)ch;
        h *= 1099511628211ULL;
    }
    return h;
}

static std::vector<u64> tokenize_prompt(const std::string &prompt, u64 vocab_size, u64 max_length) {
    const u64 bos = 0;
    const u64 pad = (vocab_size > 1) ? 1 : 0;
    const u64 eos = (vocab_size > 2) ? 2 : pad;
    const u64 token_offset = (vocab_size > 3) ? 3 : vocab_size;
    std::vector<u64> ids(max_length, pad);
    if (max_length == 0) return ids;
    if (max_length == 1) {
        ids[0] = eos;
        return ids;
    }
    ids[0] = bos;
    ids[max_length - 1] = eos;
    if (vocab_size > token_offset) {
        const u64 range = vocab_size - token_offset;
        std::vector<std::string> tokens;
        std::string token;
        for (unsigned char ch : prompt) {
            if (std::isalnum(ch) != 0) {
                token.push_back((char)std::tolower(ch));
            } else if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        }
        if (!token.empty()) {
            tokens.push_back(token);
        }
        const u64 token_count = std::min<u64>((u64)tokens.size(), max_length - 2);
        for (u64 i = 0; i < token_count; ++i) {
            ids[i + 1] = token_offset + (prompt_token_hash(tokens[i]) % range);
        }
    }
    return ids;
}

static void build_prompt_selectors(const std::vector<u64> &ids, span<u64> &token_select,
                                   span<u64> &pos_select, u64 vocab_size, u64 seq_len, int fp_bits) {
    always_assert(ids.size() == seq_len);
    always_assert(token_select.size() == seq_len * vocab_size);
    always_assert(pos_select.size() == seq_len * seq_len);
    zero_plain(token_select);
    zero_plain(pos_select);

    const u64 pad = (vocab_size > 1) ? 1 : 0;
    const u64 one_q = (u64)((int64_t)1 << fp_bits);
    u64 pos = 0;
    for (u64 i = 0; i < seq_len; ++i) {
        u64 tok = ids[i];
        if (tok >= vocab_size) tok = pad;

        token_select[i * vocab_size + tok] = one_q;

        u64 pos_id = 0;
        if (tok != pad) {
            pos_id = pos;
            pos = (pos + 1 < seq_len) ? (pos + 1) : pos;
        }
        pos_select[i * seq_len + pos_id] = one_q;
    }
}

static span<u64> normalize_image_input(const span<u64> &img, u64 H, u64 W, u64 C) {
    span<u64> out(img.size());
    const double scale = 1.0 / (double)(1ULL << f);
    const double mean[3] = {0.48145466, 0.4578275, 0.40821073};
    const double stdv[3] = {0.26862954, 0.26130258, 0.27577711};
    for (u64 h = 0; h < H; ++h) {
        for (u64 widx = 0; widx < W; ++widx) {
            for (u64 c = 0; c < C; ++c) {
                size_t idx = ((size_t)h * W + widx) * C + c;
                double v = (double)(int64_t)img[idx] * scale;
                double norm = (v - mean[c % 3]) / stdv[c % 3];
                int64_t q = (int64_t)std::llround(norm * (double)(1ULL << f));
                out[idx] = (u64)q;
            }
        }
    }
    return out;
}

// -----------------------------
// Helpers
// -----------------------------
static AuthShare make_public_const(u64 size, double v, int fp) {
    return auth_from_public_const(size, v, fp);
}

static AuthShare make_public_raw(u64 size, u64 val) {
    return auth_from_public_raw(size, val);
}

// Re-authenticate a share (no-op for now, preserves MAC).
static AuthShare reauth(const AuthShare &x) {
    return x;
}

// Protocol-safe negation: (-x)
static AuthShare neg_span(const AuthShare &x) {
    return auth_neg(x);
}

// Protocol arithmetic right shift (no-open, authenticated).
static AuthShare ss_ars(const AuthShare &x, u64 shift) {
    return auth_shift(x, shift);
}

// Secure compare to zero without opening x itself.
// Call chain: recv/send_authenticated_ashare_full() supplies the mask r,
// authenticated_reconstruct_full() opens only x + r + bias, recv_dcfring()/dcfring_eval()
// evaluate the comparison key material, and auth_sub(make_public_raw(...), lt) converts
// the shared "less-than" bit into the final ">=" bit.
static AuthShare cmp_ge_zero_noopen(const AuthShare &x) {
    u64 n = x.share.size();
    AuthShare out = auth_alloc(n);
    const u128 bias = u128(1) << 63;

    if (party == DEALER) {
        span<u128> r(n);
        span<u128> alpha(n);
        for (u64 i = 0; i < n; ++i) {
            // Dealer prepares the comparison mask and the DCF alpha value derived from r + 2^63.
            r[i] = u128(rand<u64>());
            alpha[i] = u128(getLow(r[i] + bias));
        }
        send_authenticated_ashare_full(r);
        send_dcfring(alpha, 64);
        for (u64 i = 0; i < n; ++i) {
            // Dealer keeps the same authenticated output bit so all parties share one result.
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
    // Only the masked value x + r + bias is opened, and the open is immediately surrounded
    // by MAC verification so malicious tampering aborts before the plaintext is consumed.
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
    auto one = make_public_raw(n, 1);
    // x >= 0 is equivalent to 1 - [x < 0] once the DCF has produced the less-than bit.
    out = auth_sub(one, lt);
    return out;
}

// Secure select: returns cond * x (cond is shared 0/1).
static AuthShare select_noopen(const AuthShare &cond, const AuthShare &x) {
    return MUL_CALL(cond, x);
}

// No-open wrappers for core ops.
static AuthShare mul_noopen(const AuthShare &x, const AuthShare &y) {
    return auth_mul(x, y);
}

static AuthShare matmul_noopen(u64 M, u64 K, u64 N, const AuthShare &x, const span<u64> &w) {
    return auth_matmul(M, K, N, x, w);
}

static AuthShare conv_noopen(u64 k, u64 padding, u64 stride, u64 inC, u64 outC, u64 H, u64 W,
                             const AuthShare &x, const span<u64> &w) {
    return auth_conv(k, padding, stride, inC, outC, H, W, x, w);
}

static inline u64 idx4(u64 b, u64 h, u64 w, u64 c, u64 H, u64 W, u64 C) {
    return ((b * H + h) * W + w) * C + c;
}

static bool comm_progress_enabled() {
    if (minimal_terminal_output_enabled()) return false;
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = std::getenv("SHARK_PROFILE_PROGRESS");
        if (env == nullptr || env[0] == '\0') {
            env = std::getenv("SHARK_COMM_PROGRESS");
        }
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

static bool profile_step_summary_enabled() {
    if (minimal_terminal_output_enabled()) return false;
    static int enabled = -1;
    if (enabled < 0) {
        enabled = (std::getenv("SHARK_PROFILE_STEP_SUMMARY") != nullptr) ? 1 : 0;
    }
    return enabled == 1;
}

static bool repeat_estimation_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = std::getenv("UNCLIP_ESTIMATE_REPEAT");
        if (env == nullptr || env[0] == '\0' ||
            env[0] == '0' || env[0] == 'n' || env[0] == 'N' ||
            env[0] == 'f' || env[0] == 'F') {
            enabled = 0;
        } else {
            enabled = 1;
        }
    }
    return enabled == 1;
}

static bool repeat_estimation_active_for_count(u64 total_count) {
    return repeat_estimation_enabled() && total_count > 3;
}

static u64 repeat_estimation_executed_count(u64 total_count) {
    return repeat_estimation_active_for_count(total_count) ? 3ull : total_count;
}

static bool repeat_estimation_should_execute_index(u64 total_count, u64 index) {
    if (!repeat_estimation_active_for_count(total_count)) return true;
    return index < 2ull || (index + 1ull == total_count);
}

static shark::utils::TimerStat make_manual_timer_delta(double time_ms, u64 comm_bytes, u64 rounds) {
    shark::utils::TimerStat stat{};
    stat.accumulated_time = (time_ms <= 0.0) ? 0ull : (u64)std::llround(time_ms);
    stat.accumulated_comm = comm_bytes;
    stat.accumulated_rounds = rounds;
    return stat;
}

struct StableUnCLIPBenchConfig {
    const char *profile_name = stableunclip_autogen::kProfileName;
    u64 batch = stableunclip_autogen::kBatch;
    u64 image_h = stableunclip_autogen::kImageH;
    u64 image_w = stableunclip_autogen::kImageW;
    u64 image_channels = stableunclip_autogen::kImageC;
    u64 output_channels = stableunclip_autogen::kDecodedC;
    u64 vae_scale_factor = stableunclip_autogen::kVaeScaleFactor;
    u64 out_h = stableunclip_autogen::kDecodedH;
    u64 out_w = stableunclip_autogen::kDecodedW;
    u64 latent_channels = stableunclip_autogen::kLatentC;
    u64 seq_len = stableunclip_autogen::kTextSeqLen;
    u64 text_hidden = stableunclip_autogen::kTextHiddenSize;
    u64 text_num_heads = stableunclip_autogen::kTextHeads;
    u64 cond_embed_dim = stableunclip_autogen::kFeatureOutChannels;
    u64 clip_hidden_dim = stableunclip_autogen::kVisionWidth;
    u64 image_embed_dim = stableunclip_autogen::kVisionEmbedDim;
    u64 vocab_size = stableunclip_autogen::kTextVocabSize;
    u64 patch_size = stableunclip_autogen::kVisionPatchSize;
    u64 time_embed_input_dim = stableunclip_autogen::kUnetTimeProjDim;
    u64 time_embed_dim = stableunclip_autogen::kUnetTimeEmbedDim;
    u64 class_labels_dim = stableunclip_autogen::kClassLabelsDim;
    u64 noise_level = stableunclip_autogen::kNoiseLevel;
    int num_inference_steps = (int)stableunclip_autogen::kGenerateNumInferenceSteps;
    double guidance_scale = stableunclip_autogen::kGenerateGuidanceScale;
    u64 text_layers = stableunclip_autogen::kTextLayers;
    u64 vision_layers = stableunclip_autogen::kVisionLayers;
    u64 feature_layers = stableunclip_autogen::kFeatureLayers;
    u64 unet_layers_per_block = stableunclip_autogen::kUnetLayersPerBlock;
    u64 vae_layers_per_block = stableunclip_autogen::kVaeLayersPerBlock;
    u64 superres_layers = stableunclip_autogen::kSuperresLayers;
    u64 text_ff_inner = stableunclip_autogen::kTextIntermediateSize;
    u64 vision_ff_inner = stableunclip_autogen::kVisionFfInner;
    u64 feature_hidden_dim = stableunclip_autogen::kFeatureHiddenDim;
    u64 superres_hidden_dim = stableunclip_autogen::kSuperresHiddenChannels;
    u64 cfg_copies = stableunclip_autogen::kCfgCopies;
    u64 unet_depth = stableunclip_autogen::kUnetBlockOutChannels.size();
    u64 unet_ch = stableunclip_autogen::kUnetBlockOutChannels.front();
    u64 default_norm_groups = stableunclip_autogen::kVaeNormNumGroups;
    u64 default_transformer_blocks = stableunclip_autogen::kDefaultTransformerBlocks;
    u64 default_vae_temb_dim = stableunclip_autogen::kDefaultVaeTembDim;
    u64 vae_mid_channels = stableunclip_autogen::kVaeMidChannels;
    u64 clip_num_layers = stableunclip_autogen::kVisionLayers;
    u64 clip_num_heads = stableunclip_autogen::kVisionHeads;
    u64 attention_head_dim = stableunclip_autogen::kUnetAttentionHeadDim;
    u64 num_train_timesteps = stableunclip_autogen::kNumTrainTimesteps;
    double beta_start = stableunclip_autogen::kBetaStart;
    double beta_end = stableunclip_autogen::kBetaEnd;
    std::array<u64, 4> stable_unet_block_out_channels = stableunclip_autogen::kUnetBlockOutChannels;
    std::array<u64, 4> stable_unet_down_resnet_counts = stableunclip_autogen::kUnetDownResnetCounts;
    std::array<u64, 4> stable_unet_down_transformer_blocks = stableunclip_autogen::kUnetDownTransformerBlocks;
    std::array<u64, 4> stable_unet_up_resnet_counts = stableunclip_autogen::kUnetUpResnetCounts;
    std::array<u64, 4> stable_unet_up_transformer_blocks = stableunclip_autogen::kUnetUpTransformerBlocks;
    u64 stable_unet_mid_resnet_count = stableunclip_autogen::kUnetMidResnetCount;
    u64 stable_unet_mid_transformer_blocks = stableunclip_autogen::kUnetMidTransformerBlocks;
    std::array<u64, 4> stable_vae_block_out_channels = stableunclip_autogen::kVaeBlockOutChannels;
    bool use_stable_unet_channel_plan = true;
    bool use_stable_vae_channel_plan = true;
    bool use_linear_scheduler_fallback = stableunclip_autogen::kUseLinearSchedulerFallback;
    bool unet_cross_attn_enabled = stableunclip_autogen::kUnetCrossAttentionEnabled;
    bool unet_mid_attn_enabled = stableunclip_autogen::kUnetMidAttentionEnabled;
    bool text_encoder_enabled = stableunclip_autogen::kTextEncoderEnabled;
    bool use_text_conditioning = stableunclip_autogen::kUseTextConditioning;
    bool full_secure_image_encoder_enabled = stableunclip_autogen::kFullSecureImageEncoderEnabled;
    bool vision_encoder_enabled = stableunclip_autogen::kVisionEncoderEnabled;
    bool vae_mid_attn_enabled = stableunclip_autogen::kVaeMidAttentionEnabled;
    bool enable_superres = stableunclip_autogen::kEnableSuperres;
    bool secure_prompt_lookup = stableunclip_autogen::kSecurePromptLookup;
    bool use_light_image_encoder_fallback = stableunclip_autogen::kUseLightImageEncoderFallback;
    bool text_transformer_enabled = stableunclip_autogen::kTextTransformerEnabled;
    bool vision_transformer_enabled = stableunclip_autogen::kVisionTransformerEnabled;
    bool unet_transformer_enabled = stableunclip_autogen::kUnetTransformerEnabled;
    bool unet_mid_attn_runtime_enabled = stableunclip_autogen::kUnetMidAttnRuntimeEnabled;
    bool use_unet_class_proj = stableunclip_autogen::kUseUnetClassProj;
    bool external_spec_loaded = false;
    std::string external_spec_path{};
    u64 external_spec_fingerprint = 0;
};

static void align_unclip_generate_runtime(StableUnCLIPBenchConfig &cfg) {
    // Keep the runtime generation path aligned with stableunclip.py while
    // preserving the exported topology from the autogen spec.
    cfg.use_linear_scheduler_fallback = false;
    // stableunclip.py::generate() ignores prompt embeddings and feeds image_embeds
    // directly as encoder_hidden_states, with degenerate CFG where both branches
    // are the same prediction.
    cfg.use_text_conditioning = false;
    cfg.use_unet_class_proj = false;
    cfg.cfg_copies = 1;
}

static std::string default_stableunclip_autogen_header_path() {
    namespace fs = std::filesystem;
    const std::array<fs::path, 4> candidates{{
        fs::path("..") / "artifacts" / "stableunclip" / "stableunclip_spec_autogen.hpp",
        fs::path("..") / ".." / "artifacts" / "stableunclip" / "stableunclip_spec_autogen.hpp",
        fs::path("artifacts") / "stableunclip" / "stableunclip_spec_autogen.hpp",
        fs::path("..") / ".." / ".." / "artifacts" / "stableunclip" / "stableunclip_spec_autogen.hpp",
    }};
    for (const fs::path &candidate : candidates) {
        std::error_code ec;
        if (!fs::exists(candidate, ec) || ec) {
            continue;
        }
        fs::path resolved = fs::weakly_canonical(candidate, ec);
        if (ec) {
            ec.clear();
            resolved = fs::absolute(candidate, ec);
        }
        if (ec) {
            continue;
        }
        return resolved.generic_string();
    }
    return (fs::path("artifacts") / "stableunclip" / "stableunclip_spec_autogen.hpp").generic_string();
}

static void apply_stableunclip_autogen_sections(StableUnCLIPBenchConfig &cfg) {
    cfg.profile_name = stableunclip_autogen::kProfileName;
    cfg.external_spec_loaded = true;
    cfg.external_spec_path = default_stableunclip_autogen_header_path();
    cfg.external_spec_fingerprint = stableunclip_autogen::kSpecFingerprint;

    cfg.image_h = stableunclip_autogen::kImageH;
    cfg.image_w = stableunclip_autogen::kImageW;
    cfg.image_channels = stableunclip_autogen::kImageC;
    cfg.output_channels = stableunclip_autogen::kDecodedC;
    cfg.vae_scale_factor = stableunclip_autogen::kVaeScaleFactor;
    cfg.out_h = stableunclip_autogen::kDecodedH;
    cfg.out_w = stableunclip_autogen::kDecodedW;
    cfg.latent_channels = stableunclip_autogen::kLatentC;
    cfg.seq_len = stableunclip_autogen::kTextSeqLen;
    cfg.text_hidden = stableunclip_autogen::kTextHiddenSize;
    cfg.text_num_heads = stableunclip_autogen::kTextHeads;
    cfg.cond_embed_dim = stableunclip_autogen::kFeatureOutChannels;
    cfg.clip_hidden_dim = stableunclip_autogen::kVisionWidth;
    cfg.image_embed_dim = stableunclip_autogen::kVisionEmbedDim;
    cfg.vocab_size = stableunclip_autogen::kTextVocabSize;
    cfg.patch_size = stableunclip_autogen::kVisionPatchSize;
    cfg.time_embed_input_dim = stableunclip_autogen::kUnetTimeProjDim;
    cfg.time_embed_dim = stableunclip_autogen::kUnetTimeEmbedDim;
    cfg.class_labels_dim = stableunclip_autogen::kClassLabelsDim;
    cfg.num_inference_steps = (int)stableunclip_autogen::kGenerateNumInferenceSteps;
    cfg.guidance_scale = stableunclip_autogen::kGenerateGuidanceScale;
    cfg.text_layers = stableunclip_autogen::kTextLayers;
    cfg.vision_layers = stableunclip_autogen::kVisionLayers;
    cfg.feature_layers = stableunclip_autogen::kFeatureLayers;
    cfg.unet_layers_per_block = stableunclip_autogen::kUnetLayersPerBlock;
    cfg.vae_layers_per_block = stableunclip_autogen::kVaeLayersPerBlock;
    cfg.superres_layers = stableunclip_autogen::kSuperresLayers;
    cfg.text_ff_inner = stableunclip_autogen::kTextIntermediateSize;
    cfg.vision_ff_inner = stableunclip_autogen::kVisionFfInner;
    cfg.feature_hidden_dim = stableunclip_autogen::kFeatureHiddenDim;
    cfg.superres_hidden_dim = stableunclip_autogen::kSuperresHiddenChannels;
    cfg.unet_depth = stableunclip_autogen::kUnetBlockOutChannels.size();
    cfg.unet_ch = stableunclip_autogen::kUnetBlockOutChannels.front();
    cfg.default_norm_groups = stableunclip_autogen::kVaeNormNumGroups;
    cfg.default_transformer_blocks = stableunclip_autogen::kDefaultTransformerBlocks;
    cfg.vae_mid_channels = stableunclip_autogen::kVaeMidChannels;
    cfg.clip_num_layers = stableunclip_autogen::kVisionLayers;
    cfg.clip_num_heads = stableunclip_autogen::kVisionHeads;
    cfg.attention_head_dim = stableunclip_autogen::kUnetAttentionHeadDim;
    cfg.num_train_timesteps = stableunclip_autogen::kNumTrainTimesteps;
    cfg.beta_start = stableunclip_autogen::kBetaStart;
    cfg.beta_end = stableunclip_autogen::kBetaEnd;
    cfg.batch = stableunclip_autogen::kBatch;
    cfg.noise_level = stableunclip_autogen::kNoiseLevel;
    cfg.cfg_copies = stableunclip_autogen::kCfgCopies;
    cfg.default_vae_temb_dim = stableunclip_autogen::kDefaultVaeTembDim;
    cfg.use_linear_scheduler_fallback = stableunclip_autogen::kUseLinearSchedulerFallback;
    cfg.unet_cross_attn_enabled = stableunclip_autogen::kUnetCrossAttentionEnabled;
    cfg.unet_mid_attn_enabled = stableunclip_autogen::kUnetMidAttentionEnabled;
    cfg.text_encoder_enabled = stableunclip_autogen::kTextEncoderEnabled;
    cfg.use_text_conditioning = stableunclip_autogen::kUseTextConditioning;
    cfg.full_secure_image_encoder_enabled = stableunclip_autogen::kFullSecureImageEncoderEnabled;
    cfg.vision_encoder_enabled = stableunclip_autogen::kVisionEncoderEnabled;
    cfg.vae_mid_attn_enabled = stableunclip_autogen::kVaeMidAttentionEnabled;
    cfg.enable_superres = stableunclip_autogen::kEnableSuperres;
    cfg.secure_prompt_lookup = stableunclip_autogen::kSecurePromptLookup;
    cfg.use_light_image_encoder_fallback = stableunclip_autogen::kUseLightImageEncoderFallback;
    cfg.text_transformer_enabled = stableunclip_autogen::kTextTransformerEnabled;
    cfg.vision_transformer_enabled = stableunclip_autogen::kVisionTransformerEnabled;
    cfg.unet_transformer_enabled = stableunclip_autogen::kUnetTransformerEnabled;
    cfg.unet_mid_attn_runtime_enabled = stableunclip_autogen::kUnetMidAttnRuntimeEnabled;
    cfg.use_unet_class_proj = stableunclip_autogen::kUseUnetClassProj;
    cfg.stable_unet_block_out_channels = stableunclip_autogen::kUnetBlockOutChannels;
    cfg.stable_unet_down_resnet_counts = stableunclip_autogen::kUnetDownResnetCounts;
    cfg.stable_unet_down_transformer_blocks = stableunclip_autogen::kUnetDownTransformerBlocks;
    cfg.stable_unet_up_resnet_counts = stableunclip_autogen::kUnetUpResnetCounts;
    cfg.stable_unet_up_transformer_blocks = stableunclip_autogen::kUnetUpTransformerBlocks;
    cfg.stable_unet_mid_resnet_count = stableunclip_autogen::kUnetMidResnetCount;
    cfg.stable_unet_mid_transformer_blocks = stableunclip_autogen::kUnetMidTransformerBlocks;
    cfg.stable_vae_block_out_channels = stableunclip_autogen::kVaeBlockOutChannels;
    cfg.use_stable_unet_channel_plan = true;
    cfg.use_stable_vae_channel_plan = true;
    align_unclip_generate_runtime(cfg);

    always_assert(stableunclip_autogen::kVisionImageSize == stableunclip_autogen::kImageH);
    always_assert(stableunclip_autogen::kVisionImageSize == stableunclip_autogen::kImageW);
    always_assert(stableunclip_autogen::kLatentC == stableunclip_autogen::kUnetInChannels);
    always_assert(stableunclip_autogen::kLatentC == stableunclip_autogen::kUnetOutChannels);
    always_assert(stableunclip_autogen::kFeatureOutChannels == stableunclip_autogen::kUnetCrossAttentionDim);

    if (cfg.enable_superres) {
        always_assert(stableunclip_autogen::kDecodedC == stableunclip_autogen::kSuperresInChannels);
        always_assert(stableunclip_autogen::kDecodedC == stableunclip_autogen::kSuperresOutChannels);
    }
}

static std::vector<u64> build_unet_channel_plan(const StableUnCLIPBenchConfig &cfg) {
    if (cfg.use_stable_unet_channel_plan) {
        return std::vector<u64>(cfg.stable_unet_block_out_channels.begin(),
                                cfg.stable_unet_block_out_channels.end());
    }
    return std::vector<u64>(cfg.unet_depth, cfg.unet_ch);
}

static std::vector<u64> build_vae_decoder_channel_plan(const StableUnCLIPBenchConfig &cfg) {
    if (cfg.use_stable_vae_channel_plan) {
        return std::vector<u64>(cfg.stable_vae_block_out_channels.rbegin(),
                                cfg.stable_vae_block_out_channels.rend());
    }
    return std::vector<u64>();
}

static StableUnCLIPBenchConfig load_unclip_bench_config() {
    StableUnCLIPBenchConfig cfg;
    apply_stableunclip_autogen_sections(cfg);
    align_unclip_generate_runtime(cfg);
    always_assert(cfg.external_spec_loaded);

    always_assert(std::getenv("UNCLIP_BENCH_CONFIG_JSON") == nullptr);
    always_assert(std::getenv("UNCLIP_SPEC_JSON") == nullptr);
    always_assert(std::getenv("UNCLIP_USE_LINEAR_SCHEDULER") == nullptr);
    always_assert(std::getenv("UNCLIP_DISABLE_FULL_SECURE_IMAGE_ENCODER") == nullptr);
    always_assert(std::getenv("UNCLIP_LIGHT_IMAGE_ENCODER") == nullptr);
    always_assert(std::getenv("UNCLIP_DISABLE_UNET_CROSS_ATTN") == nullptr);
    always_assert(std::getenv("UNCLIP_DISABLE_UNET_MID_ATTN") == nullptr);
    always_assert(std::getenv("UNCLIP_DISABLE_VISION_ENCODER") == nullptr);
    always_assert(std::getenv("UNCLIP_DISABLE_VAE_MID_ATTN") == nullptr);
    always_assert(cfg.text_num_heads > 0);
    always_assert(cfg.text_ff_inner > 0);
    always_assert(cfg.vision_ff_inner > 0);
    always_assert(cfg.feature_hidden_dim > 0);
    always_assert(cfg.superres_hidden_dim > 0);

    always_assert(cfg.patch_size > 0);
    always_assert(cfg.image_h % cfg.patch_size == 0);
    always_assert(cfg.image_w % cfg.patch_size == 0);
    always_assert(cfg.image_h > 0);
    always_assert(cfg.image_w > 0);
    always_assert(cfg.image_channels > 0);
    always_assert(cfg.out_h > 0);
    always_assert(cfg.out_w > 0);
    always_assert(cfg.output_channels > 0);
    always_assert(cfg.vae_scale_factor > 0);
    always_assert(is_power_of_two_u64(cfg.vae_scale_factor));
    always_assert(cfg.out_h % cfg.vae_scale_factor == 0);
    always_assert(cfg.out_w % cfg.vae_scale_factor == 0);
    always_assert(cfg.text_hidden % cfg.text_num_heads == 0);
    always_assert(cfg.cfg_copies == 1 || cfg.cfg_copies == 2);
    always_assert(!cfg.use_unet_class_proj || cfg.class_labels_dim > 0);
    always_assert(!cfg.use_unet_class_proj || cfg.class_labels_dim >= cfg.cond_embed_dim);
    always_assert(cfg.full_secure_image_encoder_enabled || cfg.cond_embed_dim == cfg.image_embed_dim);
    always_assert(cfg.full_secure_image_encoder_enabled || cfg.cond_embed_dim == cfg.image_channels);
    return cfg;
}

static u64 build_unclip_runtime_protocol_fingerprint(const StableUnCLIPBenchConfig &cfg) {
    u64 runtime_protocol_fingerprint = 0x554e434c4950ULL;
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.batch);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.image_h);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.image_w);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.image_channels);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.output_channels);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.vae_scale_factor);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.out_h);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.out_w);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.latent_channels);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.seq_len);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.text_hidden);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.text_num_heads);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.cond_embed_dim);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.clip_hidden_dim);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.image_embed_dim);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.vocab_size);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.patch_size);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.time_embed_input_dim);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.time_embed_dim);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.class_labels_dim);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.noise_level);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, (u64)cfg.num_inference_steps);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.cfg_copies);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.unet_depth);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.unet_ch);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.unet_layers_per_block);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.vae_layers_per_block);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.text_layers);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.vision_layers);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.feature_layers);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.superres_layers);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.text_ff_inner);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.vision_ff_inner);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.feature_hidden_dim);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.superres_hidden_dim);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.default_norm_groups);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.default_transformer_blocks);
    runtime_protocol_fingerprint =
        mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.stable_unet_mid_resnet_count);
    runtime_protocol_fingerprint =
        mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.stable_unet_mid_transformer_blocks);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.default_vae_temb_dim);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.vae_mid_channels);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.clip_num_layers);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.clip_num_heads);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.attention_head_dim);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.num_train_timesteps);
    runtime_protocol_fingerprint = mix_u64_fingerprint(
        runtime_protocol_fingerprint, (u64)std::llround(cfg.beta_start * 1.0e15));
    runtime_protocol_fingerprint = mix_u64_fingerprint(
        runtime_protocol_fingerprint, (u64)std::llround(cfg.beta_end * 1.0e15));
    for (u64 ch : cfg.stable_unet_block_out_channels) {
        runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, ch);
    }
    for (u64 count : cfg.stable_unet_down_resnet_counts) {
        runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, count);
    }
    for (u64 count : cfg.stable_unet_down_transformer_blocks) {
        runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, count);
    }
    for (u64 count : cfg.stable_unet_up_resnet_counts) {
        runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, count);
    }
    for (u64 count : cfg.stable_unet_up_transformer_blocks) {
        runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, count);
    }
    for (u64 ch : cfg.stable_vae_block_out_channels) {
        runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, ch);
    }
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.unet_cross_attn_enabled ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.unet_mid_attn_enabled ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.text_encoder_enabled ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.vision_encoder_enabled ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.full_secure_image_encoder_enabled ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.use_light_image_encoder_fallback ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.use_stable_vae_channel_plan ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.use_linear_scheduler_fallback ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.use_text_conditioning ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.vae_mid_attn_enabled ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, norm_use_ssrsqrt_enabled() ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, norm_force_recip_div_enabled() ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.secure_prompt_lookup ? 1 : 0);
    runtime_protocol_fingerprint = mix_u64_fingerprint(runtime_protocol_fingerprint, cfg.use_unet_class_proj ? 1 : 0);
    return runtime_protocol_fingerprint;
}

static int g_sanitized_protocol_debug_env_count = 0;

static void unset_env_if_present(const char *name) {
    if (std::getenv(name) == nullptr) return;
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
    ++g_sanitized_protocol_debug_env_count;
}

static void sanitize_protocol_debug_env() {
    static bool done = false;
    if (done) return;
    done = true;
    if (std::getenv("UNCLIP_KEEP_PROTOCOL_DEBUG") != nullptr) return;
    if (std::getenv("SHARK_DEBUG_ENABLE_BATCHCHECK") != nullptr) return;
    unset_env_if_present("SHARK_DEBUG_BATCHCHECK_FIND");
    unset_env_if_present("SHARK_DEBUG_BATCHCHECK_SCAN");
    unset_env_if_present("SHARK_DEBUG_BATCHCHECK_VERBOSE");
    unset_env_if_present("SHARK_DEBUG_BATCHCHECK_PAIR");
    unset_env_if_present("SHARK_DEBUG_KEYPOS");
}

static void profile_progress(const char *label) {
    if (!comm_progress_enabled() || party == DEALER || !peer) {
        return;
    }
    static auto t0 = std::chrono::steady_clock::now();
    static u64 last_comm = 0;
    static u64 last_rounds = 0;

    const u64 comm = peer->bytesReceived() + peer->bytesSent();
    const u64 rounds = peer->roundsReceived() + peer->roundsSent();
    const u64 delta_comm = comm - last_comm;
    const u64 delta_rounds = rounds - last_rounds;
    last_comm = comm;
    last_rounds = rounds;

    const u64 elapsed_ms = (u64)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "[PROFILE_PROGRESS] " << (label ? label : "<null>")
              << " | elapsed=" << elapsed_ms << " ms"
              << ", total_comm=" << (comm / 1024.0) << " KB"
              << ", total_rounds=" << rounds
              << ", delta_comm=" << (delta_comm / 1024.0) << " KB"
              << ", delta_rounds=" << delta_rounds
              << std::endl;
}

static const char *party_name_short() {
    switch (party) {
        case SERVER: return "server";
        case CLIENT: return "client";
        case DEALER: return "dealer";
        default: return "unknown";
    }
}

static void print_unclip_config(const StableUnCLIPBenchConfig &cfg, u64 runtime_protocol_fingerprint) {
    if (minimal_terminal_output_enabled()) return;
    if (party == DEALER) return;
    std::cout << "[UNCLIP_CONFIG][" << party_name_short() << "] " << cfg.profile_name
              << " batch=" << cfg.batch
              << " img=" << cfg.image_h << "x" << cfg.image_w << "x" << cfg.image_channels
              << " out=" << cfg.out_h << "x" << cfg.out_w << "x" << cfg.output_channels
              << " latent=" << cfg.latent_channels
              << " seq_len=" << cfg.seq_len
              << " hidden=" << cfg.text_hidden
              << " text_heads=" << cfg.text_num_heads
              << " ctx_dim=" << cfg.cond_embed_dim
              << " clip_hidden=" << cfg.clip_hidden_dim
              << " img_emb_dim=" << cfg.image_embed_dim
              << " patch=" << cfg.patch_size
              << " temb=" << cfg.time_embed_dim
              << " class_labels_dim=" << cfg.class_labels_dim
              << " train_steps=" << cfg.num_train_timesteps
              << " beta_start=" << cfg.beta_start
              << " beta_end=" << cfg.beta_end
              << " steps=" << cfg.num_inference_steps
              << " cfg_copies=" << cfg.cfg_copies
              << " unet_depth=" << cfg.unet_depth
              << " unet_ch=" << cfg.unet_ch
              << " block_out=(" << cfg.stable_unet_block_out_channels[0]
              << "," << cfg.stable_unet_block_out_channels[1]
              << "," << cfg.stable_unet_block_out_channels[2]
              << "," << cfg.stable_unet_block_out_channels[3] << ")"
              << " unet_layers_per_block=" << cfg.unet_layers_per_block
              << " vae_mid_ch=" << cfg.vae_mid_channels
              << " vae_block_out=(" << cfg.stable_vae_block_out_channels[0]
              << "," << cfg.stable_vae_block_out_channels[1]
              << "," << cfg.stable_vae_block_out_channels[2]
              << "," << cfg.stable_vae_block_out_channels[3] << ")"
              << " vae_layers_per_block=" << cfg.vae_layers_per_block
              << " text_layers=" << cfg.text_layers
              << " vision_layers=" << cfg.vision_layers
              << " feature_layers=" << cfg.feature_layers
              << " superres_layers=" << cfg.superres_layers
              << " attn_head_dim=" << cfg.attention_head_dim
              << " unet_cross_attn=" << (cfg.unet_cross_attn_enabled ? 1 : 0)
              << " unet_mid_attn=" << (cfg.unet_mid_attn_enabled ? 1 : 0)
              << " text_encoder=" << (cfg.text_encoder_enabled ? 1 : 0)
              << " vision_encoder=" << (cfg.vision_encoder_enabled ? 1 : 0)
              << " light_image_encoder=" << (cfg.use_light_image_encoder_fallback ? 1 : 0)
              << " linear_scheduler=" << (cfg.use_linear_scheduler_fallback ? 1 : 0)
              << " full_secure_image=" << (cfg.full_secure_image_encoder_enabled ? 1 : 0)
              << " use_text_cond=" << (cfg.use_text_conditioning ? 1 : 0)
              << " unet_class_proj=" << (cfg.use_unet_class_proj ? 1 : 0)
              << " vae_mid_attn=" << (cfg.vae_mid_attn_enabled ? 1 : 0)
              << " estimate_repeat=" << (repeat_estimation_enabled() ? 1 : 0)
              << " norm_ssrsqrt=" << (norm_use_ssrsqrt_enabled() ? 1 : 0)
              << " norm_force_recip_div=" << (norm_force_recip_div_enabled() ? 1 : 0)
              << " secure_prompt_lookup=" << (cfg.secure_prompt_lookup ? 1 : 0)
              << " disable_batchcheck=" << (std::getenv("UNCLIP_DISABLE_BATCHCHECK") ? 1 : 0)
              << " debug_find=" << (std::getenv("SHARK_DEBUG_BATCHCHECK_FIND") ? 1 : 0)
              << " debug_scan=" << (std::getenv("SHARK_DEBUG_BATCHCHECK_SCAN") ? 1 : 0)
              << " debug_verbose=" << (std::getenv("SHARK_DEBUG_BATCHCHECK_VERBOSE") ? 1 : 0)
              << " debug_keypos=" << (std::getenv("SHARK_DEBUG_KEYPOS") ? 1 : 0)
              << " protocol_fp=" << runtime_protocol_fingerprint;
    if (cfg.external_spec_loaded) {
        std::cout << " spec=" << cfg.external_spec_path
                  << " spec_fp=" << cfg.external_spec_fingerprint;
    }
    std::cout
              << std::endl;
}

enum class PhaseDBenchKind {
    Full = 0,
    Block = 1,
    OpReplay = 2,
};

static PhaseDBenchKind phase_d_bench_kind() {
#if defined(UNCLIP_STANDALONE_BLOCK)
    return PhaseDBenchKind::Block;
#elif defined(UNCLIP_STANDALONE_OPREPLAY)
    return PhaseDBenchKind::OpReplay;
#else
    const char *env = std::getenv("UNCLIP_BENCH_KIND");
    if (env == nullptr || env[0] == '\0') return PhaseDBenchKind::Full;
    if (std::strcmp(env, "block") == 0) return PhaseDBenchKind::Block;
    if (std::strcmp(env, "opreplay") == 0) return PhaseDBenchKind::OpReplay;
    return PhaseDBenchKind::Full;
#endif
}

static const char *phase_d_bench_kind_name() {
    switch (phase_d_bench_kind()) {
        case PhaseDBenchKind::Block: return "block";
        case PhaseDBenchKind::OpReplay: return "opreplay";
        case PhaseDBenchKind::Full:
        default: return "full";
    }
}

static std::string phase_d_block_target() {
    const char *env = std::getenv("UNCLIP_BLOCK_TARGET");
    if (env == nullptr || env[0] == '\0') return "full";
    return std::string(env);
}

static bool phase_d_block_target_is(const char *name) {
    return phase_d_bench_kind() == PhaseDBenchKind::Block &&
           phase_d_block_target() == std::string(name ? name : "");
}

static bool phase_d_profile_schema_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = std::getenv("UNCLIP_PROFILE_SCHEMA");
        if (env == nullptr || env[0] == '\0') {
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

static void light_progress_msg(const char *label) {
    if (!comm_progress_enabled() || party == DEALER) {
        return;
    }
    std::cout << "[COMM][" << party_name_short() << "] "
              << (label ? label : "<null>") << std::endl;
}

static thread_local const char *g_owner_input_trace_label = nullptr;

struct ScopedOwnerInputTraceLabel {
    const char *prev = nullptr;

    explicit ScopedOwnerInputTraceLabel(const char *label)
        : prev(g_owner_input_trace_label) {
        g_owner_input_trace_label = label;
    }

    ~ScopedOwnerInputTraceLabel() {
        g_owner_input_trace_label = prev;
    }
};

static void owner_input_progress(const char *phase) {
    if (g_owner_input_trace_label == nullptr) return;
    std::string msg = std::string(g_owner_input_trace_label) + ":" + (phase ? phase : "<null>");
    light_progress_msg(msg.c_str());
}

static u64 mix_u64_fingerprint(u64 acc, u64 v) {
    acc ^= v + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
    return acc;
}

static void verify_runtime_protocol_fingerprint(u64 fingerprint, const char *label) {
    if (party == DEALER || peer == nullptr) return;
    peer->send<u64>(fingerprint);
    u64 peer_fingerprint = peer->recv<u64>();
    if (peer_fingerprint != fingerprint) {
        std::cerr << "[UNCLIP_ERROR] protocol mismatch at "
                  << (label ? label : "<null>")
                  << " local_party=" << party_name_short()
                  << " local_fp=" << fingerprint
                  << " peer_fp=" << peer_fingerprint
                  << std::endl;
        always_assert(peer_fingerprint == fingerprint);
    }
}

static bool debug_protocol_fingerprint_enabled() {
    return std::getenv("UNCLIP_DEBUG_PROTOCOL_FP") != nullptr;
}

static bool debug_boundary_sync_enabled() {
    return std::getenv("UNCLIP_DEBUG_SYNC_BOUNDARY") != nullptr;
}

// Common malicious-check barrier used across the benchmark.
// Call chain: set_debug_batchcheck_context(label) stores a human-readable marker in the protocol
// layer, light_progress_msg() optionally emits progress, batch_check() verifies all buffered MAC
// equations and aborts on mismatch, and profile_progress(label) attributes the barrier in reports.
static void clip_batch_check(const char* label) {
    // Keep the active label in the protocol layer so abort logs point back to the benchmark stage.
    shark::protocols::set_debug_batchcheck_context(label);
    if (party != DEALER) {
        light_progress_msg(label);
        // batch_check() is the actual malicious-security gate: it verifies every pending MAC equation.
        shark::protocols::batch_check();
    }
    // The benchmark profiler treats each check barrier as a separate progress / timing point.
    profile_progress(label);
}

// Forward declarations for timed wrappers.
static AuthShare gelu_apply(const AuthShare &x);
static AuthShare silu_apply(const AuthShare &x);
static AuthShare mul_qf(const AuthShare &a, const AuthShare &b);
static AuthShare scale_public(const AuthShare &x, double v);
static AuthShare layernorm_rows(u64 rows, u64 cols, const AuthShare &X,
                                const AuthShare *weight = nullptr, const AuthShare *bias = nullptr,
                                const char *debug_name = nullptr);
static AuthShare groupnorm_apply(u64 B, u64 H, u64 W, u64 C, const AuthShare &x,
                                 const AuthShare *weight = nullptr, const AuthShare *bias = nullptr);
static AuthShare groupnorm_apply_groups(u64 B, u64 H, u64 W, u64 C, u64 groups, const AuthShare &x,
                                        const AuthShare *weight = nullptr, const AuthShare *bias = nullptr);

static void print_key_file_sizes() {
    if (minimal_terminal_output_enabled()) return;
    namespace fs = std::filesystem;
    const char *server_path = "server.dat";
    const char *client_path = "client.dat";
    if (fs::exists(server_path)) {
        auto sz = fs::file_size(server_path);
        std::cout << "[KEY] server.dat size: " << (double)sz / (1024.0 * 1024.0)
                  << " MB (" << sz << " bytes)" << std::endl;
    }
    if (fs::exists(client_path)) {
        auto sz = fs::file_size(client_path);
        std::cout << "[KEY] client.dat size: " << (double)sz / (1024.0 * 1024.0)
                  << " MB (" << sz << " bytes)" << std::endl;
    }
}

static bool keygen_progress_enabled() {
    if (party == DEALER) return true;
    if (minimal_terminal_output_enabled()) return false;
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

// Keygen progress metadata (dealer-only display path).
static u64 keygen_global_step = 0;
static u64 keygen_global_total = 0;
static u64 keygen_layer_total[8] = {0};
static u64 keygen_layer_seen[8] = {0};

static void keygen_print_plan() {
    if (party != DEALER) return;
    std::cout << "[KEYGEN_PLAN] "
              << "kind=" << phase_d_bench_kind_name()
              << " total=" << keygen_global_total
              << " init=" << keygen_layer_total[0]
              << " auth_weights=" << keygen_layer_total[1]
              << " inputs=" << keygen_layer_total[2]
              << " image_encoder=" << keygen_layer_total[3]
              << " diffusion_loop=" << keygen_layer_total[4]
              << " unet_per_step=" << keygen_layer_total[5]
              << " vae_decode=" << keygen_layer_total[6]
              << std::endl;
}

static const char *keygen_stage_name(int layer) {
    switch (layer) {
        case 0: return "Init";
        case 1: return "AuthWeights";
        case 2: return "Inputs";
        case 3: return "ImageEncoder";
        case 4: return "DiffusionLoop";
        case 5: return "UNetPerStep";
        case 6: return "VAEDecode";
        default: return "Unknown";
    }
}

static const char *keygen_default_model_name(int layer) {
    switch (layer) {
        case 0: return "System";
        case 1: return "Model";
        case 2: return "InputPipeline";
        case 3: return "ImageEncoder";
        case 4: return "Diffusion";
        case 5: return "UNet";
        case 6: return "VAE";
        default: return "Unknown";
    }
}

static std::string keygen_model_name(const std::string &label, int layer) {
    if (label.find("image_encoder") != std::string::npos) return "ImageEncoder";
    if (label.find("unet") != std::string::npos) return "UNet";
    if (label.find("vae") != std::string::npos) return "VAE";
    if (label.find("prompt") != std::string::npos || label.find("text") != std::string::npos) return "TextEncoder";
    if (label.find("input:image") != std::string::npos) return "ImageInput";
    if (label.find("input:noise") != std::string::npos || label.find("input:latents") != std::string::npos) return "DiffusionInput";
    if (label.find("start") != std::string::npos) return "System";
    return keygen_default_model_name(layer);
}

static void keygen_configure_progress(int num_inference_steps) {
    const u64 executed_diffusion_steps =
        repeat_estimation_executed_count((num_inference_steps > 0) ? (u64)num_inference_steps : 0);
    for (u64 &v : keygen_layer_total) {
        v = 0;
    }
    for (u64 &v : keygen_layer_seen) {
        v = 0;
    }
    keygen_layer_total[0] = 1;  // start
    keygen_layer_total[1] = 1;  // authenticated weights for the complete arch
    keygen_layer_total[2] = 4;  // prompt, image, noise, latents
    keygen_layer_total[3] = 3;  // patch_conv, token/transformer, cls_proj
    keygen_layer_total[4] = executed_diffusion_steps;
    keygen_layer_total[5] = 6ull * executed_diffusion_steps;
    keygen_layer_total[6] = 2;  // vae decode, optional superres
    keygen_global_total = 0;
    for (u64 i = 0; i < 8; ++i) {
        keygen_global_total += keygen_layer_total[i];
    }
    keygen_global_step = 0;
    keygen_print_plan();
}

static void keygen_ckpt(int layer, int component, const std::string &label) {
    if (party != DEALER || !keygen_progress_enabled()) {
        return;
    }
    keygen_global_step += 1;
    const u64 stage_total = (layer >= 0 && layer < 8) ? keygen_layer_total[layer] : 0;
    u64 stage_step = 0;
    if (layer >= 0 && layer < 8) {
        keygen_layer_seen[layer] += 1;
        stage_step = keygen_layer_seen[layer];
    }
    const std::string model = keygen_model_name(label, layer);

    std::cout << "[KEYGEN] "
              << "stage=" << keygen_stage_name(layer)
              << " model=" << model
              << " step=" << stage_step;
    if (stage_total > 0) {
        std::cout << "/" << stage_total;
    } else {
        std::cout << "/?";
    }
    std::cout << " global=" << keygen_global_step;
    if (keygen_global_total > 0) {
        std::cout << "/" << keygen_global_total;
    } else {
        std::cout << "/?";
    }
    std::cout << " L" << layer << "/C" << component
              << " | " << label << std::endl;
    print_key_file_sizes();
}

static std::string g_unclip_last_image_path = "unclip_out.jpg";

static std::string replace_file_extension(const char *path, const char *new_ext) {
    std::string out = path;
    size_t slash = out.find_last_of("/\\");
    size_t dot = out.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        out += new_ext;
    } else {
        out.replace(dot, std::string::npos, new_ext);
    }
    return out;
}

static void tone_map_output_for_display(span<u64> &img, int fp) {
    if (img.size() == 0) return;
    const double inv_scale = 1.0 / (double)(1ULL << fp);
    double raw_min = 0.0;
    double raw_max = 0.0;
    double max_abs = 0.0;
    for (u64 i = 0; i < img.size(); ++i) {
        double v = (double)(int64_t)img[i] * inv_scale;
        if (i == 0) {
            raw_min = v;
            raw_max = v;
        } else {
            raw_min = std::min(raw_min, v);
            raw_max = std::max(raw_max, v);
        }
        max_abs = std::max(max_abs, std::abs(v));
    }

    always_assert(std::getenv("UNCLIP_OUTPUT_TANH_GAIN") == nullptr);
    always_assert(std::getenv("UNCLIP_DISABLE_OUTPUT_AUTOSCALE") == nullptr);
    double gain = 1.0;
    const bool autoscale = true;
    if (autoscale) gain /= std::max(1.0, max_abs);

    if (!minimal_terminal_output_enabled()) {
        std::cout << "[UNCLIP] raw_output_range: min=" << raw_min
                  << " max=" << raw_max
                  << " max_abs=" << max_abs
                  << " tanh_gain=" << gain
                  << " autoscale=" << (autoscale ? 1 : 0)
                  << std::endl;
    }

    for (u64 i = 0; i < img.size(); ++i) {
        double v = (double)(int64_t)img[i] * inv_scale;
        int64_t q = (int64_t)std::llround(std::tanh(v * gain) * (double)(1ULL << fp));
        img[i] = (u64)q;
    }
}

static bool write_image_from_fixed(const span<u64> &img, u64 H, u64 W, u64 C, int fp, const char *path) {
    if (img.size() != H * W * C) return false;
    const int outC = 3;
    std::vector<unsigned char> buf((size_t)H * (size_t)W * (size_t)outC);
    const double scale = 1.0 / (double)(1ULL << fp);
    for (u64 h = 0; h < H; ++h) {
        for (u64 widx = 0; widx < W; ++widx) {
            if (C == 1) {
                size_t src = ((size_t)h * W + widx) * C;
                double v = (double)(int64_t)img[src] * scale;
                v = (v + 1.0) * 0.5;
                if (v < 0.0) v = 0.0;
                if (v > 1.0) v = 1.0;
                unsigned char u = (unsigned char)std::llround(v * 255.0);
                size_t dst = ((size_t)h * W + widx) * outC;
                buf[dst + 0] = u;
                buf[dst + 1] = u;
                buf[dst + 2] = u;
            } else {
                auto to_u8 = [&](u64 cidx) -> unsigned char {
                    size_t src = (((size_t)h * W + widx) * C) + cidx;
                    double v = (double)(int64_t)img[src] * scale;
                    v = (v + 1.0) * 0.5;
                    if (v < 0.0) v = 0.0;
                    if (v > 1.0) v = 1.0;
                    return (unsigned char)std::llround(v * 255.0);
                };
                size_t dst = ((size_t)h * W + widx) * outC;
                if (C >= 3) {
                    buf[dst + 0] = to_u8(0);
                    buf[dst + 1] = to_u8(1);
                    buf[dst + 2] = to_u8(2);
                } else if (C == 2) {
                    unsigned char u0 = to_u8(0);
                    unsigned char u1 = to_u8(1);
                    buf[dst + 0] = u0;
                    buf[dst + 1] = u1;
                    buf[dst + 2] = (unsigned char)(((unsigned int)u0 + (unsigned int)u1) / 2U);
                } else {
                    buf[dst + 0] = 0;
                    buf[dst + 1] = 0;
                    buf[dst + 2] = 0;
                }
            }
        }
    }
    g_unclip_last_image_path = path;
    const bool tiny_image = (H < 8 || W < 8);
    if (!tiny_image && stbi_write_jpg(path, (int)W, (int)H, outC, buf.data(), 95) != 0) {
        return true;
    }

    auto png_path = replace_file_extension(path, ".png");
    if (stbi_write_png(png_path.c_str(), (int)W, (int)H, outC, buf.data(), (int)(W * outC)) != 0) {
        g_unclip_last_image_path = png_path;
        return true;
    }

    auto bmp_path = replace_file_extension(path, ".bmp");
    if (stbi_write_bmp(bmp_path.c_str(), (int)W, (int)H, outC, buf.data()) != 0) {
        g_unclip_last_image_path = bmp_path;
        return true;
    }

    return false;
}

static AuthShare timed_gelu(const char *name, const AuthShare &x) {
    OpProfileScope op_profile(OP_GELU);
    static constexpr const char *kProfileGeluTimer = "gelu";
    shark::utils::start_timer(kProfileGeluTimer);
    shark::utils::start_timer(name);
    auto out = gelu_apply(x);
    shark::utils::stop_timer(name);
    shark::utils::stop_timer(kProfileGeluTimer);
    return out;
}

static AuthShare timed_silu(const char *name, const AuthShare &x) {
    OpProfileScope op_profile(OP_SILU);
    static constexpr const char *kProfileSiluTimer = "silu";
    shark::utils::start_timer(kProfileSiluTimer);
    shark::utils::start_timer(name);
    auto out = silu_apply(x);
    shark::utils::stop_timer(name);
    shark::utils::stop_timer(kProfileSiluTimer);
    return out;
}

static double time_mlp_silu_plain_value(double xv) {
    if (xv < -6.0) return 0.0;
    if (xv < -2.0) {
        return -0.52212664 + (-0.16910363 * xv) + (-0.01420163 * xv * xv);
    }
    if (xv <= 6.0) {
        const double x2 = xv * xv;
        const double x4 = x2 * x2;
        const double x6 = x2 * x4;
        return 0.03453821 + 0.49379432 * xv + 0.19784596 * x2
            - 0.00602401 * x4 + 0.00008032 * x6;
    }
    return xv;
}

static inline double q_to_double_time_mlp(u64 x) {
    return (double)(int64_t)x / (double)(1ULL << f);
}

static inline u64 double_to_q_time_mlp(double x) {
    return (u64)(int64_t)std::llround(x * (double)(1ULL << f));
}

static span<u64> plain_linear_rows_server(u64 rows,
                                          u64 in_dim,
                                          u64 out_dim,
                                          const span<u64> &x,
                                          const span<u64> &w,
                                          const span<u64> &b) {
    always_assert(x.size() == rows * in_dim);
    always_assert(w.size() == in_dim * out_dim);
    always_assert(b.size() == out_dim);
    span<u64> out(rows * out_dim);
    zero_plain(out);
    if (party != SERVER) return out;
    for (u64 r = 0; r < rows; ++r) {
        for (u64 j = 0; j < out_dim; ++j) {
            double acc = q_to_double_time_mlp(b[j]);
            for (u64 k = 0; k < in_dim; ++k) {
                acc += q_to_double_time_mlp(x[r * in_dim + k]) *
                       q_to_double_time_mlp(w[k * out_dim + j]);
            }
            out[r * out_dim + j] = double_to_q_time_mlp(acc);
        }
    }
    return out;
}

static AuthShare timed_time_mlp_silu_plain(const char *name, const AuthShare &x) {
    OpProfileScope op_profile(OP_SILU);
    static constexpr const char *kProfileSiluTimer = "silu";
    shark::utils::start_timer(kProfileSiluTimer);
    shark::utils::start_timer(name);
    auto opened = auth_open_authenticated(x);
    span<u64> out(opened.size());
    const double scale = 1.0 / (double)(1ULL << f);
    #pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)opened.size(); ++i) {
        const double xv = (double)(int64_t)opened[(u64)i] * scale;
        const double y = time_mlp_silu_plain_value(xv);
        out[(u64)i] = (u64)(int64_t)std::llround(y * (double)(1ULL << f));
    }
    auto auth = auth_from_public_span(out);
    shark::utils::stop_timer(name);
    shark::utils::stop_timer(kProfileSiluTimer);
    return auth;
}

static AuthShare timed_layernorm_rows(const char *name, u64 rows, u64 cols, const AuthShare &x,
                                      const AuthShare *weight = nullptr, const AuthShare *bias = nullptr) {
    ScopedCommTrace trace(name);
    OpProfileScope op_profile(OP_LAYERNORM);
    static constexpr const char *kProfileLayerNormTimer = "layernorm";
    shark::utils::start_timer(kProfileLayerNormTimer);
    shark::utils::start_timer(name);
    auto out = layernorm_rows(rows, cols, x, weight, bias, name);
    shark::utils::stop_timer(name);
    shark::utils::stop_timer(kProfileLayerNormTimer);
    return out;
}

static AuthShare timed_groupnorm_apply(const char *name, u64 B, u64 H, u64 W, u64 C, const AuthShare &x,
                                       const AuthShare *weight = nullptr, const AuthShare *bias = nullptr) {
    ScopedCommTrace trace(name);
    OpProfileScope op_profile(OP_GROUPNORM);
    static constexpr const char *kProfileGroupNormTimer = "groupnorm";
    shark::utils::start_timer(kProfileGroupNormTimer);
    shark::utils::start_timer(name);
    auto out = groupnorm_apply(B, H, W, C, x, weight, bias);
    shark::utils::stop_timer(name);
    shark::utils::stop_timer(kProfileGroupNormTimer);
    return out;
}

static AuthShare timed_groupnorm_apply(const char *name, u64 B, u64 H, u64 W, u64 C,
                                       u64 groups, const AuthShare &x,
                                       const AuthShare *weight = nullptr, const AuthShare *bias = nullptr) {
    OpProfileScope op_profile(OP_GROUPNORM);
    static constexpr const char *kProfileGroupNormTimer = "groupnorm";
    shark::utils::start_timer(kProfileGroupNormTimer);
    shark::utils::start_timer(name);
    auto out = groupnorm_apply_groups(B, H, W, C, groups, x, weight, bias);
    shark::utils::stop_timer(name);
    shark::utils::stop_timer(kProfileGroupNormTimer);
    return out;
}

static void print_profile_timers() {
    if (minimal_terminal_output_enabled()) return;
    using shark::utils::print_timer;
    std::cout << "[PROFILE] time(ms), comm(KB) per leaf op" << std::endl;
    print_timer("total_eval");
    print_timer("input");
    print_timer("reconstruct");
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

static void print_profile_components_table_impl(
    const char *table_prefix,
    const char *op_table_prefix,
    const char *timer_table_prefix,
    const char *reconcile_prefix,
    const std::array<OpProfileStat, OP_PROFILE_COUNT> &op_stats,
    const EvalProfileTotals &total_eval,
    const std::array<shark::utils::TimerStat, kProfileTimerRows.size()> &timer_stats,
    u64 network_total_comm_bytes,
    u64 network_total_rounds) {
    const auto old_flags = std::cout.flags();
    const auto old_prec = std::cout.precision();
    std::cout << std::fixed << std::setprecision(3);

    double accounted_time_ms = 0.0;
    u64 accounted_comm_bytes = 0;
    u64 accounted_rounds = 0;

    std::cout << table_prefix << " component,total_time_ms,total_comm_mb,total_rounds" << std::endl;
    for (size_t idx = 0; idx < OP_PROFILE_COUNT; ++idx) {
        const auto &stat = op_stats[idx];
        std::cout << table_prefix << " " << stat.name
                  << "," << stat.time_ms
                  << "," << ((double)stat.comm_bytes / (1024.0 * 1024.0))
                  << "," << stat.rounds
                  << std::endl;
        accounted_time_ms += stat.time_ms;
        accounted_comm_bytes += stat.comm_bytes;
        accounted_rounds += stat.rounds;
    }

    double other_time_ms = total_eval.time_ms - accounted_time_ms;
    if (other_time_ms < 0.0 && other_time_ms > -1e-6) other_time_ms = 0.0;
    u64 other_comm_bytes = (total_eval.comm_bytes >= accounted_comm_bytes)
        ? (total_eval.comm_bytes - accounted_comm_bytes) : 0;
    u64 other_rounds = (total_eval.rounds >= accounted_rounds)
        ? (total_eval.rounds - accounted_rounds) : 0;

    std::cout << table_prefix << " other"
              << "," << other_time_ms
              << "," << ((double)other_comm_bytes / (1024.0 * 1024.0))
              << "," << other_rounds
              << std::endl;
    std::cout << table_prefix << " leaf_accounted"
              << "," << (accounted_time_ms + other_time_ms)
              << "," << ((double)(accounted_comm_bytes + other_comm_bytes) / (1024.0 * 1024.0))
              << "," << (accounted_rounds + other_rounds)
              << std::endl;

    std::cout << op_table_prefix
              << " component,calls,total_time_ms,total_comm_mb,total_rounds,avg_time_ms,avg_comm_mb,avg_rounds"
              << std::endl;
    for (size_t idx = 0; idx < OP_PROFILE_COUNT; ++idx) {
        const auto &stat = op_stats[idx];
        const auto avg = avg_op_profile_from_stat(stat);
        std::cout << op_table_prefix << " " << stat.name
                  << "," << stat.calls
                  << "," << stat.time_ms
                  << "," << ((double)stat.comm_bytes / (1024.0 * 1024.0))
                  << "," << stat.rounds
                  << "," << avg.time_ms
                  << "," << avg.comm_mb
                  << "," << avg.rounds
                  << std::endl;
    }

    std::cout << timer_table_prefix << " component,total_time_ms,total_comm_mb,total_rounds" << std::endl;
    for (size_t i = 0; i < kProfileTimerRows.size(); ++i) {
        const auto &row = kProfileTimerRows[i];
        const auto &stat = timer_stats[i];
        std::cout << timer_table_prefix << " " << row.component
                  << "," << stat.accumulated_time
                  << "," << (double)stat.accumulated_comm / (1024.0 * 1024.0)
                  << "," << stat.accumulated_rounds
                  << std::endl;
    }

    std::cout << timer_table_prefix << " network_total"
              << ",-"
              << "," << (double)network_total_comm_bytes / (1024.0 * 1024.0)
              << "," << network_total_rounds
              << std::endl;

    std::cout << reconcile_prefix << " leaf_accounted_time_ms="
              << (accounted_time_ms + other_time_ms)
              << " total_eval_time_ms=" << total_eval.time_ms
              << std::endl;
    std::cout << reconcile_prefix << " leaf_accounted_comm_mb="
              << ((double)(accounted_comm_bytes + other_comm_bytes) / (1024.0 * 1024.0))
              << " total_eval_comm_mb="
              << ((double)total_eval.comm_bytes / (1024.0 * 1024.0))
              << std::endl;
    std::cout << reconcile_prefix << " leaf_accounted_rounds="
              << (accounted_rounds + other_rounds)
              << " total_eval_rounds=" << total_eval.rounds
              << std::endl;
    std::cout.flags(old_flags);
    std::cout.precision(old_prec);
}

static void print_profile_components_table() {
    if (minimal_terminal_output_enabled()) return;
    if (party == DEALER) return;
    print_profile_components_table_impl(
        "[PROFILE_TABLE]",
        "[PROFILE_OP_TABLE]",
        "[PROFILE_TIMER_TABLE]",
        "[PROFILE_RECONCILE]",
        g_op_stats,
        current_total_eval_totals(),
        capture_component_timer_stats(),
        current_comm_bytes(),
        current_rounds());
}

static void print_estimated_profile_components_table() {
    if (minimal_terminal_output_enabled()) return;
    if (party == DEALER || !g_estimated_profile.active) return;
    for (const auto &meta : g_estimated_profile_meta_lines) {
        std::cout << "[EST_PROFILE_META] " << meta
                  << " output=truncated_after_real_steps"
                  << std::endl;
    }
    print_profile_components_table_impl(
        "[EST_PROFILE_TABLE]",
        "[EST_PROFILE_OP_TABLE]",
        "[EST_PROFILE_TIMER_TABLE]",
        "[EST_PROFILE_RECONCILE]",
        g_estimated_profile.op_stats,
        g_estimated_profile.total_eval,
        g_estimated_profile.timers,
        g_estimated_profile.network_total_comm_bytes,
        g_estimated_profile.network_total_rounds);
}

static void print_phase_d_trace_template_summary() {
    if (minimal_terminal_output_enabled()) return;
    if (phase_d_bench_kind() != PhaseDBenchKind::OpReplay || party == DEALER) return;
    const char *path = std::getenv("UNCLIP_TRACE_TEMPLATE_JSONL");
    std::map<std::string, u64> op_counts;
    std::string line;
    auto collect_counts = [&](std::istream &in) {
        while (std::getline(in, line)) {
            const size_t pos = line.find("\"op\"");
            if (pos == std::string::npos) continue;
            size_t colon = line.find(':', pos);
            if (colon == std::string::npos) continue;
            size_t q1 = line.find('"', colon + 1);
            if (q1 == std::string::npos) continue;
            size_t q2 = line.find('"', q1 + 1);
            if (q2 == std::string::npos) continue;
            op_counts[line.substr(q1 + 1, q2 - q1 - 1)] += 1;
        }
    };
    if (path && *path) {
        std::ifstream in(path);
        if (!in) {
            std::cout << "[TRACE_TEMPLATE] missing,path=" << path << std::endl;
            return;
        }
        collect_counts(in);
    } else {
        std::istringstream in(stableunclip_autogen::kTraceTemplateJsonl);
        collect_counts(in);
    }

    std::cout << "[TRACE_TEMPLATE] op,records" << std::endl;
    for (const auto &kv : op_counts) {
        std::cout << "[TRACE_TEMPLATE] " << kv.first << "," << kv.second << std::endl;
    }
}

static void print_phase_d_profile_schema_impl(
    const char *meta_prefix,
    const char *schema_prefix,
    const StableUnCLIPBenchConfig &cfg,
    u64 runtime_protocol_fingerprint,
    const EvalProfileTotals &total_eval,
    const std::array<shark::utils::TimerStat, kProfileTimerRows.size()> &timer_stats,
    const std::array<OpProfileStat, OP_PROFILE_COUNT> &op_stats,
    const char *extra_meta = nullptr) {
    if (!phase_d_profile_schema_enabled() || party == DEALER) return;
    const auto old_flags = std::cout.flags();
    const auto old_prec = std::cout.precision();
    std::cout << std::fixed << std::setprecision(3);

    const std::string block_target = phase_d_block_target();
    std::cout << meta_prefix << " kind=" << phase_d_bench_kind_name()
              << ",block_target=" << block_target
              << ",profile=" << cfg.profile_name
              << ",protocol_fp=" << runtime_protocol_fingerprint;
    if (cfg.external_spec_loaded) {
        std::cout << ",spec=" << cfg.external_spec_path
                  << ",spec_fp=" << cfg.external_spec_fingerprint;
    }
    if (extra_meta && extra_meta[0] != '\0') {
        std::cout << "," << extra_meta;
    }
    std::cout << std::endl;

    std::cout << schema_prefix
              << " level,name,calls,total_time_ms,total_comm_mb,total_rounds,avg_time_ms,avg_comm_mb,avg_rounds"
              << std::endl;
    std::cout << schema_prefix << " end_to_end,total_eval,1,"
              << total_eval.time_ms << ","
              << ((double)total_eval.comm_bytes / (1024.0 * 1024.0)) << ","
              << total_eval.rounds << ","
              << total_eval.time_ms << ","
              << ((double)total_eval.comm_bytes / (1024.0 * 1024.0)) << ","
              << total_eval.rounds
              << std::endl;

    for (size_t i = 0; i < kProfileTimerRows.size(); ++i) {
        const auto &row = kProfileTimerRows[i];
        if (std::strcmp(row.component, "end-to-end") == 0) continue;
        const auto &stat = timer_stats[i];
        const double total_comm_mb = (double)stat.accumulated_comm / (1024.0 * 1024.0);
        std::cout << schema_prefix << " component," << row.component
                  << ",1,"
                  << stat.accumulated_time
                  << "," << total_comm_mb
                  << "," << stat.accumulated_rounds
                  << "," << stat.accumulated_time
                  << "," << total_comm_mb
                  << "," << stat.accumulated_rounds
                  << std::endl;
    }

    for (size_t idx = 0; idx < OP_PROFILE_COUNT; ++idx) {
        const auto &stat = op_stats[idx];
        const auto avg = avg_op_profile_from_stat(stat);
        std::cout << schema_prefix << " operator," << stat.name
                  << "," << stat.calls
                  << "," << stat.time_ms
                  << "," << ((double)stat.comm_bytes / (1024.0 * 1024.0))
                  << "," << stat.rounds
                  << "," << avg.time_ms
                  << "," << avg.comm_mb
                  << "," << avg.rounds
                  << std::endl;
    }

    std::cout.flags(old_flags);
    std::cout.precision(old_prec);
}

static void print_phase_d_profile_schema(const StableUnCLIPBenchConfig &cfg,
                                         u64 runtime_protocol_fingerprint) {
    if (minimal_terminal_output_enabled()) return;
    print_phase_d_profile_schema_impl(
        "[PROFILE_SCHEMA_META]",
        "[PROFILE_SCHEMA]",
        cfg,
        runtime_protocol_fingerprint,
        current_total_eval_totals(),
        capture_component_timer_stats(),
        g_op_stats);
}

static void print_estimated_phase_d_profile_schema(const StableUnCLIPBenchConfig &cfg,
                                                   u64 runtime_protocol_fingerprint) {
    if (minimal_terminal_output_enabled()) return;
    if (!g_estimated_profile.active) return;
    std::ostringstream meta;
    for (size_t i = 0; i < g_estimated_profile_meta_lines.size(); ++i) {
        if (i != 0) meta << ";";
        meta << g_estimated_profile_meta_lines[i];
    }
    if (!g_estimated_profile_meta_lines.empty()) {
        meta << ",output=truncated_after_real_steps";
    } else {
        meta << "output=truncated_after_real_steps";
    }
    const std::string meta_str = meta.str();
    print_phase_d_profile_schema_impl(
        "[EST_PROFILE_SCHEMA_META]",
        "[EST_PROFILE_SCHEMA]",
        cfg,
        runtime_protocol_fingerprint,
        g_estimated_profile.total_eval,
        g_estimated_profile.timers,
        g_estimated_profile.op_stats,
        meta_str.c_str());
}

static int finish_unclip_benchmark(const StableUnCLIPBenchConfig &cfg,
                                   u64 runtime_protocol_fingerprint) {
    if (party != DEALER) {
        shark::utils::stop_timer("total_eval");
        stop_total_eval_profile();
        if (g_estimated_profile_extras.active) {
            finalize_estimated_profile_projection(capture_profile_snapshot());
        }
        if (minimal_terminal_output_enabled()) {
            print_minimal_final_cost();
        } else if (shark::protocols::peer) {
            u64 total_comm = shark::protocols::peer->bytesReceived() + shark::protocols::peer->bytesSent();
            std::cout << "[PROFILE] total_comm: " << (double)total_comm / 1024.0
                      << " KB (" << total_comm << " bytes)" << std::endl;
        }
        print_profile_timers();
        print_profile_components_table();
        print_estimated_profile_components_table();
        print_phase_d_profile_schema(cfg, runtime_protocol_fingerprint);
        print_estimated_phase_d_profile_schema(cfg, runtime_protocol_fingerprint);
        print_legacy_profile_lines();
    }

    finalize::call();
    return 0;
}

static void print_legacy_profile_lines() {
    if (minimal_terminal_output_enabled()) return;
    if (party == DEALER) {
        return;
    }
    auto print_ms_kb = [](const char *alias, const char *timer_name) {
        shark::utils::TimerStat stat{};
        shark::utils::get_timer_stat(timer_name, stat);
        std::cout << alias << ": " << stat.accumulated_time << " ms, "
                  << (stat.accumulated_comm / 1024.0) << " KB" << std::endl;
    };

    print_ms_kb("stableunclip", "total_eval");
    print_ms_kb("input", "input");
    print_ms_kb("reconstruct", "reconstruct");
}

// -----------------------------
// Fixed-point activations
// -----------------------------
// GELU(x) = x * sigmoid(1.702 * x) (QuickGELU) with 5th-order sigmoid approximation
static AuthShare gelu_apply(const AuthShare &x) {
    auto add_const = [&](const AuthShare &src, double v) {
        auto c = make_public_const(src.share.size(), v, f);
        return ADD_CALL(src, c);
    };

    // x' = 1.702 * x
    auto xs = scale_public(x, 1.702);

    auto x2 = LRS_CALL(MUL_CALL(xs, xs), f);
    auto x3 = LRS_CALL(MUL_CALL(x2, xs), f);
    auto x4 = LRS_CALL(MUL_CALL(x2, x2), f);
    auto x5 = LRS_CALL(MUL_CALL(x4, xs), f);

    // 5th-order polynomial coefficients for sigmoid(x) approximation.
    // outer: x in [-7, -2.2), inner: x in [-2.2, 0)
    auto cO0 = make_public_const(x.share.size(), 5.90010486e-01, f);
    auto cI0 = make_public_const(x.share.size(), 4.99957471e-01, f);
    auto tO1 = scale_public(xs, 4.21319936e-01);
    auto tO2 = scale_public(x2, 1.26610092e-01);
    auto tO3 = scale_public(x3, 1.97985686e-02);
    auto tO4 = scale_public(x4, 1.59538912e-03);
    auto tO5 = scale_public(x5, 5.25625056e-05);

    auto P_outer = cO0;
    P_outer = ADD_CALL(P_outer, tO1);
    P_outer = ADD_CALL(P_outer, tO2);
    P_outer = ADD_CALL(P_outer, tO3);
    P_outer = ADD_CALL(P_outer, tO4);
    P_outer = ADD_CALL(P_outer, tO5);

    auto tI1 = scale_public(xs, 2.49155471e-01);
    auto tI2 = scale_public(x2, -4.01655846e-03);
    auto tI3 = scale_public(x3, -2.84974728e-02);
    auto tI4 = scale_public(x4, -6.76484741e-03);
    auto tI5 = scale_public(x5, -4.34163661e-04);

    auto P_inner = cI0;
    P_inner = ADD_CALL(P_inner, tI1);
    P_inner = ADD_CALL(P_inner, tI2);
    P_inner = ADD_CALL(P_inner, tI3);
    P_inner = ADD_CALL(P_inner, tI4);
    P_inner = ADD_CALL(P_inner, tI5);

    // P(-x) = c0 - c1*x + c2*x^2 - c3*x^3 + c4*x^4 - c5*x^5
    auto P_outer_neg = cO0;
    P_outer_neg = ADD_CALL(P_outer_neg, neg_span(tO1));
    P_outer_neg = ADD_CALL(P_outer_neg, tO2);
    P_outer_neg = ADD_CALL(P_outer_neg, neg_span(tO3));
    P_outer_neg = ADD_CALL(P_outer_neg, tO4);
    P_outer_neg = ADD_CALL(P_outer_neg, neg_span(tO5));

    auto P_inner_neg = cI0;
    P_inner_neg = ADD_CALL(P_inner_neg, neg_span(tI1));
    P_inner_neg = ADD_CALL(P_inner_neg, tI2);
    P_inner_neg = ADD_CALL(P_inner_neg, neg_span(tI3));
    P_inner_neg = ADD_CALL(P_inner_neg, tI4);
    P_inner_neg = ADD_CALL(P_inner_neg, neg_span(tI5));

    auto one = make_public_const(x.share.size(), 1.0, f);
    auto segL = make_public_const(x.share.size(), 0.0, f);
    auto segA = P_outer;
    auto segB = P_inner;
    auto segC = ADD_CALL(one, neg_span(P_inner_neg));
    auto segD = ADD_CALL(one, neg_span(P_outer_neg));
    auto segR = one;

    // Comparisons using compare.
    auto cmp_gt = [&](const AuthShare &src, double v) {
        auto diff = add_const(src, -v);
        return CMP_GE_ZERO_CALL(diff);
    };
    auto cmp_lt = [&](const AuthShare &src, double v) {
        auto negv = neg_span(src);
        auto diff = add_const(negv, v);
        return CMP_GE_ZERO_CALL(diff);
    };

    const double eps = 1.0 / (double)(1ull << f);
    auto lt_neg7 = cmp_lt(xs, -7.0 - eps);
    auto ge_neg2_2 = cmp_gt(xs, -2.2);
    auto ge_0 = cmp_gt(xs, 0.0);
    auto ge_2_2 = cmp_gt(xs, 2.2);
    auto ge_7 = cmp_gt(xs, 7.0);

    auto ret = segA;
    auto update = [&](AuthShare &cur, const AuthShare &cond, const AuthShare &next) {
        auto delta = ADD_CALL(next, neg_span(cur));
        auto delta_sel = SELECT_CALL(cond, delta);
        cur = ADD_CALL(cur, delta_sel);
    };

    update(ret, lt_neg7, segL);
    update(ret, ge_neg2_2, segB);
    update(ret, ge_0, segC);
    update(ret, ge_2_2, segD);
    update(ret, ge_7, segR);

    // clamp to [0, 1]
    auto lt0 = cmp_lt(ret, 0.0);
    auto gt1 = cmp_gt(ret, 1.0);
    update(ret, lt0, segL);
    update(ret, gt1, segR);

    return LRS_CALL(MUL_CALL(x, ret), f);
}

// SiLU (piecewise polynomial approximation).
// Matches the requested segmented form:
// x < -6 -> 0
// -6 <= x < -2 -> degree-2 polynomial
// -2 <= x <= 6 -> sparse degree-6 polynomial (x, x^2, x^4, x^6 terms)
// x > 6 -> identity
static AuthShare silu_apply(const AuthShare &x) {
    auto add_const = [&](const AuthShare &src, double v) {
        auto c = make_public_const(src.share.size(), v, f);
        return ADD_CALL(src, c);
    };

    auto cmp_ge = [&](const AuthShare &src, double v) {
        auto diff = add_const(src, -v);
        return CMP_GE_ZERO_CALL(diff);
    };
    auto cmp_gt = [&](const AuthShare &src, double v) {
        const double eps = 1.0 / (double)(1ull << f);
        auto diff = add_const(src, -(v + eps));
        return CMP_GE_ZERO_CALL(diff);
    };

    auto x2 = mul_qf(x, x);
    auto x4 = mul_qf(x2, x2);
    auto x6 = mul_qf(x2, x4);

    const auto a0 = make_public_const(x.share.size(), -0.52212664, f);
    auto segA = a0;
    segA = ADD_CALL(segA, scale_public(x, -0.16910363));
    segA = ADD_CALL(segA, scale_public(x2, -0.01420163));

    const auto b0 = make_public_const(x.share.size(), 0.03453821, f);
    auto segB = b0;
    segB = ADD_CALL(segB, scale_public(x, 0.49379432));
    segB = ADD_CALL(segB, scale_public(x2, 0.19784596));
    segB = ADD_CALL(segB, scale_public(x4, -0.00602401));
    segB = ADD_CALL(segB, scale_public(x6, 0.00008032));

    auto ge_neg6 = cmp_ge(x, -6.0);
    auto ge_neg2 = cmp_ge(x, -2.0);
    auto gt_6 = cmp_gt(x, 6.0);

    auto ret = make_public_const(x.share.size(), 0.0, f);
    auto update = [&](AuthShare &cur, const AuthShare &cond, const AuthShare &next) {
        auto delta = ADD_CALL(next, neg_span(cur));
        auto delta_sel = SELECT_CALL(cond, delta);
        cur = ADD_CALL(cur, delta_sel);
    };

    update(ret, ge_neg6, segA);
    update(ret, ge_neg2, segB);
    update(ret, gt_6, x);

    return ret;
}

// Fixed-point multiply with rescale back to Q(f).
static AuthShare mul_qf(const AuthShare &a, const AuthShare &b) {
    auto prod = MUL_CALL(a, b);
    return LRS_CALL(prod, f);
}

static AuthShare scale_public(const AuthShare &x, double v) {
    int64_t q = (int64_t)std::llround(v * (double)(1ULL << f));
    return LRS_CALL(auth_mul_const(x, (u64)q), f);
}

static AuthShare clamp_min_public(const AuthShare &x, double min_value, int fp_bits);

static AuthShare select_shared_value(const AuthShare &cond,
                                     const AuthShare &false_value,
                                     const AuthShare &true_value) {
    auto delta = ADD_CALL(true_value, neg_span(false_value));
    return ADD_CALL(false_value, SELECT_CALL(cond, delta));
}

static AuthShare abs_from_ge_zero(const AuthShare &x, const AuthShare &ge_zero) {
    auto neg_x = neg_span(x);
    auto abs_delta = ADD_CALL(x, neg_span(neg_x));
    return ADD_CALL(neg_x, SELECT_CALL(ge_zero, abs_delta));
}

struct GoldschmidtNormalization {
    AuthShare factor;
    AuthShare c;
};

// Secure replacement for libspu's highestOneBit/bitrev based normalization.
// It chooses a power-of-two scaling factor so that:
//   c = x_abs * factor
// lands in roughly [0.5, 1) without opening x_abs.
static GoldschmidtNormalization goldschmidt_normalize_interval_0_5_1(const AuthShare &x_abs,
                                                                     int fp_bits) {
    const u64 n = x_abs.share.size();
    auto factor = make_public_const(n, std::ldexp(1.0, fp_bits - 1), fp_bits);
    auto c = mul_qf(x_abs, factor);

    const int max_j = 48;
    for (int j = 1; j <= max_j; ++j) {
        auto threshold = make_public_raw(n, (u64)(1ULL << j));
        auto diff = ADD_CALL(x_abs, neg_span(threshold));
        auto ge_j = CMP_GE_ZERO_CALL(diff);

        auto factor_j = make_public_const(n, std::ldexp(1.0, fp_bits - j - 1), fp_bits);
        auto c_j = mul_qf(x_abs, factor_j);
        c = select_shared_value(ge_j, c, c_j);
        factor = select_shared_value(ge_j, factor, factor_j);
    }

    return GoldschmidtNormalization{std::move(factor), std::move(c)};
}

// Positive reciprocal for softmax row sums, aligned with 000000ddpm1:
// normalize into roughly [0.5, 1), run a 3-step Goldschmidt refinement,
// then multiply back the power-of-two factor.
static AuthShare reciprocal_goldschmidt(const AuthShare &x, int fp_bits, int num_iters = 2) {
    (void)fp_bits;
    (void)num_iters;

    auto add_const = [&](const AuthShare &src, double v) {
        auto c = make_public_const(src.share.size(), v, f);
        return ADD_CALL(src, c);
    };
    auto cmp_ge = [&](const AuthShare &src, double v) {
        auto diff = add_const(src, -v);
        return CMP_GE_ZERO_CALL(diff);
    };
    auto cmp_le = [&](const AuthShare &src, double v) {
        auto negv = neg_span(src);
        auto diff = add_const(negv, v);
        return CMP_GE_ZERO_CALL(diff);
    };
    auto update = [&](AuthShare &cur, const AuthShare &cond, const AuthShare &next) {
        cur = select_shared_value(cond, cur, next);
    };

    const u64 n = x.share.size();
    const double eps = 1e-6;
    const int norm_iters = 8;

    auto one = make_public_const(n, 1.0, f);
    auto init_guess_bias = make_public_const(n, 2.9142, f);
    auto eps_const = make_public_const(n, eps, f);

    AuthShare x_safe = auth_clone(eps_const);
    update(x_safe, cmp_ge(x, eps), x);

    AuthShare factor = auth_clone(one);
    AuthShare c = auth_clone(x_safe);
    for (int iter = 0; iter < norm_iters; ++iter) {
        auto lt_half = cmp_le(c, 0.5 - eps);
        update(c, lt_half, scale_public(c, 2.0));
        update(factor, lt_half, scale_public(factor, 2.0));

        auto ge_one = cmp_ge(c, 1.0);
        update(c, ge_one, scale_public(c, 0.5));
        update(factor, ge_one, scale_public(factor, 0.5));
    }

    auto r = ADD_CALL(init_guess_bias, neg_span(scale_public(c, 2.0)));
    auto e = ADD_CALL(one, neg_span(mul_qf(c, r)));
    auto r1 = mul_qf(r, ADD_CALL(one, e));
    auto e1 = mul_qf(e, e);
    auto r2 = mul_qf(r1, ADD_CALL(one, e1));
    auto e2 = mul_qf(e1, e1);
    auto r3 = mul_qf(r2, ADD_CALL(one, e2));
    return mul_qf(r3, factor);
}

// Newton-Raphson rsqrt in Q(f). Assumes x is positive and roughly normalized.
static AuthShare ss_rsqrt(const AuthShare &x) {
    auto y = make_public_const(x.share.size(), 1.0, f);
    auto half = make_public_const(x.share.size(), 0.5, f);
    auto three_halves = make_public_const(x.share.size(), 1.5, f);
    for (int iter = 0; iter < 2; ++iter) {
        auto y2 = mul_qf(y, y);
        auto xy2 = mul_qf(x, y2);
        auto term = ADD_CALL(three_halves, neg_span(mul_qf(half, xy2)));
        y = mul_qf(y, term);
    }
    return y;
}

static AuthShare inv_goldschmidt2(const AuthShare &t) {
    return reciprocal_goldschmidt(t, f, 3);
}

// Chebyshev exp approximation on x in [-14, 0] using recurrence.
static AuthShare chexp(const AuthShare &x) {
    u64 n = x.share.size();

    // transform_x: map [-14, 0] -> [-1, 1] via x/7 + 1
    const auto inv7 = make_public_const(n, 1.0 / 7.0, f);
    const auto one = make_public_const(n, 1.0, f);

    auto xt = LRS_CALL(MUL_CALL(x, inv7), f);
    xt = ADD_CALL(xt, one);

    const auto c0 = make_public_const(n, 0.14021878, f);
    const auto c1 = make_public_const(n, 0.27541278, f);
    const auto c2 = make_public_const(n, 0.22122865, f);
    const auto c3 = make_public_const(n, 0.14934221, f);
    const auto c4 = make_public_const(n, 0.09077360, f);
    const auto c5 = make_public_const(n, 0.04369614, f);
    const auto c6 = make_public_const(n, 0.02087868, f);
    const auto c7 = make_public_const(n, 0.00996535, f);

    // Chebyshev recurrence: T0=1, T1=xt, Tn = 2*xt*T_{n-1} - T_{n-2}
    auto t0 = one;
    auto t1 = xt;
    auto ex = ADD_CALL(c0, LRS_CALL(MUL_CALL(t1, c1), f));

    auto two_xt = ADD_CALL(xt, xt);
    auto t_nm2 = t0;
    auto t_nm1 = t1;

    auto next_t = LRS_CALL(MUL_CALL(two_xt, t_nm1), f);
    next_t = ADD_CALL(next_t, neg_span(t_nm2));
    ex = ADD_CALL(ex, LRS_CALL(MUL_CALL(next_t, c2), f));
    t_nm2 = t_nm1;
    t_nm1 = next_t;

    next_t = LRS_CALL(MUL_CALL(two_xt, t_nm1), f);
    next_t = ADD_CALL(next_t, neg_span(t_nm2));
    ex = ADD_CALL(ex, LRS_CALL(MUL_CALL(next_t, c3), f));
    t_nm2 = t_nm1;
    t_nm1 = next_t;

    next_t = LRS_CALL(MUL_CALL(two_xt, t_nm1), f);
    next_t = ADD_CALL(next_t, neg_span(t_nm2));
    ex = ADD_CALL(ex, LRS_CALL(MUL_CALL(next_t, c4), f));
    t_nm2 = t_nm1;
    t_nm1 = next_t;

    next_t = LRS_CALL(MUL_CALL(two_xt, t_nm1), f);
    next_t = ADD_CALL(next_t, neg_span(t_nm2));
    ex = ADD_CALL(ex, LRS_CALL(MUL_CALL(next_t, c5), f));
    t_nm2 = t_nm1;
    t_nm1 = next_t;

    next_t = LRS_CALL(MUL_CALL(two_xt, t_nm1), f);
    next_t = ADD_CALL(next_t, neg_span(t_nm2));
    ex = ADD_CALL(ex, LRS_CALL(MUL_CALL(next_t, c6), f));
    t_nm2 = t_nm1;
    t_nm1 = next_t;

    next_t = LRS_CALL(MUL_CALL(two_xt, t_nm1), f);
    next_t = ADD_CALL(next_t, neg_span(t_nm2));
    ex = ADD_CALL(ex, LRS_CALL(MUL_CALL(next_t, c7), f));
    return ex;
}

// Softmax using Chebyshev exp approximation and division.
static AuthShare rowmax_share(u64 s1, u64 s2, const AuthShare &X) {
    always_assert(X.share.size() == s1 * s2);
    AuthShare res = auth_alloc(s1 * s2);
    #pragma omp parallel for
    for (u64 i = 0; i < s1 * s2; ++i) {
        res.share[i] = X.share[i];
        res.tag[i] = X.tag[i];
    }

    u64 curr = s2;
    while (curr != 1) {
        u64 curr2 = curr / 2;
        AuthShare left = auth_alloc(s1 * curr2);
        AuthShare right = auth_alloc(s1 * curr2);
        #pragma omp parallel for collapse(2)
        for (u64 i = 0; i < s1; ++i) {
            for (u64 j = 0; j < curr2; ++j) {
                u64 idx_left = i * curr + 2 * j;
                u64 idx_right = i * curr + 2 * j + 1;
                u64 out_idx = i * curr2 + j;
                left.share[out_idx] = res.share[idx_left];
                left.tag[out_idx] = res.tag[idx_left];
                right.share[out_idx] = res.share[idx_right];
                right.tag[out_idx] = res.tag[idx_right];
            }
        }

        // max(left, right) = right + ge(left-right) * (left-right)
        auto diff = ADD_CALL(left, neg_span(right));
        auto ge = CMP_GE_ZERO_CALL(diff);
        auto sel = SELECT_CALL(ge, diff);
        auto max_lr = ADD_CALL(right, sel);

        u64 currNext = (curr % 2 == 0) ? (curr / 2) : (curr / 2 + 1);
        AuthShare resNext = auth_alloc(s1 * currNext);

        if (curr % 2 == 1) {
            #pragma omp parallel for
            for (u64 i = 0; i < s1; ++i) {
                u64 dst = i * currNext + currNext - 1;
                u64 src = i * curr + curr - 1;
                resNext.share[dst] = res.share[src];
                resNext.tag[dst] = res.tag[src];
            }
        }

        #pragma omp parallel for collapse(2)
        for (u64 i = 0; i < s1; ++i) {
            for (u64 j = 0; j < curr2; ++j) {
                u64 dst = i * currNext + j;
                u64 src = i * curr2 + j;
                resNext.share[dst] = max_lr.share[src];
                resNext.tag[dst] = max_lr.tag[src];
            }
        }

        res = std::move(resNext);
        curr = currNext;
    }

    AuthShare out = auth_alloc(s1);
    #pragma omp parallel for
    for (u64 i = 0; i < s1; ++i) {
        out.share[i] = res.share[i];
        out.tag[i] = res.tag[i];
    }
    return out;
}

static AuthShare clip_softmax(u64 a, u64 b, const AuthShare &x) {
    always_assert(x.share.size() == (a * b));

    // Step 1: Compute row-wise max and shift (secure, no-open).
    auto row_max = rowmax_share(a, b, x);
    AuthShare shifted = auth_alloc(a * b);
    #pragma omp parallel for collapse(2)
    for (u64 i = 0; i < a; ++i) {
        for (u64 j = 0; j < b; ++j) {
            u64 idx = i * b + j;
            shifted.share[idx] = x.share[idx] - row_max.share[i];
            shifted.tag[idx] = x.tag[idx] - row_max.tag[i];
        }
    }
    clip_batch_check("softmax after shift");

    // Step 2: exp(x) approximation via Chebyshev recurrence on [-14, 0].
    AuthShare exp_vals;
    {
        OpProfileScope exp_profile(OP_SOFTMAX_EXP);
        exp_vals = chexp(shifted);
    }

    // Step 3: clip tail as requested:
    // x <= -14 -> exp(x)=0
    auto add_const = [&](const AuthShare &src, double v) {
        auto c = make_public_const(src.share.size(), v, f);
        return ADD_CALL(src, c);
    };
    auto cmp_le = [&](const AuthShare &src, double v) {
        auto negv = neg_span(src);
        auto diff = add_const(negv, v);
        return CMP_GE_ZERO_CALL(diff);
    };

    auto le_neg14 = cmp_le(shifted, -14.0);
    auto zero = make_public_const(a * b, 0.0, f);
    auto exp_to_zero = ADD_CALL(zero, neg_span(exp_vals));
    auto exp_to_zero_sel = SELECT_CALL(le_neg14, exp_to_zero);
    exp_vals = ADD_CALL(exp_vals, exp_to_zero_sel);
    clip_batch_check("softmax after exp_deg5");

    // Step 4: Compute Row Sums
    AuthShare row_sums = auth_alloc(a);
    #pragma omp parallel for
    for (u64 i = 0; i < a; ++i) {
        u128 sum_share = 0;
        u128 sum_tag = 0;
        for (u64 j = 0; j < b; ++j) {
            u64 idx = i * b + j;
            sum_share += exp_vals.share[idx];
            sum_tag += exp_vals.tag[idx];
        }
        row_sums.share[i] = sum_share;
        row_sums.tag[i] = sum_tag;
    }

    // Step 5: Inverse sum with the same positive Goldschmidt reciprocal used in ddpm1.
    auto inv_sums = reciprocal_goldschmidt(row_sums, f, 3);
    clip_batch_check("softmax after inv_goldschmidt");

    // Step 6: Broadcast and Multiply
    AuthShare inv_sums_b = auth_alloc(a * b);
    #pragma omp parallel for collapse(2)
    for (u64 i = 0; i < a; ++i) {
        for (u64 j = 0; j < b; ++j) {
            u64 idx = i * b + j;
            inv_sums_b.share[idx] = inv_sums.share[i];
            inv_sums_b.tag[idx] = inv_sums.tag[i];
        }
    }

    auto result = MUL_CALL(exp_vals, inv_sums_b);
    result = LRS_CALL(result, f); // Normalize after multiplication
    clip_batch_check("softmax after mul");
    return result;
}

static AuthShare timed_clip_softmax(const char *name, u64 a, u64 b, const AuthShare &x) {
    ScopedCommTrace trace(name);
    OpProfileScope op_profile(OP_SOFTMAX);
    shark::utils::start_timer("softmax");
    shark::utils::start_timer(name);
    auto out = clip_softmax(a, b, x);
    shark::utils::stop_timer(name);
    shark::utils::stop_timer("softmax");
    return out;
}

// -----------------------------
// Linear and conv helpers
// -----------------------------
struct AuthenticatedWeightCacheEntry {
    const u64 *ptr = nullptr;
    u64 size = 0;
    AuthShare auth;
};

struct DerivedAuthenticatedWeightCacheEntry {
    const u64 *ptr = nullptr;
    u64 size = 0;
    u64 inC = 0;
    u64 outC = 0;
    AuthShare auth;
};

static std::vector<AuthenticatedWeightCacheEntry> &authenticated_weight_cache() {
    static std::vector<AuthenticatedWeightCacheEntry> cache;
    return cache;
}

static std::vector<DerivedAuthenticatedWeightCacheEntry> &derived_authenticated_weight_cache() {
    static std::vector<DerivedAuthenticatedWeightCacheEntry> cache;
    return cache;
}

static void clear_authenticated_weight_cache() {
    authenticated_weight_cache().clear();
    derived_authenticated_weight_cache().clear();
}

static AuthenticatedWeightCacheEntry *find_authenticated_weight_cache_entry(const u64 *ptr, u64 size) {
    auto &cache = authenticated_weight_cache();
    for (auto &entry : cache) {
        if (entry.ptr == ptr && entry.size == size) return &entry;
    }
    return nullptr;
}

static const AuthShare *find_authenticated_weight_cache(const span<u64> &w) {
    auto *entry = find_authenticated_weight_cache_entry(w.data(), w.size());
    return entry ? &entry->auth : nullptr;
}

static DerivedAuthenticatedWeightCacheEntry *find_derived_authenticated_weight_cache_entry(
    const u64 *ptr, u64 size, u64 inC, u64 outC
) {
    auto &cache = derived_authenticated_weight_cache();
    for (auto &entry : cache) {
        if (entry.ptr == ptr && entry.size == size && entry.inC == inC && entry.outC == outC) {
            return &entry;
        }
    }
    return nullptr;
}

static const AuthShare *find_derived_authenticated_weight_cache(const span<u64> &w, u64 inC, u64 outC) {
    auto *entry = find_derived_authenticated_weight_cache_entry(w.data(), w.size(), inC, outC);
    return entry ? &entry->auth : nullptr;
}

static void cache_authenticated_weight(const u64 *ptr, u64 size, AuthShare &&auth) {
    if (find_authenticated_weight_cache_entry(ptr, size) != nullptr) return;
    authenticated_weight_cache().push_back(
        AuthenticatedWeightCacheEntry{ptr, size, std::move(auth)});
}

static void cache_derived_authenticated_weight(const u64 *ptr, u64 size, u64 inC, u64 outC, AuthShare &&auth) {
    if (find_derived_authenticated_weight_cache_entry(ptr, size, inC, outC) != nullptr) return;
    derived_authenticated_weight_cache().push_back(
        DerivedAuthenticatedWeightCacheEntry{ptr, size, inC, outC, std::move(auth)});
}

static AuthShare share_server_owned_model_tensor(const span<u64> &x) {
    if (x.size() == 0) return auth_alloc(0);
    return authenticated_input_from_owner(x, SERVER);
}

static void preauthenticate_weight(const span<u64> &w) {
    if (w.size() == 0) return;
    if (find_authenticated_weight_cache(w) != nullptr) return;
    cache_authenticated_weight(w.data(), w.size(), share_server_owned_model_tensor(w));
}

static span<u64> reshape_conv3x3_weight_to_linear_plain(const span<u64> &w, u64 inC, u64 outC) {
    const u64 k = 3;
    always_assert(w.size() == outC * inC * k * k);
    span<u64> linear(inC * k * k * outC);
    zero_plain(linear);
    if (party != SERVER) return linear;
    for (u64 oc = 0; oc < outC; ++oc) {
        for (u64 ic = 0; ic < inC; ++ic) {
            for (u64 kh = 0; kh < k; ++kh) {
                for (u64 kw = 0; kw < k; ++kw) {
                    const u64 src = (((oc * inC + ic) * k + kh) * k + kw);
                    const u64 col = ((ic * k + kh) * k + kw);
                    const u64 dst = col * outC + oc;
                    linear[dst] = w[src];
                }
            }
        }
    }
    return linear;
}

static void preauthenticate_conv3x3_linear_weight(const span<u64> &w, u64 inC, u64 outC) {
    if (w.size() == 0) return;
    if (find_derived_authenticated_weight_cache(w, inC, outC) != nullptr) return;
    auto linear_plain = reshape_conv3x3_weight_to_linear_plain(w, inC, outC);
    cache_derived_authenticated_weight(
        w.data(), w.size(), inC, outC, share_server_owned_model_tensor(linear_plain));
}

static const AuthShare &preauthenticated_conv3x3_linear_weight(const span<u64> &w, u64 inC, u64 outC) {
    static AuthShare empty_auth{};
    if (w.size() == 0) return empty_auth;
    if (const AuthShare *cached = find_derived_authenticated_weight_cache(w, inC, outC)) {
        return *cached;
    }
    preauthenticate_conv3x3_linear_weight(w, inC, outC);
    if (const AuthShare *cached = find_derived_authenticated_weight_cache(w, inC, outC)) {
        return *cached;
    }
    always_assert(false);
    return empty_auth;
}

static const AuthShare &preauthenticated_weight(const span<u64> &w) {
    static AuthShare empty_auth{};
    if (w.size() == 0) return empty_auth;
    if (const AuthShare *cached = find_authenticated_weight_cache(w)) {
        return *cached;
    }
    preauthenticate_weight(w);
    if (const AuthShare *cached = find_authenticated_weight_cache(w)) {
        return *cached;
    }
    always_assert(false);
    return empty_auth;
}

static AuthShare linear_mat(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w, const AuthShare *b = nullptr) {
    ScopedCommTrace trace("bench:linear_mat.auth");
    OpProfileScope op_profile(OP_LINEAR);
    shark::utils::start_timer("linear");
    auto y = auth_matmul_secret(M, K, N, x, w);
    clip_batch_check("linear:after matmul");
    y = LRS_CALL(y, f);
    clip_batch_check("linear:after shift");
    if (b && b->share.size() == N) {
        AuthShare out = auth_alloc(y.share.size());
        for (u64 i = 0; i < M; ++i) {
            for (u64 j = 0; j < N; ++j) {
                u64 idx = i * N + j;
                out.share[idx] = y.share[idx] + b->share[j];
                out.tag[idx] = y.tag[idx] + b->tag[j];
            }
        }
        shark::utils::stop_timer("linear");
        return out;
    }
    shark::utils::stop_timer("linear");
    return y;
}

static AuthShare linear_mat(u64 M, u64 K, u64 N, const AuthShare &x, const span<u64> &w, const AuthShare *b = nullptr) {
    ScopedCommTrace trace("bench:linear_mat.plain");
    return linear_mat(M, K, N, x, preauthenticated_weight(w), b);
}

static AuthShare linear_matmul_apply(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w) {
    ScopedCommTrace trace("bench:linear_matmul.auth");
    OpProfileScope op_profile(OP_LINEAR);
    shark::utils::start_timer("linear");
    auto y = auth_matmul_secret(M, K, N, x, w);
    y = LRS_CALL(y, f);
    shark::utils::stop_timer("linear");
    return y;
}

static AuthShare linear_matmul_apply(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w, double post_scale) {
    ScopedCommTrace trace("bench:linear_matmul_scaled.auth");
    OpProfileScope op_profile(OP_LINEAR);
    shark::utils::start_timer("linear");
    auto y = auth_matmul_secret(M, K, N, x, w);
    y = LRS_CALL(y, f);
    y = scale_public(y, post_scale);
    shark::utils::stop_timer("linear");
    return y;
}

static AuthShare conv2d_apply(
    u64 B, u64 H, u64 W, u64 inC, u64 outC, u64 k, u64 stride, u64 padding,
    const AuthShare &x, const AuthShare &w, const AuthShare *b = nullptr
) {
    ScopedCommTrace trace("bench:conv2d_apply.auth");
    OpProfileScope op_profile(OP_CONV);
    shark::utils::start_timer("conv");
    const u64 outH = (H - k + 2 * padding) / stride + 1;
    const u64 outW = (W - k + 2 * padding) / stride + 1;
    auto y = auth_conv_secret(k, padding, stride, inC, outC, H, W, x, w);
    clip_batch_check("debug:conv2d:after conv");
    y = LRS_CALL(y, f);
    clip_batch_check("debug:conv2d:after shift");
    if (b && b->share.size() == outC) {
        AuthShare out = auth_alloc(y.share.size());
        for (u64 n = 0; n < B; ++n) {
            for (u64 h = 0; h < outH; ++h) {
                for (u64 widx = 0; widx < outW; ++widx) {
                    for (u64 c = 0; c < outC; ++c) {
                        u64 idx = ((n * outH + h) * outW + widx) * outC + c;
                        out.share[idx] = y.share[idx] + b->share[c];
                        out.tag[idx] = y.tag[idx] + b->tag[c];
                    }
                }
            }
        }
        shark::utils::stop_timer("conv");
        return out;
    }
    shark::utils::stop_timer("conv");
    return y;
}

static AuthShare conv2d_apply(
    u64 B, u64 H, u64 W, u64 inC, u64 outC, u64 k, u64 stride, u64 padding,
    const AuthShare &x, const span<u64> &w, const AuthShare *b = nullptr
) {
    ScopedCommTrace trace("bench:conv2d_apply.plain");
    return conv2d_apply(B, H, W, inC, outC, k, stride, padding, x, preauthenticated_weight(w), b);
}

// 3x3 conv with padding=1 and crop back to (H, W) to preserve shapes.
static AuthShare conv2d_apply_k3_same(
    u64 B, u64 H, u64 W, u64 inC, u64 outC, u64 stride,
    const AuthShare &x, const span<u64> &w, const AuthShare *b = nullptr
) {
    const u64 k = 3;
    const u64 padding = 1;
    auto y = conv2d_apply(B, H, W, inC, outC, k, stride, padding, x, w, b);
    u64 outH = (H - k + 2 * padding) / stride + 1;
    u64 outW = (W - k + 2 * padding) / stride + 1;
    if (outH == H && outW == W) return y;

    AuthShare out = auth_alloc(B * H * W * outC);
    for (u64 i = 0; i < out.share.size(); ++i) {
        out.share[i] = 0;
        out.tag[i] = 0;
    }
    u64 copyH = (outH < H) ? outH : H;
    u64 copyW = (outW < W) ? outW : W;
    for (u64 n = 0; n < B; ++n) {
        for (u64 h = 0; h < copyH; ++h) {
            for (u64 widx = 0; widx < copyW; ++widx) {
                for (u64 c = 0; c < outC; ++c) {
                    u64 src = ((n * outH + h) * outW + widx) * outC + c;
                    u64 dst = ((n * H + h) * W + widx) * outC + c;
                    out.share[dst] = y.share[src];
                    out.tag[dst] = y.tag[src];
                }
            }
        }
    }
    return out;
}

static AuthShare conv2d_apply_k3_same(
    u64 B, u64 H, u64 W, u64 inC, u64 outC, u64 stride,
    const AuthShare &x, const AuthShare &w, const AuthShare *b = nullptr
) {
    const u64 k = 3;
    const u64 padding = 1;
    auto y = conv2d_apply(B, H, W, inC, outC, k, stride, padding, x, w, b);
    u64 outH = (H - k + 2 * padding) / stride + 1;
    u64 outW = (W - k + 2 * padding) / stride + 1;
    if (outH == H && outW == W) return y;

    AuthShare out = auth_alloc(B * H * W * outC);
    for (u64 i = 0; i < out.share.size(); ++i) {
        out.share[i] = 0;
        out.tag[i] = 0;
    }
    u64 copyH = (outH < H) ? outH : H;
    u64 copyW = (outW < W) ? outW : W;
    for (u64 n = 0; n < B; ++n) {
        for (u64 h = 0; h < copyH; ++h) {
            for (u64 widx = 0; widx < copyW; ++widx) {
                for (u64 c = 0; c < outC; ++c) {
                    u64 src = ((n * outH + h) * outW + widx) * outC + c;
                    u64 dst = ((n * H + h) * W + widx) * outC + c;
                    out.share[dst] = y.share[src];
                    out.tag[dst] = y.tag[src];
                }
            }
        }
    }
    return out;
}

static AuthShare transpose_mat(const AuthShare &x, u64 rows, u64 cols) {
    AuthShare out = auth_alloc(rows * cols);
    for (u64 i = 0; i < rows; ++i) {
        for (u64 j = 0; j < cols; ++j) {
            u64 src = i * cols + j;
            u64 dst = j * rows + i;
            out.share[dst] = x.share[src];
            out.tag[dst] = x.tag[src];
        }
    }
    return out;
}

// -----------------------------
// Layernorm / Groupnorm (approximate)
// -----------------------------
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

static bool norm_use_ssrsqrt_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = (std::getenv("UNCLIP_NORM_USE_SSRSQRT") != nullptr) ? 1 : 0;
    }
    return enabled == 1;
}

static bool norm_force_recip_div_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char *force = std::getenv("UNCLIP_NORM_FORCE_RECIP_DIV");
        const char *allow_pow2 = std::getenv("UNCLIP_NORM_ALLOW_POW2_DIV");
        if (force != nullptr) {
            enabled = 1;
        } else if (allow_pow2 != nullptr) {
            enabled = 0;
        } else {
            // Prefer exact arithmetic right-shift when the denominator is a
            // public power of two. This avoids the LRS path for mean/variance
            // division in tiny-width debug configurations.
            enabled = 0;
        }
    }
    return enabled == 1;
}

static bool norm_use_exact_pow2_div(u64 denom) {
    return is_power_of_two_u64(denom) && !norm_force_recip_div_enabled();
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

static AuthShare broadcast_lastdim_param(u64 rows, u64 cols, const AuthShare &param) {
    always_assert(param.share.size() == cols);
    AuthShare out = auth_alloc(rows * cols);
    for (u64 i = 0; i < rows; ++i) {
        for (u64 j = 0; j < cols; ++j) {
            u64 idx = i * cols + j;
            out.share[idx] = param.share[j];
            out.tag[idx] = param.tag[j];
        }
    }
    return out;
}

static AuthShare apply_affine_lastdim(u64 rows, u64 cols, const AuthShare &x,
                                      const AuthShare *weight, const AuthShare *bias) {
    const bool has_weight = (weight != nullptr && weight->share.size() != 0);
    const bool has_bias = (bias != nullptr && bias->share.size() != 0);
    if (!has_weight && !has_bias) {
        return auth_clone(x);
    }

    AuthShare out = auth_clone(x);
    if (has_weight) {
        auto weight_b = broadcast_lastdim_param(rows, cols, *weight);
        out = mul_qf(out, weight_b);
    }
    if (has_bias) {
        auto bias_b = broadcast_lastdim_param(rows, cols, *bias);
        out = ADD_CALL(out, bias_b);
    }
    return out;
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

struct MantissaExponentRsqrtDecomposition {
    // u = 2 * mantissa in Q(fp_bits), so real(u) lies in [1, 2).
    AuthShare u;
    // scale = 2^{-0.5 * (exponent - 1)} in Q(fp_bits).
    AuthShare scale;
};

static MantissaExponentRsqrtDecomposition decompose_var_mantissa_exponent_batched(
    const AuthShare &var_clamped, int fp_bits
) {
    const u64 n = var_clamped.share.size();

    // raw_var = var_real * 2^fp_bits.
    // If var_real = mantissa * 2^exponent with mantissa in [0.5, 1), then:
    //   raw_var in [2^j, 2^(j+1)), where j = exponent + fp_bits - 1
    //   u = 2 * mantissa = raw_var / 2^j in [1, 2)
    //   1/sqrt(var) = 2^{-0.5 * (exponent - 1)} * 1/sqrt(u)
    //
    // We recover j with one batched compare pass over power-of-two thresholds,
    // then select the corresponding normalized u and scale without opening var.
    const int max_j = 48;
    AuthShare diff_all = auth_alloc(n * max_j);
    for (int j = 1; j <= max_j; ++j) {
        auto threshold = make_public_raw(n, (u64(1ULL << j)));
        auto diff = ADD_CALL(var_clamped, neg_span(threshold));
        auth_copy_into(diff_all, (u64)(j - 1) * n, diff);
    }
    auto ge_all = CMP_GE_ZERO_CALL(diff_all);

    // Default bucket j = 0 corresponds to raw_var in [1, 2).
    AuthShare u = auth_mul_const(var_clamped, (u64)(1ULL << fp_bits));
    auto scale = make_public_const(n, rsqrt_scale_from_shift(fp_bits), fp_bits);

    for (int j = 1; j <= max_j; ++j) {
        auto ge_j = auth_view(ge_all, (u64)(j - 1) * n, n);

        const int shift_j = fp_bits - j;
        AuthShare u_j;
        if (shift_j >= 0) {
            u_j = auth_mul_const(var_clamped, (u64)(1ULL << shift_j));
        } else {
            u_j = LRS_CALL(var_clamped, (u64)(-shift_j));
        }
        auto scale_j = make_public_const(n, rsqrt_scale_from_shift(shift_j), fp_bits);

        u = select_shared_value(ge_j, u, u_j);
        scale = select_shared_value(ge_j, scale, scale_j);
    }

    return MantissaExponentRsqrtDecomposition{std::move(u), std::move(scale)};
}

static AuthShare approx_inv_std_from_var(const AuthShare &var, int fp_bits,
                                         const SecureNormApproxConfig &cfg,
                                         double max_inv_std) {
    const u64 n = var.share.size();
    const double min_var = std::max(kNormVarFloor, std::ldexp(1.0, -fp_bits));

    auto var_clamped = clamp_min_public(var, min_var, fp_bits);
    if (norm_use_ssrsqrt_enabled()) {
        return clamp_max_public(ss_rsqrt(var_clamped), max_inv_std, fp_bits);
    }
    // Exact rsqrt fallback is intentionally omitted here: the secure benchmark path
    // only exposes comparison/select-based arithmetic, so we keep the API shape but
    // stay on the approximation branch for every party.
    (void)cfg.approx_mode;
    (void)cfg.use_exact_fallback;
    (void)cfg.min_denominator;

    auto decomp = decompose_var_mantissa_exponent_batched(var_clamped, fp_bits);
    auto r = rsqrt_seed_interval_1_2(decomp.u, fp_bits, cfg.poly_order);
    r = rsqrt_newton_refine(decomp.u, r, fp_bits, cfg.nr_iters);
    auto inv_std = mul_qf(decomp.scale, r);
    return clamp_max_public(inv_std, max_inv_std, fp_bits);
}

static AuthShare layernorm_rows(u64 rows, u64 cols, const AuthShare &X,
                                const AuthShare *weight, const AuthShare *bias,
                                const char *debug_name) {
    const auto cfg = make_secure_norm_approx_config();
    const double max_inv_std = 1.0 / std::sqrt(cfg.eps);
    std::string debug_storage;
    const auto debug_label = [&](const char *suffix) -> const char * {
        if (debug_name == nullptr || debug_name[0] == '\0') return suffix;
        debug_storage = std::string(debug_name) + ":" + suffix;
        return debug_storage.c_str();
    };
    clip_batch_check(debug_label("enter"));
    always_assert(cols > 1);
    always_assert(X.share.size() == rows * cols);
    always_assert(X.tag.size() == rows * cols);
    AuthShare row_sum = auth_alloc(rows);
    for (u64 i = 0; i < rows; ++i) {
        u128 acc = 0;
        u128 acc_tag = 0;
        for (u64 j = 0; j < cols; ++j) {
            u64 idx = i * cols + j;
            acc += X.share[idx];
            acc_tag += X.tag[idx];
        }
        row_sum.share[i] = acc;
        row_sum.tag[i] = acc_tag;
    }

    const double inv_n = 1.0 / (double)cols;
    const bool exact_pow2_div = norm_use_exact_pow2_div(cols);
    const u64 div_shift = exact_pow2_div ? log2_exact_u64(cols) : 0;
    clip_batch_check(debug_label("after reciprocal"));
    AuthShare mean = auth_alloc(rows);
    if (exact_pow2_div) {
        clip_batch_check(debug_label("after mean_mul"));
        mean = ss_ars(row_sum, div_shift);
    } else {
        auto prod_mean = auth_mul_const(row_sum, (u64)(int64_t)std::llround(inv_n * (double)(1ULL << f)));
        clip_batch_check(debug_label("after mean_mul"));
        mean = LRS_CALL(prod_mean, f);
    }
    clip_batch_check(debug_label("after mean_shift"));

    AuthShare mean_b = auth_alloc(rows * cols);
    for (u64 i = 0; i < rows; ++i) {
        for (u64 j = 0; j < cols; ++j) {
            u64 idx = i * cols + j;
            mean_b.share[idx] = mean.share[i];
            mean_b.tag[idx] = mean.tag[i];
        }
    }

    AuthShare centered = auth_alloc(rows * cols);
    for (u64 k = 0; k < rows * cols; ++k) {
        centered.share[k] = X.share[k] - mean_b.share[k];
        centered.tag[k] = X.tag[k] - mean_b.tag[k];
    }

    auto sqr = MUL_CALL(centered, centered);
    auto sqr_scaled = LRS_CALL(sqr, f);
    AuthShare row_sum_sqr = auth_alloc(rows);
    for (u64 i = 0; i < rows; ++i) {
        u128 acc = 0;
        u128 acc_tag = 0;
        for (u64 j = 0; j < cols; ++j) {
            u64 idx = i * cols + j;
            acc += sqr_scaled.share[idx];
            acc_tag += sqr_scaled.tag[idx];
        }
        row_sum_sqr.share[i] = acc;
        row_sum_sqr.tag[i] = acc_tag;
    }

    AuthShare var = auth_alloc(rows);
    if (exact_pow2_div) {
        var = ss_ars(row_sum_sqr, div_shift);
    } else {
        auto prod_var = auth_mul_const(row_sum_sqr, (u64)(int64_t)std::llround(inv_n * (double)(1ULL << f)));
        var = LRS_CALL(prod_var, f);
    }
    auto eps = make_public_const(rows, cfg.eps, f);
    auto var_eps = ADD_CALL(var, eps);
    auto inv_std = approx_inv_std_from_var(var_eps, f, cfg, max_inv_std);
    clip_batch_check(debug_label("after rsqrt"));
    AuthShare inv_std_broadcast = auth_alloc(rows * cols);
    for (u64 i = 0; i < rows; ++i) {
        for (u64 j = 0; j < cols; ++j) {
            u64 idx = i * cols + j;
            inv_std_broadcast.share[idx] = inv_std.share[i];
            inv_std_broadcast.tag[idx] = inv_std.tag[i];
        }
    }

    auto output = LRS_CALL(MUL_CALL(centered, inv_std_broadcast), f);
    output = apply_affine_lastdim(rows, cols, output, weight, bias);
    clip_batch_check(debug_label("after norm"));
    return output;
}

static AuthShare groupnorm_apply_groups(u64 B, u64 H, u64 W, u64 C, u64 groups, const AuthShare &x,
                                        const AuthShare *weight, const AuthShare *bias) {
    const auto cfg = make_secure_norm_approx_config();
    const double max_inv_std = 1.0 / std::sqrt(cfg.eps);
    always_assert(C > 0);
    always_assert(groups > 0);
    always_assert(C % groups == 0);
    always_assert(x.share.size() == B * H * W * C);

    const u64 ch_per_group = C / groups;
    const u64 elems_per_group = H * W * ch_per_group;

    AuthShare group_sum = auth_alloc(B * groups);
    for (u64 n = 0; n < B; ++n) {
        for (u64 g = 0; g < groups; ++g) {
            u128 acc_share = 0;
            u128 acc_tag = 0;
            for (u64 h = 0; h < H; ++h) {
                for (u64 widx = 0; widx < W; ++widx) {
                    for (u64 c = 0; c < ch_per_group; ++c) {
                        u64 global_c = g * ch_per_group + c;
                        u64 idx = idx4(n, h, widx, global_c, H, W, C);
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
    for (u64 n = 0; n < B; ++n) {
        for (u64 g = 0; g < groups; ++g) {
            u64 mean_idx = n * groups + g;
            for (u64 h = 0; h < H; ++h) {
                for (u64 widx = 0; widx < W; ++widx) {
                    for (u64 c = 0; c < ch_per_group; ++c) {
                        u64 global_c = g * ch_per_group + c;
                        u64 idx = idx4(n, h, widx, global_c, H, W, C);
                        centered.share[idx] = x.share[idx] - mean.share[mean_idx];
                        centered.tag[idx] = x.tag[idx] - mean.tag[mean_idx];
                    }
                }
            }
        }
    }

    auto sqr_scaled = mul_qf(centered, centered);

    AuthShare var_sum = auth_alloc(B * groups);
    for (u64 n = 0; n < B; ++n) {
        for (u64 g = 0; g < groups; ++g) {
            u128 acc_share = 0;
            u128 acc_tag = 0;
            for (u64 h = 0; h < H; ++h) {
                for (u64 widx = 0; widx < W; ++widx) {
                    for (u64 c = 0; c < ch_per_group; ++c) {
                        u64 global_c = g * ch_per_group + c;
                        u64 idx = idx4(n, h, widx, global_c, H, W, C);
                        acc_share += sqr_scaled.share[idx];
                        acc_tag += sqr_scaled.tag[idx];
                    }
                }
            }
            u64 out_idx = n * groups + g;
            var_sum.share[out_idx] = acc_share;
            var_sum.tag[out_idx] = acc_tag;
        }
    }

    auto var = mul_qf(var_sum, inv_n);
    auto eps = make_public_const(B * groups, cfg.eps, f);
    auto var_eps = ADD_CALL(var, eps);
    auto inv_std = approx_inv_std_from_var(var_eps, f, cfg, max_inv_std);

    AuthShare inv_std_b = auth_alloc(B * H * W * C);
    for (u64 n = 0; n < B; ++n) {
        for (u64 g = 0; g < groups; ++g) {
            u64 stat_idx = n * groups + g;
            for (u64 h = 0; h < H; ++h) {
                for (u64 widx = 0; widx < W; ++widx) {
                    for (u64 c = 0; c < ch_per_group; ++c) {
                        u64 global_c = g * ch_per_group + c;
                        u64 idx = idx4(n, h, widx, global_c, H, W, C);
                        inv_std_b.share[idx] = inv_std.share[stat_idx];
                        inv_std_b.tag[idx] = inv_std.tag[stat_idx];
                    }
                }
            }
        }
    }

    auto output = mul_qf(centered, inv_std_b);
    return apply_affine_lastdim(B * H * W, C, output, weight, bias);
}

static AuthShare groupnorm_apply(u64 B, u64 H, u64 W, u64 C, const AuthShare &x,
                                 const AuthShare *weight, const AuthShare *bias) {
    return groupnorm_apply_groups(B, H, W, C, std::min<u64>(2, C), x, weight, bias);
}

static AuthShare resize_nearest_hw(u64 B, u64 srcH, u64 srcW, u64 dstH, u64 dstW, u64 C, const AuthShare &x) {
    always_assert(x.share.size() == B * srcH * srcW * C);
    AuthShare out = auth_alloc(B * dstH * dstW * C);
    for (u64 n = 0; n < B; ++n) {
        for (u64 h = 0; h < dstH; ++h) {
            u64 src_h = (dstH == 0) ? 0 : std::min<u64>(srcH - 1, (h * srcH) / dstH);
            for (u64 widx = 0; widx < dstW; ++widx) {
                u64 src_w = (dstW == 0) ? 0 : std::min<u64>(srcW - 1, (widx * srcW) / dstW);
                for (u64 c = 0; c < C; ++c) {
                    u64 src = idx4(n, src_h, src_w, c, srcH, srcW, C);
                    u64 dst = idx4(n, h, widx, c, dstH, dstW, C);
                    out.share[dst] = x.share[src];
                    out.tag[dst] = x.tag[src];
                }
            }
        }
    }
    return out;
}

static AuthShare upsample_nearest_2x(u64 B, u64 H, u64 W, u64 C, const AuthShare &x) {
    return resize_nearest_hw(B, H, W, H * 2, W * 2, C, x);
}

static AuthShare concat_channels(u64 B, u64 H, u64 W, u64 C1, const AuthShare &a,
                                 u64 C2, const AuthShare &b) {
    always_assert(a.share.size() == B * H * W * C1);
    always_assert(b.share.size() == B * H * W * C2);
    AuthShare out = auth_alloc(B * H * W * (C1 + C2));
    for (u64 n = 0; n < B; ++n) {
        for (u64 h = 0; h < H; ++h) {
            for (u64 widx = 0; widx < W; ++widx) {
                for (u64 c = 0; c < C1; ++c) {
                    u64 src = idx4(n, h, widx, c, H, W, C1);
                    u64 dst = idx4(n, h, widx, c, H, W, C1 + C2);
                    out.share[dst] = a.share[src];
                    out.tag[dst] = a.tag[src];
                }
                for (u64 c = 0; c < C2; ++c) {
                    u64 src = idx4(n, h, widx, c, H, W, C2);
                    u64 dst = idx4(n, h, widx, C1 + c, H, W, C1 + C2);
                    out.share[dst] = b.share[src];
                    out.tag[dst] = b.tag[src];
                }
            }
        }
    }
    return out;
}

// -----------------------------
// Attention
// -----------------------------
static AuthShare attention_dot(u64 q_rows, u64 k_rows, u64 dim,
                               const AuthShare &Q, const AuthShare &K, const AuthShare &V,
                               const char *softmax_tag) {
    ScopedCommTrace trace("bench:attention_dot");
    always_assert(!(q_rows == 1 && k_rows == 1));
    auto Kt = transpose_mat(K, k_rows, dim); // dim x k_rows
    Kt = reauth(Kt);
    auto scores = (dim > 1)
        ? linear_matmul_apply(q_rows, dim, k_rows, Q, Kt, 1.0 / std::sqrt((double)dim))
        : linear_matmul_apply(q_rows, dim, k_rows, Q, Kt);
    clip_batch_check("attn:after qk");
    clip_batch_check("attn:before softmax");
    auto attn = timed_clip_softmax(softmax_tag, q_rows, k_rows, scores);
    clip_batch_check("attn:after softmax");
    auto out = linear_matmul_apply(q_rows, k_rows, dim, attn, V);
    clip_batch_check("attn:after av");
    return out;
}

static AuthShare auth_extract_attn_head(const AuthShare &x, u64 rows,
                                        u64 num_heads, u64 head_dim, u64 head_idx) {
    always_assert(head_idx < num_heads);
    always_assert(x.share.size() == rows * num_heads * head_dim);
    AuthShare out = auth_alloc(rows * head_dim);
    const u64 model_dim = num_heads * head_dim;
    for (u64 r = 0; r < rows; ++r) {
        for (u64 c = 0; c < head_dim; ++c) {
            u64 src = r * model_dim + head_idx * head_dim + c;
            u64 dst = r * head_dim + c;
            out.share[dst] = x.share[src];
            out.tag[dst] = x.tag[src];
        }
    }
    return out;
}

static void auth_scatter_attn_head(AuthShare &dst, u64 rows, u64 num_heads,
                                   u64 head_dim, u64 head_idx, const AuthShare &src) {
    always_assert(head_idx < num_heads);
    always_assert(dst.share.size() == rows * num_heads * head_dim);
    always_assert(src.share.size() == rows * head_dim);
    const u64 model_dim = num_heads * head_dim;
    for (u64 r = 0; r < rows; ++r) {
        for (u64 c = 0; c < head_dim; ++c) {
            u64 src_idx = r * head_dim + c;
            u64 dst_idx = r * model_dim + head_idx * head_dim + c;
            dst.share[dst_idx] = src.share[src_idx];
            dst.tag[dst_idx] = src.tag[src_idx];
        }
    }
}

// -----------------------------
// Basic Transformer Block (single head)
// -----------------------------
struct AttnWeights {
    u64 num_heads = 1;
    u64 head_dim = 0;
    u64 model_dim = 0;
    u64 context_dim = 0;
    span<u64> wq, wk, wv, wout;
    AuthShare wq_auth, wk_auth, wv_auth, wout_auth;
    span<u64> bq_plain, bk_plain, bv_plain, bout_plain;
    AuthShare bq, bk, bv, bout;
};

static void configure_attn_layout(AttnWeights &w, u64 model_dim, u64 context_dim,
                                  u64 attention_head_dim);

struct NormAffineWeights {
    span<u64> weight_plain, bias_plain;
    AuthShare weight, bias;
};

struct FFWeights {
    span<u64> w_up, w_down;
    AuthShare w_up_auth, w_down_auth;
    span<u64> b_up_plain, b_down_plain;
    AuthShare b_up, b_down;
    u64 inner_dim = 0;
};

struct TransformerWeights {
    NormAffineWeights norm1, norm2, norm3;
    AttnWeights self_attn;
    AttnWeights cross_attn;
    FFWeights ff;
};

struct DenseBlockWeights {
    span<u64> w1, w2;
    AuthShare w1_auth, w2_auth;
    span<u64> b1_plain, b2_plain;
    AuthShare b1, b2;
    u64 hidden_dim = 0;
};

struct ResBlock2DWeights {
    u64 in_ch = 0;
    u64 out_ch = 0;
    u64 temb_dim = 0;
    u64 norm_groups = 2;
    bool has_skip = false;
    NormAffineWeights gn1, gn2;
    span<u64> conv1_w, conv2_w, skip_w, temb_w;
    AuthShare conv1_w_auth, conv2_w_auth, skip_w_auth, temb_w_auth;
    span<u64> conv1_b_plain, conv2_b_plain, skip_b_plain, temb_b_plain;
    AuthShare conv1_b, conv2_b, skip_b, temb_b;
};

struct Transformer2DWeights {
    u64 in_ch = 0;
    u64 inner_dim = 0;
    u64 ctx_dim = 0;
    u64 norm_groups = 2;
    NormAffineWeights gn;
    span<u64> proj_in_w, proj_out_w;
    AuthShare proj_in_w_auth, proj_out_w_auth;
    span<u64> proj_in_b_plain, proj_out_b_plain;
    AuthShare proj_in_b, proj_out_b;
    std::vector<TransformerWeights> blocks;
};

struct FeatureUpscalerWeights {
    u64 in_dim = 0;
    u64 hidden_dim = 0;
    u64 intermediate_dim = 0;
    u64 out_dim = 0;
    u64 num_layers = 0;
    span<u64> proj_in_w, proj_out_w;
    AuthShare proj_in_w_auth, proj_out_w_auth;
    span<u64> proj_in_b_plain, proj_out_b_plain;
    AuthShare proj_in_b, proj_out_b;
    std::vector<DenseBlockWeights> layers;
};

struct ImageEncoderWeights {
    u64 image_channels = 0;
    u64 patch_size = 0;
    u64 hidden_dim = 0;
    u64 embed_dim = 0;
    u64 num_patches = 0;
    span<u64> patch_w, proj_w;
    AuthShare patch_w_auth, proj_w_auth;
    span<u64> patch_b_plain, cls_token_plain, pos_embed_plain, proj_b_plain;
    AuthShare patch_b, cls_token, pos_embed, proj_b;
    NormAffineWeights pre_norm, post_norm;
    std::vector<TransformerWeights> blocks;
};

struct TextEncoderWeights {
    u64 vocab_size = 0;
    u64 seq_len = 0;
    u64 hidden_dim = 0;
    u64 num_heads = 1;
    span<u64> token_embed, pos_embed;
    AuthShare token_embed_auth, pos_embed_auth;
    NormAffineWeights out_norm;
    std::vector<TransformerWeights> blocks;
};

struct TimeEmbeddingWeights {
    u64 time_in_dim = 0;
    u64 temb_dim = 0;
    u64 class_labels_dim = 0;
    span<u64> time_w1, time_b1, time_w2, time_b2;
    span<u64> class_w, class_b_plain;
    AuthShare class_w_auth;
    AuthShare class_b;
};

struct DownBlockWeights {
    u64 in_ch = 0;
    u64 out_ch = 0;
    bool has_cross_attn = false;
    bool has_downsample = false;
    std::vector<ResBlock2DWeights> resblocks;
    std::vector<Transformer2DWeights> attn;
    span<u64> down_w, down_b_plain;
    AuthShare down_w_auth;
    AuthShare down_b;
};

struct UpBlockWeights {
    u64 in_ch = 0;
    u64 out_ch = 0;
    bool has_cross_attn = false;
    bool has_upsample = false;
    std::vector<u64> skip_channels;
    std::vector<ResBlock2DWeights> resblocks;
    std::vector<Transformer2DWeights> attn;
    span<u64> up_w, up_b_plain;
    AuthShare up_w_auth;
    AuthShare up_b;
};

struct CompleteUNetWeights {
    u64 in_ch = 0;
    u64 out_ch = 0;
    u64 temb_dim = 0;
    NormAffineWeights out_norm;
    span<u64> conv_in_w, conv_out_w;
    AuthShare conv_in_w_auth, conv_out_w_auth;
    span<u64> conv_in_b_plain, conv_out_b_plain;
    AuthShare conv_in_b, conv_out_b;
    std::vector<DownBlockWeights> down_blocks;
    ResBlock2DWeights mid_res1, mid_res2;
    Transformer2DWeights mid_attn;
    std::vector<UpBlockWeights> up_blocks;
};

struct VAEUpBlockWeights {
    u64 in_ch = 0;
    u64 out_ch = 0;
    bool has_upsample = false;
    span<u64> up_w, up_b_plain;
    AuthShare up_w_auth;
    AuthShare up_b;
    std::vector<ResBlock2DWeights> resblocks;
};

struct CompleteVAEDecoderWeights {
    u64 latent_ch = 0;
    u64 mid_ch = 0;
    u64 norm_groups = 2;
    NormAffineWeights mid_attn_gn, out_norm;
    span<u64> post_quant_w, conv_in_w, conv_out_w;
    AuthShare post_quant_w_auth, conv_in_w_auth, conv_out_w_auth;
    span<u64> post_quant_b_plain, conv_in_b_plain, conv_out_b_plain;
    AuthShare post_quant_b, conv_in_b, conv_out_b;
    ResBlock2DWeights mid_res1, mid_res2;
    TransformerWeights mid_attn;
    std::vector<VAEUpBlockWeights> up_blocks;
};

struct SuperResWeights {
    u64 in_ch = 0;
    u64 hidden_ch = 0;
    u64 out_ch = 0;
    NormAffineWeights out_norm;
    span<u64> conv_in_w, conv_mid_w, conv_out_w;
    AuthShare conv_in_w_auth, conv_mid_w_auth, conv_out_w_auth;
    span<u64> conv_in_b_plain, conv_mid_b_plain, conv_out_b_plain;
    AuthShare conv_in_b, conv_mid_b, conv_out_b;
    std::vector<ResBlock2DWeights> resblocks;
};

static NormAffineWeights make_norm_affine_weights(u64 dim) {
    NormAffineWeights w;
    w.weight_plain = span<u64>(dim);
    w.bias_plain = span<u64>(dim);
    const u64 one = (u64)(1ULL << f);
    for (u64 i = 0; i < w.weight_plain.size(); ++i) {
        w.weight_plain[i] = one;
    }
    for (u64 i = 0; i < w.bias_plain.size(); ++i) {
        w.bias_plain[i] = 0;
    }
    // Match PyTorch's affine GroupNorm/LayerNorm parameter defaults:
    // weights initialize to 1 and biases initialize to 0, but they remain
    // ordinary learnable tensors in the secure model.
    return w;
}

static void fill_norm_affine_identity(NormAffineWeights &w, int fp_bits) {
    const u64 one = (u64)(1ULL << fp_bits);
    for (u64 i = 0; i < w.weight_plain.size(); ++i) {
        w.weight_plain[i] = one;
    }
    for (u64 i = 0; i < w.bias_plain.size(); ++i) {
        w.bias_plain[i] = 0;
    }
}

static void auth_norm_affine(NormAffineWeights &w) {
    if (w.weight_plain.size() == 0 && w.bias_plain.size() == 0) return;
    w.weight = share_server_owned_model_tensor(w.weight_plain);
    w.bias = share_server_owned_model_tensor(w.bias_plain);
}

static TransformerWeights make_transformer_weights(u64 dim, u64 ctx_dim, u64 ff_inner,
                                                   bool with_cross_attn, bool gated_ff,
                                                   u64 attention_head_dim = 0) {
    TransformerWeights w;
    w.norm1 = make_norm_affine_weights(dim);
    w.norm2 = make_norm_affine_weights(dim);
    w.norm3 = make_norm_affine_weights(dim);
    configure_attn_layout(w.self_attn, dim, dim, attention_head_dim);
    w.self_attn.wq = span<u64>(dim * dim);
    w.self_attn.wk = span<u64>(dim * dim);
    w.self_attn.wv = span<u64>(dim * dim);
    w.self_attn.wout = span<u64>(dim * dim);
    w.self_attn.bq_plain = span<u64>(dim);
    w.self_attn.bk_plain = span<u64>(dim);
    w.self_attn.bv_plain = span<u64>(dim);
    w.self_attn.bout_plain = span<u64>(dim);

    if (with_cross_attn) {
        configure_attn_layout(w.cross_attn, dim, ctx_dim, attention_head_dim);
        w.cross_attn.wq = span<u64>(dim * dim);
        w.cross_attn.wk = span<u64>(ctx_dim * dim);
        w.cross_attn.wv = span<u64>(ctx_dim * dim);
        w.cross_attn.wout = span<u64>(dim * dim);
        w.cross_attn.bq_plain = span<u64>(dim);
        w.cross_attn.bk_plain = span<u64>(dim);
        w.cross_attn.bv_plain = span<u64>(dim);
        w.cross_attn.bout_plain = span<u64>(dim);
    }

    w.ff.inner_dim = ff_inner;
    w.ff.w_up = span<u64>(dim * (gated_ff ? (ff_inner * 2) : ff_inner));
    w.ff.w_down = span<u64>(ff_inner * dim);
    w.ff.b_up_plain = span<u64>(gated_ff ? (ff_inner * 2) : ff_inner);
    w.ff.b_down_plain = span<u64>(dim);
    return w;
}

static DenseBlockWeights make_dense_block_weights(u64 in_dim, u64 hidden_dim, u64 out_dim) {
    DenseBlockWeights w;
    w.hidden_dim = hidden_dim;
    w.w1 = span<u64>(in_dim * hidden_dim);
    w.w2 = span<u64>(hidden_dim * out_dim);
    w.b1_plain = span<u64>(hidden_dim);
    w.b2_plain = span<u64>(out_dim);
    return w;
}

static void configure_attn_layout(AttnWeights &w, u64 model_dim, u64 context_dim,
                                  u64 attention_head_dim) {
    w.model_dim = model_dim;
    w.context_dim = context_dim;
    if (attention_head_dim > 0 && model_dim > 0 && model_dim % attention_head_dim == 0) {
        w.head_dim = attention_head_dim;
        w.num_heads = model_dim / attention_head_dim;
    } else {
        w.num_heads = 1;
        w.head_dim = model_dim;
    }
}

static ResBlock2DWeights make_resblock2d_weights(u64 in_ch, u64 out_ch, u64 temb_dim, u64 norm_groups = 2) {
    ResBlock2DWeights w;
    w.in_ch = in_ch;
    w.out_ch = out_ch;
    w.temb_dim = temb_dim;
    w.norm_groups = std::max<u64>(1, std::min<u64>(norm_groups, out_ch));
    w.has_skip = in_ch != out_ch;
    w.gn1 = make_norm_affine_weights(in_ch);
    w.gn2 = make_norm_affine_weights(out_ch);
    w.conv1_w = span<u64>(out_ch * in_ch * 3 * 3);
    w.conv2_w = span<u64>(out_ch * out_ch * 3 * 3);
    w.temb_w = span<u64>(temb_dim * out_ch);
    w.conv1_b_plain = span<u64>(out_ch);
    w.conv2_b_plain = span<u64>(out_ch);
    w.temb_b_plain = span<u64>((temb_dim > 0) ? out_ch : 0);
    if (w.has_skip) {
        w.skip_w = span<u64>(out_ch * in_ch);
        w.skip_b_plain = span<u64>(out_ch);
    }
    return w;
}

static Transformer2DWeights make_transformer2d_weights(u64 in_ch, u64 inner_dim, u64 ctx_dim,
                                                       u64 num_blocks, u64 norm_groups = 2,
                                                       u64 attention_head_dim = 0) {
    Transformer2DWeights w;
    w.in_ch = in_ch;
    w.inner_dim = inner_dim;
    w.ctx_dim = ctx_dim;
    w.norm_groups = std::max<u64>(1, std::min<u64>(norm_groups, in_ch));
    w.gn = make_norm_affine_weights(in_ch);
    w.proj_in_w = span<u64>(in_ch * inner_dim);
    w.proj_out_w = span<u64>(inner_dim * in_ch);
    w.proj_in_b_plain = span<u64>(inner_dim);
    w.proj_out_b_plain = span<u64>(in_ch);
    w.blocks.reserve(num_blocks);
    for (u64 i = 0; i < num_blocks; ++i) {
        w.blocks.push_back(make_transformer_weights(
            inner_dim, ctx_dim, inner_dim * 4, true, true, attention_head_dim));
    }
    return w;
}

static FeatureUpscalerWeights make_feature_upscaler_weights(u64 in_dim, u64 hidden_dim,
                                                            u64 out_dim, u64 num_layers) {
    FeatureUpscalerWeights w;
    w.in_dim = in_dim;
    w.hidden_dim = hidden_dim;
    w.intermediate_dim = hidden_dim * 2;
    w.out_dim = out_dim;
    w.num_layers = num_layers;
    w.proj_in_w = span<u64>(in_dim * hidden_dim);
    w.proj_out_w = span<u64>(hidden_dim * out_dim);
    w.proj_in_b_plain = span<u64>(hidden_dim);
    w.proj_out_b_plain = span<u64>(out_dim);
    w.layers.reserve(num_layers);
    for (u64 i = 0; i < num_layers; ++i) {
        w.layers.push_back(make_dense_block_weights(hidden_dim, w.intermediate_dim, hidden_dim));
    }
    return w;
}

static TextEncoderWeights make_text_encoder_weights(u64 vocab_size, u64 seq_len,
                                                    u64 hidden_dim, u64 num_heads,
                                                    u64 num_layers, u64 ff_inner) {
    TextEncoderWeights w;
    w.vocab_size = vocab_size;
    w.seq_len = seq_len;
    w.hidden_dim = hidden_dim;
    w.num_heads = std::max<u64>(1, num_heads);
    w.token_embed = span<u64>(vocab_size * hidden_dim);
    w.pos_embed = span<u64>(seq_len * hidden_dim);
    w.out_norm = make_norm_affine_weights(hidden_dim);
    const u64 text_head_dim =
        (w.num_heads > 0 && hidden_dim > 0 && hidden_dim % w.num_heads == 0)
            ? (hidden_dim / w.num_heads)
            : 0;
    w.blocks.reserve(num_layers);
    for (u64 i = 0; i < num_layers; ++i) {
        w.blocks.push_back(
            make_transformer_weights(hidden_dim, hidden_dim, ff_inner, false, false, text_head_dim));
    }
    return w;
}

static TimeEmbeddingWeights make_time_embedding_weights(u64 time_in_dim,
                                                        u64 temb_dim,
                                                        u64 class_labels_dim = 0) {
    TimeEmbeddingWeights w;
    w.time_in_dim = time_in_dim;
    w.temb_dim = temb_dim;
    w.class_labels_dim = class_labels_dim;
    w.time_w1 = span<u64>(time_in_dim * temb_dim);
    w.time_b1 = span<u64>(temb_dim);
    w.time_w2 = span<u64>(temb_dim * temb_dim);
    w.time_b2 = span<u64>(temb_dim);
    w.class_w = span<u64>(class_labels_dim * temb_dim);
    w.class_b_plain = span<u64>(class_labels_dim > 0 ? temb_dim : 0);
    return w;
}

template <typename FillKaiming, typename FillTransformer>
static void fill_text_encoder_weights(TextEncoderWeights &w,
                                      FillKaiming &&fill_kaiming_uniform,
                                      FillTransformer &&fill_transformer,
                                      int fp_bits) {
    fill_kaiming_uniform(w.token_embed, w.vocab_size);
    fill_kaiming_uniform(w.pos_embed, w.seq_len);
    fill_norm_affine_identity(w.out_norm, fp_bits);
    for (auto &blk : w.blocks) fill_transformer(blk, w.hidden_dim, w.hidden_dim, false);
}

template <typename FillKaiming, typename FillBias>
static void fill_time_embedding_weights(TimeEmbeddingWeights &w,
                                        FillKaiming &&fill_kaiming_uniform,
                                        FillBias &&fill_bias_uniform) {
    fill_kaiming_uniform(w.time_w1, w.time_in_dim);
    fill_bias_uniform(w.time_b1, w.time_in_dim);
    fill_kaiming_uniform(w.time_w2, w.temb_dim);
    fill_bias_uniform(w.time_b2, w.temb_dim);
    if (w.class_labels_dim > 0) {
        fill_kaiming_uniform(w.class_w, w.class_labels_dim);
        fill_bias_uniform(w.class_b_plain, w.class_labels_dim);
    }
}

template <typename FillKaiming, typename FillBias>
static void fill_dense_block_weights(DenseBlockWeights &w,
                                     FillKaiming &&fill_kaiming_uniform,
                                     FillBias &&fill_bias_uniform) {
    const u64 in_dim = (w.hidden_dim == 0) ? 0 : (w.w1.size() / w.hidden_dim);
    fill_kaiming_uniform(w.w1, std::max<u64>(1, in_dim));
    fill_bias_uniform(w.b1_plain, std::max<u64>(1, in_dim));
    fill_kaiming_uniform(w.w2, std::max<u64>(1, w.hidden_dim));
    fill_bias_uniform(w.b2_plain, std::max<u64>(1, w.hidden_dim));
}

template <typename FillKaiming, typename FillBias>
static void fill_feature_upscaler_weights(FeatureUpscalerWeights &w,
                                          FillKaiming &&fill_kaiming_uniform,
                                          FillBias &&fill_bias_uniform) {
    always_assert(w.layers.size() == w.num_layers);
    fill_kaiming_uniform(w.proj_in_w, w.in_dim);
    fill_bias_uniform(w.proj_in_b_plain, w.in_dim);
    for (auto &layer : w.layers) {
        always_assert(layer.hidden_dim == w.intermediate_dim);
        fill_dense_block_weights(layer, fill_kaiming_uniform, fill_bias_uniform);
    }
    fill_kaiming_uniform(w.proj_out_w, w.hidden_dim);
    fill_bias_uniform(w.proj_out_b_plain, w.hidden_dim);
}

static ImageEncoderWeights make_image_encoder_weights(u64 image_channels, u64 patch_size,
                                                      u64 hidden_dim, u64 embed_dim,
                                                      u64 num_patches, u64 num_layers,
                                                      u64 ff_inner, u64 attention_head_dim) {
    ImageEncoderWeights w;
    w.image_channels = image_channels;
    w.patch_size = patch_size;
    w.hidden_dim = hidden_dim;
    w.embed_dim = embed_dim;
    w.num_patches = num_patches;
    w.patch_w = span<u64>(hidden_dim * image_channels * patch_size * patch_size);
    w.patch_b_plain = span<u64>(hidden_dim);
    w.cls_token_plain = span<u64>(hidden_dim);
    w.pos_embed_plain = span<u64>((num_patches + 1) * hidden_dim);
    w.proj_w = span<u64>(hidden_dim * embed_dim);
    w.proj_b_plain = span<u64>(embed_dim);
    w.pre_norm = make_norm_affine_weights(hidden_dim);
    w.post_norm = make_norm_affine_weights(hidden_dim);
    w.blocks.reserve(num_layers);
    for (u64 i = 0; i < num_layers; ++i) {
        w.blocks.push_back(make_transformer_weights(
            hidden_dim, hidden_dim, ff_inner, false, false, attention_head_dim));
    }
    return w;
}

template <typename FillKaiming, typename FillBias, typename FillTransformer>
static void fill_image_encoder_weights(ImageEncoderWeights &w,
                                       FillKaiming &&fill_kaiming_uniform,
                                       FillBias &&fill_bias_uniform,
                                       FillTransformer &&fill_transformer,
                                       int fp_bits) {
    fill_kaiming_uniform(w.patch_w, w.image_channels * w.patch_size * w.patch_size);
    fill_bias_uniform(w.patch_b_plain, w.image_channels * w.patch_size * w.patch_size);
    fill_bias_uniform(w.cls_token_plain, w.hidden_dim);
    fill_kaiming_uniform(w.pos_embed_plain, w.num_patches + 1);
    fill_kaiming_uniform(w.proj_w, w.hidden_dim);
    fill_bias_uniform(w.proj_b_plain, w.hidden_dim);
    fill_norm_affine_identity(w.pre_norm, fp_bits);
    fill_norm_affine_identity(w.post_norm, fp_bits);
    for (auto &blk : w.blocks) fill_transformer(blk, w.hidden_dim, w.hidden_dim, false);
}

static void share_transformer_weights(TransformerWeights &w, bool with_cross) {
    auth_norm_affine(w.norm1);
    auth_norm_affine(w.norm2);
    auth_norm_affine(w.norm3);
    auto share_attn = [&](AttnWeights &attn) {
        attn.wq_auth = share_server_owned_model_tensor(attn.wq);
        attn.wk_auth = share_server_owned_model_tensor(attn.wk);
        attn.wv_auth = share_server_owned_model_tensor(attn.wv);
        attn.wout_auth = share_server_owned_model_tensor(attn.wout);
        attn.bq = share_server_owned_model_tensor(attn.bq_plain);
        attn.bk = share_server_owned_model_tensor(attn.bk_plain);
        attn.bv = share_server_owned_model_tensor(attn.bv_plain);
        attn.bout = share_server_owned_model_tensor(attn.bout_plain);
    };
    share_attn(w.self_attn);
    if (with_cross) share_attn(w.cross_attn);
    w.ff.w_up_auth = share_server_owned_model_tensor(w.ff.w_up);
    w.ff.w_down_auth = share_server_owned_model_tensor(w.ff.w_down);
    w.ff.b_up = share_server_owned_model_tensor(w.ff.b_up_plain);
    w.ff.b_down = share_server_owned_model_tensor(w.ff.b_down_plain);
}

static void clear_transformer_weights_auth(TransformerWeights &w, bool with_cross) {
    w.norm1.weight = auth_alloc(0);
    w.norm1.bias = auth_alloc(0);
    w.norm2.weight = auth_alloc(0);
    w.norm2.bias = auth_alloc(0);
    w.norm3.weight = auth_alloc(0);
    w.norm3.bias = auth_alloc(0);
    auto clear_attn = [&](AttnWeights &attn) {
        attn.wq_auth = auth_alloc(0);
        attn.wk_auth = auth_alloc(0);
        attn.wv_auth = auth_alloc(0);
        attn.wout_auth = auth_alloc(0);
        attn.bq = auth_alloc(0);
        attn.bk = auth_alloc(0);
        attn.bv = auth_alloc(0);
        attn.bout = auth_alloc(0);
    };
    clear_attn(w.self_attn);
    clear_attn(w.cross_attn);
    w.ff.w_up_auth = auth_alloc(0);
    w.ff.w_down_auth = auth_alloc(0);
    w.ff.b_up = auth_alloc(0);
    w.ff.b_down = auth_alloc(0);
    (void)with_cross;
}

static void share_dense_weights(DenseBlockWeights &w) {
    w.w1_auth = share_server_owned_model_tensor(w.w1);
    w.w2_auth = share_server_owned_model_tensor(w.w2);
    w.b1 = share_server_owned_model_tensor(w.b1_plain);
    w.b2 = share_server_owned_model_tensor(w.b2_plain);
}

static void clear_dense_weights_auth(DenseBlockWeights &w) {
    w.w1_auth = auth_alloc(0);
    w.w2_auth = auth_alloc(0);
    w.b1 = auth_alloc(0);
    w.b2 = auth_alloc(0);
}

static void share_transformer_stack_estimated(std::vector<TransformerWeights> &blocks, bool with_cross) {
    const u64 total_layers = (u64)blocks.size();
    for (u64 i = 0; i < total_layers; ++i) {
        if (repeat_estimation_should_execute_index(total_layers, i)) {
            share_transformer_weights(blocks[i], with_cross);
        } else {
            clear_transformer_weights_auth(blocks[i], with_cross);
        }
    }
}

static void share_dense_stack_estimated(std::vector<DenseBlockWeights> &layers) {
    const u64 total_layers = (u64)layers.size();
    for (u64 i = 0; i < total_layers; ++i) {
        if (repeat_estimation_should_execute_index(total_layers, i)) {
            share_dense_weights(layers[i]);
        } else {
            clear_dense_weights_auth(layers[i]);
        }
    }
}

static void share_feature_upscaler_weights(FeatureUpscalerWeights &w) {
    w.proj_in_w_auth = share_server_owned_model_tensor(w.proj_in_w);
    w.proj_out_w_auth = share_server_owned_model_tensor(w.proj_out_w);
    w.proj_in_b = share_server_owned_model_tensor(w.proj_in_b_plain);
    w.proj_out_b = share_server_owned_model_tensor(w.proj_out_b_plain);
    share_dense_stack_estimated(w.layers);
}

static void clear_feature_upscaler_auth(FeatureUpscalerWeights &w) {
    w.proj_in_w_auth = auth_alloc(0);
    w.proj_out_w_auth = auth_alloc(0);
    w.proj_in_b = auth_alloc(0);
    w.proj_out_b = auth_alloc(0);
    for (auto &layer : w.layers) clear_dense_weights_auth(layer);
}

static void share_text_encoder_weights(TextEncoderWeights &w) {
    w.token_embed_auth = share_server_owned_model_tensor(w.token_embed);
    w.pos_embed_auth = share_server_owned_model_tensor(w.pos_embed);
    auth_norm_affine(w.out_norm);
    share_transformer_stack_estimated(w.blocks, false);
}

static void clear_text_encoder_auth(TextEncoderWeights &w) {
    w.token_embed_auth = auth_alloc(0);
    w.pos_embed_auth = auth_alloc(0);
    w.out_norm.weight = auth_alloc(0);
    w.out_norm.bias = auth_alloc(0);
    for (auto &blk : w.blocks) clear_transformer_weights_auth(blk, false);
}

static void share_resblock2d_weights(ResBlock2DWeights &w) {
    auth_norm_affine(w.gn1);
    auth_norm_affine(w.gn2);
    w.conv1_w_auth = share_server_owned_model_tensor(w.conv1_w);
    w.conv2_w_auth = share_server_owned_model_tensor(w.conv2_w);
    w.conv1_b = share_server_owned_model_tensor(w.conv1_b_plain);
    w.conv2_b = share_server_owned_model_tensor(w.conv2_b_plain);
    if (w.temb_dim > 0) {
        w.temb_w_auth = share_server_owned_model_tensor(w.temb_w);
        w.temb_b = share_server_owned_model_tensor(w.temb_b_plain);
    } else {
        w.temb_w_auth = auth_alloc(0);
        w.temb_b = auth_alloc(0);
    }
    if (w.has_skip) {
        w.skip_w_auth = share_server_owned_model_tensor(w.skip_w);
        w.skip_b = share_server_owned_model_tensor(w.skip_b_plain);
    } else {
        w.skip_w_auth = auth_alloc(0);
        w.skip_b = auth_alloc(0);
    }
}

static void clear_resblock2d_weights_auth(ResBlock2DWeights &w) {
    w.gn1.weight = auth_alloc(0);
    w.gn1.bias = auth_alloc(0);
    w.gn2.weight = auth_alloc(0);
    w.gn2.bias = auth_alloc(0);
    w.conv1_w_auth = auth_alloc(0);
    w.conv2_w_auth = auth_alloc(0);
    w.skip_w_auth = auth_alloc(0);
    w.temb_w_auth = auth_alloc(0);
    w.conv1_b = auth_alloc(0);
    w.conv2_b = auth_alloc(0);
    w.skip_b = auth_alloc(0);
    w.temb_b = auth_alloc(0);
}

static void share_resblock_stack_estimated(std::vector<ResBlock2DWeights> &blocks) {
    const u64 total_layers = (u64)blocks.size();
    for (u64 i = 0; i < total_layers; ++i) {
        if (repeat_estimation_should_execute_index(total_layers, i)) {
            share_resblock2d_weights(blocks[i]);
        } else {
            clear_resblock2d_weights_auth(blocks[i]);
        }
    }
}

static void share_resblock_stack_full(std::vector<ResBlock2DWeights> &blocks) {
    for (auto &blk : blocks) share_resblock2d_weights(blk);
}

static void share_transformer2d_weights(Transformer2DWeights &w) {
    auth_norm_affine(w.gn);
    w.proj_in_w_auth = share_server_owned_model_tensor(w.proj_in_w);
    w.proj_out_w_auth = share_server_owned_model_tensor(w.proj_out_w);
    w.proj_in_b = share_server_owned_model_tensor(w.proj_in_b_plain);
    w.proj_out_b = share_server_owned_model_tensor(w.proj_out_b_plain);
    share_transformer_stack_estimated(w.blocks, true);
}

static void clear_transformer2d_weights_auth(Transformer2DWeights &w) {
    w.gn.weight = auth_alloc(0);
    w.gn.bias = auth_alloc(0);
    w.proj_in_w_auth = auth_alloc(0);
    w.proj_out_w_auth = auth_alloc(0);
    w.proj_in_b = auth_alloc(0);
    w.proj_out_b = auth_alloc(0);
    for (auto &blk : w.blocks) clear_transformer_weights_auth(blk, true);
}

static void share_transformer2d_stack_estimated(std::vector<Transformer2DWeights> &blocks) {
    const u64 total_layers = (u64)blocks.size();
    for (u64 i = 0; i < total_layers; ++i) {
        if (repeat_estimation_should_execute_index(total_layers, i)) {
            share_transformer2d_weights(blocks[i]);
        } else {
            clear_transformer2d_weights_auth(blocks[i]);
        }
    }
}

static void share_transformer2d_stack_full(std::vector<Transformer2DWeights> &blocks) {
    for (auto &blk : blocks) share_transformer2d_weights(blk);
}

static void share_image_encoder_weights(ImageEncoderWeights &w) {
    w.patch_w_auth = share_server_owned_model_tensor(w.patch_w);
    w.proj_w_auth = share_server_owned_model_tensor(w.proj_w);
    w.patch_b = share_server_owned_model_tensor(w.patch_b_plain);
    w.cls_token = share_server_owned_model_tensor(w.cls_token_plain);
    w.pos_embed = share_server_owned_model_tensor(w.pos_embed_plain);
    w.proj_b = share_server_owned_model_tensor(w.proj_b_plain);
    auth_norm_affine(w.pre_norm);
    auth_norm_affine(w.post_norm);
    share_transformer_stack_estimated(w.blocks, false);
}

static void clear_image_encoder_auth(ImageEncoderWeights &w) {
    w.patch_w_auth = auth_alloc(0);
    w.proj_w_auth = auth_alloc(0);
    w.patch_b = auth_alloc(0);
    w.cls_token = auth_alloc(0);
    w.pos_embed = auth_alloc(0);
    w.proj_b = auth_alloc(0);
    w.pre_norm.weight = auth_alloc(0);
    w.pre_norm.bias = auth_alloc(0);
    w.post_norm.weight = auth_alloc(0);
    w.post_norm.bias = auth_alloc(0);
    for (auto &blk : w.blocks) clear_transformer_weights_auth(blk, false);
}

static AuthShare apply_attn(const AuthShare &x, u64 rows, u64 dim,
                            const AuthShare &context, u64 ctx_rows, u64 ctx_dim,
                            const AttnWeights &w, const char *softmax_tag) {
    const u64 model_dim = (w.model_dim == 0) ? dim : w.model_dim;
    const u64 context_dim = (w.context_dim == 0) ? ctx_dim : w.context_dim;
    const u64 num_heads = std::max<u64>(1, w.num_heads);
    const u64 head_dim = (w.head_dim == 0) ? (model_dim / num_heads) : w.head_dim;
    always_assert(model_dim == dim);
    always_assert(context_dim == ctx_dim);
    always_assert(num_heads * head_dim == model_dim);
    always_assert(ctx_rows > 0);
    clip_batch_check("attn:before v");
    auto v = linear_mat(ctx_rows, ctx_dim, dim, context, w.wv_auth, &w.bv);
    clip_batch_check("attn:after v");
    v = reauth(v);
    AuthShare attn_out = auth_alloc(rows * model_dim);
    if (ctx_rows == 1) {
        // With a single context row, softmax([score]) = 1, so the attention
        // output is just that lone V row broadcast to every query row.
        for (u64 r = 0; r < rows; ++r) {
            for (u64 c = 0; c < model_dim; ++c) {
                const u64 dst = r * model_dim + c;
                attn_out.share[dst] = v.share[c];
                attn_out.tag[dst] = v.tag[c];
            }
        }
    } else {
        clip_batch_check("attn:before q");
        auto q = linear_mat(rows, dim, dim, x, w.wq_auth, &w.bq);
        clip_batch_check("attn:after q");
        clip_batch_check("attn:before k");
        auto k = linear_mat(ctx_rows, ctx_dim, dim, context, w.wk_auth, &w.bk);
        clip_batch_check("attn:after k");
        q = reauth(q);
        k = reauth(k);
        if (num_heads == 1) {
            attn_out = attention_dot(rows, ctx_rows, head_dim, q, k, v, softmax_tag);
        } else {
            for (u64 head = 0; head < num_heads; ++head) {
                auto q_head = auth_extract_attn_head(q, rows, num_heads, head_dim, head);
                auto k_head = auth_extract_attn_head(k, ctx_rows, num_heads, head_dim, head);
                auto v_head = auth_extract_attn_head(v, ctx_rows, num_heads, head_dim, head);
                auto out_head = attention_dot(rows, ctx_rows, head_dim, q_head, k_head, v_head, softmax_tag);
                auth_scatter_attn_head(attn_out, rows, num_heads, head_dim, head, out_head);
            }
        }
    }
    return linear_mat(rows, dim, dim, attn_out, w.wout_auth, &w.bout);
}

static AuthShare basic_transformer_block(u64 rows, u64 dim, u64 ctx_rows, u64 ctx_dim,
                                         const AuthShare &x_in,
                                         const AuthShare &context_in,
                                         const TransformerWeights &w) {
    auto x = x_in;
    clip_batch_check("attn:before norm1");
    auto n1 = timed_layernorm_rows("layernorm:unet.mid_attn.norm1", rows, dim, x,
                                   &w.norm1.weight, &w.norm1.bias);
    clip_batch_check("attn:after norm1");
    auto a1 = apply_attn(n1, rows, dim, n1, rows, dim, w.self_attn, "softmax:unet.mid_attn.self");
    x = ADD_CALL(x, a1);

    auto n2 = timed_layernorm_rows("layernorm:unet.mid_attn.norm2", rows, dim, x,
                                   &w.norm2.weight, &w.norm2.bias);
    clip_batch_check("attn:after norm2");
    auto a2 = apply_attn(n2, rows, dim, context_in, ctx_rows, ctx_dim, w.cross_attn, "softmax:unet.mid_attn.cross");
    x = ADD_CALL(x, a2);

    auto n3 = timed_layernorm_rows("layernorm:unet.mid_attn.norm3", rows, dim, x,
                                   &w.norm3.weight, &w.norm3.bias);
    shark::utils::start_timer("linear2_attn.ff.gate");
    auto ff_up = linear_mat(rows, dim, w.ff.inner_dim * 2, n3, w.ff.w_up_auth, &w.ff.b_up);
    shark::utils::stop_timer("linear2_attn.ff.gate");
    AuthShare a = auth_alloc(rows * w.ff.inner_dim);
    AuthShare gate = auth_alloc(rows * w.ff.inner_dim);
    for (u64 i = 0; i < rows; ++i) {
        for (u64 j = 0; j < w.ff.inner_dim; ++j) {
            u64 src_base = i * (w.ff.inner_dim * 2);
            u64 dst = i * w.ff.inner_dim + j;
            a.share[dst] = ff_up.share[src_base + j];
            a.tag[dst] = ff_up.tag[src_base + j];
            gate.share[dst] = ff_up.share[src_base + w.ff.inner_dim + j];
            gate.tag[dst] = ff_up.tag[src_base + w.ff.inner_dim + j];
        }
    }
    auto gate_act = timed_gelu("gelu:unet.mid_attn.ff.gate", gate);
    shark::utils::start_timer("multiplication_attn.ff.gate");
    auto prod = MUL_CALL(a, gate_act);
    clip_batch_check("attn:before ff_mul_shift");
    auto prod_scaled = LRS_CALL(prod, f);
    clip_batch_check("attn:after ff_mul_shift");
    shark::utils::stop_timer("multiplication_attn.ff.gate");
    shark::utils::start_timer("linear4_attn.ff.gate");
    auto ff_out = linear_mat(rows, w.ff.inner_dim, dim, prod_scaled, w.ff.w_down_auth, &w.ff.b_down);
    shark::utils::stop_timer("linear4_attn.ff.gate");
    x = ADD_CALL(x, ff_out);
    return x;
}

// -----------------------------
// Simple scheduler coefficients (public)
// -----------------------------
static std::vector<double> build_alphas_cumprod(int num_train_timesteps, double beta_start, double beta_end) {
    std::vector<double> betas(num_train_timesteps);
    for (int i = 0; i < num_train_timesteps; ++i) {
        double t = (double)i / (double)(num_train_timesteps - 1);
        double b = std::pow(std::sqrt(beta_start) * (1.0 - t) + std::sqrt(beta_end) * t, 2.0);
        betas[i] = b;
    }
    std::vector<double> alphas(num_train_timesteps);
    for (int i = 0; i < num_train_timesteps; ++i) alphas[i] = 1.0 - betas[i];
    std::vector<double> alphas_cumprod(num_train_timesteps);
    double acc = 1.0;
    for (int i = 0; i < num_train_timesteps; ++i) {
        acc *= alphas[i];
        alphas_cumprod[i] = acc;
    }
    return alphas_cumprod;
}

static std::vector<double> build_alphas_cumprod_cos(int num_train_timesteps) {
    // squaredcos_cap_v2 schedule from python
    std::vector<double> betas(num_train_timesteps);
    auto alpha_bar = [](double t) {
        const double kPi = 3.14159265358979323846;
        double c = std::cos((t + 0.008) / 1.008 * kPi / 2.0);
        return c * c;
    };
    for (int i = 0; i < num_train_timesteps; ++i) {
        double t1 = (double)i / (double)num_train_timesteps;
        double t2 = (double)(i + 1) / (double)num_train_timesteps;
        double beta = std::min(1.0 - alpha_bar(t2) / alpha_bar(t1), 0.999);
        betas[i] = beta;
    }
    std::vector<double> alphas(num_train_timesteps);
    for (int i = 0; i < num_train_timesteps; ++i) alphas[i] = 1.0 - betas[i];
    std::vector<double> alphas_cumprod(num_train_timesteps);
    double acc = 1.0;
    for (int i = 0; i < num_train_timesteps; ++i) {
        acc *= alphas[i];
        alphas_cumprod[i] = acc;
    }
    return alphas_cumprod;
}

struct PNDMSchedulerState {
    std::vector<int> timesteps;
    std::vector<AuthShare> ets;
    AuthShare cur_sample = auth_alloc(0);
    bool has_cur_sample = false;
    int counter = 0;
    int step_ratio = 1;
};

static PNDMSchedulerState scheduler_make_pndm_state(int num_train_timesteps,
                                                    int num_inference_steps) {
    always_assert(num_train_timesteps > 0);
    always_assert(num_inference_steps > 0);
    always_assert(num_inference_steps <= num_train_timesteps);

    // Match the Stable UnCLIP default diffusers scheduler path:
    // PNDMScheduler(skip_prk_steps=true, steps_offset=1, timestep_spacing="leading").
    const int steps_offset = 1;
    const int step_ratio = num_train_timesteps / num_inference_steps;
    always_assert(step_ratio > 0);

    std::vector<int> base_timesteps(num_inference_steps);
    for (int i = 0; i < num_inference_steps; ++i) {
        base_timesteps[(size_t)i] = i * step_ratio + steps_offset;
    }

    std::vector<int> timesteps;
    if (num_inference_steps == 1) {
        timesteps.push_back(base_timesteps[0]);
    } else {
        timesteps.reserve((size_t)num_inference_steps + 1);
        for (int i = 0; i < num_inference_steps - 1; ++i) {
            timesteps.push_back(base_timesteps[(size_t)i]);
        }
        timesteps.push_back(base_timesteps[(size_t)num_inference_steps - 2]);
        timesteps.push_back(base_timesteps.back());
        std::reverse(timesteps.begin(), timesteps.end());
    }

    PNDMSchedulerState state;
    state.timesteps = std::move(timesteps);
    state.step_ratio = step_ratio;
    return state;
}

static double scheduler_lookup_alpha_cumprod_pndm(const std::vector<double> &alphas_cumprod,
                                                  int timestep) {
    always_assert(!alphas_cumprod.empty());
    if (timestep < 0) {
        // set_alpha_to_one=false for Stable UnCLIP, so the virtual "previous" alpha
        // at the final step is alpha_cumprod[0], not 1.0.
        return alphas_cumprod[0];
    }
    if (timestep >= (int)alphas_cumprod.size()) {
        timestep = (int)alphas_cumprod.size() - 1;
    }
    return alphas_cumprod[(size_t)timestep];
}

static AuthShare add_noise_ddpm(const AuthShare &x, const AuthShare &noise, double alpha_cumprod) {
    OpProfileScope op_profile(OP_SCHEDULER);
    shark::utils::start_timer("scheduler");
    double a = std::sqrt(alpha_cumprod);
    double b = std::sqrt(1.0 - alpha_cumprod);
    auto a_const = make_public_const(x.share.size(), a, f);
    auto b_const = make_public_const(x.share.size(), b, f);
    auto ax = LRS_CALL(MUL_CALL(x, a_const), f);
    auto bn = LRS_CALL(MUL_CALL(noise, b_const), f);
    auto out = ADD_CALL(ax, bn);
    shark::utils::stop_timer("scheduler");
    return out;
}

static AuthShare scheduler_step_linear(const AuthShare &sample, const AuthShare &model_output,
                                       double alpha_t, double alpha_prev) {
    OpProfileScope op_profile(OP_SCHEDULER);
    shark::utils::start_timer("scheduler");
    double sqrt_alpha_t = std::sqrt(alpha_t);
    double sqrt_alpha_prev = std::sqrt(alpha_prev);
    double sqrt_one_minus_alpha_t = std::sqrt(1.0 - alpha_t);
    double sqrt_one_minus_alpha_prev = std::sqrt(1.0 - alpha_prev);

    double a = sqrt_alpha_prev / sqrt_alpha_t;
    double b = sqrt_one_minus_alpha_prev - (sqrt_alpha_prev * sqrt_one_minus_alpha_t / sqrt_alpha_t);

    auto a_const = make_public_const(sample.share.size(), a, f);
    auto b_const = make_public_const(sample.share.size(), b, f);
    auto term_a = LRS_CALL(MUL_CALL(sample, a_const), f);
    auto term_b = LRS_CALL(MUL_CALL(model_output, b_const), f);
    auto out = ADD_CALL(term_a, term_b);
    shark::utils::stop_timer("scheduler");
    return out;
}

static AuthShare scheduler_get_prev_sample_pndm(const AuthShare &sample,
                                                const AuthShare &model_output,
                                                int timestep,
                                                int prev_timestep,
                                                const std::vector<double> &alphas_cumprod) {
    OpProfileScope op_profile(OP_SCHEDULER);
    shark::utils::start_timer("scheduler");
    const double alpha_prod_t = scheduler_lookup_alpha_cumprod_pndm(alphas_cumprod, timestep);
    const double alpha_prod_t_prev = scheduler_lookup_alpha_cumprod_pndm(alphas_cumprod, prev_timestep);
    const double beta_prod_t = std::max(0.0, 1.0 - alpha_prod_t);
    const double beta_prod_t_prev = std::max(0.0, 1.0 - alpha_prod_t_prev);

    // Stable UnCLIP uses prediction_type="v_prediction" in the default PNDM config.
    auto model_output_eps = ADD_CALL(
        scale_public(model_output, std::sqrt(alpha_prod_t)),
        scale_public(sample, std::sqrt(beta_prod_t)));

    const double sample_coeff = std::sqrt(alpha_prod_t_prev / alpha_prod_t);
    const double model_output_denom_coeff =
        alpha_prod_t * std::sqrt(beta_prod_t_prev) +
        std::sqrt(alpha_prod_t * beta_prod_t * alpha_prod_t_prev);

    auto prev_sample = ADD_CALL(
        scale_public(sample, sample_coeff),
        scale_public(model_output_eps, -(alpha_prod_t_prev - alpha_prod_t) / model_output_denom_coeff));
    shark::utils::stop_timer("scheduler");
    return prev_sample;
}

static AuthShare scheduler_step_pndm_plms(PNDMSchedulerState &scheduler,
                                          const AuthShare &sample,
                                          const AuthShare &model_output_in,
                                          int timestep,
                                          const std::vector<double> &alphas_cumprod) {
    always_assert(scheduler.step_ratio > 0);

    int prev_timestep = timestep - scheduler.step_ratio;
    AuthShare model_output = model_output_in;
    const AuthShare *sample_for_step = &sample;

    if (scheduler.counter != 1) {
        while (scheduler.ets.size() > 3) {
            scheduler.ets.erase(scheduler.ets.begin());
        }
        scheduler.ets.push_back(model_output_in);
    } else {
        prev_timestep = timestep;
        timestep = timestep + scheduler.step_ratio;
    }

    if (scheduler.ets.size() == 1 && scheduler.counter == 0) {
        scheduler.cur_sample = sample;
        scheduler.has_cur_sample = true;
        model_output = scheduler.ets.back();
    } else if (scheduler.ets.size() == 1 && scheduler.counter == 1) {
        always_assert(scheduler.has_cur_sample);
        sample_for_step = &scheduler.cur_sample;
        scheduler.has_cur_sample = false;
        model_output = scale_public(ADD_CALL(model_output_in, scheduler.ets.back()), 0.5);
    } else if (scheduler.ets.size() == 2) {
        model_output = ADD_CALL(
            scale_public(scheduler.ets[1], 3.0 / 2.0),
            scale_public(scheduler.ets[0], -1.0 / 2.0));
    } else if (scheduler.ets.size() == 3) {
        model_output = ADD_CALL(
            ADD_CALL(
                scale_public(scheduler.ets[2], 23.0 / 12.0),
                scale_public(scheduler.ets[1], -16.0 / 12.0)),
            scale_public(scheduler.ets[0], 5.0 / 12.0));
    } else {
        always_assert(scheduler.ets.size() >= 4);
        const size_t n = scheduler.ets.size();
        model_output = ADD_CALL(
            ADD_CALL(
                scale_public(scheduler.ets[n - 1], 55.0 / 24.0),
                scale_public(scheduler.ets[n - 2], -59.0 / 24.0)),
            ADD_CALL(
                scale_public(scheduler.ets[n - 3], 37.0 / 24.0),
                scale_public(scheduler.ets[n - 4], -9.0 / 24.0)));
    }

    auto prev_sample = scheduler_get_prev_sample_pndm(
        *sample_for_step, model_output, timestep, prev_timestep, alphas_cumprod);
    scheduler.counter += 1;
    return prev_sample;
}

static AuthShare dense_block_apply(u64 rows, u64 in_dim, u64 out_dim,
                                   const AuthShare &x, const DenseBlockWeights &w,
                                   const std::string &prefix) {
    auto h1 = linear_mat(rows, in_dim, w.hidden_dim, x, w.w1_auth, &w.b1);
    auto act1 = timed_gelu((prefix + ".gelu1").c_str(), h1);
    auto h2 = linear_mat(rows, w.hidden_dim, out_dim, act1, w.w2_auth, &w.b2);
    return timed_gelu((prefix + ".gelu2").c_str(), h2);
}

static AuthShare residual_self_attention_block(u64 rows, u64 dim, const AuthShare &x_in,
                                               const TransformerWeights &w,
                                               const std::string &prefix) {
    auto x = x_in;
    auto n1 = timed_layernorm_rows((prefix + ".norm1").c_str(), rows, dim, x,
                                   &w.norm1.weight, &w.norm1.bias);
    auto a1 = apply_attn(n1, rows, dim, n1, rows, dim, w.self_attn, "softmax:self_stack");
    x = ADD_CALL(x, a1);

    auto n2 = timed_layernorm_rows((prefix + ".norm2").c_str(), rows, dim, x,
                                   &w.norm2.weight, &w.norm2.bias);
    auto ff_up = linear_mat(rows, dim, w.ff.inner_dim, n2, w.ff.w_up_auth, &w.ff.b_up);
    auto ff_act = timed_gelu((prefix + ".ff").c_str(), ff_up);
    auto ff_out = linear_mat(rows, w.ff.inner_dim, dim, ff_act, w.ff.w_down_auth, &w.ff.b_down);
    return ADD_CALL(x, ff_out);
}

static AuthShare apply_batched_self_transformer_stack(u64 B, u64 rows, u64 dim,
                                                      const AuthShare &x_in,
                                                      const std::vector<TransformerWeights> &blocks,
                                                      const std::string &prefix) {
    if (blocks.empty()) return x_in;
    const char *segment_name = nullptr;
    int manual_timer_row_idx = -1;
    if (prefix == "image_encoder") {
        segment_name = "vision_transformer";
        manual_timer_row_idx = profile_timer_row_index_by_component("vision");
    } else if (prefix == "text_encoder") {
        segment_name = "text_transformer";
        manual_timer_row_idx = profile_timer_row_index_by_component("text");
    }
    const u64 total_layers = (u64)blocks.size();
    const u64 executed_layers =
        (segment_name != nullptr) ? repeat_estimation_executed_count(total_layers)
                                  : total_layers;
    ProfileSnapshot template_before{};
    ProfileSnapshot template_after{};
    shark::utils::TimerStat manual_timer_delta{};
    bool captured_template = false;
    AuthShare out = x_in;
    for (u64 layer = 0; layer < total_layers; ++layer) {
        const bool execute_this_layer =
            (segment_name == nullptr) || repeat_estimation_should_execute_index(total_layers, layer);
        if (!execute_this_layer) continue;
        const bool capture_this_layer =
            (segment_name != nullptr) && repeat_estimation_active_for_count(total_layers) &&
            (layer + 1 == 2);
        const auto manual_t0 = capture_this_layer ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point{};
        const u64 manual_comm0 = capture_this_layer ? current_comm_bytes() : 0;
        const u64 manual_rounds0 = capture_this_layer ? current_rounds() : 0;
        if (capture_this_layer) {
            template_before = capture_profile_snapshot();
        }
        AuthShare next = auth_alloc(B * rows * dim);
        for (u64 n = 0; n < B; ++n) {
            u64 off = n * rows * dim;
            auto x_b = auth_view(out, off, rows * dim);
            auto y_b = residual_self_attention_block(rows, dim, x_b, blocks[layer],
                                                     prefix + ".layer" + std::to_string(layer));
            auth_copy_into(next, off, y_b);
        }
        out = std::move(next);
        if (capture_this_layer) {
            template_after = capture_profile_snapshot();
            const auto manual_t1 = std::chrono::steady_clock::now();
            const double manual_time_ms =
                std::chrono::duration<double, std::milli>(manual_t1 - manual_t0).count();
            manual_timer_delta = make_manual_timer_delta(
                manual_time_ms,
                nonnegative_delta_u64(current_comm_bytes(), manual_comm0),
                nonnegative_delta_u64(current_rounds(), manual_rounds0));
            captured_template = true;
        }
    }
    if (captured_template) {
        accumulate_estimated_profile_repeat_segment(
            segment_name,
            total_layers,
            executed_layers,
            2,
            template_before,
            template_after,
            manual_timer_row_idx,
            &manual_timer_delta);
    }
    return out;
}

static AuthShare apply_batched_cross_transformer_stack(u64 B, u64 rows, u64 dim,
                                                       u64 ctx_rows, u64 ctx_dim,
                                                       const AuthShare &x_in,
                                                       const AuthShare &context,
                                                       const std::vector<TransformerWeights> &blocks) {
    if (blocks.empty()) return x_in;
    const u64 total_layers = (u64)blocks.size();
    AuthShare out = x_in;
    for (u64 layer = 0; layer < total_layers; ++layer) {
        AuthShare next = auth_alloc(B * rows * dim);
        for (u64 n = 0; n < B; ++n) {
            u64 x_off = n * rows * dim;
            u64 ctx_off = n * ctx_rows * ctx_dim;
            auto x_b = auth_view(out, x_off, rows * dim);
            auto ctx_b = auth_view(context, ctx_off, ctx_rows * ctx_dim);
            auto y_b = basic_transformer_block(rows, dim, ctx_rows, ctx_dim, x_b, ctx_b, blocks[layer]);
            auth_copy_into(next, x_off, y_b);
        }
        out = std::move(next);
    }
    return out;
}

static double fixed_to_double_local(u64 x) {
    return (double)(int64_t)x / (double)(1ULL << f);
}

static u64 double_to_fixed_local(double x) {
    return (u64)(int64_t)std::llround(x * (double)(1ULL << f));
}

static AuthTensor4D resblock2d_apply(const AuthTensor4D &x, const AuthShare &temb,
                                     const ResBlock2DWeights &w, const std::string &prefix) {
    always_assert(x.C == w.in_ch);
    auto n1 = timed_groupnorm_apply((prefix + ".gn1").c_str(), x.B, x.H, x.W, x.C,
                                    std::max<u64>(1, std::min<u64>(w.norm_groups, x.C)), x.data,
                                    &w.gn1.weight, &w.gn1.bias);
    auto a1 = timed_silu((prefix + ".act1").c_str(), n1);
    auto h1 = conv2d_apply_k3_same(x.B, x.H, x.W, x.C, w.out_ch, 1, a1, w.conv1_w_auth, &w.conv1_b);

    if (w.temb_dim > 0) {
        auto temb_proj = linear_mat(x.B, w.temb_dim, w.out_ch, temb, w.temb_w_auth, &w.temb_b);
        for (u64 n = 0; n < x.B; ++n) {
            for (u64 c = 0; c < w.out_ch; ++c) {
                u64 t_idx = n * w.out_ch + c;
                for (u64 h = 0; h < x.H; ++h) {
                    for (u64 widx = 0; widx < x.W; ++widx) {
                        u64 idx = idx4(n, h, widx, c, x.H, x.W, w.out_ch);
                        h1.share[idx] += temb_proj.share[t_idx];
                        h1.tag[idx] += temb_proj.tag[t_idx];
                    }
                }
            }
        }
    }

    auto n2 = timed_groupnorm_apply((prefix + ".gn2").c_str(), x.B, x.H, x.W, w.out_ch,
                                    std::max<u64>(1, std::min<u64>(w.norm_groups, w.out_ch)), h1,
                                    &w.gn2.weight, &w.gn2.bias);
    auto a2 = timed_silu((prefix + ".act2").c_str(), n2);
    auto h2 = conv2d_apply_k3_same(x.B, x.H, x.W, w.out_ch, w.out_ch, 1, a2, w.conv2_w_auth, &w.conv2_b);

    AuthShare skip = x.data;
    if (w.has_skip) {
        skip = conv2d_apply(x.B, x.H, x.W, x.C, w.out_ch, 1, 1, 0, x.data, w.skip_w_auth, &w.skip_b);
    }

    return AuthTensor4D{ADD_CALL(h2, skip), x.B, x.H, x.W, w.out_ch};
}

static AuthTensor4D transformer2d_apply(const AuthTensor4D &x, const AuthShare &context,
                                        u64 ctx_rows, u64 ctx_dim,
                                        const Transformer2DWeights &w, const std::string &prefix) {
    always_assert(x.C == w.in_ch);
    auto gn = timed_groupnorm_apply((prefix + ".gn").c_str(), x.B, x.H, x.W, x.C,
                                    std::max<u64>(1, std::min<u64>(w.norm_groups, x.C)), x.data,
                                    &w.gn.weight, &w.gn.bias);

    const u64 rows = x.H * x.W;
    AuthShare flat = auth_alloc(x.B * rows * x.C);
    for (u64 n = 0; n < x.B; ++n) {
        for (u64 h = 0; h < x.H; ++h) {
            for (u64 widx = 0; widx < x.W; ++widx) {
                for (u64 c = 0; c < x.C; ++c) {
                    u64 src = idx4(n, h, widx, c, x.H, x.W, x.C);
                    u64 dst = (n * rows + (h * x.W + widx)) * x.C + c;
                    flat.share[dst] = gn.share[src];
                    flat.tag[dst] = gn.tag[src];
                }
            }
        }
    }

    auto proj_in = linear_mat(x.B * rows, x.C, w.inner_dim, flat, w.proj_in_w_auth, &w.proj_in_b);
    auto proj_mid = apply_batched_cross_transformer_stack(x.B, rows, w.inner_dim, ctx_rows, ctx_dim,
                                                          proj_in, context, w.blocks);
    auto proj_out = linear_mat(x.B * rows, w.inner_dim, x.C, proj_mid, w.proj_out_w_auth, &w.proj_out_b);

    AuthShare reshaped = auth_alloc(x.B * x.H * x.W * x.C);
    for (u64 n = 0; n < x.B; ++n) {
        for (u64 h = 0; h < x.H; ++h) {
            for (u64 widx = 0; widx < x.W; ++widx) {
                for (u64 c = 0; c < x.C; ++c) {
                    u64 dst = idx4(n, h, widx, c, x.H, x.W, x.C);
                    u64 src = (n * rows + (h * x.W + widx)) * x.C + c;
                    reshaped.share[dst] = proj_out.share[src] + x.data.share[dst];
                    reshaped.tag[dst] = proj_out.tag[src] + x.data.tag[dst];
                }
            }
        }
    }

    return AuthTensor4D{std::move(reshaped), x.B, x.H, x.W, x.C};
}

static AuthTensor4D down_block_apply(const AuthTensor4D &x_in, const AuthShare &temb,
                                     const AuthShare &context, u64 ctx_rows, u64 ctx_dim,
                                     const DownBlockWeights &w, std::vector<AuthTensor4D> &skips,
                                     const std::string &prefix) {
    AuthTensor4D x = x_in;
    const u64 total_layers = (u64)w.resblocks.size();
    for (u64 i = 0; i < total_layers; ++i) {
        x = resblock2d_apply(x, temb, w.resblocks[i], prefix + ".res" + std::to_string(i));
        if (w.has_cross_attn && i < w.attn.size()) {
            x = transformer2d_apply(x, context, ctx_rows, ctx_dim, w.attn[i],
                                    prefix + ".attn" + std::to_string(i));
        }
        skips.push_back(x);
    }
    if (w.has_downsample) {
        auto y = conv2d_apply(x.B, x.H, x.W, x.C, x.C, 3, 2, 1, x.data, w.down_w_auth, &w.down_b);
        u64 outH = (x.H - 3 + 2) / 2 + 1;
        u64 outW = (x.W - 3 + 2) / 2 + 1;
        x = AuthTensor4D{std::move(y), x.B, outH, outW, x.C};
        skips.push_back(x);
    }
    return x;
}

static AuthTensor4D up_block_apply(const AuthTensor4D &x_in, std::vector<AuthTensor4D> &skips,
                                   const AuthShare &temb, const AuthShare &context,
                                   u64 ctx_rows, u64 ctx_dim,
                                   const UpBlockWeights &w, const std::string &prefix) {
    AuthTensor4D x = x_in;
    const u64 total_layers = (u64)w.resblocks.size();
    for (u64 i = 0; i < total_layers; ++i) {
        always_assert(!skips.empty());
        auto skip = skips.back();
        skips.pop_back();
        if (skip.H != x.H || skip.W != x.W) {
            auto resized = resize_nearest_hw(x.B, x.H, x.W, skip.H, skip.W, x.C, x.data);
            x = AuthTensor4D{std::move(resized), x.B, skip.H, skip.W, x.C};
        }
        auto cat = concat_channels(x.B, x.H, x.W, x.C, x.data, skip.C, skip.data);
        AuthTensor4D cat_t{std::move(cat), x.B, x.H, x.W, x.C + skip.C};
        x = resblock2d_apply(cat_t, temb, w.resblocks[i], prefix + ".res" + std::to_string(i));
        if (w.has_cross_attn && i < w.attn.size()) {
            x = transformer2d_apply(x, context, ctx_rows, ctx_dim, w.attn[i],
                                    prefix + ".attn" + std::to_string(i));
        }
    }
    if (w.has_upsample) {
        auto up = upsample_nearest_2x(x.B, x.H, x.W, x.C, x.data);
        auto conv = conv2d_apply_k3_same(x.B, x.H * 2, x.W * 2, x.C, x.C, 1, up, w.up_w_auth, &w.up_b);
        x = AuthTensor4D{std::move(conv), x.B, x.H * 2, x.W * 2, x.C};
    }
    return x;
}

static AuthShare feature_upscaler_apply(const AuthShare &x, u64 rows,
                                        const FeatureUpscalerWeights &w) {
    always_assert(w.layers.size() == w.num_layers);
    auto h = linear_mat(rows, w.in_dim, w.hidden_dim, x, w.proj_in_w_auth, &w.proj_in_b);
    const u64 total_layers = (u64)w.layers.size();
    const u64 executed_layers = repeat_estimation_executed_count(total_layers);
    ProfileSnapshot template_before{};
    ProfileSnapshot template_after{};
    shark::utils::TimerStat manual_timer_delta{};
    bool captured_template = false;
    const int manual_timer_row_idx = profile_timer_row_index_by_component("vision");
    for (u64 i = 0; i < total_layers; ++i) {
        if (!repeat_estimation_should_execute_index(total_layers, i)) continue;
        const bool capture_this_layer =
            repeat_estimation_active_for_count(total_layers) && (i + 1 == 2);
        const auto manual_t0 = capture_this_layer ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point{};
        const u64 manual_comm0 = capture_this_layer ? current_comm_bytes() : 0;
        const u64 manual_rounds0 = capture_this_layer ? current_rounds() : 0;
        if (capture_this_layer) {
            template_before = capture_profile_snapshot();
        }
        always_assert(w.layers[i].hidden_dim == w.intermediate_dim);
        h = dense_block_apply(rows, w.hidden_dim, w.hidden_dim, h, w.layers[i],
                              "gelu:feature_upscaler.layer" + std::to_string(i));
        if (capture_this_layer) {
            template_after = capture_profile_snapshot();
            const auto manual_t1 = std::chrono::steady_clock::now();
            const double manual_time_ms =
                std::chrono::duration<double, std::milli>(manual_t1 - manual_t0).count();
            manual_timer_delta = make_manual_timer_delta(
                manual_time_ms,
                nonnegative_delta_u64(current_comm_bytes(), manual_comm0),
                nonnegative_delta_u64(current_rounds(), manual_rounds0));
            captured_template = true;
        }
    }
    if (captured_template) {
        accumulate_estimated_profile_repeat_segment(
            "feature_dense",
            total_layers,
            executed_layers,
            2,
            template_before,
            template_after,
            manual_timer_row_idx,
            &manual_timer_delta);
    }
    return linear_mat(rows, w.hidden_dim, w.out_dim, h, w.proj_out_w_auth, &w.proj_out_b);
}

static AuthShare build_image_condition_secure(const AuthShare &img_embed_noised,
                                              u64 batch,
                                              const FeatureUpscalerWeights &feature_upscaler) {
    clip_batch_check("debug:before image_cond_auth");
    auto image_cond = feature_upscaler_apply(img_embed_noised, batch, feature_upscaler);
    clip_batch_check("debug:after image_cond_auth");
    return image_cond;
}

static AuthTensor4D vae_up_block_apply(const AuthTensor4D &x_in, const VAEUpBlockWeights &w,
                                       const std::string &prefix) {
    AuthTensor4D x = x_in;
    const u64 total_layers = (u64)w.resblocks.size();
    for (u64 i = 0; i < total_layers; ++i) {
        x = resblock2d_apply(x, make_public_const(x.B * w.resblocks[i].temb_dim, 0.0, f),
                             w.resblocks[i], prefix + ".res" + std::to_string(i));
    }
    if (w.has_upsample) {
        auto up = upsample_nearest_2x(x.B, x.H, x.W, x.C, x.data);
        auto conv = conv2d_apply_k3_same(x.B, x.H * 2, x.W * 2, x.C, x.C, 1, up, w.up_w_auth, &w.up_b);
        x = AuthTensor4D{std::move(conv), x.B, x.H * 2, x.W * 2, x.C};
    }
    return x;
}

static AuthTensor4D vae_decoder_apply(const AuthTensor4D &z, const CompleteVAEDecoderWeights &w) {
    auto z_post = conv2d_apply(z.B, z.H, z.W, z.C, w.latent_ch, 1, 1, 0,
                               z.data, w.post_quant_w_auth, &w.post_quant_b);
    auto h0 = conv2d_apply_k3_same(z.B, z.H, z.W, w.latent_ch, w.mid_ch, 1,
                                   z_post, w.conv_in_w_auth, &w.conv_in_b);
    AuthTensor4D h{std::move(h0), z.B, z.H, z.W, w.mid_ch};

    auto zero_temb = make_public_const(z.B * w.mid_res1.temb_dim, 0.0, f);
    h = resblock2d_apply(h, zero_temb, w.mid_res1, "vae.mid.res1");
    const bool disable_vae_mid_attn = (w.mid_attn.self_attn.wq_auth.share.size() == 0);
    if (!disable_vae_mid_attn) {
        const u64 gn_groups = std::max<u64>(1, std::min<u64>(w.norm_groups, h.C));
        auto gn = timed_groupnorm_apply("layernorm:vae.mid_attn.gn", h.B, h.H, h.W, h.C, gn_groups, h.data,
                                        &w.mid_attn_gn.weight, &w.mid_attn_gn.bias);
        const u64 rows = h.H * h.W;
        AuthShare flat = auth_alloc(h.B * rows * h.C);
        for (u64 n = 0; n < h.B; ++n) {
            for (u64 hh = 0; hh < h.H; ++hh) {
                for (u64 ww = 0; ww < h.W; ++ww) {
                    for (u64 c = 0; c < h.C; ++c) {
                        u64 src = idx4(n, hh, ww, c, h.H, h.W, h.C);
                        u64 dst = (n * rows + (hh * h.W + ww)) * h.C + c;
                        flat.share[dst] = gn.share[src];
                        flat.tag[dst] = gn.tag[src];
                    }
                }
            }
        }
        auto flat_attn = apply_batched_self_transformer_stack(
            h.B, rows, h.C, flat, std::vector<TransformerWeights>{w.mid_attn}, "vae.mid_attn");
        AuthShare h_attn = auth_alloc(h.B * h.H * h.W * h.C);
        for (u64 n = 0; n < h.B; ++n) {
            for (u64 hh = 0; hh < h.H; ++hh) {
                for (u64 ww = 0; ww < h.W; ++ww) {
                    for (u64 c = 0; c < h.C; ++c) {
                        u64 dst = idx4(n, hh, ww, c, h.H, h.W, h.C);
                        u64 src = (n * rows + (hh * h.W + ww)) * h.C + c;
                        h_attn.share[dst] = flat_attn.share[src] + h.data.share[dst];
                        h_attn.tag[dst] = flat_attn.tag[src] + h.data.tag[dst];
                    }
                }
            }
        }
        h = AuthTensor4D{std::move(h_attn), h.B, h.H, h.W, h.C};
    }
    h = resblock2d_apply(h, zero_temb, w.mid_res2, "vae.mid.res2");

    for (u64 i = 0; i < w.up_blocks.size(); ++i) {
        h = vae_up_block_apply(h, w.up_blocks[i], "vae.up" + std::to_string(i));
    }

    const u64 out_groups = std::max<u64>(1, std::min<u64>(w.norm_groups, h.C));
    auto nout = timed_groupnorm_apply("layernorm:vae.out", h.B, h.H, h.W, h.C, out_groups, h.data,
                                      &w.out_norm.weight, &w.out_norm.bias);
    auto aout = timed_silu("silu:vae.out", nout);
    const u64 out_ch = w.conv_out_b.share.size();
    auto out = conv2d_apply_k3_same(h.B, h.H, h.W, h.C, out_ch, 1, aout, w.conv_out_w_auth, &w.conv_out_b);
    return AuthTensor4D{std::move(out), h.B, h.H, h.W, out_ch};
}

static AuthTensor4D superres_apply(const AuthTensor4D &x_in, const SuperResWeights &w) {
    auto conv_in = conv2d_apply_k3_same(x_in.B, x_in.H, x_in.W, x_in.C, w.hidden_ch, 1,
                                        x_in.data, w.conv_in_w_auth, &w.conv_in_b);
    AuthTensor4D x{std::move(conv_in), x_in.B, x_in.H, x_in.W, w.hidden_ch};
    const u64 total_layers = (u64)w.resblocks.size();
    const u64 executed_layers = repeat_estimation_executed_count(total_layers);
    ProfileSnapshot template_before{};
    ProfileSnapshot template_after{};
    shark::utils::TimerStat manual_timer_delta{};
    bool captured_template = false;
    const int manual_timer_row_idx = profile_timer_row_index_by_component("superres");
    for (u64 i = 0; i < total_layers; ++i) {
        if (!repeat_estimation_should_execute_index(total_layers, i)) continue;
        const bool capture_this_layer =
            repeat_estimation_active_for_count(total_layers) && (i + 1 == 2);
        const auto manual_t0 = capture_this_layer ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point{};
        const u64 manual_comm0 = capture_this_layer ? current_comm_bytes() : 0;
        const u64 manual_rounds0 = capture_this_layer ? current_rounds() : 0;
        if (capture_this_layer) {
            template_before = capture_profile_snapshot();
        }
        auto zero_temb = make_public_const(x.B * w.resblocks[i].temb_dim, 0.0, f);
        x = resblock2d_apply(x, zero_temb, w.resblocks[i], "superres.res" + std::to_string(i));
        if (capture_this_layer) {
            template_after = capture_profile_snapshot();
            const auto manual_t1 = std::chrono::steady_clock::now();
            const double manual_time_ms =
                std::chrono::duration<double, std::milli>(manual_t1 - manual_t0).count();
            manual_timer_delta = make_manual_timer_delta(
                manual_time_ms,
                nonnegative_delta_u64(current_comm_bytes(), manual_comm0),
                nonnegative_delta_u64(current_rounds(), manual_rounds0));
            captured_template = true;
        }
    }
    if (captured_template) {
        accumulate_estimated_profile_repeat_segment(
            "superres_resblock",
            total_layers,
            executed_layers,
            2,
            template_before,
            template_after,
            manual_timer_row_idx,
            &manual_timer_delta);
    }
    auto nout = timed_groupnorm_apply("layernorm:superres.out", x.B, x.H, x.W, x.C, x.data,
                                      &w.out_norm.weight, &w.out_norm.bias);
    auto aout = timed_silu("silu:superres.out", nout);
    auto mid = conv2d_apply_k3_same(x.B, x.H, x.W, x.C, x.C, 1, aout, w.conv_mid_w_auth, &w.conv_mid_b);
    auto up = upsample_nearest_2x(x.B, x.H, x.W, x.C, mid);
    auto out = conv2d_apply_k3_same(x.B, x.H * 2, x.W * 2, x.C, w.out_ch, 1, up, w.conv_out_w_auth, &w.conv_out_b);
    return AuthTensor4D{std::move(out), x.B, x.H * 2, x.W * 2, w.out_ch};
}

static AuthTensor4D decode_latents_secure(const AuthShare &latents,
                                          u64 batch, u64 H, u64 W, u64 latent_C,
                                          const CompleteVAEDecoderWeights &vae_full) {
    // Match stableunclip.py::decode_latents(): images = vae.decode(latents / 0.18215).sample.
    const double scaling_factor = 0.18215;
    auto inv_scale = make_public_const(latents.share.size(), 1.0 / scaling_factor, f);
    auto latents_scaled = LRS_CALL(MUL_CALL(latents, inv_scale), f);
    AuthTensor4D latent_tensor{std::move(latents_scaled), batch, H, W, latent_C};
    return vae_decoder_apply(latent_tensor, vae_full);
}

static AuthShare extract_cls_tokens(u64 B, u64 tokens, u64 dim, const AuthShare &x) {
    always_assert(x.share.size() == B * tokens * dim);
    AuthShare out = auth_alloc(B * dim);
    for (u64 n = 0; n < B; ++n) {
        u64 src_off = n * tokens * dim;
        u64 dst_off = n * dim;
        for (u64 i = 0; i < dim; ++i) {
            out.share[dst_off + i] = x.share[src_off + i];
            out.tag[dst_off + i] = x.tag[src_off + i];
        }
    }
    return out;
}

static AuthShare encode_image_secure(const AuthShare &image_input_a,
                                     u64 batch, u64 img_H, u64 img_W,
                                     u64 noise_level, u64 num_train_timesteps,
                                     u64 base_rng_seed,
                                     const ImageEncoderWeights &w,
                                     bool vision_transformer_enabled) {
    (void)num_train_timesteps;
    const u64 patch_vec_dim = w.patch_size * w.patch_size * w.image_channels;
    const u64 outH = img_H / w.patch_size;
    const u64 outW = img_W / w.patch_size;
    const u64 N = w.num_patches;

    AuthShare patch_rows = auth_alloc(batch * N * patch_vec_dim);
    for (u64 n = 0; n < batch; ++n) {
        for (u64 oh = 0; oh < outH; ++oh) {
            for (u64 ow = 0; ow < outW; ++ow) {
                u64 row = n * N + oh * outW + ow;
                for (u64 kh = 0; kh < w.patch_size; ++kh) {
                    for (u64 kw = 0; kw < w.patch_size; ++kw) {
                        for (u64 c = 0; c < w.image_channels; ++c) {
                            u64 src_h = oh * w.patch_size + kh;
                            u64 src_w = ow * w.patch_size + kw;
                            u64 src = idx4(n, src_h, src_w, c, img_H, img_W, w.image_channels);
                            u64 dst = row * patch_vec_dim + (kh * w.patch_size + kw) * w.image_channels + c;
                            patch_rows.share[dst] = image_input_a.share[src];
                            patch_rows.tag[dst] = image_input_a.tag[src];
                        }
                    }
                }
            }
        }
    }

    auto patch = linear_mat(batch * N, patch_vec_dim, w.hidden_dim, patch_rows, w.patch_w_auth, &w.patch_b);
    clip_batch_check("debug:after image_patch_proj");
    keygen_ckpt(3, 1, "image_encoder:patch_conv");

    AuthShare img_tokens = auth_alloc(batch * (N + 1) * w.hidden_dim);
    for (u64 n = 0; n < batch; ++n) {
        u64 token_base = n * (N + 1) * w.hidden_dim;
        u64 patch_base = n * N * w.hidden_dim;
        for (u64 c = 0; c < w.hidden_dim; ++c) {
            img_tokens.share[token_base + c] = w.cls_token.share[c] + w.pos_embed.share[c];
            img_tokens.tag[token_base + c] = w.cls_token.tag[c] + w.pos_embed.tag[c];
        }
        for (u64 i = 0; i < N; ++i) {
            for (u64 c = 0; c < w.hidden_dim; ++c) {
                u64 dst = token_base + (i + 1) * w.hidden_dim + c;
                u64 src = patch_base + i * w.hidden_dim + c;
                u64 pos = (i + 1) * w.hidden_dim + c;
                img_tokens.share[dst] = patch.share[src] + w.pos_embed.share[pos];
                img_tokens.tag[dst] = patch.tag[src] + w.pos_embed.tag[pos];
            }
        }
    }

    AuthShare img_tokens_post = timed_layernorm_rows("layernorm:image_encoder.tokens",
                                                     batch * (N + 1), w.hidden_dim, img_tokens);
    if (vision_transformer_enabled) {
        auto img_tokens_pre = timed_layernorm_rows("layernorm:image_encoder.pre",
                                                   batch * (N + 1), w.hidden_dim, img_tokens,
                                                   &w.pre_norm.weight, &w.pre_norm.bias);
        auto img_tokens_vit = apply_batched_self_transformer_stack(batch, N + 1, w.hidden_dim,
                                                                   img_tokens_pre, w.blocks,
                                                                   "image_encoder");
        img_tokens_post = timed_layernorm_rows("layernorm:image_encoder.post",
                                               batch * (N + 1), w.hidden_dim, img_tokens_vit,
                                               &w.post_norm.weight, &w.post_norm.bias);
    }
    clip_batch_check("debug:after image_tokens_ln");
    keygen_ckpt(3, 2, "image_encoder:token_ln");

    auto cls_tokens = extract_cls_tokens(batch, N + 1, w.hidden_dim, img_tokens_post);
    auto img_embed = linear_mat(batch, w.hidden_dim, w.embed_dim, cls_tokens, w.proj_w_auth, &w.proj_b);
    clip_batch_check("debug:after image_cls_proj");
    keygen_ckpt(3, 3, "image_encoder:cls_proj");

    if (noise_level == 0) {
        return img_embed;
    }

    // Match stableunclip.py encode_image(): image_embeds += randn_like(image_embeds) * noise_level / 1000.0
    span<u64> noise(img_embed.share.size());
    zero_plain(noise);
    if (party == CLIENT) {
        current_rng_seed = base_rng_seed;
        for (u64 i = 0; i < noise.size(); ++i) {
            noise[i] = qrand_normal(f, 1.0);
        }
    }
    keygen_ckpt(2, 3, "input:noise");
    auto noise_a = profiled_authenticated_input_from_owner(noise, CLIENT, OP_INPUT);
    auto scaled_noise = scale_public(noise_a, (double)noise_level / 1000.0);
    return ADD_CALL(img_embed, scaled_noise);
}

static AuthShare encode_image_condition_secure(const AuthShare &image_input_a,
                                               u64 batch, u64 img_H, u64 img_W,
                                               u64 noise_level, u64 num_train_timesteps,
                                               u64 base_rng_seed,
                                               const ImageEncoderWeights &image_encoder,
                                               bool vision_transformer_enabled,
                                               const FeatureUpscalerWeights &feature_upscaler) {
    auto img_embed_noised = encode_image_secure(image_input_a, batch, img_H, img_W,
                                                noise_level, num_train_timesteps,
                                                base_rng_seed, image_encoder,
                                                vision_transformer_enabled);
    return build_image_condition_secure(img_embed_noised, batch, feature_upscaler);
}

static AuthShare build_prompt_embeds_secure_lookup(const std::vector<u64> &ids,
                                                   const TextEncoderWeights &text_encoder) {
    const u64 seq_len = text_encoder.seq_len;
    const u64 hidden = text_encoder.hidden_dim;
    const u64 vocab_size = text_encoder.vocab_size;
    always_assert(text_encoder.token_embed_auth.share.size() == vocab_size * hidden);
    always_assert(text_encoder.pos_embed_auth.share.size() == seq_len * hidden);

    span<u64> token_select(seq_len * vocab_size);
    span<u64> pos_select(seq_len * seq_len);
    zero_plain(token_select);
    zero_plain(pos_select);
    if (party == CLIENT) {
        build_prompt_selectors(ids, token_select, pos_select, vocab_size, seq_len, f);
    }

    AuthShare token_select_auth;
    AuthShare pos_select_auth;
    {
        ScopedOwnerInputTraceLabel trace_label("prompt_token_select_auth");
        token_select_auth = profiled_authenticated_input_from_owner(token_select, CLIENT, OP_INPUT);
    }
    {
        ScopedOwnerInputTraceLabel trace_label("prompt_pos_select_auth");
        pos_select_auth = profiled_authenticated_input_from_owner(pos_select, CLIENT, OP_INPUT);
    }

    auto token_lookup = linear_mat(seq_len, vocab_size, hidden,
                                   token_select_auth, text_encoder.token_embed_auth);
    auto pos_lookup = linear_mat(seq_len, seq_len, hidden,
                                 pos_select_auth, text_encoder.pos_embed_auth);
    return ADD_CALL(token_lookup, pos_lookup);
}

static AuthShare build_prompt_context_secure(u64 cfg_copies, u64 batch,
                                             const std::vector<u64> &prompt_ids,
                                             const std::vector<u64> &neg_prompt_ids,
                                             const TextEncoderWeights &text_encoder,
                                             bool text_transformer_enabled,
                                             bool secure_prompt_lookup) {
    always_assert(secure_prompt_lookup);
    const u64 seq_len = text_encoder.seq_len;
    const u64 hidden = text_encoder.hidden_dim;
    const u64 prompt_elems = seq_len * hidden;

    auto pe = build_prompt_embeds_secure_lookup(prompt_ids, text_encoder);
    auto ne = build_prompt_embeds_secure_lookup(neg_prompt_ids, text_encoder);

    AuthShare prompt_ctx = auth_alloc(cfg_copies * batch * prompt_elems);
    for (u64 b = 0; b < batch; ++b) {
        if (cfg_copies == 1) {
            const u64 pos_base = b * prompt_elems;
            for (u64 i = 0; i < prompt_elems; ++i) {
                prompt_ctx.share[pos_base + i] = pe.share[i];
                prompt_ctx.tag[pos_base + i] = pe.tag[i];
            }
        } else {
            const u64 neg_base = b * prompt_elems;
            const u64 pos_base = (batch + b) * prompt_elems;
            for (u64 i = 0; i < prompt_elems; ++i) {
                prompt_ctx.share[neg_base + i] = ne.share[i];
                prompt_ctx.tag[neg_base + i] = ne.tag[i];
                prompt_ctx.share[pos_base + i] = pe.share[i];
                prompt_ctx.tag[pos_base + i] = pe.tag[i];
            }
        }
    }

    if (text_transformer_enabled) {
        prompt_ctx = apply_batched_self_transformer_stack(
            cfg_copies * batch, seq_len, hidden, prompt_ctx, text_encoder.blocks, "text_encoder");
        AuthShare prompt_ctx_ln = auth_alloc(cfg_copies * batch * prompt_elems);
        for (u64 n = 0; n < cfg_copies * batch; ++n) {
            const u64 off = n * prompt_elems;
            auto xb = auth_view(prompt_ctx, off, prompt_elems);
            auto yb = timed_layernorm_rows(
                ("layernorm:text_encoder.out.b" + std::to_string(n)).c_str(),
                seq_len, hidden, xb, &text_encoder.out_norm.weight, &text_encoder.out_norm.bias);
            auth_copy_into(prompt_ctx_ln, off, yb);
        }
        prompt_ctx = std::move(prompt_ctx_ln);
    }
    return prompt_ctx;
}

static AuthShare build_time_embedding_secure(u64 Bcfg, int timestep,
                                             const TimeEmbeddingWeights &w) {
    span<u64> t_embed_plain(Bcfg * w.time_in_dim);
    auto t_base = make_timestep_embedding(timestep, w.time_in_dim, f, true);
    for (u64 n = 0; n < Bcfg; ++n) {
        for (u64 i = 0; i < w.time_in_dim; ++i) {
            t_embed_plain[n * w.time_in_dim + i] = t_base[i];
        }
    }
    shark::utils::start_timer("linear1_mlp");
    auto t1_plain = plain_linear_rows_server(Bcfg, w.time_in_dim, w.temb_dim,
                                             t_embed_plain, w.time_w1, w.time_b1);
    shark::utils::stop_timer("linear1_mlp");
    shark::utils::start_timer("gelu:unet.time_mlp");
    shark::utils::start_timer("exp_mlp");
    if (party == SERVER) {
        for (u64 i = 0; i < t1_plain.size(); ++i) {
            t1_plain[i] = double_to_q_time_mlp(
                time_mlp_silu_plain_value(q_to_double_time_mlp(t1_plain[i])));
        }
    }
    shark::utils::stop_timer("exp_mlp");
    shark::utils::stop_timer("gelu:unet.time_mlp");
    shark::utils::start_timer("linear3_mlp");
    auto temb_plain = plain_linear_rows_server(Bcfg, w.temb_dim, w.temb_dim,
                                               t1_plain, w.time_w2, w.time_b2);
    shark::utils::stop_timer("linear3_mlp");
    return authenticated_input_from_owner(temb_plain, SERVER);
}

static AuthShare build_class_labels_auth(const AuthShare &image_cond,
                                         u64 cfg_copies,
                                         u64 batch,
                                         u64 image_dim,
                                         u64 class_labels_dim,
                                         u64 noise_level) {
    always_assert(image_cond.share.size() == batch * image_dim);
    always_assert(class_labels_dim >= image_dim);
    const u64 noise_dim = class_labels_dim - image_dim;
    AuthShare class_labels = auth_alloc(batch * class_labels_dim);
    for (u64 n = 0; n < batch; ++n) {
        auth_copy_into(class_labels, n * class_labels_dim,
                       auth_view(image_cond, n * image_dim, image_dim));
    }
    if (noise_dim > 0) {
        span<u64> noise_plain(batch * noise_dim);
        auto noise_base = make_timestep_embedding((int)noise_level, noise_dim, f, true);
        for (u64 n = 0; n < batch; ++n) {
            for (u64 i = 0; i < noise_dim; ++i) {
                noise_plain[n * noise_dim + i] = noise_base[i];
            }
        }
        auto noise_auth = authenticated_input_from_owner(noise_plain, SERVER);
        for (u64 n = 0; n < batch; ++n) {
            auth_copy_into(class_labels, n * class_labels_dim + image_dim,
                           auth_view(noise_auth, n * noise_dim, noise_dim));
        }
    }

    AuthShare cfg_class_labels = make_public_const(cfg_copies * batch * class_labels_dim, 0.0, f);
    if (cfg_copies == 1) {
        auth_copy_into(cfg_class_labels, 0, class_labels);
    } else {
        auth_copy_into(cfg_class_labels, batch * class_labels_dim, class_labels);
    }
    return cfg_class_labels;
}

static AuthShare add_class_embedding_to_temb(const AuthShare &temb,
                                             const AuthShare &class_labels,
                                             u64 Bcfg,
                                             const TimeEmbeddingWeights &w) {
    if (w.class_labels_dim == 0) {
        return temb;
    }
    always_assert(class_labels.share.size() == Bcfg * w.class_labels_dim);
    auto class_emb = linear_mat(Bcfg, w.class_labels_dim, w.temb_dim,
                                class_labels, w.class_w_auth, &w.class_b);
    return ADD_CALL(temb, class_emb);
}

static AuthShare build_cfg_condition_auth(const AuthShare &src,
                                          u64 cfg_copies,
                                          u64 rows,
                                          u64 row_dim) {
    always_assert(src.share.size() == rows * row_dim);
    AuthShare out = make_public_const(cfg_copies * rows * row_dim, 0.0, f);
    for (u64 copy = 0; copy < cfg_copies; ++copy) {
        auth_copy_into(out, copy * rows * row_dim, src);
    }
    return out;
}

static int run_complete_arch(u64 base_rng_seed);

static int run_complete_arch(u64 base_rng_seed) {
    const StableUnCLIPBenchConfig cfg = load_unclip_bench_config();
    const u64 batch = cfg.batch;
    const u64 img_H = cfg.image_h;
    const u64 img_W = cfg.image_w;
    const u64 input_C = cfg.image_channels;
    const u64 output_C = cfg.output_channels;
    const u64 vae_scale_factor = cfg.vae_scale_factor;
    const u64 out_H = cfg.out_h;
    const u64 out_W = cfg.out_w;
    const u64 H = out_H / vae_scale_factor;
    const u64 W = out_W / vae_scale_factor;
    const u64 latent_C = cfg.latent_channels;
    const u64 seq_len = cfg.seq_len;
    const u64 hidden = cfg.text_hidden;
    const u64 text_num_heads = cfg.text_num_heads;
    const u64 ctx_dim = cfg.cond_embed_dim;
    const u64 img_hidden = cfg.clip_hidden_dim;
    const u64 img_emb_dim = cfg.image_embed_dim;
    const u64 vocab = cfg.vocab_size;
    const u64 patch_size = cfg.patch_size;
    const u64 time_in_dim = cfg.time_embed_input_dim;
    const u64 temb_dim = cfg.time_embed_dim;
    const u64 class_labels_dim = cfg.class_labels_dim;
    const u64 noise_level = cfg.noise_level;
    const int num_inference_steps = cfg.num_inference_steps;
    const double guidance_scale = cfg.guidance_scale;
    const u64 text_layers = cfg.text_layers;
    const u64 vision_layers = cfg.vision_layers;
    const u64 feature_layers = cfg.feature_layers;
    const u64 unet_layers_per_block = cfg.unet_layers_per_block;
    const u64 superres_layers = cfg.superres_layers;
    const u64 text_ff_inner = cfg.text_ff_inner;
    const u64 vision_ff_inner = cfg.vision_ff_inner;
    const u64 feature_hidden_dim = cfg.feature_hidden_dim;
    const u64 superres_hidden_dim = cfg.superres_hidden_dim;
    const u64 cfg_copies = cfg.cfg_copies;
    const u64 num_train_timesteps = cfg.num_train_timesteps;
    const double beta_start = cfg.beta_start;
    const double beta_end = cfg.beta_end;
    const bool use_linear_scheduler_fallback = cfg.use_linear_scheduler_fallback;
    const bool unet_cross_attn_enabled = cfg.unet_cross_attn_enabled;
    const bool unet_mid_attn_enabled = cfg.unet_mid_attn_enabled;
    const bool text_encoder_enabled = cfg.text_encoder_enabled;
    const bool use_text_conditioning = cfg.use_text_conditioning;
    const bool full_secure_image_encoder_enabled = cfg.full_secure_image_encoder_enabled;
    const bool vision_encoder_enabled = cfg.vision_encoder_enabled;
    const bool vae_mid_attn_enabled = cfg.vae_mid_attn_enabled;
    const bool enable_superres = cfg.enable_superres;
    const bool secure_prompt_lookup = cfg.secure_prompt_lookup;
    const bool text_transformer_enabled = cfg.text_transformer_enabled;
    const bool vision_transformer_enabled = cfg.vision_transformer_enabled;
    const bool unet_transformer_enabled = cfg.unet_transformer_enabled;
    const bool unet_mid_attn_runtime_enabled = cfg.unet_mid_attn_runtime_enabled;
    const bool use_unet_class_proj = cfg.use_unet_class_proj;
    always_assert(cfg_copies == 1 || cfg_copies == 2);
    if (text_encoder_enabled && use_text_conditioning && unet_transformer_enabled) {
        always_assert(hidden == ctx_dim);
    }
    const u64 default_norm_groups = cfg.default_norm_groups;
    const u64 default_transformer_blocks = cfg.default_transformer_blocks;
    const u64 default_vae_temb_dim = cfg.default_vae_temb_dim;
    const u64 vae_mid_channels = cfg.vae_mid_channels;
    const u64 vae_layers_per_block = cfg.vae_layers_per_block;
    const u64 vision_head_dim =
        (cfg.clip_num_heads > 0 && img_hidden % cfg.clip_num_heads == 0)
            ? (img_hidden / cfg.clip_num_heads)
            : cfg.attention_head_dim;
    const std::vector<u64> unet_channels = build_unet_channel_plan(cfg);
    const auto unet_down_resnet_counts = cfg.stable_unet_down_resnet_counts;
    const auto unet_down_transformer_blocks = cfg.stable_unet_down_transformer_blocks;
    const auto unet_up_resnet_counts = cfg.stable_unet_up_resnet_counts;
    const auto unet_up_transformer_blocks = cfg.stable_unet_up_transformer_blocks;
    const u64 unet_mid_resnet_count = std::max<u64>(1, cfg.stable_unet_mid_resnet_count);
    const u64 unet_mid_transformer_blocks = std::max<u64>(1, cfg.stable_unet_mid_transformer_blocks);
    const u64 unet_base_ch = unet_channels.empty() ? cfg.unet_ch : unet_channels.front();
    const std::vector<u64> vae_channels = build_vae_decoder_channel_plan(cfg);
    const u64 vae_mid_ch = vae_channels.empty() ? latent_C : vae_mid_channels;
    const u64 runtime_protocol_fingerprint = build_unclip_runtime_protocol_fingerprint(cfg);
    if (debug_protocol_fingerprint_enabled()) {
        verify_runtime_protocol_fingerprint(runtime_protocol_fingerprint, "run_complete_arch");
    }
    if (cfg.external_spec_loaded) {
        always_assert(unet_mid_resnet_count == 2);
        always_assert(unet_mid_transformer_blocks == 1);
    }

    print_unclip_config(cfg, runtime_protocol_fingerprint);
    // Reject debug shortcuts that would weaken the malicious-security path described in the paper.
    always_assert(std::getenv("SHARK_PUBLIC_ALPHA") == nullptr);
    always_assert(std::getenv("UNCLIP_DISABLE_BATCHCHECK") == nullptr);
    if (!minimal_terminal_output_enabled() && party == CLIENT && g_sanitized_protocol_debug_env_count > 0) {
        std::cout << "[UNCLIP_INFO] disabled " << g_sanitized_protocol_debug_env_count
                  << " protocol debug env flag(s) for benchmark fast path; "
                     "set UNCLIP_KEEP_PROTOCOL_DEBUG=1 to keep them."
                  << std::endl;
    }
    const bool block_text_only = phase_d_block_target_is("text");
    const bool block_vision_only = phase_d_block_target_is("vision");
    const bool block_unet_step_only = phase_d_block_target_is("unet_step");
    const bool block_vae_only = phase_d_block_target_is("vae");
    const bool block_superres_only = phase_d_block_target_is("superres");
    reset_estimated_profile_projection();
    bool diffusion_repeat_estimation_active = false;
    int diffusion_template_step = 0;
    int executed_diffusion_steps = 0;

    keygen_configure_progress(num_inference_steps);
    keygen_ckpt(0, 1, "start:complete_arch");
    clear_input_snapshots().clear();
    clear_authenticated_input_cache();
    clear_authenticated_weight_cache();
    const u64 img_grid_h = img_H / patch_size;
    const u64 img_grid_w = img_W / patch_size;
    const u64 num_patches = img_grid_h * img_grid_w;

    auto fill_kaiming_uniform = [&](span<u64> &buf, u64 fan_in) {
        double bound = std::sqrt(6.0 / (double)std::max<u64>(1, fan_in));
        for (u64 i = 0; i < buf.size(); ++i) buf[i] = qrand_uniform_symmetric(f, bound);
    };
    auto fill_bias_uniform = [&](span<u64> &buf, u64 fan_in) {
        double bound = 1.0 / std::sqrt((double)std::max<u64>(1, fan_in));
        for (u64 i = 0; i < buf.size(); ++i) buf[i] = qrand_uniform_symmetric(f, bound);
    };

    auto text_encoder = make_text_encoder_weights(vocab, seq_len, hidden, text_num_heads,
                                                  text_layers, text_ff_inner);
    auto image_encoder = make_image_encoder_weights(input_C, patch_size, img_hidden, img_emb_dim,
                                                    num_patches, vision_layers, vision_ff_inner,
                                                    vision_head_dim);

    auto feature_upscaler = make_feature_upscaler_weights(img_emb_dim, feature_hidden_dim, ctx_dim, feature_layers);
    auto time_embedder = make_time_embedding_weights(
        time_in_dim, temb_dim, use_unet_class_proj ? class_labels_dim : 0);

    CompleteUNetWeights unet_full;
    unet_full.in_ch = latent_C;
    unet_full.out_ch = latent_C;
    unet_full.temb_dim = temb_dim;
    unet_full.out_norm = make_norm_affine_weights(unet_base_ch);
    unet_full.conv_in_w = span<u64>(unet_base_ch * latent_C * 3 * 3);
    unet_full.conv_in_b_plain = span<u64>(unet_base_ch);
    unet_full.conv_out_w = span<u64>(latent_C * unet_base_ch * 3 * 3);
    unet_full.conv_out_b_plain = span<u64>(latent_C);

    u64 curr_ch = unet_base_ch;
    std::vector<u64> skip_channels;
    if (!unet_channels.empty()) {
        skip_channels.push_back(curr_ch);
    }
    for (u64 i = 0; i < unet_channels.size(); ++i) {
        DownBlockWeights block;
        block.in_ch = curr_ch;
        block.out_ch = unet_channels[i];
        block.has_cross_attn = unet_cross_attn_enabled;
        block.has_downsample = (i + 1 < unet_channels.size());
        const u64 block_num_res =
            std::max<u64>(1, i < unet_down_resnet_counts.size()
                                 ? unet_down_resnet_counts[i]
                                 : unet_layers_per_block);
        const u64 block_num_attn =
            std::max<u64>(1, i < unet_down_transformer_blocks.size()
                                 ? unet_down_transformer_blocks[i]
                                 : block_num_res);
        u64 block_ch = curr_ch;
        for (u64 layer = 0; layer < block_num_res; ++layer) {
            block.resblocks.push_back(make_resblock2d_weights(block_ch, block.out_ch, temb_dim, default_norm_groups));
            if (layer < block_num_attn) {
                block.attn.push_back(make_transformer2d_weights(block.out_ch, block.out_ch, ctx_dim,
                                                                default_transformer_blocks, default_norm_groups,
                                                                cfg.attention_head_dim));
            }
            block_ch = block.out_ch;
            skip_channels.push_back(block.out_ch);
        }
        if (block.has_downsample) {
            block.down_w = span<u64>(block.out_ch * block.out_ch * 3 * 3);
            block.down_b_plain = span<u64>(block.out_ch);
            skip_channels.push_back(block.out_ch);
        }
        unet_full.down_blocks.push_back(std::move(block));
        curr_ch = unet_channels[i];
    }

    unet_full.mid_res1 = make_resblock2d_weights(curr_ch, curr_ch, temb_dim, default_norm_groups);
    unet_full.mid_attn = make_transformer2d_weights(curr_ch, curr_ch, ctx_dim,
                                                    default_transformer_blocks, default_norm_groups,
                                                    cfg.attention_head_dim);
    unet_full.mid_res2 = make_resblock2d_weights(curr_ch, curr_ch, temb_dim, default_norm_groups);

    auto rev_channels = unet_channels;
    std::reverse(rev_channels.begin(), rev_channels.end());
    u64 up_ch = curr_ch;
    for (u64 i = 0; i < rev_channels.size(); ++i) {
        UpBlockWeights block;
        block.in_ch = up_ch;
        block.out_ch = rev_channels[i];
        block.has_cross_attn = unet_cross_attn_enabled;
        block.has_upsample = (i + 1 < rev_channels.size());
        const u64 block_num_res =
            std::max<u64>(1, i < unet_up_resnet_counts.size()
                                 ? unet_up_resnet_counts[i]
                                 : (unet_layers_per_block + 1));
        const u64 block_num_attn =
            std::max<u64>(1, i < unet_up_transformer_blocks.size()
                                 ? unet_up_transformer_blocks[i]
                                 : block_num_res);
        for (u64 j = 0; j < block_num_res; ++j) {
            always_assert(!skip_channels.empty());
            u64 skip_ch = skip_channels.back();
            skip_channels.pop_back();
            block.skip_channels.push_back(skip_ch);
            block.resblocks.push_back(make_resblock2d_weights(up_ch + skip_ch, block.out_ch, temb_dim, default_norm_groups));
            if (j < block_num_attn) {
                block.attn.push_back(make_transformer2d_weights(block.out_ch, block.out_ch, ctx_dim,
                                                                default_transformer_blocks, default_norm_groups,
                                                                cfg.attention_head_dim));
            }
            up_ch = block.out_ch;
        }
        if (block.has_upsample) {
            block.up_w = span<u64>(block.out_ch * block.out_ch * 3 * 3);
            block.up_b_plain = span<u64>(block.out_ch);
        }
        unet_full.up_blocks.push_back(std::move(block));
    }
    always_assert(skip_channels.empty());

    CompleteVAEDecoderWeights vae_full;
    vae_full.latent_ch = latent_C;
    vae_full.mid_ch = vae_mid_ch;
    vae_full.norm_groups = std::max<u64>(1, default_norm_groups);
    vae_full.mid_attn_gn = make_norm_affine_weights(vae_full.mid_ch);
    vae_full.post_quant_w = span<u64>(latent_C * latent_C);
    vae_full.conv_in_w = span<u64>(vae_full.mid_ch * latent_C * 3 * 3);
    vae_full.post_quant_b_plain = span<u64>(latent_C);
    vae_full.conv_in_b_plain = span<u64>(vae_full.mid_ch);
    vae_full.mid_res1 = make_resblock2d_weights(vae_full.mid_ch, vae_full.mid_ch, 0, vae_full.norm_groups);
    vae_full.mid_res2 = make_resblock2d_weights(vae_full.mid_ch, vae_full.mid_ch, 0, vae_full.norm_groups);
    vae_full.mid_attn = make_transformer_weights(vae_full.mid_ch, vae_full.mid_ch,
                                                 vae_full.mid_ch * 4, false, false,
                                                 vae_full.mid_ch);
    u64 vae_ch = vae_full.mid_ch;
    always_assert(is_power_of_two_u64(vae_scale_factor));
    const u64 vae_upsample_stages = log2_exact_u64(vae_scale_factor);
    always_assert(vae_channels.size() >= vae_upsample_stages + 1);
    for (u64 i = 0; i < vae_channels.size(); ++i) {
        VAEUpBlockWeights block;
        block.in_ch = vae_ch;
        block.out_ch = vae_channels[i];
        block.has_upsample = (i < vae_upsample_stages) && (i + 1 < vae_channels.size());
        u64 block_in_ch = block.in_ch;
        const u64 block_num_res = std::max<u64>(1, vae_layers_per_block + 1);
        for (u64 layer = 0; layer < block_num_res; ++layer) {
            block.resblocks.push_back(make_resblock2d_weights(block_in_ch, block.out_ch,
                                                              0, vae_full.norm_groups));
            block_in_ch = block.out_ch;
        }
        if (block.has_upsample) {
            block.up_w = span<u64>(block.out_ch * block.out_ch * 3 * 3);
            block.up_b_plain = span<u64>(block.out_ch);
        }
        vae_ch = block.out_ch;
        vae_full.up_blocks.push_back(std::move(block));
    }
    vae_full.out_norm = make_norm_affine_weights(vae_ch);
    vae_full.conv_out_w = span<u64>(output_C * vae_ch * 3 * 3);
    vae_full.conv_out_b_plain = span<u64>(output_C);

    SuperResWeights superres_w;
    superres_w.in_ch = output_C;
    superres_w.hidden_ch = superres_hidden_dim;
    superres_w.out_ch = output_C;
    superres_w.out_norm = make_norm_affine_weights(superres_w.hidden_ch);
    superres_w.conv_in_w = span<u64>(superres_w.hidden_ch * output_C * 3 * 3);
    superres_w.conv_in_b_plain = span<u64>(superres_w.hidden_ch);
    superres_w.conv_mid_w = span<u64>(superres_w.hidden_ch * superres_w.hidden_ch * 3 * 3);
    superres_w.conv_mid_b_plain = span<u64>(superres_w.hidden_ch);
    superres_w.conv_out_w = span<u64>(output_C * superres_w.hidden_ch * 3 * 3);
    superres_w.conv_out_b_plain = span<u64>(output_C);
    for (u64 i = 0; i < superres_layers; ++i) {
        superres_w.resblocks.push_back(
            make_resblock2d_weights(superres_w.hidden_ch, superres_w.hidden_ch,
                                    default_vae_temb_dim, default_norm_groups));
    }

    auto fill_attn = [&](AttnWeights &w, u64 dim, u64 ctx_in) {
        fill_kaiming_uniform(w.wq, dim);
        fill_kaiming_uniform(w.wk, ctx_in);
        fill_kaiming_uniform(w.wv, ctx_in);
        fill_kaiming_uniform(w.wout, dim);
        fill_bias_uniform(w.bq_plain, dim);
        fill_bias_uniform(w.bk_plain, ctx_in);
        fill_bias_uniform(w.bv_plain, ctx_in);
        fill_bias_uniform(w.bout_plain, dim);
    };
    auto fill_ff = [&](FFWeights &w, u64 in_dim) {
        fill_kaiming_uniform(w.w_up, in_dim);
        fill_bias_uniform(w.b_up_plain, in_dim);
        fill_kaiming_uniform(w.w_down, w.inner_dim);
        fill_bias_uniform(w.b_down_plain, w.inner_dim);
    };
    auto fill_transformer = [&](TransformerWeights &w, u64 dim, u64 ctx_in, bool with_cross) {
        fill_norm_affine_identity(w.norm1, f);
        fill_norm_affine_identity(w.norm2, f);
        fill_norm_affine_identity(w.norm3, f);
        fill_attn(w.self_attn, dim, dim);
        if (with_cross) fill_attn(w.cross_attn, dim, ctx_in);
        fill_ff(w.ff, dim);
    };
    auto fill_resblock = [&](ResBlock2DWeights &w) {
        fill_norm_affine_identity(w.gn1, f);
        fill_norm_affine_identity(w.gn2, f);
        fill_kaiming_uniform(w.conv1_w, w.in_ch * 3 * 3);
        fill_bias_uniform(w.conv1_b_plain, w.in_ch * 3 * 3);
        fill_kaiming_uniform(w.conv2_w, w.out_ch * 3 * 3);
        fill_bias_uniform(w.conv2_b_plain, w.out_ch * 3 * 3);
        if (w.temb_dim > 0) {
            fill_kaiming_uniform(w.temb_w, std::max<u64>(1, w.temb_dim));
            fill_bias_uniform(w.temb_b_plain, std::max<u64>(1, w.temb_dim));
        }
        if (w.has_skip) {
            fill_kaiming_uniform(w.skip_w, w.in_ch);
            fill_bias_uniform(w.skip_b_plain, w.in_ch);
        }
    };
    auto fill_transformer2d = [&](Transformer2DWeights &w) {
        fill_norm_affine_identity(w.gn, f);
        fill_kaiming_uniform(w.proj_in_w, w.in_ch);
        fill_bias_uniform(w.proj_in_b_plain, w.in_ch);
        fill_kaiming_uniform(w.proj_out_w, w.inner_dim);
        fill_bias_uniform(w.proj_out_b_plain, w.inner_dim);
        for (auto &blk : w.blocks) fill_transformer(blk, w.inner_dim, w.ctx_dim, true);
    };

    // Match ddpm1's secure ownership pattern: model parameters are synthesized
    // only on SERVER and then authenticated into MPC.
    if (party == SERVER) {
        current_rng_seed = base_rng_seed;
        if (text_encoder_enabled && use_text_conditioning && unet_transformer_enabled) {
            fill_text_encoder_weights(text_encoder, fill_kaiming_uniform, fill_transformer, f);
        }
        fill_image_encoder_weights(image_encoder, fill_kaiming_uniform, fill_bias_uniform,
                                   fill_transformer, f);
        fill_feature_upscaler_weights(feature_upscaler, fill_kaiming_uniform, fill_bias_uniform);
        fill_time_embedding_weights(time_embedder, fill_kaiming_uniform, fill_bias_uniform);
        fill_kaiming_uniform(unet_full.conv_in_w, latent_C * 3 * 3);
        fill_bias_uniform(unet_full.conv_in_b_plain, latent_C * 3 * 3);
        fill_kaiming_uniform(unet_full.conv_out_w, unet_base_ch * 3 * 3);
        fill_bias_uniform(unet_full.conv_out_b_plain, unet_base_ch * 3 * 3);
        fill_norm_affine_identity(unet_full.out_norm, f);
        for (auto &blk : unet_full.down_blocks) {
            for (auto &res : blk.resblocks) fill_resblock(res);
            for (auto &attn : blk.attn) fill_transformer2d(attn);
            if (blk.has_downsample) {
                fill_kaiming_uniform(blk.down_w, blk.out_ch * 3 * 3);
                fill_bias_uniform(blk.down_b_plain, blk.out_ch * 3 * 3);
            }
        }
        fill_resblock(unet_full.mid_res1);
        fill_transformer2d(unet_full.mid_attn);
        fill_resblock(unet_full.mid_res2);
        for (auto &blk : unet_full.up_blocks) {
            for (auto &res : blk.resblocks) fill_resblock(res);
            for (auto &attn : blk.attn) fill_transformer2d(attn);
            if (blk.has_upsample) {
                fill_kaiming_uniform(blk.up_w, blk.out_ch * 3 * 3);
                fill_bias_uniform(blk.up_b_plain, blk.out_ch * 3 * 3);
            }
        }
        fill_kaiming_uniform(vae_full.post_quant_w, latent_C);
        fill_bias_uniform(vae_full.post_quant_b_plain, latent_C);
        fill_kaiming_uniform(vae_full.conv_in_w, latent_C * 3 * 3);
        fill_bias_uniform(vae_full.conv_in_b_plain, latent_C * 3 * 3);
        fill_resblock(vae_full.mid_res1);
        fill_norm_affine_identity(vae_full.mid_attn_gn, f);
        fill_transformer(vae_full.mid_attn, vae_full.mid_ch, vae_full.mid_ch, false);
        fill_resblock(vae_full.mid_res2);
        for (auto &blk : vae_full.up_blocks) {
            if (blk.has_upsample) {
                fill_kaiming_uniform(blk.up_w, blk.out_ch * 3 * 3);
                fill_bias_uniform(blk.up_b_plain, blk.out_ch * 3 * 3);
            }
            for (auto &res : blk.resblocks) fill_resblock(res);
        }
        fill_norm_affine_identity(vae_full.out_norm, f);
        fill_kaiming_uniform(vae_full.conv_out_w, vae_ch * 3 * 3);
        fill_bias_uniform(vae_full.conv_out_b_plain, vae_ch * 3 * 3);
        fill_kaiming_uniform(superres_w.conv_in_w, output_C * 3 * 3);
        fill_bias_uniform(superres_w.conv_in_b_plain, output_C * 3 * 3);
        fill_kaiming_uniform(superres_w.conv_mid_w, superres_w.hidden_ch * 3 * 3);
        fill_bias_uniform(superres_w.conv_mid_b_plain, superres_w.hidden_ch * 3 * 3);
        fill_kaiming_uniform(superres_w.conv_out_w, superres_w.hidden_ch * 3 * 3);
        fill_bias_uniform(superres_w.conv_out_b_plain, superres_w.hidden_ch * 3 * 3);
        fill_norm_affine_identity(superres_w.out_norm, f);
        for (auto &res : superres_w.resblocks) fill_resblock(res);
    }

    clip_batch_check("share_model:begin");
    if (text_encoder_enabled && use_text_conditioning && unet_transformer_enabled) {
        clip_batch_check("share_model:text_encoder");
        share_text_encoder_weights(text_encoder);
    } else {
        clear_text_encoder_auth(text_encoder);
    }
    if (full_secure_image_encoder_enabled) {
        clip_batch_check("share_model:image_encoder");
        share_image_encoder_weights(image_encoder);
        share_feature_upscaler_weights(feature_upscaler);
    } else {
        clear_image_encoder_auth(image_encoder);
        clear_feature_upscaler_auth(feature_upscaler);
    }

    clip_batch_check("share_model:time1");
    if (time_embedder.class_labels_dim > 0) {
        time_embedder.class_w_auth = share_server_owned_model_tensor(time_embedder.class_w);
        time_embedder.class_b = share_server_owned_model_tensor(time_embedder.class_b_plain);
    } else {
        time_embedder.class_w_auth = auth_alloc(0);
        time_embedder.class_b = auth_alloc(0);
    }
    clip_batch_check("share_model:time2");

    clip_batch_check("share_model:conv_in");
    unet_full.conv_in_w_auth = share_server_owned_model_tensor(unet_full.conv_in_w);
    clip_batch_check("share_model:conv_out");
    unet_full.conv_out_w_auth = share_server_owned_model_tensor(unet_full.conv_out_w);
    unet_full.conv_in_b = share_server_owned_model_tensor(unet_full.conv_in_b_plain);
    unet_full.conv_out_b = share_server_owned_model_tensor(unet_full.conv_out_b_plain);
    auth_norm_affine(unet_full.out_norm);
    for (auto &blk : unet_full.down_blocks) {
        clip_batch_check("share_model:down_block");
        share_resblock_stack_full(blk.resblocks);
        if (blk.has_cross_attn && unet_transformer_enabled) {
            share_transformer2d_stack_full(blk.attn);
        } else {
            for (auto &attn : blk.attn) clear_transformer2d_weights_auth(attn);
        }
        if (blk.has_downsample) {
            blk.down_w_auth = share_server_owned_model_tensor(blk.down_w);
            blk.down_b = share_server_owned_model_tensor(blk.down_b_plain);
        } else {
            blk.down_w_auth = auth_alloc(0);
            blk.down_b = auth_alloc(0);
        }
    }
    clip_batch_check("share_model:mid_res1");
    share_resblock2d_weights(unet_full.mid_res1);
    clip_batch_check("share_model:mid_attn");
    if (unet_mid_attn_runtime_enabled) share_transformer2d_weights(unet_full.mid_attn);
    clip_batch_check("share_model:mid_res2");
    share_resblock2d_weights(unet_full.mid_res2);
    for (auto &blk : unet_full.up_blocks) {
        clip_batch_check("share_model:up_block");
        share_resblock_stack_full(blk.resblocks);
        if (blk.has_cross_attn && unet_transformer_enabled) {
            share_transformer2d_stack_full(blk.attn);
        } else {
            for (auto &attn : blk.attn) clear_transformer2d_weights_auth(attn);
        }
        if (blk.has_upsample) {
            blk.up_w_auth = share_server_owned_model_tensor(blk.up_w);
            blk.up_b = share_server_owned_model_tensor(blk.up_b_plain);
        } else {
            blk.up_w_auth = auth_alloc(0);
            blk.up_b = auth_alloc(0);
        }
    }
    clip_batch_check("share_model:vae");
    vae_full.post_quant_w_auth = share_server_owned_model_tensor(vae_full.post_quant_w);
    vae_full.post_quant_b = share_server_owned_model_tensor(vae_full.post_quant_b_plain);
    vae_full.conv_in_w_auth = share_server_owned_model_tensor(vae_full.conv_in_w);
    vae_full.conv_in_b = share_server_owned_model_tensor(vae_full.conv_in_b_plain);
    share_resblock2d_weights(vae_full.mid_res1);
    if (vae_mid_attn_enabled) {
        auth_norm_affine(vae_full.mid_attn_gn);
        share_transformer_weights(vae_full.mid_attn, false);
    } else {
        vae_full.mid_attn_gn.weight = auth_alloc(0);
        vae_full.mid_attn_gn.bias = auth_alloc(0);
    }
    share_resblock2d_weights(vae_full.mid_res2);
    for (auto &blk : vae_full.up_blocks) {
        if (blk.has_upsample) {
            blk.up_w_auth = share_server_owned_model_tensor(blk.up_w);
            blk.up_b = share_server_owned_model_tensor(blk.up_b_plain);
        } else {
            blk.up_w_auth = auth_alloc(0);
            blk.up_b = auth_alloc(0);
        }
        share_resblock_stack_full(blk.resblocks);
    }
    auth_norm_affine(vae_full.out_norm);
    vae_full.conv_out_w_auth = share_server_owned_model_tensor(vae_full.conv_out_w);
    vae_full.conv_out_b = share_server_owned_model_tensor(vae_full.conv_out_b_plain);
    if (enable_superres) {
        clip_batch_check("share_model:superres");
        superres_w.conv_in_w_auth = share_server_owned_model_tensor(superres_w.conv_in_w);
        superres_w.conv_mid_w_auth = share_server_owned_model_tensor(superres_w.conv_mid_w);
        superres_w.conv_out_w_auth = share_server_owned_model_tensor(superres_w.conv_out_w);
        superres_w.conv_in_b = share_server_owned_model_tensor(superres_w.conv_in_b_plain);
        superres_w.conv_mid_b = share_server_owned_model_tensor(superres_w.conv_mid_b_plain);
        superres_w.conv_out_b = share_server_owned_model_tensor(superres_w.conv_out_b_plain);
        auth_norm_affine(superres_w.out_norm);
        share_resblock_stack_estimated(superres_w.resblocks);
    }
    clip_batch_check("share_model:end");
    keygen_ckpt(1, 1, "auth_weights:complete_arch");

    if (party != DEALER) {
        shark::utils::start_timer("total_eval");
        start_total_eval_profile();
    }
    shark::utils::start_timer("input");

    // -----------------------------
    // Inputs: prompt + image
    // -----------------------------
    AuthShare prompt_ctx = auth_alloc(0);
    shark::utils::start_timer("block_text");
    if (text_encoder_enabled && use_text_conditioning && unet_transformer_enabled) {
        std::string prompt;
        if (party == CLIENT) {
            const char *prompt_env = std::getenv("UNCLIP_PROMPT");
            prompt = (prompt_env && *prompt_env) ? std::string(prompt_env) : std::string("");
        }
        auto prompt_ids = tokenize_prompt(prompt, vocab, seq_len);
        auto neg_prompt_ids = tokenize_prompt("", vocab, seq_len);
        keygen_ckpt(2, 1, "input:prompt_ctx");
        prompt_ctx = build_prompt_context_secure(
            cfg_copies, batch, prompt_ids, neg_prompt_ids, text_encoder,
            text_transformer_enabled, secure_prompt_lookup);
        clip_batch_check("debug:after prompt_ctx");
        if (debug_boundary_sync_enabled() && party != DEALER && peer != nullptr) {
            clip_batch_check("debug:before prompt_to_client_sync");
            peer->sync();
            clip_batch_check("debug:after prompt_to_client_sync");
        }
    }
    shark::utils::stop_timer("block_text");
    if (block_text_only) {
        shark::utils::stop_timer("input");
        return finish_unclip_benchmark(cfg, runtime_protocol_fingerprint);
    }

    AuthShare image_cond = make_public_const(batch * ctx_dim, 0.0, f);
    shark::utils::start_timer("block_vision");
        span<u64> image_input(batch * img_H * img_W * input_C);
        zero_plain(image_input);
        if (party == CLIENT) {
            current_rng_seed = base_rng_seed;
            for (u64 i = 0; i < image_input.size(); ++i) {
                image_input[i] = qrand_uniform_symmetric(f, 1.0);
            }
        }
        if (party == CLIENT && std::getenv("UNCLIP_APPLY_CLIP_IMAGE_NORM")) {
            image_input = normalize_image_input(image_input, img_H, img_W, input_C);
            clip_batch_check("debug:after image_norm");
        }
        keygen_ckpt(2, 2, "input:image");
        clip_batch_check("debug:after image_input");
        clip_batch_check("debug:before image_input_auth");
        AuthShare image_input_a;
        {
            ScopedOwnerInputTraceLabel trace_label("image_input_auth");
            image_input_a = profiled_authenticated_input_from_owner(image_input, CLIENT, OP_INPUT);
        }
        clip_batch_check("debug:after image_input_auth");
        if (!full_secure_image_encoder_enabled) {
            image_cond = auth_alloc(batch * ctx_dim);
            for (u64 i = 0; i < image_cond.share.size(); ++i) {
                image_cond.share[i] = 0;
                image_cond.tag[i] = 0;
            }
            for (u64 n = 0; n < batch; ++n) {
                for (u64 h = 0; h < img_H; ++h) {
                    for (u64 widx = 0; widx < img_W; ++widx) {
                        for (u64 c = 0; c < ctx_dim; ++c) {
                            u64 src = idx4(n, h, widx, c, img_H, img_W, input_C);
                            u64 dst = n * ctx_dim + c;
                            image_cond.share[dst] += image_input_a.share[src];
                            image_cond.tag[dst] += image_input_a.tag[src];
                        }
                    }
                }
            }
            const u64 denom = img_H * img_W;
            clip_batch_check("debug:before image_cond_reduce");
            if (is_power_of_two_u64(denom)) {
                u64 shift = 0;
                while ((u64(1) << shift) < denom) ++shift;
                image_cond = LRS_CALL(image_cond, shift);
            } else {
                image_cond = scale_public(image_cond, 1.0 / (double)denom);
            }
            clip_batch_check("debug:after image_cond_reduce");
            keygen_ckpt(3, 1, "image_encoder:direct_cond");
            keygen_ckpt(3, 2, "image_encoder:direct_cond");
            keygen_ckpt(3, 3, "image_encoder:direct_cond");
        } else {
            image_cond = encode_image_condition_secure(image_input_a, batch, img_H, img_W,
                                                       noise_level, cfg.num_train_timesteps,
                                                       base_rng_seed, image_encoder,
                                                       vision_transformer_enabled,
                                                       feature_upscaler);
        }
    shark::utils::stop_timer("block_vision");
    if (block_vision_only) {
        shark::utils::stop_timer("input");
        return finish_unclip_benchmark(cfg, runtime_protocol_fingerprint);
    }

    AuthShare image_cond_cfg = build_cfg_condition_auth(image_cond, cfg_copies, batch, ctx_dim);
    AuthShare class_labels_cfg =
        use_unet_class_proj
            ? build_class_labels_auth(image_cond, cfg_copies, batch, ctx_dim, class_labels_dim, noise_level)
            : auth_alloc(0);
    const u64 prompt_ctx_rows = (prompt_ctx.share.size() == 0) ? 0 : seq_len;

    // -----------------------------
    // Latents
    // -----------------------------
    span<u64> latents(batch * H * W * latent_C);
    zero_plain(latents);
    if (party == CLIENT) {
        current_rng_seed = base_rng_seed;
        for (u64 i = 0; i < latents.size(); ++i) {
            latents[i] = qrand_normal(f, 1.0);
        }
    }
    keygen_ckpt(2, 4, "input:latents");
    clip_batch_check("debug:before latents_auth");
    auto latents_a = profiled_authenticated_input_from_owner(latents, CLIENT, OP_INPUT);
    clip_batch_check("debug:after latents_auth");
    shark::utils::stop_timer("input");

    if (block_vae_only || block_superres_only) {
        shark::utils::start_timer("block_vae");
        AuthTensor4D final_img = decode_latents_secure(latents_a, batch, H, W, latent_C,
                                                       vae_full);
        shark::utils::stop_timer("block_vae");
        if (block_superres_only) {
            shark::utils::start_timer("block_superres");
            final_img = superres_apply(final_img, superres_w);
            shark::utils::stop_timer("block_superres");
        }
        return finish_unclip_benchmark(cfg, runtime_protocol_fingerprint);
    }

    auto alphas_cumprod = build_alphas_cumprod((int)num_train_timesteps, beta_start, beta_end);
    auto scheduler = scheduler_make_pndm_state((int)num_train_timesteps, num_inference_steps);
    const std::vector<int> &timesteps = scheduler.timesteps;
    const int num_scheduler_steps = (int)timesteps.size();
    diffusion_repeat_estimation_active =
        repeat_estimation_active_for_count((u64)num_scheduler_steps);
    diffusion_template_step = diffusion_repeat_estimation_active ? 2 : 0;
    executed_diffusion_steps = diffusion_repeat_estimation_active ? 3 : num_scheduler_steps;

    // -----------------------------
    // UNet forward (complete small arch)
    // -----------------------------
    // Match stableunclip.py::generate(): encoder_hidden_states = image_embeds.
    (void)prompt_ctx_rows;
    AuthShare cross_ctx_cfg = image_cond_cfg;
    const u64 cross_ctx_rows = 1;

    auto unet_forward = [&](const AuthShare &lat_in, const AuthShare &encoder_ctx_in,
                            int timestep,
                            int diffusion_step, int diffusion_total) -> AuthShare {
        const u64 Bcfg = cfg_copies * batch;

        auto unet_step_ckpt = [&](int component, const char *op_name) {
            std::ostringstream oss;
            oss << "unet:" << op_name
                << " step " << diffusion_step << "/" << diffusion_total
                << " (t=" << timestep << ")";
            keygen_ckpt(5, component, oss.str());
        };

        clip_batch_check("debug:before temb_auth");
        auto temb = build_time_embedding_secure(Bcfg, timestep, time_embedder);
        if (use_unet_class_proj) {
            temb = add_class_embedding_to_temb(temb, class_labels_cfg, Bcfg, time_embedder);
        }
        clip_batch_check("debug:after temb_auth");
        unet_step_ckpt(1, "time_mlp");

        auto h0 = conv2d_apply_k3_same(Bcfg, H, W, latent_C, unet_base_ch, 1,
                                       lat_in, unet_full.conv_in_w_auth, &unet_full.conv_in_b);
        unet_step_ckpt(2, "conv_in");

        AuthTensor4D x{std::move(h0), Bcfg, H, W, unet_base_ch};
        std::vector<AuthTensor4D> skips;
        skips.reserve(16);
        if (!unet_full.up_blocks.empty()) {
            skips.push_back(x);
        }
        for (u64 i = 0; i < unet_full.down_blocks.size(); ++i) {
            x = down_block_apply(x, temb, encoder_ctx_in, cross_ctx_rows, ctx_dim,
                                 unet_full.down_blocks[i], skips,
                                 "unet.down" + std::to_string(i));
        }
        unet_step_ckpt(3, "down_path");

        x = resblock2d_apply(x, temb, unet_full.mid_res1, "unet.mid.res1");
        if (unet_mid_attn_enabled) {
            x = transformer2d_apply(x, encoder_ctx_in, cross_ctx_rows, ctx_dim,
                                    unet_full.mid_attn, "unet.mid.attn");
        }
        x = resblock2d_apply(x, temb, unet_full.mid_res2, "unet.mid.res2");
        unet_step_ckpt(4, "mid");

        for (u64 i = 0; i < unet_full.up_blocks.size(); ++i) {
            x = up_block_apply(x, skips, temb, encoder_ctx_in, cross_ctx_rows, ctx_dim,
                               unet_full.up_blocks[i], "unet.up" + std::to_string(i));
        }
        always_assert(skips.empty());
        unet_step_ckpt(5, "up_path");

        auto out_norm = timed_groupnorm_apply("layernorm:unet.out", x.B, x.H, x.W, x.C, x.data,
                                              &unet_full.out_norm.weight, &unet_full.out_norm.bias);
        auto out_act = timed_silu("silu:unet.out", out_norm);
        auto out = conv2d_apply_k3_same(x.B, x.H, x.W, x.C, unet_full.out_ch, 1,
                                        out_act, unet_full.conv_out_w_auth, &unet_full.conv_out_b);
        unet_step_ckpt(6, "conv_out");
        return out;
    };

    ProfileSnapshot estimated_repeat_before{};
    ProfileSnapshot estimated_repeat_after{};
    EstimatedProfileProjection estimated_repeat_extras_before{};
    EstimatedProfileProjection estimated_repeat_extras_after{};
    bool estimated_repeat_captured = false;
    // Diffusion loop.
    for (int step = 0; step < num_scheduler_steps; ++step) {
        if (!repeat_estimation_should_execute_index((u64)num_scheduler_steps, (u64)step)) continue;
        if (diffusion_repeat_estimation_active && (step + 1) == diffusion_template_step) {
            estimated_repeat_before = capture_profile_snapshot();
            estimated_repeat_extras_before = g_estimated_profile_extras;
        }
        shark::utils::start_timer("block_unet_step");
        AuthShare latents_cfg = auth_alloc(cfg_copies * latents_a.share.size());
        for (u64 i = 0; i < latents_a.share.size(); ++i) {
            latents_cfg.share[i] = latents_a.share[i];
            latents_cfg.tag[i] = latents_a.tag[i];
            if (cfg_copies > 1) {
                latents_cfg.share[latents_a.share.size() + i] = latents_a.share[i];
                latents_cfg.tag[latents_a.share.size() + i] = latents_a.tag[i];
            }
        }

        int t = timesteps[step];
        auto noise_pred = unet_forward(latents_cfg, cross_ctx_cfg,
                                       t, step + 1, num_scheduler_steps);

        const u64 lat_elems = latents_a.share.size();
        always_assert(noise_pred.share.size() == cfg_copies * lat_elems);

        AuthShare guided = auth_alloc(lat_elems);
        if (cfg_copies == 1) {
            for (u64 i = 0; i < lat_elems; ++i) {
                guided.share[i] = noise_pred.share[i];
                guided.tag[i] = noise_pred.tag[i];
            }
        } else {
            always_assert(cfg_copies >= 2);
            AuthShare uncond = auth_alloc(lat_elems);
            AuthShare text = auth_alloc(lat_elems);
            for (u64 i = 0; i < lat_elems; ++i) {
                uncond.share[i] = noise_pred.share[i];
                uncond.tag[i] = noise_pred.tag[i];
                text.share[i] = noise_pred.share[lat_elems + i];
                text.tag[i] = noise_pred.tag[lat_elems + i];
            }

            auto diff = ADD_CALL(text, neg_span(uncond));
            auto scale = make_public_const(diff.share.size(), guidance_scale, f);
            auto scaled = LRS_CALL(MUL_CALL(diff, scale), f);
            guided = ADD_CALL(uncond, scaled);
        }

        always_assert(!use_linear_scheduler_fallback);
        latents_a = scheduler_step_pndm_plms(scheduler, latents_a, guided, t, alphas_cumprod);

        std::ostringstream step_label;
        step_label << "diffusion:step " << (step + 1) << "/" << num_scheduler_steps
                   << " (t=" << t << ")";
        keygen_ckpt(4, step + 1, step_label.str());
        if (minimal_terminal_output_enabled() && party != DEALER) {
            print_minimal_step((u64)step + 1);
        } else if (profile_step_summary_enabled() && party != DEALER) {
            std::cout << "[PROFILE_STEP] " << step_label.str() << std::endl;
            if (shark::protocols::peer) {
                u64 total_comm = shark::protocols::peer->bytesReceived() + shark::protocols::peer->bytesSent();
                u64 total_rounds = shark::protocols::peer->roundsReceived() + shark::protocols::peer->roundsSent();
                std::cout << "[PROFILE_STEP] total_comm: " << (double)total_comm / 1024.0 << " KB (" << total_comm
                          << " bytes), total_rounds: " << total_rounds << std::endl;
            }
            print_profile_timers();
        }
        shark::utils::stop_timer("block_unet_step");
        if (diffusion_repeat_estimation_active && (step + 1) == diffusion_template_step) {
            estimated_repeat_after = capture_profile_snapshot();
            estimated_repeat_extras_after = g_estimated_profile_extras;
            estimated_repeat_captured = true;
        }
        if (block_unet_step_only) {
            return finish_unclip_benchmark(cfg, runtime_protocol_fingerprint);
        }
    }

    shark::utils::start_timer("block_vae");
    AuthTensor4D final_img = decode_latents_secure(latents_a, batch, H, W, latent_C,
                                                   vae_full);
    shark::utils::stop_timer("block_vae");
    keygen_ckpt(6, 1, "vae:decode");
    if (enable_superres) {
        shark::utils::start_timer("block_superres");
        final_img = superres_apply(final_img, superres_w);
        shark::utils::stop_timer("block_superres");
        keygen_ckpt(6, 2, "superres:decode");
    }

    // Final barrier: all authenticated arithmetic and masked opens above must pass MAC
    // verification before the benchmark reveals the image. The benchmark then opens
    // final_img + r publicly while only CLIENT receives the clear one-time pad r.
    clip_batch_check("output");
    shark::utils::start_timer("reconstruct");
    auto out_plain = profiled_auth_open_authenticated_to_client(final_img.data, OP_RECONSTRUCT);
    shark::utils::stop_timer("reconstruct");
    if (party == CLIENT) {
        tone_map_output_for_display(out_plain, f);
        const bool ok = write_image_from_fixed(out_plain, final_img.H, final_img.W, final_img.C, f, "unclip_out.jpg");
        if (!minimal_terminal_output_enabled()) {
            std::cout << "[UNCLIP] output first 8 values: ";
            for (u64 i = 0; i < std::min<u64>(8, out_plain.size()); ++i) {
                std::cout << (int64_t)out_plain[i] << " ";
            }
            std::cout << std::endl;
            if (ok) {
                std::cout << "[UNCLIP] wrote " << g_unclip_last_image_path << std::endl;
            } else {
                std::cout << "[UNCLIP] failed to write unclip_out.jpg/.png/.bmp" << std::endl;
            }
        }
    } else if (!minimal_terminal_output_enabled() && party == SERVER) {
        std::cout << "[UNCLIP] output revealed to CLIENT only" << std::endl;
    }

    if (diffusion_repeat_estimation_active && estimated_repeat_captured) {
        accumulate_estimated_profile_repeat_segment(
            "diffusion_step",
            (u64)num_scheduler_steps,
            (u64)executed_diffusion_steps,
            (u64)diffusion_template_step,
            estimated_repeat_before,
            estimated_repeat_after,
            -1,
            nullptr,
            &estimated_repeat_extras_before,
            &estimated_repeat_extras_after);
    }

    return finish_unclip_benchmark(cfg, runtime_protocol_fingerprint);
}

int main(int argc, char **argv) {
    sanitize_protocol_debug_env();
    init::from_args(argc, argv);
    if (minimal_terminal_output_enabled() || keygen_progress_enabled() || comm_progress_enabled()) {
        std::cout.setf(std::ios::unitbuf);
    }
    mpspdz_32bit_compaison = false;
    print_phase_d_trace_template_summary();

    print_key_file_sizes();
    const char *seed_env = std::getenv("UNCLIP_SEED");
    if (seed_env && *seed_env) {
        current_rng_seed = std::strtoull(seed_env, nullptr, 0);
    }
    const u64 base_rng_seed = current_rng_seed;

    return run_complete_arch(base_rng_seed);
}
