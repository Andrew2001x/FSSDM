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
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace shark;
using namespace shark::protocols;

// Plaintexts are encoded as 24-bit fixed-point payloads over Z_{2^64}.
// For AuthSS, we use a 64-bit MAC key and place authenticated shares, tags,
// and integrity checks in the extended ring Z_{2^128} = Z_{2^{64+64}}.
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
    OP_EXP = 0,
    OP_SILU = 1,
    OP_SOFTMAX = 2,
    OP_RECIPROCAL = 3,
    OP_LINEAR = 4,
    OP_GROUPNORM = 5,
    OP_LAYERNORM = 6,
    OP_PROFILE_COUNT = 7
};

static std::array<OpProfileStat, OP_PROFILE_COUNT> g_op_stats{{
    {"exp", 0, 0, 0, 0.0},
    {"silu", 0, 0, 0, 0.0},
    {"softmax", 0, 0, 0, 0.0},
    {"reciprocal", 0, 0, 0, 0.0},
    {"linear", 0, 0, 0, 0.0},
    {"groupnorm", 0, 0, 0, 0.0},
    {"layernorm", 0, 0, 0, 0.0},
}};

static inline u64 current_comm_bytes() {
    if (peer == nullptr) return 0;
    return peer->bytesReceived() + peer->bytesSent();
}

static inline u64 current_rounds() {
    if (peer == nullptr) return 0;
    return peer->roundsReceived() + peer->roundsSent();
}

struct OpProfileScope {
    size_t idx = 0;
    u64 comm0 = 0;
    u64 rounds0 = 0;
    std::chrono::steady_clock::time_point t0{};
    bool active = false;

    explicit OpProfileScope(size_t idx_in)
        : idx(idx_in), comm0(current_comm_bytes()), rounds0(current_rounds()),
          t0(std::chrono::steady_clock::now()), active(party != DEALER) {}

    ~OpProfileScope() {
        if (!active) return;
        const auto t1 = std::chrono::steady_clock::now();
        auto &stat = g_op_stats[idx];
        stat.calls += 1;
        stat.comm_bytes += current_comm_bytes() - comm0;
        stat.rounds += current_rounds() - rounds0;
        stat.time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

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
    std::cout << "[PROFILE] time(ms), comm(KB) per op" << std::endl;
    print_timer("total_eval");
    print_timer("input");
    print_timer("reconstruct");
    print_avg_op_profile_line(OP_EXP);
    print_avg_op_profile_line(OP_SILU);
    print_avg_op_profile_line(OP_SOFTMAX);
    print_avg_op_profile_line(OP_RECIPROCAL);
    print_avg_op_profile_line(OP_LINEAR);
    print_avg_op_profile_line(OP_GROUPNORM);
    print_avg_op_profile_line(OP_LAYERNORM);
}

static void print_profile_components_table() {
    if (party == DEALER) return;

    struct TimerRow {
        const char *component;
        const char *timer_name;
    };

    const TimerRow timer_rows[] = {
        {"input", "input"},
        {"reconstruct", "reconstruct"},
        {"end-to-end", "total_eval"},
    };

    std::cout << "[PROFILE_TABLE] component,time_ms,comm_mb,rounds" << std::endl;
    for (size_t idx = 0; idx < OP_PROFILE_COUNT; ++idx) {
        const auto avg = avg_op_profile(idx);
        std::cout << "[PROFILE_TABLE] " << g_op_stats[idx].name
                  << "," << avg.time_ms
                  << "," << avg.comm_mb
                  << "," << avg.rounds
                  << std::endl;
    }
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
    for (const auto &row : timer_rows) {
        shark::utils::TimerStat stat{};
        shark::utils::get_timer_stat(row.timer_name, stat);
        std::cout << "[PROFILE_TABLE] " << row.component
                  << "," << stat.accumulated_time
                  << "," << (double)stat.accumulated_comm / (1024.0 * 1024.0)
                  << "," << stat.accumulated_rounds
                  << std::endl;
    }

    if (peer) {
        u64 total_comm = peer->bytesReceived() + peer->bytesSent();
        u64 total_rounds = peer->roundsReceived() + peer->roundsSent();
        std::cout << "[PROFILE_TABLE] network_total"
                  << ",-"
                  << "," << (double)total_comm / (1024.0 * 1024.0)
                  << "," << total_rounds
                  << std::endl;
    }
}

static void print_legacy_profile_lines() {
    if (party == DEALER) return;

    auto print_ms_kb = [](const char *alias, const char *timer_name) {
        shark::utils::TimerStat stat{};
        shark::utils::get_timer_stat(timer_name, stat);
        std::cout << alias << ": " << stat.accumulated_time << " ms, "
                  << (stat.accumulated_comm / 1024.0) << " KB" << std::endl;
    };

    print_ms_kb("ddpm", "total_eval");
    print_ms_kb("input", "input");
    print_ms_kb("reconstruct", "reconstruct");
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
        "share_model.down0_r0", "share_model.down0_down",
        "share_model.down1_r0",
        "share_model.mid_r0", "share_model.mid_r1",
        "share_model.up0_r0", "share_model.up0_r1", "share_model.up0_up",
        "share_model.up1_r0", "share_model.up1_r1",
    };
    return kComponents;
}

static const std::vector<std::string> &keygen_step_components() {
    static const std::vector<std::string> kComponents = {
        "time_embedding",
        "unet.conv_in",
        "unet.down0_r0", "unet.down0_down",
        "unet.down1_r0",
        "unet.mid_r0", "unet.mid_r1",
        "unet.up0_r0", "unet.up0_r1", "unet.up0_up",
        "unet.up1_r0", "unet.up1_r1",
        "unet.out",
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

static void keygen_progress_begin(u64 denoise_steps) {
    if (party != DEALER) return;
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
struct ResnetBlockWeights;
struct SelfAttnWeights;
struct DDPMWeights;
static void zero_plain(span<u64> &x);
static void init_attn(SelfAttnWeights &a, u64 dim);
static AuthShare conv_apply(u64 B, u64 H, u64 W, const AuthShare &x, const ConvWeights &w);
static AuthShare auth_from_plain_open(const span<u64> &x);
static AuthShare auth_matmul_secret(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w);
static AuthShare auth_conv_secret(u64 k, u64 padding, u64 stride, u64 inC, u64 outC, u64 H, u64 W,
                                  const AuthShare &x, const AuthShare &w);

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

static span<u64> auth_low(const AuthShare &x) {
    span<u64> out(x.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < x.share.size(); ++i) out[i] = getLow(x.share[i]);
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
        out.share[i] = a.share[i] * u128(c);
        out.tag[i] = a.tag[i] * u128(c);
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

static span<u128> open_plain_full(const span<u128> &share) {
    span<u128> out(share.size());
    if (party == DEALER) return out;
    span<u128> tmp(share.size());
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            peer->send_array(share);
        }
        #pragma omp section
        {
            peer->recv_array(tmp);
        }
    }
    #pragma omp parallel for
    for (u64 i = 0; i < share.size(); ++i) out[i] = share[i] + tmp[i];
    return out;
}

static span<u64> auth_open_authenticated(const AuthShare &x) {
    span<u64> out(x.share.size());
    if (party == DEALER) {
        #pragma omp parallel for
        for (u64 i = 0; i < x.share.size(); ++i) out[i] = getLow(x.share[i]);
        return out;
    }
    auto tmp = auth_clone(x);
    auto opened = authenticated_reconstruct_full(tmp.share, tmp.tag);
    #pragma omp parallel for
    for (u64 i = 0; i < opened.size(); ++i) out[i] = getLow(opened[i]);
    return out;
}

// Lift a 64-bit plaintext payload vector into 128-bit authenticated shares.
static AuthShare auth_from_plain_open(const span<u64> &x) {
    AuthShare out = auth_alloc(x.size());
    #pragma omp parallel for
    for (u64 i = 0; i < x.size(); ++i) {
        out.share[i] = (party == SERVER || party == DEALER) ? u128(x[i]) : u128(0);
        out.tag[i] = mac_mul_u128(u128(x[i]));
    }
    return out;
}

static AuthShare authenticated_input_from_owner(const span<u64> &x, int owner) {
    always_assert(owner == SERVER || owner == CLIENT);

    const u64 size = x.size();
    AuthShare out = auth_alloc(size);

    if (party == DEALER) {
        span<u64> r_clear_low(size);
        randomize(r_clear_low);
        span<u128> r_clear(size);
        #pragma omp parallel for
        for (u64 i = 0; i < size; ++i) r_clear[i] = u128(r_clear_low[i]);
        send_authenticated_ashare_full(r_clear);
        if (owner == SERVER) {
            server->send_array(r_clear_low);
        } else {
            client->send_array(r_clear_low);
        }
        #pragma omp parallel for
        for (u64 i = 0; i < size; ++i) {
            out.share[i] = r_clear[i];
            out.tag[i] = mac_mul_u128(r_clear[i]);
        }
        return out;
    }

    auto [r_share, r_tag] = recv_authenticated_ashare_full(size);
    span<u128> d(size);
    #pragma omp parallel for
    for (u64 i = 0; i < size; ++i) d[i] = u128(0);

    if (party == owner) {
        auto r_clear = dealer->recv_array<u64>(size);
        #pragma omp parallel for
        for (u64 i = 0; i < size; ++i) {
            d[i] = u128(x[i] - r_clear[i]);
        }
        peer->send_array(d);
    } else {
        peer->recv_array(d);
    }

    #pragma omp parallel for
    for (u64 i = 0; i < size; ++i) {
        out.share[i] = r_share[i] + ((party == SERVER) ? d[i] : u128(0));
        out.tag[i] = mac_add_u128(r_tag[i], mac_mul_u128(d[i]));
    }
    return out;
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

static AuthShare auth_matmul(u64 M, u64 K, u64 N, const AuthShare &x, const span<u64> &w) {
    auto w_auth = auth_from_plain_open(w);
    return auth_matmul_secret(M, K, N, x, w_auth);
}

static AuthShare auth_matmul_secret(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w) {
    auto tmp = matmul::call_share_secret_full(M, K, N, x.share, x.tag, w.share, w.tag);
    return AuthShare{std::move(tmp.share), std::move(tmp.tag)};
}

static AuthShare auth_conv(u64 k, u64 padding, u64 stride, u64 inC, u64 outC, u64 H, u64 W,
                           const AuthShare &x, const span<u64> &w) {
    auto w_auth = auth_from_plain_open(w);
    return auth_conv_secret(k, padding, stride, inC, outC, H, W, x, w_auth);
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

static AuthShare reauth(const AuthShare &x) {
    return auth_clone(x);
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

static AuthShare matmul_noopen(u64 M, u64 K, u64 N, const AuthShare &x, const span<u64> &w) {
    return auth_matmul(M, K, N, x, w);
}

static AuthShare matmul_secret_noopen(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w) {
    return auth_matmul_secret(M, K, N, x, w);
}

static AuthShare conv_noopen(u64 k, u64 padding, u64 stride, u64 inC, u64 outC, u64 H, u64 W,
                             const AuthShare &x, const span<u64> &w) {
    return auth_conv(k, padding, stride, inC, outC, H, W, x, w);
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
    auto masked_tmp = auth_clone(masked_auth);
    auto masked = authenticated_reconstruct_full(masked_tmp.share, masked_tmp.tag);

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
#define MATMUL_CALL(...) matmul_noopen(__VA_ARGS__)
#define CONV_CALL(...) conv_noopen(__VA_ARGS__)
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
    shark::protocols::debug_batch_check(label);
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

static bool load_fixed_list(const std::string &path, int fp_bits, std::vector<u64> &out, size_t limit = 0) {
    std::ifstream in(path);
    if (!in) return false;
    out.clear();
    std::string tok;
    while (in >> tok) {
        while (!tok.empty()) {
            char c = tok.front();
            if (c == '[' || c == ']' || c == '(' || c == ')' || c == ',' || c == ';') tok.erase(tok.begin());
            else break;
        }
        while (!tok.empty()) {
            char c = tok.back();
            if (c == '[' || c == ']' || c == '(' || c == ')' || c == ',' || c == ';') tok.pop_back();
            else break;
        }
        if (tok.empty()) continue;
        char *end = nullptr;
        double v = std::strtod(tok.c_str(), &end);
        if (end == tok.c_str()) continue;
        int64_t q = (int64_t)std::llround(v * (double)(1ULL << fp_bits));
        out.push_back((u64)q);
        if (limit && out.size() >= limit) break;
    }
    return !out.empty();
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

static AuthShare chexp(const AuthShare &x);

static double approx1_time_exp_plain(double x) {
    double xt = 2.0 * (x - (-14.0)) / (0.0 - (-14.0)) - 1.0;
    double t0 = 1.0;
    double t1 = xt;
    double ex = 0.14021878 * t0 + 0.27541278 * t1;

    double two_xt = 2.0 * xt;
    double t_nm2 = t0;
    double t_nm1 = t1;
    const double coeffs[6] = {0.22122865, 0.14934221, 0.09077360, 0.04369614, 0.02087868, 0.00996535};
    for (int i = 0; i < 6; ++i) {
        double next_t = two_xt * t_nm1 - t_nm2;
        ex += coeffs[i] * next_t;
        t_nm2 = t_nm1;
        t_nm1 = next_t;
    }
    return ex;
}

static double timestep_log_freq_step_plain(u64 half) {
    if (half == 0) return 0.0;
    return -std::log(10000.0) / (double)half;
}

static std::vector<u64> make_timestep_embedding_plain(u64 timestep, u64 dim) {
    std::vector<u64> out(dim, 0);
    u64 half = dim / 2;
    if (half == 0) return out;
    // The exponent grid is {0, step, 2*step, ...} with step < 0, so the maximum is 0.
    const double max_exponent = 0.0;
    double freq_step = timestep_log_freq_step_plain(half) - max_exponent;
    double freq_base = approx1_time_exp_plain(freq_step);
    double freq = 1.0;
    for (u64 i = 0; i < half; ++i) {
        double arg = (double)timestep * freq;
        out[i] = (u64)(int64_t)std::llround(std::sin(arg) * (double)(1ULL << f));
        out[i + half] = (u64)(int64_t)std::llround(std::cos(arg) * (double)(1ULL << f));
        freq *= freq_base;
    }
    if (dim & 1) out.back() = 0;
    return out;
}

static AuthShare make_timestep_embedding(u64 timestep, u64 dim) {
    std::vector<u64> out(dim, 0);
    u64 half = dim / 2;
    if (half == 0) return auth_from_public_vec(out);

    // Mirror the Python time-embedding stabilization by subtracting the maximum exponent.
    const double max_exponent = 0.0;
    auto freq_step = make_public_const(1, timestep_log_freq_step_plain(half) - max_exponent, f);
    if (party == DEALER) {
        (void)chexp(freq_step);
        return auth_from_public_vec(make_timestep_embedding_plain(timestep, dim));
    }
    auto freq_base_share = chexp(freq_step);
    // The timestep is public; we only reconstruct the public frequency base because
    // the benchmark has no secure sin/cos primitive.
    auto freq_base_plain = auth_open_authenticated(freq_base_share);

    const double scale = 1.0 / (double)(1ULL << f);
    double freq_base = (double)(int64_t)freq_base_plain[0] * scale;
    double freq = 1.0;
    for (u64 i = 0; i < half; ++i) {
        double arg = (double)timestep * freq;
        out[i] = (u64)(int64_t)std::llround(std::sin(arg) * (double)(1ULL << f));
        out[i + half] = (u64)(int64_t)std::llround(std::cos(arg) * (double)(1ULL << f));
        freq *= freq_base;
    }
    if (dim & 1) out.back() = 0;
    return auth_from_public_vec(out);
}

static std::vector<int> build_timesteps(u64 num_train_timesteps, u64 steps) {
    always_assert(num_train_timesteps > 0);
    always_assert(steps > 0);
    std::vector<int> out(steps);
    if (steps == 1) {
        out[0] = (int)(num_train_timesteps - 1);
        return out;
    }
    int prev = (int)num_train_timesteps;
    for (u64 i = 0; i < steps; ++i) {
        double frac = (double)(steps - 1 - i) / (double)(steps - 1);
        int t = (int)std::llround(frac * (double)(num_train_timesteps - 1));
        if (t >= prev) t = prev - 1;
        if (t < 0) t = 0;
        out[i] = t;
        prev = t;
    }
    return out;
}

static std::vector<double> build_alphas_cumprod_linear(u64 num_timesteps) {
    std::vector<double> out(num_timesteps);
    const double beta_start = 1e-4;
    const double beta_end = 2e-2;
    double acc = 1.0;
    for (u64 i = 0; i < num_timesteps; ++i) {
        double t = (num_timesteps == 1) ? 0.0 : (double)i / (double)(num_timesteps - 1);
        double beta = beta_start + (beta_end - beta_start) * t;
        acc *= (1.0 - beta);
        out[i] = acc;
    }
    return out;
}

static std::vector<double> build_alphas_cumprod_scaled_linear(u64 num_timesteps) {
    std::vector<double> out(num_timesteps);
    const double beta_start = std::sqrt(1e-4);
    const double beta_end = std::sqrt(2e-2);
    double acc = 1.0;
    for (u64 i = 0; i < num_timesteps; ++i) {
        double t = (num_timesteps == 1) ? 0.0 : (double)i / (double)(num_timesteps - 1);
        double beta = beta_start + (beta_end - beta_start) * t;
        beta = beta * beta;
        acc *= (1.0 - beta);
        out[i] = acc;
    }
    return out;
}

static std::vector<double> build_alphas_cumprod_cos(u64 num_timesteps) {
    std::vector<double> out(num_timesteps);
    auto alpha_bar = [](double t) {
        double x = (t + 0.008) / 1.008;
        return std::pow(std::cos(x * 3.14159265358979323846 * 0.5), 2.0);
    };
    double acc = 1.0;
    for (u64 i = 0; i < num_timesteps; ++i) {
        double t1 = (double)i / (double)num_timesteps;
        double t2 = (double)(i + 1) / (double)num_timesteps;
        double beta = std::min(0.999, 1.0 - alpha_bar(t2) / alpha_bar(t1));
        acc *= (1.0 - beta);
        out[i] = acc;
    }
    return out;
}

static std::vector<double> build_alphas_cumprod(const std::string &schedule, u64 num_timesteps) {
    if (schedule == "scaled_linear") return build_alphas_cumprod_scaled_linear(num_timesteps);
    if (schedule == "squaredcos_cap_v2") return build_alphas_cumprod_cos(num_timesteps);
    return build_alphas_cumprod_linear(num_timesteps);
}

static AuthShare mul_qf(const AuthShare &a, const AuthShare &b) {
    return LRS_CALL(MUL_CALL(a, b), f);
}

static AuthShare scale_public(const AuthShare &x, double v) {
    int64_t q = (int64_t)std::llround(v * (double)(1ULL << f));
    return LRS_CALL(auth_mul_const(x, (u64)q), f);
}

static AuthShare ss_reciprocal(const AuthShare &x) {
    OpProfileScope profile(OP_RECIPROCAL);
    shark::utils::start_timer("reciprocal");

    const u64 n = x.share.size();
    const double eps = 1.0 / 256.0;
    const int norm_iters = 8;

    auto one = make_public_const(n, 1.0, f);
    auto neg_one = make_public_const(n, -1.0, f);
    auto init_guess_bias = make_public_const(n, 2.9142, f);
    auto eps_const = make_public_const(n, eps, f);

    // Match reciprocal_goldschmidt_real():
    // sign = where(x >= 0, 1, -1), x_abs = abs(x)
    auto ge_zero = cmp_ge_public(x, 0.0);
    AuthShare sign = auth_clone(neg_one);
    update_if(sign, ge_zero, one);

    AuthShare x_abs = neg_span(x);
    update_if(x_abs, ge_zero, x);

    // Softmax denominator is strictly positive; keep a small clamp for safety.
    auto ge_eps = cmp_ge_public(x_abs, eps);
    AuthShare x_safe = auth_clone(eps_const);
    update_if(x_safe, ge_eps, x_abs);

    // Secure range normalization equivalent to:
    //   m = floor(log2(x_abs))
    //   factor = 2^{-m}
    //   c = x_abs * factor in [0.5, 1)
    AuthShare factor = auth_clone(one);
    AuthShare c = auth_clone(x_safe);
    for (int iter = 0; iter < norm_iters; ++iter) {
        auto lt_half = cmp_le_public(c, 0.5 - eps);
        auto c_up = scale_public(c, 2.0);
        auto factor_up = scale_public(factor, 2.0);
        update_if(c, lt_half, c_up);
        update_if(factor, lt_half, factor_up);

        auto ge_one = cmp_ge_public(c, 1.0);
        auto c_down = scale_public(c, 0.5);
        auto factor_down = scale_public(factor, 0.5);
        update_if(c, ge_one, c_down);
        update_if(factor, ge_one, factor_down);
    }

    // Initial guess: r = 2.9142 - 2c
    auto r = ADD_CALL(init_guess_bias, neg_span(scale_public(c, 2.0)));
    auto e = ADD_CALL(one, neg_span(mul_qf(c, r)));

    for (int iter = 0; iter < 2; ++iter) {
        r = mul_qf(r, ADD_CALL(one, e));
        e = mul_qf(e, e);
    }

    auto out = mul_qf(mul_qf(sign, r), factor);
    shark::utils::stop_timer("reciprocal");
    return out;
}

static AuthShare ss_rsqrt(const AuthShare &x) {
    auto y = make_public_const(x.share.size(), 1.0, f);
    auto half = make_public_const(x.share.size(), 0.5, f);
    auto three_halves = make_public_const(x.share.size(), 1.5, f);
    for (int iter = 0; iter < 3; ++iter) {
        auto y2 = mul_qf(y, y);
        auto xy2 = mul_qf(x, y2);
        auto term = ADD_CALL(three_halves, neg_span(scale_public(xy2, 0.5)));
        y = mul_qf(y, term);
    }
    return y;
}

static AuthShare silu_apply(const AuthShare &x) {
    OpProfileScope profile(OP_SILU);
    shark::utils::start_timer("silu");
    auto x2 = mul_qf(x, x);
    auto x4 = mul_qf(x2, x2);
    auto x6 = mul_qf(x2, x4);

    auto a0 = make_public_const(x.share.size(), -0.52212664, f);
    auto segA = a0;
    segA = ADD_CALL(segA, scale_public(x, -0.16910363));
    segA = ADD_CALL(segA, scale_public(x2, -0.01420163));

    auto b0 = make_public_const(x.share.size(), 0.03453821, f);
    auto segB = b0;
    segB = ADD_CALL(segB, scale_public(x, 0.49379432));
    segB = ADD_CALL(segB, scale_public(x2, 0.19784596));
    segB = ADD_CALL(segB, scale_public(x4, -0.00602401));
    segB = ADD_CALL(segB, scale_public(x6, 0.00008032));

    auto ge_neg6 = cmp_ge_public(x, -6.0);
    auto ge_neg2 = cmp_ge_public(x, -2.0);
    auto gt_6 = cmp_gt_public(x, 6.0);

    auto ret = make_public_const(x.share.size(), 0.0, f);
    update_if(ret, ge_neg6, segA);
    update_if(ret, ge_neg2, segB);
    update_if(ret, gt_6, x);
    shark::utils::stop_timer("silu");
    return ret;
}

static AuthShare time_silu_apply(const AuthShare &x) {
    // Match ApproxSiLUApprox1 in the Python time_mlp path.
    return silu_apply(x);
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

static AuthShare chexp(const AuthShare &x) {
    OpProfileScope profile(OP_EXP);
    shark::utils::start_timer("exp");
    u64 n = x.share.size();
    auto xt = ADD_CALL(scale_public(x, 1.0 / 7.0), make_public_const(n, 1.0, f));
    auto t0 = make_public_const(n, 1.0, f);
    auto t1 = xt;
    auto ex = ADD_CALL(make_public_const(n, 0.14021878, f), scale_public(t1, 0.27541278));

    auto two_xt = ADD_CALL(xt, xt);
    auto t_nm2 = t0;
    auto t_nm1 = t1;
    const double coeffs[6] = {0.22122865, 0.14934221, 0.09077360, 0.04369614, 0.02087868, 0.00996535};
    for (int i = 0; i < 6; ++i) {
        auto next_t = ADD_CALL(mul_qf(two_xt, t_nm1), neg_span(t_nm2));
        ex = ADD_CALL(ex, scale_public(next_t, coeffs[i]));
        t_nm2 = t_nm1;
        t_nm1 = next_t;
    }
    shark::utils::stop_timer("exp");
    return ex;
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

    auto exp_vals = chexp(shifted);
    auto le_neg14 = cmp_le_public(shifted, -14.0);
    exp_vals = ADD_CALL(exp_vals, SELECT_CALL(le_neg14, neg_span(exp_vals)));

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

static AuthShare linear_apply(u64 M, u64 K, u64 N, const AuthShare &x, const span<u64> &w, const AuthShare *b = nullptr) {
    OpProfileScope profile(OP_LINEAR);
    shark::utils::start_timer("linear");
    auto y = LRS_CALL(MATMUL_CALL(M, K, N, x, w), f);
    if (b && b->share.size() == N) y = ADD_CALL(y, broadcast_row_vector(M, N, *b));
    shark::utils::stop_timer("linear");
    return y;
}

static AuthShare linear_apply(u64 M, u64 K, u64 N, const AuthShare &x, const AuthShare &w, const AuthShare *b = nullptr) {
    OpProfileScope profile(OP_LINEAR);
    shark::utils::start_timer("linear");
    auto y = LRS_CALL(matmul_secret_noopen(M, K, N, x, w), f);
    if (b && b->share.size() == N) y = ADD_CALL(y, broadcast_row_vector(M, N, *b));
    shark::utils::stop_timer("linear");
    return y;
}

static AuthShare conv_apply(u64 B, u64 H, u64 W, u64 inC, u64 outC, u64 k, u64 stride, u64 padding,
                            const AuthShare &x, const span<u64> &w, const AuthShare *b = nullptr) {
    auto y = LRS_CALL(CONV_CALL(k, padding, stride, inC, outC, H, W, x, w), f);
    if (b && b->share.size() == outC) {
        u64 outH = (H - k + 2 * padding) / stride + 1;
        u64 outW = (W - k + 2 * padding) / stride + 1;
        y = ADD_CALL(y, broadcast_channels_nhwc(B, outH, outW, outC, *b));
    }
    return y;
}

static AuthShare conv_apply(u64 B, u64 H, u64 W, u64 inC, u64 outC, u64 k, u64 stride, u64 padding,
                            const AuthShare &x, const AuthShare &w, const AuthShare *b = nullptr) {
    auto y = LRS_CALL(conv_secret_noopen(k, padding, stride, inC, outC, H, W, x, w), f);
    if (b && b->share.size() == outC) {
        u64 outH = (H - k + 2 * padding) / stride + 1;
        u64 outW = (W - k + 2 * padding) / stride + 1;
        y = ADD_CALL(y, broadcast_channels_nhwc(B, outH, outW, outC, *b));
    }
    return y;
}

static AuthShare flatten_spatial(u64 B, u64 H, u64 W, u64 C, const AuthShare &x) {
    always_assert(x.share.size() == B * H * W * C);
    return auth_clone(x);
}

static AuthShare unflatten_spatial(u64 B, u64 H, u64 W, u64 C, const AuthShare &x) {
    always_assert(x.share.size() == B * H * W * C);
    return auth_clone(x);
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

static AuthShare upsample_zero_insert2x(u64 B, u64 H, u64 W, u64 C, const AuthShare &x) {
    always_assert(x.share.size() == B * H * W * C);
    AuthShare out = auth_alloc(B * (H * 2) * (W * 2) * C);
    #pragma omp parallel for collapse(4)
    for (u64 n = 0; n < B; ++n) {
        for (u64 h = 0; h < H * 2; ++h) {
            for (u64 widx = 0; widx < W * 2; ++widx) {
                for (u64 c = 0; c < C; ++c) {
                    u64 dst = idx4(n, h, widx, c, H * 2, W * 2, C);
                    if (((h & 1) == 0) && ((widx & 1) == 0)) {
                        u64 src = idx4(n, h / 2, widx / 2, c, H, W, C);
                        out.share[dst] = x.share[src];
                        out.tag[dst] = x.tag[src];
                    } else {
                        out.share[dst] = 0;
                        out.tag[dst] = 0;
                    }
                }
            }
        }
    }
    return out;
}

static AuthShare conv_transpose2x_apply(u64 B, u64 H, u64 W, const AuthShare &x, const ConvWeights &w) {
    always_assert(B * H * W > 0);
    always_assert(x.share.size() % (B * H * W) == 0);
    const u64 inC = x.share.size() / (B * H * W);
    auto expanded = upsample_zero_insert2x(B, H, W, inC, x);
    return conv_apply(B, H * 2, W * 2, expanded, w);
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

static AuthShare rsqrt_seed_interval_1_2(const AuthShare &u, int fp_bits, int poly_order = kNormPolyOrder) {
    auto one = make_public_const(u.share.size(), 1.0, fp_bits);
    auto z = ADD_CALL(u, neg_span(one));
    auto z2 = mul_qf(z, z);

    auto out = ADD_CALL(one, neg_span(scale_public(z, 0.5)));
    out = ADD_CALL(out, scale_public(z2, 0.375));

    if (poly_order >= 3) {
        auto z3 = mul_qf(z2, z);
        out = ADD_CALL(out, neg_span(scale_public(z3, 0.3125)));
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
        r = mul_qf(scale_public(r, 0.5), corr);
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
    auto inv_std = scale_public(r, rsqrt_scale_from_shift(fp_bits));
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

    auto mean = scale_public(group_sum, 1.0 / (double)elems_per_group);

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

    auto var = scale_public(var_sum, 1.0 / (double)elems_per_group);
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

struct SelfAttnWeights {
    u64 dim = 0;
    LinearWeights q;
    LinearWeights k;
    LinearWeights v;
    LinearWeights proj;
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
    bool class_cond = false;
    u64 inC = 1;
    u64 outC = 1;
    u64 image_h = 28;
    u64 image_w = 28;
    u64 time_in = 64;
    u64 temb = 256;
    u64 c0 = 64;
    u64 c1 = 128;
    u64 norm_groups = 32;

    LinearWeights time1;
    LinearWeights time2;
    LinearWeights class_emb;

    ConvWeights conv_in;
    ConvWeights conv_out;

    ResnetBlockWeights down0_r0;
    ConvWeights down0_down;

    ResnetBlockWeights down1_r0;

    ResnetBlockWeights mid_r0;
    ResnetBlockWeights mid_r1;

    ResnetBlockWeights up0_r0;
    ResnetBlockWeights up0_r1;
    ConvWeights up0_up;

    ResnetBlockWeights up1_r0;
    ResnetBlockWeights up1_r1;
};

static AuthShare linear_apply(u64 M, const AuthShare &x, const LinearWeights &w) {
    always_assert(w.w_auth.share.size() == w.in_dim * w.out_dim);
    return linear_apply(M, w.in_dim, w.out_dim, x, w.w_auth, &w.b);
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
    init_linear(a.q, dim, dim);
    init_linear(a.k, dim, dim);
    init_linear(a.v, dim, dim);
    init_linear(a.proj, dim, dim);
}

static void fill_attn(SelfAttnWeights &a);
static void share_attn(SelfAttnWeights &a);

static void fill_linear(LinearWeights &w) {
    if (party != SERVER) return;
    double bound = std::sqrt(6.0 / (double)std::max<u64>(1, w.in_dim));
    for (u64 i = 0; i < w.w.size(); ++i) w.w[i] = qrand_uniform_symmetric(f, bound);
    for (u64 i = 0; i < w.b_plain.size(); ++i) w.b_plain[i] = 0;
}

static void fill_conv(ConvWeights &w) {
    if (party != SERVER) return;
    double fan_in = (double)(w.k * w.k * w.inC);
    double bound = std::sqrt(6.0 / std::max(1.0, fan_in));
    for (u64 i = 0; i < w.w.size(); ++i) w.w[i] = qrand_uniform_symmetric(f, bound);
    for (u64 i = 0; i < w.b_plain.size(); ++i) w.b_plain[i] = 0;
}

static void fill_resnet(ResnetBlockWeights &rb) {
    fill_conv(rb.conv1);
    fill_conv(rb.conv2);
    fill_linear(rb.temb_proj);
    if (rb.use_shortcut) fill_conv(rb.shortcut);
    if (rb.use_attention) fill_attn(rb.attn);
}

static void fill_attn(SelfAttnWeights &a) {
    fill_linear(a.q);
    fill_linear(a.k);
    fill_linear(a.v);
    fill_linear(a.proj);
}

static void init_model(DDPMWeights &m, u64 inC, bool class_cond, u64 image_hw, u64 c0, u64 c1, u64 temb) {
    always_assert(image_hw >= 2);
    always_assert((image_hw % 2) == 0);
    always_assert(c0 >= 1);
    always_assert(c1 >= 1);
    always_assert(temb >= 1);

    m.class_cond = class_cond;
    m.inC = inC;
    m.outC = inC;
    m.image_h = image_hw;
    m.image_w = image_hw;
    m.time_in = c0;
    m.temb = temb;
    m.c0 = c0;
    m.c1 = c1;
    m.norm_groups = 32;

    init_linear(m.time1, m.time_in, m.temb);
    init_linear(m.time2, m.temb, m.temb);
    if (m.class_cond) init_linear(m.class_emb, 10, m.temb);

    init_conv(m.conv_in, 3, inC, m.c0, 1, 1);
    init_conv(m.conv_out, 3, m.c0, m.outC, 1, 1);

    init_resnet(m.down0_r0, m.c0, m.c0, m.temb, false);
    init_conv(m.down0_down, 3, m.c0, m.c0, 2, 1);

    init_resnet(m.down1_r0, m.c0, m.c1, m.temb, true);

    init_resnet(m.mid_r0, m.c1, m.c1, m.temb, true);
    init_resnet(m.mid_r1, m.c1, m.c1, m.temb, false);

    init_resnet(m.up0_r0, m.c1 + m.c1, m.c1, m.temb, true);
    init_resnet(m.up0_r1, m.c1 + m.c0, m.c1, m.temb, true);
    init_conv(m.up0_up, 3, m.c1, m.c1, 1, 1);

    init_resnet(m.up1_r0, m.c1 + m.c0, m.c0, m.temb, false);
    init_resnet(m.up1_r1, m.c0 + m.c0, m.c0, m.temb, false);
}

static void fill_model(DDPMWeights &m) {
    fill_linear(m.time1);
    fill_linear(m.time2);
    if (m.class_cond) fill_linear(m.class_emb);
    fill_conv(m.conv_in);
    fill_conv(m.conv_out);
    fill_resnet(m.down0_r0);
    fill_conv(m.down0_down);
    fill_resnet(m.down1_r0);
    fill_resnet(m.mid_r0);
    fill_resnet(m.mid_r1);
    fill_resnet(m.up0_r0);
    fill_resnet(m.up0_r1);
    fill_conv(m.up0_up);
    fill_resnet(m.up1_r0);
    fill_resnet(m.up1_r1);
}

static void share_linear(LinearWeights &w) {
    w.w_auth = authenticated_input_from_owner(w.w, SERVER);
    w.b = authenticated_input_from_owner(w.b_plain, SERVER);
}

static void share_conv(ConvWeights &w) {
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
    share_linear(a.q);
    share_linear(a.k);
    share_linear(a.v);
    share_linear(a.proj);
}

static void share_model(DDPMWeights &m) {
    keygen_progress_tick("share_model.time1");
    share_linear(m.time1);
    keygen_progress_tick("share_model.time2");
    share_linear(m.time2);
    if (m.class_cond) {
        keygen_progress_tick("share_model.class_emb");
        share_linear(m.class_emb);
    }
    keygen_progress_tick("share_model.conv_in");
    share_conv(m.conv_in);
    keygen_progress_tick("share_model.conv_out");
    share_conv(m.conv_out);
    keygen_progress_tick("share_model.down0_r0");
    share_resnet(m.down0_r0);
    keygen_progress_tick("share_model.down0_down");
    share_conv(m.down0_down);
    keygen_progress_tick("share_model.down1_r0");
    share_resnet(m.down1_r0);
    keygen_progress_tick("share_model.mid_r0");
    share_resnet(m.mid_r0);
    keygen_progress_tick("share_model.mid_r1");
    share_resnet(m.mid_r1);
    keygen_progress_tick("share_model.up0_r0");
    share_resnet(m.up0_r0);
    keygen_progress_tick("share_model.up0_r1");
    share_resnet(m.up0_r1);
    keygen_progress_tick("share_model.up0_up");
    share_conv(m.up0_up);
    keygen_progress_tick("share_model.up1_r0");
    share_resnet(m.up1_r0);
    keygen_progress_tick("share_model.up1_r1");
    share_resnet(m.up1_r1);
}

static AuthShare self_attn_rows(u64 rows, u64 dim, const AuthShare &x, const SelfAttnWeights &w) {
    auto q = linear_apply(rows, x, w.q);
    auto k = linear_apply(rows, x, w.k);
    auto v = linear_apply(rows, x, w.v);
    auto kt = transpose_auth_matrix(k, rows, dim);
    auto scores = LRS_CALL(matmul_secret_noopen(rows, dim, rows, q, kt), f);
    scores = scale_public(scores, 1.0 / std::sqrt((double)dim));
    auto probs = softmax_cheb(rows, rows, scores);
    auto ctx = LRS_CALL(matmul_secret_noopen(rows, rows, dim, probs, v), f);
    return linear_apply(rows, ctx, w.proj);
}

static AuthShare self_attn_spatial(u64 B, u64 H, u64 W, u64 C, const AuthShare &x, const SelfAttnWeights &w) {
    always_assert(B == 1);
    auto flat = flatten_spatial(B, H, W, C, x);
    auto out = self_attn_rows(H * W, C, flat, w);
    return unflatten_spatial(B, H, W, C, out);
}

static AuthShare resnet_forward(u64 B, u64 H, u64 W, const AuthShare &x, const AuthShare &temb, const ResnetBlockWeights &rb) {
    always_assert(x.share.size() == B * H * W * rb.inC);
    auto h = groupnorm_apply(B, H, W, rb.inC, x, 2);
    h = silu_apply(h);
    h = conv_apply(B, H, W, h, rb.conv1);

    auto temb_h = silu_apply(temb);
    auto temb_proj = linear_apply(B, temb_h, rb.temb_proj);
    h = ADD_CALL(h, broadcast_batch_vector_nhwc(B, H, W, rb.outC, temb_proj));

    h = groupnorm_apply(B, H, W, rb.outC, h, 2);
    h = silu_apply(h);
    h = conv_apply(B, H, W, h, rb.conv2);

    AuthShare shortcut = rb.use_shortcut
        ? conv_apply(B, H, W, x, rb.shortcut)
        : auth_clone(x);
    h = ADD_CALL(h, shortcut);
    if (rb.use_attention) {
        h = ADD_CALL(h, self_attn_spatial(B, H, W, rb.outC, groupnorm_apply(B, H, W, rb.outC, h, 2), rb.attn));
    }
    return h;
}

static AuthShare build_class_onehot(int digit) {
    span<u64> onehot(10);
    zero_plain(onehot);
    if (party == CLIENT && digit >= 0 && digit < 10) onehot[(u64)digit] = (1ULL << f);
    return authenticated_input_from_owner(onehot, CLIENT);
}

static std::vector<u64> make_synthetic_image(u64 H, u64 W, int digit) {
    std::vector<u64> img(H * W, (u64)(int64_t)std::llround(-1.0 * (double)(1ULL << f)));
    u64 on = (u64)(int64_t)std::llround(1.0 * (double)(1ULL << f));
    auto draw_rect = [&](int y0, int y1, int x0, int x1) {
        for (int y = std::max(0, y0); y < std::min((int)H, y1); ++y) {
            for (int x = std::max(0, x0); x < std::min((int)W, x1); ++x) {
                img[(u64)y * W + (u64)x] = on;
            }
        }
    };

    if (H == 28 && W == 28) {
        static const bool segs[10][7] = {
            {1,1,1,1,1,1,0}, {0,1,1,0,0,0,0}, {1,1,0,1,1,0,1}, {1,1,1,1,0,0,1}, {0,1,1,0,0,1,1},
            {1,0,1,1,0,1,1}, {1,0,1,1,1,1,1}, {1,1,1,0,0,0,0}, {1,1,1,1,1,1,1}, {1,1,1,1,0,1,1}
        };
        if (digit < 0 || digit > 9) digit = 0;
        if (segs[digit][0]) draw_rect(4, 7, 8, 24);
        if (segs[digit][1]) draw_rect(7, 16, 23, 26);
        if (segs[digit][2]) draw_rect(16, 25, 23, 26);
        if (segs[digit][3]) draw_rect(25, 28, 8, 24);
        if (segs[digit][4]) draw_rect(16, 25, 6, 9);
        if (segs[digit][5]) draw_rect(7, 16, 6, 9);
        if (segs[digit][6]) draw_rect(14, 18, 8, 24);
        return img;
    }

    int box_h = std::max<int>(1, (int)H / 2);
    int box_w = std::max<int>(1, (int)W / 2);
    int y0 = ((int)H - box_h) / 2;
    int x0 = ((int)W - box_w) / 2;
    int y_shift = (digit % 3) - 1;
    int x_shift = ((digit / 3) % 3) - 1;
    draw_rect(y0 + y_shift, y0 + y_shift + box_h, x0 + x_shift, x0 + x_shift + box_w);
    return img;
}

static AuthShare input_client_image(const std::string &path, int digit, u64 image_h, u64 image_w) {
    (void)digit;
    std::vector<u64> vals;
    const size_t image_size = (size_t)image_h * (size_t)image_w;
    if (party == CLIENT) {
        if (!path.empty()) load_fixed_list(path, f, vals, image_size);
        if (vals.size() != image_size) {
            vals.assign(image_size, 0);
            for (size_t i = 0; i < image_size; ++i) vals[i] = qrand_uniform_symmetric(f, 1.0);
        }
    } else {
        vals.assign(image_size, 0);
    }
    span<u64> img((u64)vals.size());
    for (u64 i = 0; i < vals.size(); ++i) img[i] = vals[i];
    return authenticated_input_from_owner(img, CLIENT);
}

static AuthShare input_client_noise(u64 size, double stddev = 1.0) {
    span<u64> noise(size);
    zero_plain(noise);
    if (party == CLIENT && stddev > 0.0) {
        for (u64 i = 0; i < size; ++i) noise[i] = qrand_normal(f, stddev);
    }
    return authenticated_input_from_owner(noise, CLIENT);
}

static AuthShare build_time_embedding(const DDPMWeights &m, u64 timestep) {
    auto t0 = make_timestep_embedding(timestep, m.time_in);
    auto t1 = time_silu_apply(linear_apply(1, t0, m.time1));
    return linear_apply(1, t1, m.time2);
}

static AuthShare add_class_embedding(const DDPMWeights &m, const AuthShare &temb, const AuthShare *class_onehot) {
    if (!m.class_cond || class_onehot == nullptr) return auth_clone(temb);
    auto cls = linear_apply(1, *class_onehot, m.class_emb);
    return ADD_CALL(temb, cls);
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
    h = resnet_forward(B, H, W, h, temb, m.down0_r0);
    skips.push_back(auth_clone(h));
    keygen_progress_tick("unet.down0_down");
    h = conv_apply(B, H, W, h, m.down0_down);
    H = conv_out_dim(H, m.down0_down.k, m.down0_down.stride, m.down0_down.padding);
    W = conv_out_dim(W, m.down0_down.k, m.down0_down.stride, m.down0_down.padding);
    skips.push_back(auth_clone(h));

    keygen_progress_tick("unet.down1_r0");
    h = resnet_forward(B, H, W, h, temb, m.down1_r0);
    skips.push_back(auth_clone(h));

    keygen_progress_tick("unet.mid_r0");
    h = resnet_forward(B, H, W, h, temb, m.mid_r0);
    keygen_progress_tick("unet.mid_r1");
    h = resnet_forward(B, H, W, h, temb, m.mid_r1);

    keygen_progress_tick("unet.up0_r0");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c1, h, m.c1, pop_skip()), temb, m.up0_r0);
    keygen_progress_tick("unet.up0_r1");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c1, h, m.c0, pop_skip()), temb, m.up0_r1);
    H *= 2;
    W *= 2;
    keygen_progress_tick("unet.up0_up");
    h = conv_transpose2x_apply(B, H / 2, W / 2, h, m.up0_up);

    keygen_progress_tick("unet.up1_r0");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c1, h, m.c0, pop_skip()), temb, m.up1_r0);
    keygen_progress_tick("unet.up1_r1");
    h = resnet_forward(B, H, W, concat_channels(B, H, W, m.c0, h, m.c0, pop_skip()), temb, m.up1_r1);

    keygen_progress_tick("unet.out");
    h = silu_apply(groupnorm_apply(B, H, W, m.c0, h, m.norm_groups));
    return conv_apply(B, H, W, h, m.conv_out);
}

static AuthShare add_noise_ddpm(const AuthShare &x0, const AuthShare &noise, double alpha_t) {
    return ADD_CALL(scale_public(x0, std::sqrt(alpha_t)),
                    scale_public(noise, std::sqrt(std::max(0.0, 1.0 - alpha_t))));
}

static AuthShare scheduler_step(const AuthShare &sample, const AuthShare &eps, double alpha_t, double alpha_prev) {
    double sqrt_alpha_t = std::sqrt(std::max(1e-12, alpha_t));
    double sqrt_one_minus_t = std::sqrt(std::max(0.0, 1.0 - alpha_t));
    auto pred_x0 = ADD_CALL(scale_public(sample, 1.0 / sqrt_alpha_t),
                            scale_public(eps, -sqrt_one_minus_t / sqrt_alpha_t));
    if (alpha_prev < 0.0) return pred_x0;
    return ADD_CALL(scale_public(pred_x0, std::sqrt(std::max(0.0, alpha_prev))),
                    scale_public(eps, std::sqrt(std::max(0.0, 1.0 - alpha_prev))));
}

int main(int argc, char **argv) {
    init::from_args(argc, argv);
    if (party == DEALER) {
        std::cout.setf(std::ios::unitbuf);
    }
    mpspdz_32bit_compaison = false;

    if (has_flag(argc, argv, "--help")) {
        if (party == CLIENT) {
            std::cout
                << "Usage: benchmark-ddpm [--digit 0] [--steps 5]\n"
                << "                      [--num_timesteps 1000] [--beta_schedule linear|scaled_linear|squaredcos_cap_v2]\n"
                << "                      [--strength 0.35] [--image cond.txt] [--out ddpm_out.jpg]\n"
                << "                      [--cond_out ddpm_cond.jpg] [--seed 0]\n"
                << "                      [--tiny] [--image_hw 28] [--channels 64] [--base_ch 64] [--mid_ch 128] [--temb 256]\n";
        }
        finalize::call();
        return 0;
    }

    const int digit = (int)get_arg_u64(argc, argv, "--digit", 0);
    const bool tiny = has_flag(argc, argv, "--tiny");
    u64 image_hw = 28;
    u64 base_ch = 64;
    u64 mid_ch = 128;
    u64 temb_dim = 256;
    u64 steps = 5;
    u64 num_timesteps = 1000;
    const std::string beta_schedule = get_arg_string(argc, argv, "--beta_schedule", "linear");
    double strength_in = 0.35;
    const std::string image_path = get_arg_string(argc, argv, "--image", "");
    const std::string out_path = get_arg_string(argc, argv, "--out", "ddpm_out.jpg");
    const std::string cond_out_path = get_arg_string(argc, argv, "--cond_out", "ddpm_cond.jpg");
    (void)tiny;

    if (find_arg(argc, argv, "--image_hw") >= 0) image_hw = get_arg_u64(argc, argv, "--image_hw", image_hw);
    if (find_arg(argc, argv, "--channels") >= 0) {
        u64 channels = get_arg_u64(argc, argv, "--channels", base_ch);
        base_ch = channels;
        mid_ch = channels * 2;
        temb_dim = channels * 4;
    }
    if (find_arg(argc, argv, "--base_ch") >= 0) base_ch = get_arg_u64(argc, argv, "--base_ch", base_ch);
    if (find_arg(argc, argv, "--mid_ch") >= 0) mid_ch = get_arg_u64(argc, argv, "--mid_ch", mid_ch);
    if (find_arg(argc, argv, "--temb") >= 0) temb_dim = get_arg_u64(argc, argv, "--temb", temb_dim);
    if (find_arg(argc, argv, "--steps") >= 0) steps = get_arg_u64(argc, argv, "--steps", steps);
    if (find_arg(argc, argv, "--num_timesteps") >= 0) num_timesteps = get_arg_u64(argc, argv, "--num_timesteps", num_timesteps);
    if (find_arg(argc, argv, "--strength") >= 0) strength_in = get_arg_double(argc, argv, "--strength", strength_in);

    always_assert(image_hw >= 2);
    always_assert((image_hw % 2) == 0);
    always_assert(base_ch >= 1);
    always_assert(mid_ch >= 1);
    always_assert(temb_dim >= 1);
    always_assert(steps >= 1);
    always_assert(num_timesteps >= 1);

    current_rng_seed = get_arg_u64(argc, argv, "--seed", 0);

    auto alphas_cumprod = build_alphas_cumprod(beta_schedule, num_timesteps);
    auto timesteps = build_timesteps(num_timesteps, steps);
    double strength = std::max(0.0, std::min(1.0, strength_in));
    u64 start_idx = (u64)std::llround(strength * (double)(steps - 1));
    if (start_idx >= steps) start_idx = steps - 1;
    u64 denoise_steps = (u64)timesteps.size() - start_idx;
    keygen_progress_begin(denoise_steps);

    if (party != DEALER) shark::utils::start_timer("total_eval");

    DDPMWeights model;
    init_model(model, 2, false, image_hw, base_ch, mid_ch, temb_dim);
    fill_model(model);
    share_model(model);

    if (party == CLIENT) {
        std::cout << "[ddpm] config image_hw=" << image_hw
                  << " base_ch=" << base_ch
                  << " mid_ch=" << mid_ch
                  << " temb=" << temb_dim
                  << " steps=" << steps
                  << " num_timesteps=" << num_timesteps
                  << " strength=" << strength
                  << std::endl;
    }

    if (party != DEALER) shark::utils::start_timer("input");
    keygen_progress_tick("input.cond_image");
    AuthShare cond_img = input_client_image(image_path, digit, image_hw, image_hw);
    if (party != DEALER) clip_batch_check("ddpm:after_cond_img");

    keygen_progress_tick("input.init_noise");
    AuthShare noise = input_client_noise(image_hw * image_hw, 1.0);
    if (party != DEALER) clip_batch_check("ddpm:after_noise");
    const double input_alpha = alphas_cumprod[(u64)timesteps[start_idx]];
    AuthShare sample_x0 = scale_public(cond_img, std::sqrt(input_alpha));
    if (party != DEALER) clip_batch_check("ddpm:after_scale_x0");
    AuthShare sample_noise = scale_public(noise, std::sqrt(std::max(0.0, 1.0 - input_alpha)));
    if (party != DEALER) clip_batch_check("ddpm:after_scale_noise");
    AuthShare sample = ADD_CALL(sample_x0, sample_noise);
    if (party != DEALER) clip_batch_check("ddpm:after_add_noise");
    if (party != DEALER) shark::utils::stop_timer("input");
    if (party != DEALER) clip_batch_check("ddpm:after_input");

    for (u64 i = start_idx; i < timesteps.size(); ++i) {
        int t = timesteps[i];
        keygen_progress_set_step(i - start_idx + 1, denoise_steps, t);
        keygen_progress_tick("time_embedding");
        auto temb = build_time_embedding(model, (u64)t);
        if (party != DEALER) clip_batch_check("ddpm:after_temb");
        AuthShare model_in = concat_channels(1, image_hw, image_hw, 1, sample, 1, cond_img);
        auto eps_full = unet_forward(model, model_in, temb);
        if (party != DEALER) clip_batch_check("ddpm:after_unet");
        auto eps = select_channel(1, image_hw, image_hw, model.outC, eps_full, 0);
        double alpha_t = alphas_cumprod[(u64)t];
        double alpha_prev = (i + 1 < timesteps.size()) ? alphas_cumprod[(u64)timesteps[i + 1]] : -1.0;
        keygen_progress_tick("scheduler_step");
        sample = scheduler_step(sample, eps, alpha_t, alpha_prev);
        if (party != DEALER) clip_batch_check("ddpm:after_scheduler");
        keygen_progress_clear_step();

        if (party == CLIENT) {
            std::cout << "[ddpm] step " << (i - start_idx + 1) << "/" << (timesteps.size() - start_idx)
                      << " t=" << t << std::endl;
        }
    }

    if (party != DEALER) {
        clip_batch_check("ddpm:final");
        shark::utils::start_timer("reconstruct");
        auto out_plain = auth_open_authenticated(sample);
        auto cond_plain = auth_open_authenticated(cond_img);
        shark::utils::stop_timer("reconstruct");
        if (party == CLIENT) {
            bool ok = write_jpg_from_fixed(out_plain, image_hw, image_hw, 1, f, out_path.c_str());
            std::cout << (ok ? "[ddpm] wrote output: " : "[ddpm] failed output: ") << out_path << std::endl;
        }
        if (party == CLIENT) {
            bool ok = write_jpg_from_fixed(cond_plain, image_hw, image_hw, 1, f, cond_out_path.c_str());
            std::cout << (ok ? "[ddpm] wrote cond: " : "[ddpm] failed cond: ") << cond_out_path << std::endl;
        }
        shark::utils::stop_timer("total_eval");
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
