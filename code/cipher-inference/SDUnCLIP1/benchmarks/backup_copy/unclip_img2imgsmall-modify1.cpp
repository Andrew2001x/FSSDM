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
#include <shark/protocols/reciprocal.hpp>
#include <shark/protocols/sigmoid.hpp>
#include <shark/protocols/tanh.hpp>
#include <shark/protocols/softmax.hpp>
#include <shark/protocols/drelu.hpp>
#include <shark/protocols/select.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/comm.hpp>
#include <shark/utils/timer.hpp>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace shark;
using namespace shark::protocols;

// Fixed-point fractional bits.
static int f = 24;

struct AuthShare {
    span<u128> share;
    span<u128> tag;
};

static AuthShare auth_alloc(u64 size) {
    return AuthShare{span<u128>(size), span<u128>(size)};
}

static span<u64> auth_low(const AuthShare &x) {
    span<u64> out(x.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < x.share.size(); ++i) out[i] = getLow(x.share[i]);
    return out;
}

static AuthShare auth_from_public_raw(u64 size, u64 val) {
    AuthShare out = auth_alloc(size);
    u128 tag_val = ring_key * val;
    #pragma omp parallel for
    for (u64 i = 0; i < size; ++i) {
        if (party == SERVER || party == DEALER) {
            out.share[i] = val;
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

static span<u64> open_plain(const span<u64> &share) {
    span<u64> out(share.size());
    if (party == DEALER) {
        return out;
    }
    span<u64> tmp(share.size());
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
    for (u64 i = 0; i < share.size(); ++i) {
        out[i] = share[i] + tmp[i];
    }
    return out;
}

// Authenticate a plain share via masked opening with authenticated randomness.
static AuthShare auth_from_plain_open(const span<u64> &x) {
    AuthShare out = auth_alloc(x.size());
    if (party == DEALER) {
        span<u64> r(x.size());
        randomize(r);
        send_authenticated_ashare(r);
        #pragma omp parallel for
        for (u64 i = 0; i < x.size(); ++i) {
            out.share[i] = x[i];
            out.tag[i] = ring_key * x[i];
        }
        return out;
    }

    auto [r_share, r_tag] = recv_authenticated_ashare(x.size());
    span<u64> d_share(x.size());
    #pragma omp parallel for
    for (u64 i = 0; i < x.size(); ++i) {
        d_share[i] = x[i] - getLow(r_share[i]);
    }
    auto d = open_plain(d_share);
    #pragma omp parallel for
    for (u64 i = 0; i < x.size(); ++i) {
        out.share[i] = r_share[i] + d[i];
        out.tag[i] = r_tag[i] + ring_key * d[i];
    }
    return out;
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
        out.share[i] = -a.share[i];
        out.tag[i] = -a.tag[i];
    }
    return out;
}

static AuthShare auth_mul_const(const AuthShare &a, u64 c) {
    AuthShare out = auth_alloc(a.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < a.share.size(); ++i) {
        out.share[i] = a.share[i] * c;
        out.tag[i] = a.tag[i] * c;
    }
    return out;
}

static AuthShare auth_mul(const AuthShare &a, const AuthShare &b) {
    auto a_low = auth_low(a);
    auto b_low = auth_low(b);
    auto tmp = mul::call_share(a_low, b_low);
    AuthShare out{std::move(tmp.share), std::move(tmp.tag)};
    return out;
}

static AuthShare auth_matmul(u64 M, u64 K, u64 N, const AuthShare &x, const span<u64> &w) {
    auto x_low = auth_low(x);
    auto tmp = matmul::call_share(M, K, N, x_low, w);
    AuthShare out{std::move(tmp.share), std::move(tmp.tag)};
    return out;
}

static AuthShare auth_conv(u64 k, u64 padding, u64 stride, u64 inC, u64 outC, u64 H, u64 W,
                           const AuthShare &x, const span<u64> &w) {
    auto x_low = auth_low(x);
    auto tmp = conv::call_share(k, padding, stride, inC, outC, H, W, x_low, w);
    AuthShare out{std::move(tmp.share), std::move(tmp.tag)};
    return out;
}

static AuthShare auth_shift(const AuthShare &x, u64 shift) {
    auto x_low = auth_low(x);
    auto tmp = ars::call_share(x_low, (int)shift);
    AuthShare out{std::move(tmp.share), std::move(tmp.tag)};
    return out;
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
#define RECIPROCAL_CALL(...) reciprocal::call(__VA_ARGS__)

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

static inline u64 qrand_normal(int fp, double stddev) {
    const double kTwoPi = 6.2831853071795864769;
    double u1 = rng_u01(current_rng_seed);
    double u2 = rng_u01(current_rng_seed);
    double r = std::sqrt(-2.0 * std::log(u1));
    double z = r * std::cos(kTwoPi * u2);
    double x = z * stddev;
    double cap = 3.0 * stddev;
    if (x > cap) x = cap;
    if (x < -cap) x = -cap;
    int64_t q = (int64_t)std::llround(x * (double)(1ULL << fp));
    return (u64)q;
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

// Secure compare to zero without opening the true value.
// Uses masked-open + DCF ring to get a shared bit (0/1).
static AuthShare cmp_ge_zero_noopen(const AuthShare &x) {
    u64 n = x.share.size();
    AuthShare out = auth_alloc(n);
    const u64 bias = 1ull << 63;

    if (party == DEALER) {
        span<u64> r(n);
        randomize(r);
        span<u64> alpha(n);
        for (u64 i = 0; i < n; ++i) {
            alpha[i] = bias + r[i];
        }
        send_authenticated_ashare(r);
        send_dcfring(alpha, 64);
        for (u64 i = 0; i < n; ++i) {
            u64 x_low = getLow(x.share[i]);
            u64 bit = (int64_t)x_low >= 0 ? 1ull : 0ull;
            out.share[i] = bit;
            out.tag[i] = ring_key * bit;
        }
        return out;
    }

    auto x_low = auth_low(x);
    auto [r_share, r_tag] = recv_authenticated_ashare(n);
    (void)r_tag;

    span<u64> masked_share(n);
    for (u64 i = 0; i < n; ++i) {
        u64 xb = x_low[i] + bias;
        masked_share[i] = xb + getLow(r_share[i]);
    }

    span<u64> masked(n);
    {
        span<u64> tmp(n);
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                peer->send_array(masked_share);
            }
            #pragma omp section
            {
                peer->recv_array(tmp);
            }
        }
        for (u64 i = 0; i < n; ++i) {
            masked[i] = masked_share[i] + tmp[i];
        }
    }
    auto keys = recv_dcfring(n, 64);

    span<u128> lt_share(n);
    span<u128> lt_tag(n);
    for (u64 i = 0; i < n; ++i) {
        auto [s_share, s_tag] = crypto::dcfring_eval(party, keys[i], masked[i], false);
        (void)s_tag;
        lt_share[i] = s_share;
        lt_tag[i] = s_tag;
    }

    AuthShare lt{std::move(lt_share), std::move(lt_tag)};
    auto one = make_public_raw(n, 1);
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

static void clip_batch_check(const char* label) {
    shark::protocols::debug_batch_check(label);
}

// Forward declarations for timed wrappers.
static AuthShare gelu_apply(const AuthShare &x);
static AuthShare silu_apply(const AuthShare &x);
static AuthShare layernorm_rows(u64 rows, u64 cols, const AuthShare &X);
static AuthShare groupnorm_apply(u64 B, u64 H, u64 W, u64 C, const AuthShare &x);

static void print_key_file_sizes() {
    namespace fs = std::filesystem;
    const char *server_path = "server.dat";
    const char *client_path = "client.dat";
    if (fs::exists(server_path)) {
        auto sz = fs::file_size(server_path);
        std::cout << "[KEY] server.dat size: " << (double)sz / (1024.0 * 1024.0) << " MB (" << sz << " bytes)" << std::endl;
    }
    if (fs::exists(client_path)) {
        auto sz = fs::file_size(client_path);
        std::cout << "[KEY] client.dat size: " << (double)sz / (1024.0 * 1024.0) << " MB (" << sz << " bytes)" << std::endl;
    }
}

static bool write_jpg_from_fixed(const span<u64> &img, u64 H, u64 W, u64 C, int fp, const char *path) {
    if (img.size() != H * W * C) return false;
    const int outC = (C == 1) ? 3 : (int)C;
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
                for (u64 c = 0; c < C; ++c) {
                    size_t src = (((size_t)h * W + widx) * C) + c;
                    double v = (double)(int64_t)img[src] * scale;
                    v = (v + 1.0) * 0.5;
                    if (v < 0.0) v = 0.0;
                    if (v > 1.0) v = 1.0;
                    unsigned char u = (unsigned char)std::llround(v * 255.0);
                    size_t dst = ((size_t)h * W + widx) * outC + c;
                    buf[dst] = u;
                }
            }
        }
    }
    return stbi_write_jpg(path, (int)W, (int)H, outC, buf.data(), 95) != 0;
}

static AuthShare timed_gelu(const char *name, const AuthShare &x) {
    shark::utils::start_timer(name);
    auto out = gelu_apply(x);
    shark::utils::stop_timer(name);
    return out;
}

static AuthShare timed_silu(const char *name, const AuthShare &x) {
    shark::utils::start_timer(name);
    auto out = silu_apply(x);
    shark::utils::stop_timer(name);
    return out;
}

static AuthShare timed_layernorm_rows(const char *name, u64 rows, u64 cols, const AuthShare &x) {
    shark::utils::start_timer(name);
    auto out = layernorm_rows(rows, cols, x);
    shark::utils::stop_timer(name);
    return out;
}

static AuthShare timed_groupnorm_apply(const char *name, u64 B, u64 H, u64 W, u64 C, const AuthShare &x) {
    shark::utils::start_timer(name);
    auto out = groupnorm_apply(B, H, W, C, x);
    shark::utils::stop_timer(name);
    return out;
}

static void print_profile_timers() {
    using shark::utils::print_timer;
    std::cout << "[PROFILE] time(ms), comm(KB) per op" << std::endl;
    print_timer("total_eval");
    print_timer("silu:unet.time_mlp");
    print_timer("gelu:unet.mid_attn.ff.gate");
    print_timer("silu:unet.rb1.act1");
    print_timer("silu:unet.rb1.act2");
    print_timer("silu:unet.rb2.act1");
    print_timer("silu:unet.rb2.act2");
    print_timer("silu:vae.res1.act1");
    print_timer("silu:vae.res1.act2");
    print_timer("silu:vae.res2.act1");
    print_timer("silu:vae.res2.act2");
    print_timer("layernorm:image_encoder.tokens");
    print_timer("layernorm:unet.rb1.gn1");
    print_timer("layernorm:unet.rb1.gn2");
    print_timer("layernorm:unet.mid_attn.gn");
    print_timer("layernorm:unet.mid_attn.norm1");
    print_timer("layernorm:unet.mid_attn.norm2");
    print_timer("layernorm:unet.mid_attn.norm3");
    print_timer("layernorm:unet.rb2.gn1");
    print_timer("layernorm:unet.rb2.gn2");
    print_timer("layernorm:vae.res1.gn1");
    print_timer("layernorm:vae.res1.gn2");
    print_timer("layernorm:vae.attn.gn");
    print_timer("layernorm:vae.res2.gn1");
    print_timer("layernorm:vae.res2.gn2");
    print_timer("softmax:unet.mid_attn.self");
    print_timer("softmax:unet.mid_attn.cross");
    print_timer("softmax:vae.self_attn");
}

// -----------------------------
// Fixed-point activations
// -----------------------------
static AuthShare sigmoid_apply(const AuthShare &x) {
    // Unused in this benchmark; keep a minimal authenticated wrapper if needed.
    auto x_low = auth_low(x);
    auto out_plain = sigmoid::call(f, x_low);
    return auth_from_plain_open(out_plain);
}

// GELU(x) = x * sigmoid(1.702 * x) (QuickGELU) with 5th-order sigmoid approximation
static AuthShare gelu_apply(const AuthShare &x) {
    auto add_const = [&](const AuthShare &src, double v) {
        auto c = make_public_const(src.share.size(), v, f);
        return ADD_CALL(src, c);
    };

    // x' = 1.702 * x
    auto scaled = make_public_const(x.share.size(), 1.702, f);
    // Keep intermediate scale at Q(f) to avoid overflow and keep comparisons consistent.
    auto xs = LRS_CALL(MUL_CALL(x, scaled), f);

    auto x2 = LRS_CALL(MUL_CALL(xs, xs), f);
    auto x3 = LRS_CALL(MUL_CALL(x2, xs), f);
    auto x4 = LRS_CALL(MUL_CALL(x2, x2), f);
    auto x5 = LRS_CALL(MUL_CALL(x4, xs), f);

    // 5th-order polynomial coefficients for sigmoid(x) approximation.
    // outer: x in [-7, -2.2), inner: x in [-2.2, 0)
    auto cO0 = make_public_const(x.share.size(), 5.90010486e-01, f);
    auto cO1 = make_public_const(x.share.size(), 4.21319936e-01, f);
    auto cO2 = make_public_const(x.share.size(), 1.26610092e-01, f);
    auto cO3 = make_public_const(x.share.size(), 1.97985686e-02, f);
    auto cO4 = make_public_const(x.share.size(), 1.59538912e-03, f);
    auto cO5 = make_public_const(x.share.size(), 5.25625056e-05, f);

    auto cI0 = make_public_const(x.share.size(), 4.99957471e-01, f);
    auto cI1 = make_public_const(x.share.size(), 2.49155471e-01, f);
    auto cI2 = make_public_const(x.share.size(), -4.01655846e-03, f);
    auto cI3 = make_public_const(x.share.size(), -2.84974728e-02, f);
    auto cI4 = make_public_const(x.share.size(), -6.76484741e-03, f);
    auto cI5 = make_public_const(x.share.size(), -4.34163661e-04, f);

    auto tO1 = LRS_CALL(MUL_CALL(cO1, xs), f);
    auto tO2 = LRS_CALL(MUL_CALL(cO2, x2), f);
    auto tO3 = LRS_CALL(MUL_CALL(cO3, x3), f);
    auto tO4 = LRS_CALL(MUL_CALL(cO4, x4), f);
    auto tO5 = LRS_CALL(MUL_CALL(cO5, x5), f);

    auto P_outer = cO0;
    P_outer = ADD_CALL(P_outer, tO1);
    P_outer = ADD_CALL(P_outer, tO2);
    P_outer = ADD_CALL(P_outer, tO3);
    P_outer = ADD_CALL(P_outer, tO4);
    P_outer = ADD_CALL(P_outer, tO5);

    auto tI1 = LRS_CALL(MUL_CALL(cI1, xs), f);
    auto tI2 = LRS_CALL(MUL_CALL(cI2, x2), f);
    auto tI3 = LRS_CALL(MUL_CALL(cI3, x3), f);
    auto tI4 = LRS_CALL(MUL_CALL(cI4, x4), f);
    auto tI5 = LRS_CALL(MUL_CALL(cI5, x5), f);

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

    auto lt_neg7 = cmp_lt(xs, -7.0);
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
static AuthShare silu_apply(const AuthShare &x) {
    auto add_const = [&](const AuthShare &src, double v) {
        auto c = make_public_const(src.share.size(), v, f);
        return ADD_CALL(src, c);
    };

    auto x2 = LRS_CALL(MUL_CALL(x, x), f);
    auto x4 = LRS_CALL(MUL_CALL(x2, x2), f);
    auto x6 = LRS_CALL(MUL_CALL(x2, x4), f);

    const auto a0 = make_public_const(x.share.size(), -0.52212664, f);
    const auto a1 = make_public_const(x.share.size(), -0.16910363, f);
    const auto a2 = make_public_const(x.share.size(), -0.01420163, f);

    const auto b0 = make_public_const(x.share.size(), 0.03453821, f);
    const auto b1 = make_public_const(x.share.size(), 0.49379432, f);
    const auto b2 = make_public_const(x.share.size(), 0.19784596, f);
    const auto b3 = make_public_const(x.share.size(), -0.00602401, f);
    const auto b4 = make_public_const(x.share.size(), 0.00008032, f);

    auto segA = LRS_CALL(MUL_CALL(x2, a2), f);
    segA = ADD_CALL(segA, LRS_CALL(MUL_CALL(x, a1), f));
    segA = ADD_CALL(segA, a0);

    auto segB = LRS_CALL(MUL_CALL(x6, b4), f);
    segB = ADD_CALL(segB, LRS_CALL(MUL_CALL(x4, b3), f));
    segB = ADD_CALL(segB, LRS_CALL(MUL_CALL(x2, b2), f));
    segB = ADD_CALL(segB, LRS_CALL(MUL_CALL(x, b1), f));
    segB = ADD_CALL(segB, b0);

    auto zero = make_public_const(x.share.size(), 0.0, f);
    auto segL = zero;
    auto segR = x;

    auto cmp_gt = [&](const AuthShare &src, double v) {
        auto diff = add_const(src, -v);
        return CMP_GE_ZERO_CALL(diff);
    };
    auto cmp_lt = [&](const AuthShare &src, double v) {
        auto negv = neg_span(src);
        auto diff = add_const(negv, v);
        return CMP_GE_ZERO_CALL(diff);
    };

    auto lt_neg6 = cmp_lt(x, -6.0);
    auto ge_neg2 = cmp_gt(x, -2.0);
    const double eps = 1.0 / (double)(1ull << f);
    auto ge_6 = cmp_gt(x, 6.0 + eps); // x > 6

    auto ret = segA;
    auto update = [&](AuthShare &cur, const AuthShare &cond, const AuthShare &next) {
        auto delta = ADD_CALL(next, neg_span(cur));
        auto delta_sel = SELECT_CALL(cond, delta);
        cur = ADD_CALL(cur, delta_sel);
    };

    update(ret, lt_neg6, segL);
    update(ret, ge_neg2, segB);
    update(ret, ge_6, segR);

    return ret;
}

static AuthShare tanh_apply(const AuthShare &x) {
    auto x_low = auth_low(x);
    auto out_plain = tanh::call(f, x_low);
    return auth_from_plain_open(out_plain);
}

// Fixed-point multiply with rescale back to Q(f).
static AuthShare mul_qf(const AuthShare &a, const AuthShare &b) {
    auto prod = MUL_CALL(a, b);
    return LRS_CALL(prod, f);
}

// Newton-Raphson reciprocal in Q(f). Assumes x is positive and roughly normalized.
static AuthShare ss_reciprocal(const AuthShare &x) {
    auto y = make_public_const(x.share.size(), 1.0, f);
    auto two = make_public_const(x.share.size(), 2.0, f);
    for (int iter = 0; iter < 2; ++iter) {
        auto xy = mul_qf(x, y);
        auto diff = ADD_CALL(two, neg_span(xy));
        y = mul_qf(y, diff);
    }
    return y;
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

// Secure Exp using Pade/Taylor approximation for 2^x with Range Reduction
// Mimics exp_pade_combined logic: 2^(I+F) = 2^I * Poly(F)
// Supports input range approx [-30, 0] (Softmax typical range)
#if 0
[[maybe_unused]] static AuthShare exp_pade_combined(const AuthShare &x) {
    int fp_bits = f;
    u64 n = x.size();

    // 1. Scale x by log2(e) to get base-2 exponent
    // log2(e) = 1.44269504
    auto log2e = make_public_const(n, 1.44269504, fp_bits);
    auto x_scaled = MUL_CALL(x, log2e);
    x_scaled = LRS_CALL(x_scaled, fp_bits);

    // 2. Integer Part Separation (Range Reduction)
    // We scan intervals [-1, 0), [-2, -1), ... [-30, -29)
    span<u64> I_val(n);
    span<u64> PowI(n);
    for (u64 i = 0; i < n; ++i) { I_val[i] = 0; PowI[i] = 0; }

    // Default for x >= 0 is I=0, PowI=1.
    auto zero_const = make_public_const(n, 0.0, fp_bits);
    auto c_next = CMP_GE_ZERO_CALL(x_scaled);
    auto mask_0 = c_next;

    auto one_const = make_public_const(n, 1.0, fp_bits);
    auto term_pow = SELECT_CALL(mask_0, one_const);
    PowI = ADD_CALL(PowI, term_pow);

    int min_k = -30;
    for (int k = -1; k >= min_k; --k) {
        auto k_const = make_public_const(n, (double)k, fp_bits);
        auto neg_k_const = make_public_const(n, (double)(-k), fp_bits);
        auto diff = ADD_CALL(x_scaled, neg_k_const);

        auto c_curr = CMP_GE_ZERO_CALL(diff);
        auto bit_curr = SELECT_CALL(c_curr, one_const);
        auto bit_next = SELECT_CALL(c_next, one_const);

        auto mask_k = ADD_CALL(bit_curr, neg_span(bit_next));

        double pow2k = std::ldexp(1.0, k);
        auto pow2k_const = make_public_const(n, pow2k, fp_bits);
        auto term_p = MUL_CALL(mask_k, pow2k_const);
        PowI = ADD_CALL(PowI, term_p);

        auto term_i = MUL_CALL(mask_k, k_const);
        I_val = ADD_CALL(I_val, term_i);

        c_next = c_curr;
    }

    // 3. Fractional Part: F = x_scaled - I
    span<u64> neg_I(n);
    #pragma omp parallel for
    for (u64 i = 0; i < n; ++i) neg_I[i] = -I_val[i];
    auto F = ADD_CALL(x_scaled, neg_I);

    // 4. Polynomial Evaluation Poly(F) ≈ 2^F for F in [0, 1)
    auto p0 = make_public_const(n, 1.0, fp_bits);
    auto p1 = make_public_const(n, 0.69314718, fp_bits);
    auto p2 = make_public_const(n, 0.24022650, fp_bits);
    auto p3 = make_public_const(n, 0.05550410, fp_bits);
    auto p4 = make_public_const(n, 0.00961812, fp_bits);
    auto p5 = make_public_const(n, 0.00133335, fp_bits);

    auto res = p5;
    res = MUL_CALL(res, F); res = LRS_CALL(res, fp_bits);
    res = ADD_CALL(res, p4);
    res = MUL_CALL(res, F); res = LRS_CALL(res, fp_bits);
    res = ADD_CALL(res, p3);
    res = MUL_CALL(res, F); res = LRS_CALL(res, fp_bits);
    res = ADD_CALL(res, p2);
    res = MUL_CALL(res, F); res = LRS_CALL(res, fp_bits);
    res = ADD_CALL(res, p1);
    res = MUL_CALL(res, F); res = LRS_CALL(res, fp_bits);
    res = ADD_CALL(res, p0);

    // 5. Final Result = Poly(F) * 2^I
    auto final_res = MUL_CALL(res, PowI);
    final_res = LRS_CALL(final_res, fp_bits);
    return final_res;
}
#endif

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

    // Step 2: Chebyshev exp (clip x <= -14 to 0)
    auto exp_vals = chexp(shifted);
    auto cmp_gt = [&](const AuthShare &src, double v) {
        const auto c = make_public_const(src.share.size(), v, f);
        auto diff = ADD_CALL(src, neg_span(c));
        return CMP_GE_ZERO_CALL(diff);
    };
    const double eps = 1.0 / (double)(1ull << f);
    auto mask = cmp_gt(shifted, -14.0 + eps); // x > -14
    auto exp_masked = SELECT_CALL(mask, exp_vals);
    clip_batch_check("softmax after chexp");

    // Step 3: Compute Row Sums
    AuthShare row_sums = auth_alloc(a);
    #pragma omp parallel for
    for (u64 i = 0; i < a; ++i) {
        u128 sum_share = 0;
        u128 sum_tag = 0;
        for (u64 j = 0; j < b; ++j) {
            u64 idx = i * b + j;
            sum_share += exp_masked.share[idx];
            sum_tag += exp_masked.tag[idx];
        }
        row_sums.share[i] = sum_share;
        row_sums.tag[i] = sum_tag;
    }

    // Step 4: Inverse Sum
    auto inv_sums = ss_reciprocal(row_sums);
    clip_batch_check("softmax after reciprocal");

    // Step 5: Broadcast and Multiply
    AuthShare inv_sums_b = auth_alloc(a * b);
    #pragma omp parallel for collapse(2)
    for (u64 i = 0; i < a; ++i) {
        for (u64 j = 0; j < b; ++j) {
            u64 idx = i * b + j;
            inv_sums_b.share[idx] = inv_sums.share[i];
            inv_sums_b.tag[idx] = inv_sums.tag[i];
        }
    }

    auto result = MUL_CALL(exp_masked, inv_sums_b);
    result = LRS_CALL(result, f); // Normalize after multiplication
    clip_batch_check("softmax after mul");
    return result;
}

static AuthShare timed_clip_softmax(const char *name, u64 a, u64 b, const AuthShare &x) {
    shark::utils::start_timer(name);
    auto out = clip_softmax(a, b, x);
    shark::utils::stop_timer(name);
    return out;
}

// -----------------------------
// Linear and conv helpers
// -----------------------------
static AuthShare linear_mat(u64 M, u64 K, u64 N, const AuthShare &x, const span<u64> &w, const AuthShare *b = nullptr) {
    auto y = MATMUL_CALL(M, K, N, x, w);
    y = LRS_CALL(y, f);
    if (b && b->share.size() == N) {
        AuthShare out = auth_alloc(y.share.size());
        for (u64 i = 0; i < M; ++i) {
            for (u64 j = 0; j < N; ++j) {
                u64 idx = i * N + j;
                out.share[idx] = y.share[idx] + b->share[j];
                out.tag[idx] = y.tag[idx] + b->tag[j];
            }
        }
        return out;
    }
    return y;
}

static AuthShare conv2d_apply(
    u64 B, u64 H, u64 W, u64 inC, u64 outC, u64 k, u64 stride, u64 padding,
    const AuthShare &x, const span<u64> &w, const AuthShare *b = nullptr
) {
    auto y = CONV_CALL(k, padding, stride, inC, outC, H, W, x, w);
    clip_batch_check("debug:conv2d:after conv");
    y = LRS_CALL(y, f);
    clip_batch_check("debug:conv2d:after shift");
    if (b && b->share.size() == outC) {
        u64 outH = (H - k + 2 * padding) / stride + 1;
        u64 outW = (W - k + 2 * padding) / stride + 1;
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
        return out;
    }
    return y;
}

// 2x2 conv with padding=1 and crop back to (H, W) to preserve shapes.
static AuthShare conv2d_apply_k2_same(
    u64 B, u64 H, u64 W, u64 inC, u64 outC, u64 stride,
    const AuthShare &x, const span<u64> &w, const AuthShare *b = nullptr
) {
    const u64 k = 2;
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
static AuthShare rsqrt_poly_normalized(const AuthShare &u, int fp_bits) {
    // rsqrt(u) polynomial in Qfp.
    auto c0 = make_public_const(u.share.size(), 4.14285016, fp_bits);
    auto c1 = make_public_const(u.share.size(), -15.47994394, fp_bits);
    auto c2 = make_public_const(u.share.size(), 38.4714796, fp_bits);
    auto c3 = make_public_const(u.share.size(), -49.86605845, fp_bits);
    auto c4 = make_public_const(u.share.size(), 26.02942339, fp_bits);

    auto res = c4;
    res = ADD_CALL(LRS_CALL(MUL_CALL(res, u), fp_bits), c3);
    res = ADD_CALL(LRS_CALL(MUL_CALL(res, u), fp_bits), c2);
    res = ADD_CALL(LRS_CALL(MUL_CALL(res, u), fp_bits), c1);
    res = ADD_CALL(LRS_CALL(MUL_CALL(res, u), fp_bits), c0);
    return res;
}

static AuthShare rsqrt_spu_poly(const AuthShare &x, int fp_bits) {
    int shift_0 = fp_bits - 2;
    AuthShare u;
    if (shift_0 >= 0) {
        u = auth_mul_const(x, (u64)(1ULL << shift_0));
    } else {
        u = LRS_CALL(x, (u64)(-shift_0));
    }

    int adj_0 = fp_bits - 2;
    int m_0 = adj_0 / 2;
    bool odd_0 = (adj_0 & 1) != 0;
    double comp_real_0 = std::ldexp(1.0, m_0);
    if (odd_0) comp_real_0 *= std::sqrt(2.0);
    auto comp = make_public_const(x.share.size(), comp_real_0, fp_bits);

    const int max_j = 48;
    for (int j = 1; j <= max_j; ++j) {
        u64 threshold_val = (1ULL << j) - 1ULL;
        u64 neg_thresh_val = (u64)(-(int64_t)threshold_val);
        auto neg_thresh_share = make_public_raw(x.share.size(), neg_thresh_val);
        auto diff = ADD_CALL(x, neg_thresh_share);
        auto ge_j = CMP_GE_ZERO_CALL(diff);

        int shift_j = fp_bits - j - 2;
        AuthShare u_j;
        if (shift_j >= 0) {
            u_j = auth_mul_const(x, (u64)(1ULL << shift_j));
        } else {
            u_j = LRS_CALL(x, (u64)(-shift_j));
        }

        int adj_j = fp_bits - j - 2;
        double comp_real;
        if (adj_j >= 0) {
            int m_j = adj_j / 2;
            bool odd_j = (adj_j & 1) != 0;
            comp_real = std::ldexp(1.0, m_j);
            if (odd_j) comp_real *= std::sqrt(2.0);
        } else {
            int m_j = (-adj_j) / 2;
            bool odd_j = ((-adj_j) & 1) != 0;
            comp_real = std::ldexp(1.0, -m_j);
            if (odd_j) comp_real /= std::sqrt(2.0);
        }
        auto comp_j = make_public_const(x.share.size(), comp_real, fp_bits);

        auto u_delta = ADD_CALL(u_j, neg_span(u));
        auto u_delta_sel = SELECT_CALL(ge_j, u_delta);
        u = ADD_CALL(u, u_delta_sel);

        auto comp_delta = ADD_CALL(comp_j, neg_span(comp));
        auto comp_delta_sel = SELECT_CALL(ge_j, comp_delta);
        comp = ADD_CALL(comp, comp_delta_sel);
    }

    auto r = rsqrt_poly_normalized(u, fp_bits);
    auto prod = MUL_CALL(r, comp);
    return LRS_CALL(prod, fp_bits);
}

static AuthShare layernorm_rows(u64 rows, u64 cols, const AuthShare &X) {
    clip_batch_check("layernorm:enter");
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

    auto inv_n = make_public_const(rows, 1.0 / (double)cols, f);
    clip_batch_check("layernorm:after reciprocal");
    auto prod_mean = MUL_CALL(row_sum, inv_n);
    clip_batch_check("layernorm:after mean_mul");
    auto mean = LRS_CALL(prod_mean, f);
    clip_batch_check("layernorm:after mean_shift");

    AuthShare mean_b = auth_alloc(rows * cols);
    for (u64 i = 0; i < rows; ++i) {
        for (u64 j = 0; j < cols; ++j) {
            u64 idx = i * cols + j;
            mean_b.share[idx] = mean.share[i];
            mean_b.tag[idx] = mean.tag[i];
        }
    }

    AuthShare d = auth_alloc(rows * cols);
    for (u64 k = 0; k < rows * cols; ++k) {
        d.share[k] = X.share[k] - mean_b.share[k];
        d.tag[k] = X.tag[k] - mean_b.tag[k];
    }

    auto sqr = MUL_CALL(d, d);
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

    auto prod_var = MUL_CALL(row_sum_sqr, inv_n);
    auto var = LRS_CALL(prod_var, f);

    auto eps = make_public_const(rows, 1e-5, f);
    auto var_eps = ADD_CALL(var, eps);

    auto rsqrt_var = rsqrt_spu_poly(var_eps, f);
    clip_batch_check("layernorm:after rsqrt");
    AuthShare rsqrt_var_broadcast = auth_alloc(rows * cols);
    for (u64 i = 0; i < rows; ++i) {
        for (u64 j = 0; j < cols; ++j) {
            u64 idx = i * cols + j;
            rsqrt_var_broadcast.share[idx] = rsqrt_var.share[i];
            rsqrt_var_broadcast.tag[idx] = rsqrt_var.tag[i];
        }
    }

    auto normalized = MUL_CALL(d, rsqrt_var_broadcast);
    auto out = LRS_CALL(normalized, f);
    clip_batch_check("layernorm:after norm");
    return out;
}

static AuthShare groupnorm_apply(u64 B, u64 H, u64 W, u64 C, const AuthShare &x) {
    always_assert(C > 0);
    // For small model, we assume groups=1 and normalize across C*H*W per batch.
    u64 cols = H * W * C;
    AuthShare out = layernorm_rows(B, cols, x);
    return out;
}

// -----------------------------
// Attention (single-head, small dim)
// -----------------------------
static AuthShare attention_dot(u64 q_rows, u64 k_rows, u64 dim,
                               const AuthShare &Q, const AuthShare &K, const AuthShare &V,
                               const char *softmax_tag) {
    auto Kt = transpose_mat(K, k_rows, dim); // dim x k_rows
    Kt = reauth(Kt);
    auto Kt_low = auth_low(Kt);
    auto scores = MATMUL_CALL(q_rows, dim, k_rows, Q, Kt_low);
    scores = LRS_CALL(scores, f);
    clip_batch_check("attn:after qk");
    // dim=1 => scale=1.0
    clip_batch_check("attn:before softmax");
    auto attn = timed_clip_softmax(softmax_tag, q_rows, k_rows, scores);
    clip_batch_check("attn:after softmax");
    auto V_low = auth_low(V);
    auto out = MATMUL_CALL(q_rows, k_rows, dim, attn, V_low);
    out = LRS_CALL(out, f);
    clip_batch_check("attn:after av");
    return out;
}

// -----------------------------
// Basic Transformer Block (single head)
// -----------------------------
struct AttnWeights {
    span<u64> wq, wk, wv, wout;
    AuthShare bq, bk, bv, bout;
};

struct FFWeights {
    span<u64> w_up, w_down;
    AuthShare b_up, b_down;
    u64 inner_dim = 0;
};

struct TransformerWeights {
    AttnWeights self_attn;
    AttnWeights cross_attn;
    FFWeights ff;
};

static AuthShare apply_attn(const AuthShare &x, u64 rows, u64 dim,
                            const AuthShare &context, u64 ctx_rows,
                            const AttnWeights &w, const char *softmax_tag) {
    auto q = linear_mat(rows, dim, dim, x, w.wq, &w.bq);
    clip_batch_check("attn:after q");
    auto k = linear_mat(ctx_rows, dim, dim, context, w.wk, &w.bk);
    clip_batch_check("attn:after k");
    auto v = linear_mat(ctx_rows, dim, dim, context, w.wv, &w.bv);
    clip_batch_check("attn:after v");
    q = reauth(q);
    k = reauth(k);
    v = reauth(v);
    auto attn_out = attention_dot(rows, ctx_rows, dim, q, k, v, softmax_tag);
    return linear_mat(rows, dim, dim, attn_out, w.wout, &w.bout);
}

static AuthShare basic_transformer_block(u64 rows, u64 dim, u64 ctx_rows,
                                         const AuthShare &x_in,
                                         const AuthShare &context_in,
                                         const TransformerWeights &w) {
    auto x = x_in;
    clip_batch_check("attn:before norm1");
    auto n1 = timed_layernorm_rows("layernorm:unet.mid_attn.norm1", rows, dim, x);
    clip_batch_check("attn:after norm1");
    auto a1 = apply_attn(n1, rows, dim, n1, rows, w.self_attn, "softmax:unet.mid_attn.self");
    x = ADD_CALL(x, a1);

    auto n2 = timed_layernorm_rows("layernorm:unet.mid_attn.norm2", rows, dim, x);
    clip_batch_check("attn:after norm2");
    auto a2 = apply_attn(n2, rows, dim, context_in, ctx_rows, w.cross_attn, "softmax:unet.mid_attn.cross");
    x = ADD_CALL(x, a2);

    auto n3 = timed_layernorm_rows("layernorm:unet.mid_attn.norm3", rows, dim, x);
    auto ff_up = linear_mat(rows, dim, w.ff.inner_dim * 2, n3, w.ff.w_up, &w.ff.b_up);
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
    auto prod = MUL_CALL(a, gate_act);
    auto prod_scaled = LRS_CALL(prod, f);
    auto ff_out = linear_mat(rows, w.ff.inner_dim, dim, prod_scaled, w.ff.w_down, &w.ff.b_down);
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
        return std::pow(std::cos((t + 0.008) / 1.008 * kPi / 2.0), 2.0);
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

static AuthShare add_noise_ddpm(const AuthShare &x, const AuthShare &noise, double alpha_cumprod) {
    double a = std::sqrt(alpha_cumprod);
    double b = std::sqrt(1.0 - alpha_cumprod);
    auto a_const = make_public_const(x.share.size(), a, f);
    auto b_const = make_public_const(x.share.size(), b, f);
    auto ax = LRS_CALL(MUL_CALL(x, a_const), f);
    auto bn = LRS_CALL(MUL_CALL(noise, b_const), f);
    return ADD_CALL(ax, bn);
}

static AuthShare scheduler_step_linear(const AuthShare &sample, const AuthShare &model_output,
                                       double alpha_t, double alpha_prev) {
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
    return ADD_CALL(term_a, term_b);
}

// -----------------------------
// Data loading
// -----------------------------
static bool load_fixed_list(const std::string &path, int fp, std::vector<u64> &out, size_t limit = 0) {
    std::ifstream in(path);
    if (!in) return false;
    out.clear();
    std::string tok;
    while (in >> tok) {
        // Strip common punctuation to tolerate formats like "[1.0," or "2.0]"
        while (!tok.empty()) {
            char c = tok.front();
            if (c == '[' || c == ']' || c == '(' || c == ')' || c == ',' || c == ';') {
                tok.erase(tok.begin());
            } else {
                break;
            }
        }
        while (!tok.empty()) {
            char c = tok.back();
            if (c == '[' || c == ']' || c == '(' || c == ')' || c == ',' || c == ';') {
                tok.pop_back();
            } else {
                break;
            }
        }
        if (tok.empty()) continue;
        char *end = nullptr;
        double v = std::strtod(tok.c_str(), &end);
        if (end == tok.c_str()) continue; // not a number token
        int64_t q = (int64_t)std::llround(v * (double)(1ULL << fp));
        out.push_back((u64)q);
        if (limit && out.size() >= limit) break;
    }
    return !out.empty();
}

int main(int argc, char **argv) {
    init::from_args(argc, argv);
    mpspdz_32bit_compaison = false;

    print_key_file_sizes();
    if (party != DEALER) {
        shark::utils::start_timer("total_eval");
    }

    const u64 batch = 1;
    // Minimal spatial and channel sizes for a tiny toy model.
    const u64 H = 8;
    const u64 W = 8;
    const u64 C = 1;
    const u64 latent_C = 1;
    const u64 seq_len = 1;
    const u64 hidden = 1;
    const u64 ctx_dim = 1;
    const u64 img_emb_dim = 1;
    const u64 vocab = 1;
    const u64 noise_level = 10;
    const int num_inference_steps = 12;
    const double guidance_scale = 7.5;

    const char *seed_env = std::getenv("UNCLIP_SEED");
    if (seed_env && *seed_env) {
        current_rng_seed = std::strtoull(seed_env, nullptr, 0);
    }
    const u64 base_rng_seed = current_rng_seed;

    // -----------------------------
    // Model weights (SERVER-owned)
    // -----------------------------
    // Text embeddings (toy tokenizer with fixed token ids).
    span<u64> text_token_embed(vocab * hidden);
    span<u64> text_pos_embed(seq_len * hidden);

    // Image encoder weights.
    span<u64> img_patch_w(img_emb_dim * C * 1 * 1); // out=1, k=1
    span<u64> img_patch_b(img_emb_dim);
    span<u64> img_cls(img_emb_dim);
    span<u64> img_proj_w(img_emb_dim * img_emb_dim);
    span<u64> img_proj_b(img_emb_dim);

    // UNet weights.
    span<u64> unet_conv_in_w(latent_C * latent_C * 2 * 2);
    span<u64> unet_conv_in_b(latent_C);
    span<u64> unet_conv_out_w(latent_C * latent_C * 2 * 2);
    span<u64> unet_conv_out_b(latent_C);

    // ResBlock weights (two blocks).
    span<u64> rb1_conv1_w(latent_C * latent_C * 2 * 2);
    span<u64> rb1_conv1_b(latent_C);
    span<u64> rb1_conv2_w(latent_C * latent_C * 2 * 2);
    span<u64> rb1_conv2_b(latent_C);
    span<u64> rb1_temb_w(1 * latent_C);
    span<u64> rb1_temb_b(latent_C);

    span<u64> rb2_conv1_w(latent_C * latent_C * 2 * 2);
    span<u64> rb2_conv1_b(latent_C);
    span<u64> rb2_conv2_w(latent_C * latent_C * 2 * 2);
    span<u64> rb2_conv2_b(latent_C);
    span<u64> rb2_temb_w(1 * latent_C);
    span<u64> rb2_temb_b(latent_C);

    // Time MLP and image projection.
    span<u64> time_w1(1 * 1);
    span<u64> time_b1(1);
    span<u64> time_w2(1 * 1);
    span<u64> time_b2(1);
    span<u64> img_proj_w2(2 * 1);
    span<u64> img_proj_b2(1);

    // Transformer2D weights (mid attention).
    TransformerWeights mid_attn_w;
    span<u64> mid_self_bq(1), mid_self_bk(1), mid_self_bv(1), mid_self_bout(1);
    span<u64> mid_cross_bq(1), mid_cross_bk(1), mid_cross_bv(1), mid_cross_bout(1);
    span<u64> mid_ff_b_up(2), mid_ff_b_down(1);

    mid_attn_w.self_attn.wq = span<u64>(1 * 1);
    mid_attn_w.self_attn.wk = span<u64>(1 * 1);
    mid_attn_w.self_attn.wv = span<u64>(1 * 1);
    mid_attn_w.self_attn.wout = span<u64>(1 * 1);

    mid_attn_w.cross_attn.wq = span<u64>(1 * 1);
    mid_attn_w.cross_attn.wk = span<u64>(1 * 1);
    mid_attn_w.cross_attn.wv = span<u64>(1 * 1);
    mid_attn_w.cross_attn.wout = span<u64>(1 * 1);

    mid_attn_w.ff.inner_dim = 1;
    mid_attn_w.ff.w_up = span<u64>(1 * (mid_attn_w.ff.inner_dim * 2));
    mid_attn_w.ff.w_down = span<u64>(mid_attn_w.ff.inner_dim * 1);

    // VAE decode weights (simplified).
    span<u64> vae_post_w(latent_C * latent_C);
    span<u64> vae_post_b(latent_C);
    span<u64> vae_res1_c1_w(latent_C * latent_C * 2 * 2);
    span<u64> vae_res1_c1_b(latent_C);
    span<u64> vae_res1_c2_w(latent_C * latent_C * 2 * 2);
    span<u64> vae_res1_c2_b(latent_C);
    span<u64> vae_res2_c1_w(latent_C * latent_C * 2 * 2);
    span<u64> vae_res2_c1_b(latent_C);
    span<u64> vae_res2_c2_w(latent_C * latent_C * 2 * 2);
    span<u64> vae_res2_c2_b(latent_C);

    AttnWeights vae_attn_w;
    span<u64> vae_attn_bq(1), vae_attn_bk(1), vae_attn_bv(1), vae_attn_bout(1);
    vae_attn_w.wq = span<u64>(1 * 1);
    vae_attn_w.wk = span<u64>(1 * 1);
    vae_attn_w.wv = span<u64>(1 * 1);
    vae_attn_w.wout = span<u64>(1 * 1);

    span<u64> vae_out_w(C * latent_C * 2 * 2); // out_channels=C (minimal)
    span<u64> vae_out_b(C);

    auto fill_normal = [&](span<u64> &buf, double stddev) {
        for (u64 i = 0; i < buf.size(); ++i) buf[i] = qrand_normal(f, stddev);
    };

    if (party == SERVER || party == DEALER) {
        current_rng_seed = base_rng_seed;
        double stddev = 0.02;
        fill_normal(text_token_embed, stddev);
        fill_normal(text_pos_embed, stddev);
        fill_normal(img_patch_w, stddev);
        fill_normal(img_patch_b, stddev);
        fill_normal(img_cls, stddev);
        fill_normal(img_proj_w, stddev);
        fill_normal(img_proj_b, stddev);
        fill_normal(unet_conv_in_w, stddev);
        fill_normal(unet_conv_in_b, stddev);
        fill_normal(unet_conv_out_w, stddev);
        fill_normal(unet_conv_out_b, stddev);
        fill_normal(rb1_conv1_w, stddev);
        fill_normal(rb1_conv1_b, stddev);
        fill_normal(rb1_conv2_w, stddev);
        fill_normal(rb1_conv2_b, stddev);
        fill_normal(rb1_temb_w, stddev);
        fill_normal(rb1_temb_b, stddev);
        fill_normal(rb2_conv1_w, stddev);
        fill_normal(rb2_conv1_b, stddev);
        fill_normal(rb2_conv2_w, stddev);
        fill_normal(rb2_conv2_b, stddev);
        fill_normal(rb2_temb_w, stddev);
        fill_normal(rb2_temb_b, stddev);
        fill_normal(time_w1, stddev);
        fill_normal(time_b1, stddev);
        fill_normal(time_w2, stddev);
        fill_normal(time_b2, stddev);
        fill_normal(img_proj_w2, stddev);
        fill_normal(img_proj_b2, stddev);
        fill_normal(mid_attn_w.self_attn.wq, stddev);
        fill_normal(mid_self_bq, stddev);
        fill_normal(mid_attn_w.self_attn.wk, stddev);
        fill_normal(mid_self_bk, stddev);
        fill_normal(mid_attn_w.self_attn.wv, stddev);
        fill_normal(mid_self_bv, stddev);
        fill_normal(mid_attn_w.self_attn.wout, stddev);
        fill_normal(mid_self_bout, stddev);
        fill_normal(mid_attn_w.cross_attn.wq, stddev);
        fill_normal(mid_cross_bq, stddev);
        fill_normal(mid_attn_w.cross_attn.wk, stddev);
        fill_normal(mid_cross_bk, stddev);
        fill_normal(mid_attn_w.cross_attn.wv, stddev);
        fill_normal(mid_cross_bv, stddev);
        fill_normal(mid_attn_w.cross_attn.wout, stddev);
        fill_normal(mid_cross_bout, stddev);
        fill_normal(mid_attn_w.ff.w_up, stddev);
        fill_normal(mid_ff_b_up, stddev);
        fill_normal(mid_attn_w.ff.w_down, stddev);
        fill_normal(mid_ff_b_down, stddev);
        fill_normal(vae_post_w, stddev);
        fill_normal(vae_post_b, stddev);
        fill_normal(vae_res1_c1_w, stddev);
        fill_normal(vae_res1_c1_b, stddev);
        fill_normal(vae_res1_c2_w, stddev);
        fill_normal(vae_res1_c2_b, stddev);
        fill_normal(vae_res2_c1_w, stddev);
        fill_normal(vae_res2_c1_b, stddev);
        fill_normal(vae_res2_c2_w, stddev);
        fill_normal(vae_res2_c2_b, stddev);
        fill_normal(vae_attn_w.wq, stddev);
        fill_normal(vae_attn_bq, stddev);
        fill_normal(vae_attn_w.wk, stddev);
        fill_normal(vae_attn_bk, stddev);
        fill_normal(vae_attn_w.wv, stddev);
        fill_normal(vae_attn_bv, stddev);
        fill_normal(vae_attn_w.wout, stddev);
        fill_normal(vae_attn_bout, stddev);
        fill_normal(vae_out_w, stddev);
        fill_normal(vae_out_b, stddev);
    }

    // Share model weights.
    input::call(text_token_embed, SERVER);
    input::call(text_pos_embed, SERVER);
    input::call(img_patch_w, SERVER);
    input::call(img_patch_b, SERVER);
    input::call(img_cls, SERVER);
    input::call(img_proj_w, SERVER);
    input::call(img_proj_b, SERVER);
    input::call(unet_conv_in_w, SERVER);
    input::call(unet_conv_in_b, SERVER);
    input::call(unet_conv_out_w, SERVER);
    input::call(unet_conv_out_b, SERVER);
    input::call(rb1_conv1_w, SERVER);
    input::call(rb1_conv1_b, SERVER);
    input::call(rb1_conv2_w, SERVER);
    input::call(rb1_conv2_b, SERVER);
    input::call(rb1_temb_w, SERVER);
    input::call(rb1_temb_b, SERVER);
    input::call(rb2_conv1_w, SERVER);
    input::call(rb2_conv1_b, SERVER);
    input::call(rb2_conv2_w, SERVER);
    input::call(rb2_conv2_b, SERVER);
    input::call(rb2_temb_w, SERVER);
    input::call(rb2_temb_b, SERVER);
    input::call(time_w1, SERVER);
    input::call(time_b1, SERVER);
    input::call(time_w2, SERVER);
    input::call(time_b2, SERVER);
    input::call(img_proj_w2, SERVER);
    input::call(img_proj_b2, SERVER);
    input::call(mid_attn_w.self_attn.wq, SERVER);
    input::call(mid_self_bq, SERVER);
    input::call(mid_attn_w.self_attn.wk, SERVER);
    input::call(mid_self_bk, SERVER);
    input::call(mid_attn_w.self_attn.wv, SERVER);
    input::call(mid_self_bv, SERVER);
    input::call(mid_attn_w.self_attn.wout, SERVER);
    input::call(mid_self_bout, SERVER);
    input::call(mid_attn_w.cross_attn.wq, SERVER);
    input::call(mid_cross_bq, SERVER);
    input::call(mid_attn_w.cross_attn.wk, SERVER);
    input::call(mid_cross_bk, SERVER);
    input::call(mid_attn_w.cross_attn.wv, SERVER);
    input::call(mid_cross_bv, SERVER);
    input::call(mid_attn_w.cross_attn.wout, SERVER);
    input::call(mid_cross_bout, SERVER);
    input::call(mid_attn_w.ff.w_up, SERVER);
    input::call(mid_ff_b_up, SERVER);
    input::call(mid_attn_w.ff.w_down, SERVER);
    input::call(mid_ff_b_down, SERVER);
    input::call(vae_post_w, SERVER);
    input::call(vae_post_b, SERVER);
    input::call(vae_res1_c1_w, SERVER);
    input::call(vae_res1_c1_b, SERVER);
    input::call(vae_res1_c2_w, SERVER);
    input::call(vae_res1_c2_b, SERVER);
    input::call(vae_res2_c1_w, SERVER);
    input::call(vae_res2_c1_b, SERVER);
    input::call(vae_res2_c2_w, SERVER);
    input::call(vae_res2_c2_b, SERVER);
    input::call(vae_attn_w.wq, SERVER);
    input::call(vae_attn_bq, SERVER);
    input::call(vae_attn_w.wk, SERVER);
    input::call(vae_attn_bk, SERVER);
    input::call(vae_attn_w.wv, SERVER);
    input::call(vae_attn_bv, SERVER);
    input::call(vae_attn_w.wout, SERVER);
    input::call(vae_attn_bout, SERVER);
    input::call(vae_out_w, SERVER);
    input::call(vae_out_b, SERVER);

    // Authenticate biases and additive parameters once.
    auto img_patch_b_a = auth_from_plain_open(img_patch_b);
    auto img_cls_a = auth_from_plain_open(img_cls);
    auto img_proj_b_a = auth_from_plain_open(img_proj_b);
    auto img_proj_b2_a = auth_from_plain_open(img_proj_b2);
    auto unet_conv_in_b_a = auth_from_plain_open(unet_conv_in_b);
    auto unet_conv_out_b_a = auth_from_plain_open(unet_conv_out_b);
    auto rb1_conv1_b_a = auth_from_plain_open(rb1_conv1_b);
    auto rb1_conv2_b_a = auth_from_plain_open(rb1_conv2_b);
    auto rb1_temb_b_a = auth_from_plain_open(rb1_temb_b);
    auto rb2_conv1_b_a = auth_from_plain_open(rb2_conv1_b);
    auto rb2_conv2_b_a = auth_from_plain_open(rb2_conv2_b);
    auto rb2_temb_b_a = auth_from_plain_open(rb2_temb_b);
    auto time_b1_a = auth_from_plain_open(time_b1);
    auto time_b2_a = auth_from_plain_open(time_b2);
    auto vae_post_b_a = auth_from_plain_open(vae_post_b);
    auto vae_res1_c1_b_a = auth_from_plain_open(vae_res1_c1_b);
    auto vae_res1_c2_b_a = auth_from_plain_open(vae_res1_c2_b);
    auto vae_res2_c1_b_a = auth_from_plain_open(vae_res2_c1_b);
    auto vae_res2_c2_b_a = auth_from_plain_open(vae_res2_c2_b);
    auto vae_out_b_a = auth_from_plain_open(vae_out_b);

    mid_attn_w.self_attn.bq = auth_from_plain_open(mid_self_bq);
    mid_attn_w.self_attn.bk = auth_from_plain_open(mid_self_bk);
    mid_attn_w.self_attn.bv = auth_from_plain_open(mid_self_bv);
    mid_attn_w.self_attn.bout = auth_from_plain_open(mid_self_bout);
    mid_attn_w.cross_attn.bq = auth_from_plain_open(mid_cross_bq);
    mid_attn_w.cross_attn.bk = auth_from_plain_open(mid_cross_bk);
    mid_attn_w.cross_attn.bv = auth_from_plain_open(mid_cross_bv);
    mid_attn_w.cross_attn.bout = auth_from_plain_open(mid_cross_bout);
    mid_attn_w.ff.b_up = auth_from_plain_open(mid_ff_b_up);
    mid_attn_w.ff.b_down = auth_from_plain_open(mid_ff_b_down);

    vae_attn_w.bq = auth_from_plain_open(vae_attn_bq);
    vae_attn_w.bk = auth_from_plain_open(vae_attn_bk);
    vae_attn_w.bv = auth_from_plain_open(vae_attn_bv);
    vae_attn_w.bout = auth_from_plain_open(vae_attn_bout);

    // -----------------------------
    // Inputs: prompt + image
    // -----------------------------
    // Prompt token ids (toy, fixed length 4).
    std::vector<u64> prompt_ids(seq_len, 0);
    span<u64> prompt_embeds(batch * seq_len * hidden);
    if (party == SERVER || party == DEALER) {
        for (u64 i = 0; i < seq_len; ++i) {
            u64 tok = prompt_ids[i] % vocab;
            prompt_embeds[i] = text_token_embed[tok * hidden] + text_pos_embed[i * hidden];
        }
    }
    input::call(prompt_embeds, SERVER);
    clip_batch_check("debug:after prompt_embeds");

    auto prompt_embeds_a = auth_from_plain_open(prompt_embeds);
    auto neg_prompt_embeds_a = make_public_raw(batch * seq_len * hidden, 0);

    // Concatenate for classifier-free guidance.
    AuthShare prompt_embeds_cfg = auth_alloc(2 * seq_len * hidden);
    for (u64 i = 0; i < seq_len * hidden; ++i) {
        prompt_embeds_cfg.share[i] = neg_prompt_embeds_a.share[i];
        prompt_embeds_cfg.tag[i] = neg_prompt_embeds_a.tag[i];
        prompt_embeds_cfg.share[seq_len * hidden + i] = prompt_embeds_a.share[i];
        prompt_embeds_cfg.tag[seq_len * hidden + i] = prompt_embeds_a.tag[i];
    }

    // Image input: [B, H, W, C] in HWC layout.
    span<u64> image_input(batch * H * W * C);
    if (party == CLIENT || party == DEALER) {
        current_rng_seed = base_rng_seed;
        std::vector<u64> image_values;
        const size_t needed = (size_t)(H * W * C);
        bool ok = load_fixed_list("image.txt", f, image_values, needed);
        if (!ok || image_values.size() < needed) {
            image_values.resize(needed);
            for (size_t i = 0; i < needed; ++i) image_values[i] = qrand_normal(f, 0.5);
        }
        for (u64 h = 0; h < H; ++h) {
            for (u64 widx = 0; widx < W; ++widx) {
                for (u64 c = 0; c < C; ++c) {
                    size_t src = ((size_t)h * W + widx) * C + c;
                    size_t dst = ((size_t)h * W + widx) * C + c;
                    image_input[dst] = image_values[src];
                }
            }
        }
    }
    input::call(image_input, CLIENT);
    clip_batch_check("debug:after image_input");
    auto image_input_a = auth_from_plain_open(image_input);

    // -----------------------------
    // Image encoder (patch conv + proj)
    // -----------------------------
    // Patch conv (k=1) => output [B, H, W, 1]
    auto patch = conv2d_apply(batch, H, W, C, img_emb_dim, 1, 1, 0, image_input_a, img_patch_w, &img_patch_b_a);
    clip_batch_check("debug:after patch conv");

    // Flatten patches to [N, 1], add cls token and (zero) pos embeds.
    const u64 N = H * W;
    AuthShare img_tokens = auth_alloc((N + 1) * img_emb_dim);
    for (u64 i = 0; i < img_emb_dim; ++i) {
        img_tokens.share[i] = img_cls_a.share[i];
        img_tokens.tag[i] = img_cls_a.tag[i];
    }
    for (u64 i = 0; i < N; ++i) {
        for (u64 c = 0; c < img_emb_dim; ++c) {
            u64 dst = (i + 1) * img_emb_dim + c;
            u64 src = i * img_emb_dim + c;
            img_tokens.share[dst] = patch.share[src];
            img_tokens.tag[dst] = patch.tag[src];
        }
    }
    // Layernorm across tokens (approx).
    auto img_tokens_ln = timed_layernorm_rows("layernorm:image_encoder.tokens", N + 1, img_emb_dim, img_tokens);
    // CLS token -> projection
    AuthShare cls_token = auth_alloc(img_emb_dim);
    for (u64 i = 0; i < img_emb_dim; ++i) {
        cls_token.share[i] = img_tokens_ln.share[i];
        cls_token.tag[i] = img_tokens_ln.tag[i];
    }
    auto img_embed = linear_mat(1, img_emb_dim, img_emb_dim, cls_token, img_proj_w, &img_proj_b_a);

    // -----------------------------
    // Image noising (DDPM)
    // -----------------------------
    auto alphas_cumprod_ddpm = build_alphas_cumprod_cos(1000);
    double alpha_noise = alphas_cumprod_ddpm[(int)noise_level];
    span<u64> noise(img_embed.share.size());
    if (party == CLIENT || party == DEALER) {
        current_rng_seed = base_rng_seed;
        for (u64 i = 0; i < noise.size(); ++i) noise[i] = qrand_normal(f, 1.0);
    }
    input::call(noise, CLIENT);
    auto noise_a = auth_from_plain_open(noise);
    auto img_embed_noised = add_noise_ddpm(img_embed, noise_a, alpha_noise);

    // Noise-level embedding dimension is 1; using zero as embedding for dim=1.
    auto noise_level_embed = make_public_raw(img_embed.share.size(), 0);

    AuthShare class_labels = auth_alloc(2 * img_emb_dim);
    class_labels.share[0] = img_embed_noised.share[0];
    class_labels.tag[0] = img_embed_noised.tag[0];
    class_labels.share[1] = noise_level_embed.share[0];
    class_labels.tag[1] = noise_level_embed.tag[0];

    // CFG: prepend zeros.
    AuthShare class_labels_cfg = auth_alloc(2 * class_labels.share.size());
    for (u64 i = 0; i < class_labels.share.size(); ++i) {
        class_labels_cfg.share[i] = 0;
        class_labels_cfg.tag[i] = 0;
        class_labels_cfg.share[class_labels.share.size() + i] = class_labels.share[i];
        class_labels_cfg.tag[class_labels.share.size() + i] = class_labels.tag[i];
    }

    // -----------------------------
    // Latents (random)
    // -----------------------------
    span<u64> latents(batch * H * W * latent_C);
    if (party == CLIENT || party == DEALER) {
        current_rng_seed = base_rng_seed;
        for (u64 i = 0; i < latents.size(); ++i) latents[i] = qrand_normal(f, 1.0);
    }
    input::call(latents, CLIENT);
    auto latents_a = auth_from_plain_open(latents);

    // Scheduler setup (PNDM-like, simplified).
    auto alphas_cumprod = build_alphas_cumprod(1000, 0.00085, 0.012);
    std::vector<int> timesteps(num_inference_steps);
    for (int i = 0; i < num_inference_steps; ++i) {
        double t = (double)(1000 - 1) * (1.0 - (double)i / (double)(num_inference_steps - 1));
        timesteps[i] = (int)std::llround(t);
    }

    // -----------------------------
    // UNet forward (minimal)
    // -----------------------------
    auto unet_forward = [&](const AuthShare &lat_in, const AuthShare &prompt_ctx, const AuthShare &cls_labels) -> AuthShare {
        u64 B = 2;
        u64 Ht = H;
        u64 Wt = W;
        // time embedding: dim=1 -> zero.
        auto t_embed = make_public_raw(B * 1, 0);

        auto t1 = linear_mat(B, 1, 1, t_embed, time_w1, &time_b1_a);
        auto t1_act = timed_silu("silu:unet.time_mlp", t1);
        auto temb = linear_mat(B, 1, 1, t1_act, time_w2, &time_b2_a);

        auto imgp = linear_mat(B, 2, 1, cls_labels, img_proj_w2, &img_proj_b2_a);
        temb = ADD_CALL(temb, imgp);

        // conv_in
        auto h0 = conv2d_apply_k2_same(B, Ht, Wt, latent_C, latent_C, 1, lat_in, unet_conv_in_w, &unet_conv_in_b_a);

        // ResBlock1
        auto h1n = timed_groupnorm_apply("layernorm:unet.rb1.gn1", B, Ht, Wt, latent_C, h0);
        auto h1a = timed_silu("silu:unet.rb1.act1", h1n);
        auto h1c1 = conv2d_apply_k2_same(B, Ht, Wt, latent_C, latent_C, 1, h1a, rb1_conv1_w, &rb1_conv1_b_a);
        // temb broadcast
        auto temb1 = linear_mat(B, 1, latent_C, temb, rb1_temb_w, &rb1_temb_b_a);
        for (u64 n = 0; n < B; ++n) {
            for (u64 c = 0; c < latent_C; ++c) {
                u64 t_idx = n * latent_C + c;
                for (u64 h = 0; h < Ht; ++h) {
                    for (u64 widx = 0; widx < Wt; ++widx) {
                        u64 idx = idx4(n, h, widx, c, Ht, Wt, latent_C);
                        h1c1.share[idx] += temb1.share[t_idx];
                        h1c1.tag[idx] += temb1.tag[t_idx];
                    }
                }
            }
        }
        auto h1n2 = timed_groupnorm_apply("layernorm:unet.rb1.gn2", B, Ht, Wt, latent_C, h1c1);
        auto h1a2 = timed_silu("silu:unet.rb1.act2", h1n2);
        auto h1c2 = conv2d_apply_k2_same(B, Ht, Wt, latent_C, latent_C, 1, h1a2, rb1_conv2_w, &rb1_conv2_b_a);
        auto h1 = ADD_CALL(h1c2, h0);

        // Mid attention (Transformer2D)
        auto h1n3 = timed_groupnorm_apply("layernorm:unet.mid_attn.gn", B, Ht, Wt, latent_C, h1);
        // flatten to [B, N, C]
        u64 Nseq = Ht * Wt;
        AuthShare h_flat = auth_alloc(B * Nseq * latent_C);
        for (u64 n = 0; n < B; ++n) {
            for (u64 h = 0; h < Ht; ++h) {
                for (u64 widx = 0; widx < Wt; ++widx) {
                    for (u64 c = 0; c < latent_C; ++c) {
                        u64 src = idx4(n, h, widx, c, Ht, Wt, latent_C);
                        u64 dst = (n * Nseq + (h * Wt + widx)) * latent_C + c;
                        h_flat.share[dst] = h1n3.share[src];
                        h_flat.tag[dst] = h1n3.tag[src];
                    }
                }
            }
        }

        // transformer per batch
        AuthShare h_out_flat = auth_alloc(B * Nseq * latent_C);
        for (u64 n = 0; n < B; ++n) {
            u64 offset = n * Nseq * latent_C;
            AuthShare x_b{
                span<u128>(h_flat.share.data() + offset, Nseq * latent_C),
                span<u128>(h_flat.tag.data() + offset, Nseq * latent_C)
            };
            u64 ctx_offset = n * seq_len * ctx_dim;
            AuthShare ctx_b{
                span<u128>(prompt_ctx.share.data() + ctx_offset, seq_len * ctx_dim),
                span<u128>(prompt_ctx.tag.data() + ctx_offset, seq_len * ctx_dim)
            };
            auto x_b_out = basic_transformer_block(Nseq, latent_C, seq_len, x_b, ctx_b, mid_attn_w);
            for (u64 i = 0; i < Nseq * latent_C; ++i) {
                h_out_flat.share[offset + i] = x_b_out.share[i];
                h_out_flat.tag[offset + i] = x_b_out.tag[i];
            }
        }

        // reshape back and residual
        AuthShare h_attn = auth_alloc(B * Ht * Wt * latent_C);
        for (u64 n = 0; n < B; ++n) {
            for (u64 h = 0; h < Ht; ++h) {
                for (u64 widx = 0; widx < Wt; ++widx) {
                    for (u64 c = 0; c < latent_C; ++c) {
                        u64 dst = idx4(n, h, widx, c, Ht, Wt, latent_C);
                        u64 src = (n * Nseq + (h * Wt + widx)) * latent_C + c;
                        h_attn.share[dst] = h_out_flat.share[src] + h1.share[dst];
                        h_attn.tag[dst] = h_out_flat.tag[src] + h1.tag[dst];
                    }
                }
            }
        }

        // ResBlock2
        auto h2n = timed_groupnorm_apply("layernorm:unet.rb2.gn1", B, Ht, Wt, latent_C, h_attn);
        auto h2a = timed_silu("silu:unet.rb2.act1", h2n);
        auto h2c1 = conv2d_apply_k2_same(B, Ht, Wt, latent_C, latent_C, 1, h2a, rb2_conv1_w, &rb2_conv1_b_a);
        auto temb2 = linear_mat(B, 1, latent_C, temb, rb2_temb_w, &rb2_temb_b_a);
        for (u64 n = 0; n < B; ++n) {
            for (u64 c = 0; c < latent_C; ++c) {
                u64 t_idx = n * latent_C + c;
                for (u64 h = 0; h < Ht; ++h) {
                    for (u64 widx = 0; widx < Wt; ++widx) {
                        u64 idx = idx4(n, h, widx, c, Ht, Wt, latent_C);
                        h2c1.share[idx] += temb2.share[t_idx];
                        h2c1.tag[idx] += temb2.tag[t_idx];
                    }
                }
            }
        }
        auto h2n2 = timed_groupnorm_apply("layernorm:unet.rb2.gn2", B, Ht, Wt, latent_C, h2c1);
        auto h2a2 = timed_silu("silu:unet.rb2.act2", h2n2);
        auto h2c2 = conv2d_apply_k2_same(B, Ht, Wt, latent_C, latent_C, 1, h2a2, rb2_conv2_w, &rb2_conv2_b_a);
        auto h2 = ADD_CALL(h2c2, h_attn);

        auto out = conv2d_apply_k2_same(B, Ht, Wt, latent_C, latent_C, 1, h2, unet_conv_out_w, &unet_conv_out_b_a);
        return out;
    };

    // Diffusion loop
    for (int step = 0; step < num_inference_steps; ++step) {
        // Duplicate latents for CFG
        AuthShare latents_cfg = auth_alloc(2 * latents_a.share.size());
        for (u64 i = 0; i < latents_a.share.size(); ++i) {
            latents_cfg.share[i] = latents_a.share[i];
            latents_cfg.tag[i] = latents_a.tag[i];
            latents_cfg.share[latents_a.share.size() + i] = latents_a.share[i];
            latents_cfg.tag[latents_a.share.size() + i] = latents_a.tag[i];
        }

        auto noise_pred = unet_forward(latents_cfg, prompt_embeds_cfg, class_labels_cfg);

        // Split and combine
        AuthShare uncond = auth_alloc(latents_a.share.size());
        AuthShare text = auth_alloc(latents_a.share.size());
        for (u64 i = 0; i < latents_a.share.size(); ++i) {
            uncond.share[i] = noise_pred.share[i];
            uncond.tag[i] = noise_pred.tag[i];
            text.share[i] = noise_pred.share[latents_a.share.size() + i];
            text.tag[i] = noise_pred.tag[latents_a.share.size() + i];
        }

        auto diff = ADD_CALL(text, neg_span(uncond));
        auto scale = make_public_const(diff.share.size(), guidance_scale, f);
        auto scaled = LRS_CALL(MUL_CALL(diff, scale), f);
        auto guided = ADD_CALL(uncond, scaled);

        int t = timesteps[step];
        int t_prev = std::max(t - 1, 0);
        double alpha_t = alphas_cumprod[t];
        double alpha_prev = alphas_cumprod[t_prev];
        latents_a = scheduler_step_linear(latents_a, guided, alpha_t, alpha_prev);
    }

    // -----------------------------
    // VAE decode (lightweight but uses secure ops)
    // -----------------------------
    double scaling_factor = 0.18215;
    auto inv_scale = make_public_const(latents_a.share.size(), 1.0 / scaling_factor, f);
    auto latents_scaled = LRS_CALL(MUL_CALL(latents_a, inv_scale), f);

    auto z = conv2d_apply(batch, H, W, latent_C, latent_C, 1, 1, 0, latents_scaled, vae_post_w, &vae_post_b_a);

    auto r1n = timed_groupnorm_apply("layernorm:vae.res1.gn1", batch, H, W, latent_C, z);
    auto r1a = timed_silu("silu:vae.res1.act1", r1n);
    auto r1c1 = conv2d_apply_k2_same(batch, H, W, latent_C, latent_C, 1, r1a, vae_res1_c1_w, &vae_res1_c1_b_a);
    auto r1n2 = timed_groupnorm_apply("layernorm:vae.res1.gn2", batch, H, W, latent_C, r1c1);
    auto r1a2 = timed_silu("silu:vae.res1.act2", r1n2);
    auto r1c2 = conv2d_apply_k2_same(batch, H, W, latent_C, latent_C, 1, r1a2, vae_res1_c2_w, &vae_res1_c2_b_a);
    auto r1 = ADD_CALL(r1c2, z);

    // VAE self-attn
    auto r1n3 = timed_groupnorm_apply("layernorm:vae.attn.gn", batch, H, W, latent_C, r1);
    u64 Nseq = H * W;
    AuthShare r1_flat = auth_alloc(batch * Nseq * latent_C);
    for (u64 n = 0; n < batch; ++n) {
        for (u64 h = 0; h < H; ++h) {
            for (u64 widx = 0; widx < W; ++widx) {
                for (u64 c = 0; c < latent_C; ++c) {
                    u64 src = idx4(n, h, widx, c, H, W, latent_C);
                    u64 dst = (n * Nseq + (h * W + widx)) * latent_C + c;
                    r1_flat.share[dst] = r1n3.share[src];
                    r1_flat.tag[dst] = r1n3.tag[src];
                }
            }
        }
    }
    AuthShare attn_out = auth_alloc(batch * Nseq * latent_C);
    for (u64 n = 0; n < batch; ++n) {
        u64 offset = n * Nseq * latent_C;
        AuthShare x_b{
            span<u128>(r1_flat.share.data() + offset, Nseq * latent_C),
            span<u128>(r1_flat.tag.data() + offset, Nseq * latent_C)
        };
        auto x_b_out = apply_attn(x_b, Nseq, latent_C, x_b, Nseq, vae_attn_w, "softmax:vae.self_attn");
        for (u64 i = 0; i < Nseq * latent_C; ++i) {
            attn_out.share[offset + i] = x_b_out.share[i];
            attn_out.tag[offset + i] = x_b_out.tag[i];
        }
    }
    AuthShare r1_attn = auth_alloc(batch * H * W * latent_C);
    for (u64 n = 0; n < batch; ++n) {
        for (u64 h = 0; h < H; ++h) {
            for (u64 widx = 0; widx < W; ++widx) {
                for (u64 c = 0; c < latent_C; ++c) {
                    u64 dst = idx4(n, h, widx, c, H, W, latent_C);
                    u64 src = (n * Nseq + (h * W + widx)) * latent_C + c;
                    r1_attn.share[dst] = attn_out.share[src] + r1.share[dst];
                    r1_attn.tag[dst] = attn_out.tag[src] + r1.tag[dst];
                }
            }
        }
    }

    auto r2n = timed_groupnorm_apply("layernorm:vae.res2.gn1", batch, H, W, latent_C, r1_attn);
    auto r2a = timed_silu("silu:vae.res2.act1", r2n);
    auto r2c1 = conv2d_apply_k2_same(batch, H, W, latent_C, latent_C, 1, r2a, vae_res2_c1_w, &vae_res2_c1_b_a);
    auto r2n2 = timed_groupnorm_apply("layernorm:vae.res2.gn2", batch, H, W, latent_C, r2c1);
    auto r2a2 = timed_silu("silu:vae.res2.act2", r2n2);
    auto r2c2 = conv2d_apply_k2_same(batch, H, W, latent_C, latent_C, 1, r2a2, vae_res2_c2_w, &vae_res2_c2_b_a);
    auto r2 = ADD_CALL(r2c2, r1_attn);

    auto out_img = conv2d_apply_k2_same(batch, H, W, latent_C, C, 1, r2, vae_out_w, &vae_out_b_a);
    auto out_tanh = tanh_apply(out_img);

    // Reveal first few pixels for debug.
    if (party != DEALER) {
        auto out_plain = authenticated_reconstruct(out_tanh.share, out_tanh.tag);
        clip_batch_check("output");
        std::cout << "[UNCLIP] output first 8 values: ";
        for (u64 i = 0; i < std::min<u64>(8, out_plain.size()); ++i) {
            std::cout << (int64_t)out_plain[i] << " ";
        }
        std::cout << std::endl;
        if (party == CLIENT) {
            if (write_jpg_from_fixed(out_plain, H, W, C, f, "unclip_out.jpg")) {
                std::cout << "[UNCLIP] wrote unclip_out.jpg" << std::endl;
            } else {
                std::cout << "[UNCLIP] failed to write unclip_out.jpg" << std::endl;
            }
        }
    }

    if (party != DEALER) {
        shark::utils::stop_timer("total_eval");
        if (shark::protocols::peer) {
            u64 total_comm = shark::protocols::peer->bytesReceived() + shark::protocols::peer->bytesSent();
            u64 total_rounds = shark::protocols::peer->roundsReceived() + shark::protocols::peer->roundsSent();
            std::cout << "[PROFILE] total_comm: " << (double)total_comm / 1024.0 << " KB (" << total_comm << " bytes)" << std::endl;
            std::cout << "[PROFILE] total_rounds: " << total_rounds << std::endl;
        }
        print_profile_timers();
    }

    finalize::call();
    return 0;
}
