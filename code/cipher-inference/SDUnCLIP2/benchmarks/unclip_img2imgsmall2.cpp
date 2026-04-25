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
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
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

struct AuthTensor4D {
    AuthShare data;
    u64 B = 0;
    u64 H = 0;
    u64 W = 0;
    u64 C = 0;
};

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

static span<u64> auth_low(const AuthShare &x) {
    span<u64> out(x.share.size());
    #pragma omp parallel for
    for (u64 i = 0; i < x.share.size(); ++i) out[i] = getLow(x.share[i]);
    return out;
}

static AuthShare auth_from_public_raw(u64 size, u64 val) {
    AuthShare out = auth_alloc(size);
    u128 tag_val = mac_mul_u64(val);
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
            out.tag[i] = mac_mul_u64(x[i]);
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
        u64 d_local = (party == CLIENT) ? d[i] : 0;
        out.share[i] = r_share[i] + d_local;
        out.tag[i] = mac_add(r_tag[i], mac_mul_u64(d[i]));
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

static inline u64 qrand_uniform(int fp) {
    double u = rng_u01(current_rng_seed);
    int64_t q = (int64_t)std::llround(u * (double)(1ULL << fp));
    return (u64)q;
}

static inline u64 qrand_uniform_scaled(int fp, double maxv) {
    double u = rng_u01(current_rng_seed);
    double x = u * maxv;
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

static span<u64> make_timestep_embedding(int timestep, u64 dim, int fp, bool flip_sin_to_cos = true) {
    span<u64> out(dim);
    if (dim == 0) return out;
    if (party == SERVER || party == DEALER) {
        const u64 half = dim / 2;
        std::vector<double> vals(dim, 0.0);
        double log_max = std::log(10000.0);
        double denom = (half > 1) ? (half - 1) : 1;
        for (u64 i = 0; i < half; ++i) {
            double freq = std::exp(-log_max * (double)i / denom);
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
    } else {
        for (u64 i = 0; i < dim; ++i) out[i] = 0;
    }
    return out;
}

static u64 fnv1a_hash(const std::string &s) {
    const u64 kOffset = 1469598103934665603ull;
    const u64 kPrime = 1099511628211ull;
    u64 h = kOffset;
    for (unsigned char ch : s) {
        h ^= (u64)ch;
        h *= kPrime;
    }
    return h;
}

static std::string to_lower_ascii(const std::string &s) {
    std::string out = s;
    for (char &ch : out) {
        ch = (char)std::tolower((unsigned char)ch);
    }
    return out;
}

static std::vector<u64> tokenize_prompt(const std::string &prompt, u64 vocab_size, u64 max_length) {
    const u64 bos = 0;
    const u64 eos = 2;
    const u64 pad = 1;
    const u64 token_offset = 3;
    std::vector<u64> ids;
    ids.reserve(max_length);
    ids.push_back(bos);
    std::istringstream iss(to_lower_ascii(prompt));
    std::string word;
    while (iss >> word) {
        u64 h = fnv1a_hash(word);
        u64 tok = token_offset + (h % (vocab_size - token_offset));
        ids.push_back(tok);
    }
    ids.push_back(eos);
    if (ids.size() > max_length) {
        ids.resize(max_length);
        ids[max_length - 1] = eos;
    }
    while (ids.size() < max_length) ids.push_back(pad);
    return ids;
}

static span<u64> build_prompt_embeds(const std::vector<u64> &ids, const span<u64> &token_embed,
                                     const span<u64> &pos_embed, u64 hidden, u64 seq_len) {
    span<u64> out(seq_len * hidden);
    const u64 pad = 1;
    u64 pos = 0;
    for (u64 i = 0; i < seq_len; ++i) {
        u64 tok = ids[i];
        u64 pos_id = 0;
        if (tok != pad) {
            pos_id = pos;
            pos = (pos + 1 < seq_len) ? (pos + 1) : pos;
        }
        u64 tok_base = tok * hidden;
        u64 pos_base = pos_id * hidden;
        for (u64 c = 0; c < hidden; ++c) {
            out[i * hidden + c] = token_embed[tok_base + c] + pos_embed[pos_base + c];
        }
    }
    return out;
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
            out.tag[i] = mac_mul_u64(bit);
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

static bool profile_progress_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = (std::getenv("SHARK_PROFILE_PROGRESS") != nullptr) ? 1 : 0;
    }
    return enabled == 1;
}

static bool profile_step_summary_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = (std::getenv("SHARK_PROFILE_STEP_SUMMARY") != nullptr) ? 1 : 0;
    }
    return enabled == 1;
}

static void profile_progress(const char *label) {
    if (!profile_progress_enabled() || party == DEALER || !peer) {
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
    std::cout << "[PROFILE_PROGRESS] " << label
              << " | elapsed=" << elapsed_ms << " ms"
              << ", total_comm=" << (comm / 1024.0) << " KB"
              << ", total_rounds=" << rounds
              << ", delta_comm=" << (delta_comm / 1024.0) << " KB"
              << ", delta_rounds=" << delta_rounds
              << std::endl;
}

static void clip_batch_check(const char* label) {
    shark::protocols::debug_batch_check(label);
    profile_progress(label);
}

// Forward declarations for timed wrappers.
static AuthShare gelu_apply(const AuthShare &x);
static AuthShare silu_apply(const AuthShare &x);
static AuthShare layernorm_rows(u64 rows, u64 cols, const AuthShare &X,
                                const AuthShare *weight = nullptr, const AuthShare *bias = nullptr);
static AuthShare groupnorm_apply(u64 B, u64 H, u64 W, u64 C, const AuthShare &x,
                                 const AuthShare *weight = nullptr, const AuthShare *bias = nullptr);
static AuthShare groupnorm_apply_groups(u64 B, u64 H, u64 W, u64 C, u64 groups, const AuthShare &x,
                                        const AuthShare *weight = nullptr, const AuthShare *bias = nullptr);

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

static bool keygen_progress_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = (std::getenv("SHARK_KEYGEN_PROGRESS") != nullptr) ? 1 : 0;
    }
    return enabled == 1;
}

// Keygen progress metadata (dealer-only display path).
static u64 keygen_global_step = 0;
static u64 keygen_global_total = 0;
static u64 keygen_layer_total[8] = {0};
static u64 keygen_layer_seen[8] = {0};

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
    keygen_layer_total[4] = (num_inference_steps > 0) ? (u64)num_inference_steps : 0;
    keygen_layer_total[5] = (num_inference_steps > 0) ? 6ull * (u64)num_inference_steps : 0;
    keygen_layer_total[6] = 2;  // vae decode, optional superres
    keygen_global_total = 0;
    for (u64 i = 0; i < 8; ++i) {
        keygen_global_total += keygen_layer_total[i];
    }
    keygen_global_step = 0;
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

static AuthShare timed_layernorm_rows(const char *name, u64 rows, u64 cols, const AuthShare &x,
                                      const AuthShare *weight = nullptr, const AuthShare *bias = nullptr) {
    shark::utils::start_timer(name);
    auto out = layernorm_rows(rows, cols, x, weight, bias);
    shark::utils::stop_timer(name);
    return out;
}

static AuthShare timed_groupnorm_apply(const char *name, u64 B, u64 H, u64 W, u64 C, const AuthShare &x,
                                       const AuthShare *weight = nullptr, const AuthShare *bias = nullptr) {
    shark::utils::start_timer(name);
    auto out = groupnorm_apply(B, H, W, C, x, weight, bias);
    shark::utils::stop_timer(name);
    return out;
}

static AuthShare timed_groupnorm_apply(const char *name, u64 B, u64 H, u64 W, u64 C,
                                       u64 groups, const AuthShare &x,
                                       const AuthShare *weight = nullptr, const AuthShare *bias = nullptr) {
    shark::utils::start_timer(name);
    auto out = groupnorm_apply_groups(B, H, W, C, groups, x, weight, bias);
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

static void print_profile_components_table() {
    if (party == DEALER) {
        return;
    }

    struct Row {
        const char *component;
        const char *timer_name;
    };

    // Rows aligned with the paper-style component list.
    const Row rows[] = {
        {"exp_mlp", "exp_mlp"},
        {"multiplication_attn.ff.gate", "multiplication_attn.ff.gate"},
        {"linear1_mlp", "linear1_mlp"},
        {"linear2_attn.ff.gate", "linear2_attn.ff.gate"},
        {"linear3_mlp", "linear3_mlp"},
        {"linear4_attn.ff.gate", "linear4_attn.ff.gate"},
        {"gelu:unet.time_mlp", "gelu:unet.time_mlp"},
        {"gelu:unet.mid_attn.ff.gate", "gelu:unet.mid_attn.ff.gate"},
        {"silu:unet.rb1.act1", "silu:unet.rb1.act1"},
        {"silu:vae.res1.act1", "silu:vae.res1.act1"},
        {"layernorm:image_encoder.tokens", "layernorm:image_encoder.tokens"},
        {"layernorm:unet.rb1.gn1", "layernorm:unet.rb1.gn1"},
        {"layernorm:unet.rb1.gn2", "layernorm:unet.rb1.gn2"},
        {"layernorm:unet.mid_attn.gn", "layernorm:unet.mid_attn.gn"},
        {"layernorm:unet.mid_attn.norm1", "layernorm:unet.mid_attn.norm1"},
        {"layernorm:unet.rb2.gn1", "layernorm:unet.rb2.gn1"},
        {"layernorm:vae.res1.gn1", "layernorm:vae.res1.gn1"},
        {"softmax:unet.mid_attn.self", "softmax:unet.mid_attn.self"},
        {"softmax:unet.mid_attn.cross", "softmax:unet.mid_attn.cross"},
        {"softmax:vae.self_attn", "softmax:vae.self_attn"},
        {"end-to-end", "total_eval"},
    };

    std::cout << "[PROFILE_TABLE] component,time_ms,comm_mb,rounds" << std::endl;
    for (const auto &row : rows) {
        shark::utils::TimerStat stat{};
        shark::utils::get_timer_stat(row.timer_name, stat);
        std::cout << "[PROFILE_TABLE] " << row.component
                  << "," << stat.accumulated_time
                  << "," << (double)stat.accumulated_comm / (1024.0 * 1024.0)
                  << "," << stat.accumulated_rounds
                  << std::endl;
    }

    if (shark::protocols::peer) {
        u64 total_comm = shark::protocols::peer->bytesReceived() + shark::protocols::peer->bytesSent();
        u64 total_rounds = shark::protocols::peer->roundsReceived() + shark::protocols::peer->roundsSent();
        std::cout << "[PROFILE_TABLE] network_total"
                  << ",-"
                  << "," << (double)total_comm / (1024.0 * 1024.0)
                  << "," << total_rounds
                  << std::endl;
    }
}

static void print_legacy_profile_lines() {
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
    print_ms_kb("key_read", "key_read");
    print_ms_kb("key_read-input", "key_read-input");
    print_ms_kb("reconstruct", "reconstruct");
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
static AuthShare silu_apply(const AuthShare &x) {
    auto add_const = [&](const AuthShare &src, double v) {
        auto c = make_public_const(src.share.size(), v, f);
        return ADD_CALL(src, c);
    };

    auto x2 = LRS_CALL(MUL_CALL(x, x), f);
    auto x3 = LRS_CALL(MUL_CALL(x2, x), f);
    auto x4 = LRS_CALL(MUL_CALL(x2, x2), f);
    auto x5 = LRS_CALL(MUL_CALL(x4, x), f);

    // 5th-order polynomial coefficients for SiLU(x)=x*sigmoid(x), on negative side:
    // outer in [-7, -2.2), inner in [-2.2, 0).
    const auto cO0 = make_public_const(x.share.size(), -3.24570114e-01, f);
    const auto cO1 = make_public_const(x.share.size(), 1.00306737e-01, f);
    const auto cO2 = make_public_const(x.share.size(), 1.23002598e-01, f);
    const auto cO3 = make_public_const(x.share.size(), 3.25959390e-02, f);
    const auto cO4 = make_public_const(x.share.size(), 3.60808855e-03, f);
    const auto cO5 = make_public_const(x.share.size(), 1.48142321e-04, f);

    const auto cI0 = make_public_const(x.share.size(), 7.17527925e-05, f);
    const auto cI1 = make_public_const(x.share.size(), 5.01226523e-01, f);
    const auto cI2 = make_public_const(x.share.size(), 2.54580274e-01, f);
    const auto cI3 = make_public_const(x.share.size(), 5.36077284e-03, f);
    const auto cI4 = make_public_const(x.share.size(), -2.08367638e-02, f);
    const auto cI5 = make_public_const(x.share.size(), -3.80900179e-03, f);

    auto tO1 = LRS_CALL(MUL_CALL(cO1, x), f);
    auto tO2 = LRS_CALL(MUL_CALL(cO2, x2), f);
    auto tO3 = LRS_CALL(MUL_CALL(cO3, x3), f);
    auto tO4 = LRS_CALL(MUL_CALL(cO4, x4), f);
    auto tO5 = LRS_CALL(MUL_CALL(cO5, x5), f);

    auto tI1 = LRS_CALL(MUL_CALL(cI1, x), f);
    auto tI2 = LRS_CALL(MUL_CALL(cI2, x2), f);
    auto tI3 = LRS_CALL(MUL_CALL(cI3, x3), f);
    auto tI4 = LRS_CALL(MUL_CALL(cI4, x4), f);
    auto tI5 = LRS_CALL(MUL_CALL(cI5, x5), f);

    auto seg_outer = cO0;
    seg_outer = ADD_CALL(seg_outer, tO1);
    seg_outer = ADD_CALL(seg_outer, tO2);
    seg_outer = ADD_CALL(seg_outer, tO3);
    seg_outer = ADD_CALL(seg_outer, tO4);
    seg_outer = ADD_CALL(seg_outer, tO5);

    auto seg_inner = cI0;
    seg_inner = ADD_CALL(seg_inner, tI1);
    seg_inner = ADD_CALL(seg_inner, tI2);
    seg_inner = ADD_CALL(seg_inner, tI3);
    seg_inner = ADD_CALL(seg_inner, tI4);
    seg_inner = ADD_CALL(seg_inner, tI5);

    // P(-x) = c0 - c1*x + c2*x^2 - c3*x^3 + c4*x^4 - c5*x^5
    auto seg_inner_f = cI0;
    seg_inner_f = ADD_CALL(seg_inner_f, neg_span(tI1));
    seg_inner_f = ADD_CALL(seg_inner_f, tI2);
    seg_inner_f = ADD_CALL(seg_inner_f, neg_span(tI3));
    seg_inner_f = ADD_CALL(seg_inner_f, tI4);
    seg_inner_f = ADD_CALL(seg_inner_f, neg_span(tI5));

    auto seg_outer_f = cO0;
    seg_outer_f = ADD_CALL(seg_outer_f, neg_span(tO1));
    seg_outer_f = ADD_CALL(seg_outer_f, tO2);
    seg_outer_f = ADD_CALL(seg_outer_f, neg_span(tO3));
    seg_outer_f = ADD_CALL(seg_outer_f, tO4);
    seg_outer_f = ADD_CALL(seg_outer_f, neg_span(tO5));

    auto zero = make_public_const(x.share.size(), 0.0, f);
    auto segL = zero;                      // x < -7
    auto segA = seg_outer;                 // [-7, -2.2)
    auto segB = seg_inner;                 // [-2.2, 0)
    auto segC = ADD_CALL(x, seg_inner_f);  // [0, 2.2)
    auto segD = ADD_CALL(x, seg_outer_f);  // [2.2, 7)
    auto segR = x;                         // x >= 7

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
    auto lt_neg7 = cmp_lt(x, -7.0 - eps);
    auto ge_neg2_2 = cmp_gt(x, -2.2);
    auto ge_0 = cmp_gt(x, 0.0);
    auto ge_2_2 = cmp_gt(x, 2.2);
    auto ge_7 = cmp_gt(x, 7.0);

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

// Degree-5 polynomial approximation of exp(u) on u in roughly [-7.5, 0], with P(0)=1.
static AuthShare poly5_exp_half(const AuthShare &u) {
    u64 n = u.share.size();

    auto c0 = make_public_const(n, 1.0, f);
    auto c1 = make_public_const(n, 0.9736789799002139, f);
    auto c2 = make_public_const(n, 0.4349844528710021, f);
    auto c3 = make_public_const(n, 0.10637438059443326, f);
    auto c4 = make_public_const(n, 0.013503439864618789, f);
    auto c5 = make_public_const(n, 0.000682062067913879, f);

    auto u2 = LRS_CALL(MUL_CALL(u, u), f);
    auto u4 = LRS_CALL(MUL_CALL(u2, u2), f);

    auto a0 = ADD_CALL(c0, LRS_CALL(MUL_CALL(c1, u), f));
    auto a1 = ADD_CALL(c2, LRS_CALL(MUL_CALL(c3, u), f));
    auto a2 = ADD_CALL(c4, LRS_CALL(MUL_CALL(c5, u), f));

    auto term1 = LRS_CALL(MUL_CALL(a1, u2), f);
    auto term2 = LRS_CALL(MUL_CALL(a2, u4), f);

    auto out = ADD_CALL(a0, term1);
    out = ADD_CALL(out, term2);
    return out;
}

static AuthShare approx_exp_sqrtexp_deg5(const AuthShare &x) {
    auto half = make_public_const(x.share.size(), 0.5, f);
    auto u = LRS_CALL(MUL_CALL(x, half), f);
    auto y = poly5_exp_half(u);
    auto y2 = LRS_CALL(MUL_CALL(y, y), f);
    return y2;
}

// Approximate 1/t with two Newton-Raphson steps: r <- r * (2 - t*r), r0=1.
static AuthShare inv_newton2(const AuthShare &t) {
    auto one = make_public_const(t.share.size(), 1.0, f);
    auto two = make_public_const(t.share.size(), 2.0, f);
    auto r = one;
    for (int i = 0; i < 2; ++i) {
        auto tr = mul_qf(t, r);
        auto two_minus = ADD_CALL(two, neg_span(tr));
        r = mul_qf(r, two_minus);
    }
    return r;
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
[[maybe_unused]] static AuthShare chexp(const AuthShare &x) {
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

    // Step 2: exp(x) approx via sqrt-exp trick:
    // u = 0.5 * x, y = poly5_exp_half(u), exp(x) ~= y^2
    auto exp_vals = approx_exp_sqrtexp_deg5(shifted);
    clip_batch_check("softmax after exp_deg5");

    // Step 3: Compute Row Sums
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

    // Step 4: Inverse Sum with NR2 on scaled sum:
    // alpha = 1/256, t = sum_exp * alpha, inv_sum = inv_newton2(t) * alpha.
    auto alpha = make_public_const(a, 1.0 / 256.0, f);
    auto t = LRS_CALL(MUL_CALL(row_sums, alpha), f);
    auto inv_t = inv_newton2(t);
    auto inv_sums = LRS_CALL(MUL_CALL(inv_t, alpha), f);
    clip_batch_check("softmax after inv_nr2");

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

    auto result = MUL_CALL(exp_vals, inv_sums_b);
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
        return auth_view(x, 0, x.share.size());
    }

    AuthShare out = auth_view(x, 0, x.share.size());
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

static AuthShare approx_inv_std_from_var(const AuthShare &var, int fp_bits) {
    const u64 n = var.share.size();
    const double min_var = std::max(kNormVarFloor, std::ldexp(1.0, -fp_bits));
    const double max_inv_std = 1.0 / std::sqrt(kNormEps);

    auto var_clamped = clamp_min_public(var, min_var, fp_bits);
    // Python keeps an exact rsqrt fallback here. This benchmark currently does not:
    // the repo's secure invertsqrt protocol is unfinished, so we only keep the
    // approximation path plus public clamping for numerical stabilization.

    AuthShare u = auth_mul_const(var_clamped, (u64)(1ULL << fp_bits));
    auto scale = make_public_const(n, rsqrt_scale_from_shift(fp_bits), fp_bits);

    // Find floor(log2(raw_var)) without opening the value and normalize to u in [1, 2).
    const int max_j = 48;
    for (int j = 1; j <= max_j; ++j) {
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

    auto r = rsqrt_seed_interval_1_2(u, fp_bits);
    r = rsqrt_newton_refine(u, r, fp_bits);
    auto inv_std = mul_qf(scale, r);
    return clamp_max_public(inv_std, max_inv_std, fp_bits);
}

static AuthShare layernorm_rows(u64 rows, u64 cols, const AuthShare &X,
                                const AuthShare *weight, const AuthShare *bias) {
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

    auto rsqrt_var = approx_inv_std_from_var(var_eps, f);
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
    out = apply_affine_lastdim(rows, cols, out, weight, bias);
    clip_batch_check("layernorm:after norm");
    return out;
}

static AuthShare groupnorm_apply_groups(u64 B, u64 H, u64 W, u64 C, u64 groups, const AuthShare &x,
                                        const AuthShare *weight, const AuthShare *bias) {
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
    auto mean = LRS_CALL(MUL_CALL(group_sum, inv_n), f);

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

    auto sqr = MUL_CALL(centered, centered);
    auto sqr_scaled = LRS_CALL(sqr, f);

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

    auto var = LRS_CALL(MUL_CALL(var_sum, inv_n), f);
    auto eps = make_public_const(B * groups, kNormEps, f);
    auto var_eps = ADD_CALL(var, eps);
    auto inv_std = approx_inv_std_from_var(var_eps, f);

    AuthShare out = auth_alloc(B * H * W * C);
    for (u64 n = 0; n < B; ++n) {
        for (u64 g = 0; g < groups; ++g) {
            u64 stat_idx = n * groups + g;
            for (u64 h = 0; h < H; ++h) {
                for (u64 widx = 0; widx < W; ++widx) {
                    for (u64 c = 0; c < ch_per_group; ++c) {
                        u64 global_c = g * ch_per_group + c;
                        u64 idx = idx4(n, h, widx, global_c, H, W, C);
                        out.share[idx] = centered.share[idx];
                        out.tag[idx] = centered.tag[idx];
                    }
                }
            }
        }
    }

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

    auto normed = LRS_CALL(MUL_CALL(out, inv_std_b), f);
    return apply_affine_lastdim(B * H * W, C, normed, weight, bias);
}

static AuthShare groupnorm_apply(u64 B, u64 H, u64 W, u64 C, const AuthShare &x,
                                 const AuthShare *weight, const AuthShare *bias) {
    return groupnorm_apply_groups(B, H, W, C, 1, x, weight, bias);
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
    span<u64> bq_plain, bk_plain, bv_plain, bout_plain;
    AuthShare bq, bk, bv, bout;
};

struct NormAffineWeights {
    span<u64> weight_plain, bias_plain;
    AuthShare weight, bias;
};

struct FFWeights {
    span<u64> w_up, w_down;
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
    span<u64> b1_plain, b2_plain;
    AuthShare b1, b2;
    u64 hidden_dim = 0;
};

struct ResBlock2DWeights {
    u64 in_ch = 0;
    u64 out_ch = 0;
    u64 temb_dim = 0;
    u64 norm_groups = 1;
    bool has_skip = false;
    NormAffineWeights gn1, gn2;
    span<u64> conv1_w, conv2_w, skip_w, temb_w;
    span<u64> conv1_b_plain, conv2_b_plain, skip_b_plain, temb_b_plain;
    AuthShare conv1_b, conv2_b, skip_b, temb_b;
};

struct Transformer2DWeights {
    u64 in_ch = 0;
    u64 inner_dim = 0;
    u64 ctx_dim = 0;
    u64 norm_groups = 1;
    NormAffineWeights gn;
    span<u64> proj_in_w, proj_out_w;
    span<u64> proj_in_b_plain, proj_out_b_plain;
    AuthShare proj_in_b, proj_out_b;
    std::vector<TransformerWeights> blocks;
};

struct FeatureUpscalerWeights {
    u64 in_dim = 0;
    u64 hidden_dim = 0;
    u64 out_dim = 0;
    span<u64> proj_in_w, proj_out_w;
    span<u64> proj_in_b_plain, proj_out_b_plain;
    AuthShare proj_in_b, proj_out_b;
    std::vector<DenseBlockWeights> layers;
};

struct DownBlockWeights {
    u64 in_ch = 0;
    u64 out_ch = 0;
    bool has_cross_attn = false;
    bool has_downsample = false;
    std::vector<ResBlock2DWeights> resblocks;
    std::vector<Transformer2DWeights> attn;
    span<u64> down_w, down_b_plain;
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
    AuthShare up_b;
};

struct CompleteUNetWeights {
    u64 in_ch = 0;
    u64 out_ch = 0;
    u64 temb_dim = 0;
    NormAffineWeights out_norm;
    span<u64> conv_in_w, conv_out_w;
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
    AuthShare up_b;
    std::vector<ResBlock2DWeights> resblocks;
};

struct CompleteVAEDecoderWeights {
    u64 latent_ch = 0;
    u64 mid_ch = 0;
    NormAffineWeights mid_attn_gn, out_norm;
    span<u64> post_quant_w, conv_out_w;
    span<u64> post_quant_b_plain, conv_out_b_plain;
    AuthShare post_quant_b, conv_out_b;
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
    span<u64> conv_in_b_plain, conv_mid_b_plain, conv_out_b_plain;
    AuthShare conv_in_b, conv_mid_b, conv_out_b;
    std::vector<ResBlock2DWeights> resblocks;
};

static NormAffineWeights make_norm_affine_weights(u64 dim) {
    NormAffineWeights w;
    w.weight_plain = span<u64>(dim);
    w.bias_plain = span<u64>(dim);
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

static void io_norm_affine(NormAffineWeights &w) {
    if (w.weight_plain.size() == 0 && w.bias_plain.size() == 0) return;
    input::call(w.weight_plain, SERVER);
    input::call(w.bias_plain, SERVER);
}

static void auth_norm_affine(NormAffineWeights &w) {
    if (w.weight_plain.size() == 0 && w.bias_plain.size() == 0) return;
    w.weight = auth_from_plain_open(w.weight_plain);
    w.bias = auth_from_plain_open(w.bias_plain);
}

static TransformerWeights make_transformer_weights(u64 dim, u64 ctx_dim, u64 ff_inner,
                                                   bool with_cross_attn, bool gated_ff) {
    TransformerWeights w;
    w.norm1 = make_norm_affine_weights(dim);
    w.norm2 = make_norm_affine_weights(dim);
    w.norm3 = make_norm_affine_weights(dim);
    w.self_attn.wq = span<u64>(dim * dim);
    w.self_attn.wk = span<u64>(dim * dim);
    w.self_attn.wv = span<u64>(dim * dim);
    w.self_attn.wout = span<u64>(dim * dim);
    w.self_attn.bq_plain = span<u64>(dim);
    w.self_attn.bk_plain = span<u64>(dim);
    w.self_attn.bv_plain = span<u64>(dim);
    w.self_attn.bout_plain = span<u64>(dim);

    if (with_cross_attn) {
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

static ResBlock2DWeights make_resblock2d_weights(u64 in_ch, u64 out_ch, u64 temb_dim, u64 norm_groups = 1) {
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
    w.temb_b_plain = span<u64>(out_ch);
    if (w.has_skip) {
        w.skip_w = span<u64>(out_ch * in_ch);
        w.skip_b_plain = span<u64>(out_ch);
    }
    return w;
}

static Transformer2DWeights make_transformer2d_weights(u64 in_ch, u64 inner_dim, u64 ctx_dim,
                                                       u64 num_blocks, u64 norm_groups = 1) {
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
        w.blocks.push_back(make_transformer_weights(inner_dim, ctx_dim, inner_dim * 4, true, true));
    }
    return w;
}

static FeatureUpscalerWeights make_feature_upscaler_weights(u64 in_dim, u64 hidden_dim,
                                                            u64 out_dim, u64 num_layers) {
    FeatureUpscalerWeights w;
    w.in_dim = in_dim;
    w.hidden_dim = hidden_dim;
    w.out_dim = out_dim;
    w.proj_in_w = span<u64>(in_dim * hidden_dim);
    w.proj_out_w = span<u64>(hidden_dim * out_dim);
    w.proj_in_b_plain = span<u64>(hidden_dim);
    w.proj_out_b_plain = span<u64>(out_dim);
    w.layers.reserve(num_layers);
    for (u64 i = 0; i < num_layers; ++i) {
        w.layers.push_back(make_dense_block_weights(hidden_dim, hidden_dim * 2, hidden_dim));
    }
    return w;
}

static AuthShare apply_attn(const AuthShare &x, u64 rows, u64 dim,
                            const AuthShare &context, u64 ctx_rows, u64 ctx_dim,
                            const AttnWeights &w, const char *softmax_tag) {
    auto q = linear_mat(rows, dim, dim, x, w.wq, &w.bq);
    clip_batch_check("attn:after q");
    auto k = linear_mat(ctx_rows, ctx_dim, dim, context, w.wk, &w.bk);
    clip_batch_check("attn:after k");
    auto v = linear_mat(ctx_rows, ctx_dim, dim, context, w.wv, &w.bv);
    clip_batch_check("attn:after v");
    q = reauth(q);
    k = reauth(k);
    v = reauth(v);
    auto attn_out = attention_dot(rows, ctx_rows, dim, q, k, v, softmax_tag);
    return linear_mat(rows, dim, dim, attn_out, w.wout, &w.bout);
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
    auto ff_up = linear_mat(rows, dim, w.ff.inner_dim * 2, n3, w.ff.w_up, &w.ff.b_up);
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
    auto prod_scaled = LRS_CALL(prod, f);
    shark::utils::stop_timer("multiplication_attn.ff.gate");
    shark::utils::start_timer("linear4_attn.ff.gate");
    auto ff_out = linear_mat(rows, w.ff.inner_dim, dim, prod_scaled, w.ff.w_down, &w.ff.b_down);
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

static AuthShare dense_block_apply(u64 rows, u64 in_dim, u64 out_dim,
                                   const AuthShare &x, const DenseBlockWeights &w,
                                   const std::string &prefix) {
    auto h1 = linear_mat(rows, in_dim, w.hidden_dim, x, w.w1, &w.b1);
    auto act1 = timed_gelu((prefix + ".gelu1").c_str(), h1);
    auto h2 = linear_mat(rows, w.hidden_dim, out_dim, act1, w.w2, &w.b2);
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
    auto ff_up = linear_mat(rows, dim, w.ff.inner_dim, n2, w.ff.w_up, &w.ff.b_up);
    auto ff_act = timed_gelu((prefix + ".ff").c_str(), ff_up);
    auto ff_out = linear_mat(rows, w.ff.inner_dim, dim, ff_act, w.ff.w_down, &w.ff.b_down);
    return ADD_CALL(x, ff_out);
}

static AuthShare apply_batched_self_transformer_stack(u64 B, u64 rows, u64 dim,
                                                      const AuthShare &x_in,
                                                      const std::vector<TransformerWeights> &blocks,
                                                      const std::string &prefix) {
    if (blocks.empty()) return x_in;
    AuthShare out = x_in;
    for (u64 layer = 0; layer < blocks.size(); ++layer) {
        AuthShare next = auth_alloc(B * rows * dim);
        for (u64 n = 0; n < B; ++n) {
            u64 off = n * rows * dim;
            auto x_b = auth_view(out, off, rows * dim);
            auto y_b = residual_self_attention_block(rows, dim, x_b, blocks[layer],
                                                     prefix + ".layer" + std::to_string(layer));
            auth_copy_into(next, off, y_b);
        }
        out = std::move(next);
    }
    return out;
}

static AuthShare apply_batched_cross_transformer_stack(u64 B, u64 rows, u64 dim,
                                                       u64 ctx_rows, u64 ctx_dim,
                                                       const AuthShare &x_in,
                                                       const AuthShare &context,
                                                       const std::vector<TransformerWeights> &blocks) {
    if (blocks.empty()) return x_in;
    AuthShare out = x_in;
    for (u64 layer = 0; layer < blocks.size(); ++layer) {
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

static AuthTensor4D resblock2d_apply(const AuthTensor4D &x, const AuthShare &temb,
                                     const ResBlock2DWeights &w, const std::string &prefix) {
    always_assert(x.C == w.in_ch);
    auto n1 = timed_groupnorm_apply((prefix + ".gn1").c_str(), x.B, x.H, x.W, x.C,
                                    std::max<u64>(1, std::min<u64>(w.norm_groups, x.C)), x.data,
                                    &w.gn1.weight, &w.gn1.bias);
    auto a1 = timed_silu((prefix + ".act1").c_str(), n1);
    auto h1 = conv2d_apply_k3_same(x.B, x.H, x.W, x.C, w.out_ch, 1, a1, w.conv1_w, &w.conv1_b);

    auto temb_proj = linear_mat(x.B, w.temb_dim, w.out_ch, temb, w.temb_w, &w.temb_b);
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

    auto n2 = timed_groupnorm_apply((prefix + ".gn2").c_str(), x.B, x.H, x.W, w.out_ch,
                                    std::max<u64>(1, std::min<u64>(w.norm_groups, w.out_ch)), h1,
                                    &w.gn2.weight, &w.gn2.bias);
    auto a2 = timed_silu((prefix + ".act2").c_str(), n2);
    auto h2 = conv2d_apply_k3_same(x.B, x.H, x.W, w.out_ch, w.out_ch, 1, a2, w.conv2_w, &w.conv2_b);

    AuthShare skip = x.data;
    if (w.has_skip) {
        skip = conv2d_apply(x.B, x.H, x.W, x.C, w.out_ch, 1, 1, 0, x.data, w.skip_w, &w.skip_b);
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

    auto proj_in = linear_mat(x.B * rows, x.C, w.inner_dim, flat, w.proj_in_w, &w.proj_in_b);
    auto proj_mid = apply_batched_cross_transformer_stack(x.B, rows, w.inner_dim, ctx_rows, ctx_dim,
                                                          proj_in, context, w.blocks);
    auto proj_out = linear_mat(x.B * rows, w.inner_dim, x.C, proj_mid, w.proj_out_w, &w.proj_out_b);

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
    for (u64 i = 0; i < w.resblocks.size(); ++i) {
        x = resblock2d_apply(x, temb, w.resblocks[i], prefix + ".res" + std::to_string(i));
        if (w.has_cross_attn && i < w.attn.size()) {
            x = transformer2d_apply(x, context, ctx_rows, ctx_dim, w.attn[i],
                                    prefix + ".attn" + std::to_string(i));
        }
        skips.push_back(x);
    }

    if (w.has_downsample) {
        auto y = conv2d_apply(x.B, x.H, x.W, x.C, x.C, 3, 2, 1, x.data, w.down_w, &w.down_b);
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
    for (u64 i = 0; i < w.resblocks.size(); ++i) {
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
        auto conv = conv2d_apply_k3_same(x.B, x.H * 2, x.W * 2, x.C, x.C, 1, up, w.up_w, &w.up_b);
        x = AuthTensor4D{std::move(conv), x.B, x.H * 2, x.W * 2, x.C};
    }
    return x;
}

static AuthShare feature_upscaler_apply(const AuthShare &x, u64 rows,
                                        const FeatureUpscalerWeights &w) {
    auto h = linear_mat(rows, w.in_dim, w.hidden_dim, x, w.proj_in_w, &w.proj_in_b);
    for (u64 i = 0; i < w.layers.size(); ++i) {
        h = dense_block_apply(rows, w.hidden_dim, w.hidden_dim, h, w.layers[i],
                              "gelu:feature_upscaler.layer" + std::to_string(i));
    }
    return linear_mat(rows, w.hidden_dim, w.out_dim, h, w.proj_out_w, &w.proj_out_b);
}

static AuthTensor4D vae_up_block_apply(const AuthTensor4D &x_in, const VAEUpBlockWeights &w,
                                       const std::string &prefix) {
    AuthTensor4D x = x_in;
    if (w.has_upsample) {
        auto up = upsample_nearest_2x(x.B, x.H, x.W, x.C, x.data);
        auto conv = conv2d_apply_k3_same(x.B, x.H * 2, x.W * 2, x.C, w.out_ch, 1, up, w.up_w, &w.up_b);
        x = AuthTensor4D{std::move(conv), x.B, x.H * 2, x.W * 2, w.out_ch};
    }
    for (u64 i = 0; i < w.resblocks.size(); ++i) {
        x = resblock2d_apply(x, make_public_const(x.B * w.resblocks[i].temb_dim, 0.0, f),
                             w.resblocks[i], prefix + ".res" + std::to_string(i));
    }
    return x;
}

static AuthTensor4D vae_decoder_apply(const AuthTensor4D &z, const CompleteVAEDecoderWeights &w) {
    auto h0 = conv2d_apply(z.B, z.H, z.W, z.C, w.mid_ch, 1, 1, 0, z.data, w.post_quant_w, &w.post_quant_b);
    AuthTensor4D h{std::move(h0), z.B, z.H, z.W, w.mid_ch};

    auto zero_temb = make_public_const(z.B * w.mid_ch, 0.0, f);
    h = resblock2d_apply(h, zero_temb, w.mid_res1, "vae.mid.res1");
    auto gn = timed_groupnorm_apply("layernorm:vae.mid_attn.gn", h.B, h.H, h.W, h.C, h.data,
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
    auto flat_attn = apply_batched_self_transformer_stack(h.B, rows, h.C, flat, std::vector<TransformerWeights>{w.mid_attn}, "vae.mid_attn");
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
    h = resblock2d_apply(h, zero_temb, w.mid_res2, "vae.mid.res2");

    for (u64 i = 0; i < w.up_blocks.size(); ++i) {
        h = vae_up_block_apply(h, w.up_blocks[i], "vae.up" + std::to_string(i));
    }

    auto nout = timed_groupnorm_apply("layernorm:vae.out", h.B, h.H, h.W, h.C, h.data,
                                      &w.out_norm.weight, &w.out_norm.bias);
    auto aout = timed_silu("silu:vae.out", nout);
    auto out = conv2d_apply_k3_same(h.B, h.H, h.W, h.C, 1, 1, aout, w.conv_out_w, &w.conv_out_b);
    return AuthTensor4D{std::move(out), h.B, h.H, h.W, 1};
}

static AuthTensor4D superres_apply(const AuthTensor4D &x_in, const SuperResWeights &w) {
    auto conv_in = conv2d_apply_k3_same(x_in.B, x_in.H, x_in.W, x_in.C, w.hidden_ch, 1,
                                        x_in.data, w.conv_in_w, &w.conv_in_b);
    AuthTensor4D x{std::move(conv_in), x_in.B, x_in.H, x_in.W, w.hidden_ch};
    auto zero_temb = make_public_const(x.B * w.hidden_ch, 0.0, f);
    for (u64 i = 0; i < w.resblocks.size(); ++i) {
        x = resblock2d_apply(x, zero_temb, w.resblocks[i], "superres.res" + std::to_string(i));
    }
    auto nout = timed_groupnorm_apply("layernorm:superres.out", x.B, x.H, x.W, x.C, x.data,
                                      &w.out_norm.weight, &w.out_norm.bias);
    auto aout = timed_silu("silu:superres.out", nout);
    auto mid = conv2d_apply_k3_same(x.B, x.H, x.W, x.C, x.C, 1, aout, w.conv_mid_w, &w.conv_mid_b);
    auto up = upsample_nearest_2x(x.B, x.H, x.W, x.C, mid);
    auto out = conv2d_apply_k3_same(x.B, x.H * 2, x.W * 2, x.C, w.out_ch, 1, up, w.conv_out_w, &w.conv_out_b);
    return AuthTensor4D{std::move(out), x.B, x.H * 2, x.W * 2, w.out_ch};
}

static AuthShare batched_layernorm_rows(u64 B, u64 rows, u64 cols, const AuthShare &x,
                                        const std::string &prefix) {
    AuthShare out = auth_alloc(B * rows * cols);
    for (u64 n = 0; n < B; ++n) {
        u64 off = n * rows * cols;
        auto xb = auth_view(x, off, rows * cols);
        auto yb = timed_layernorm_rows((prefix + ".b" + std::to_string(n)).c_str(), rows, cols, xb);
        auth_copy_into(out, off, yb);
    }
    return out;
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

static int run_complete_arch(u64 base_rng_seed);

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

static int run_complete_arch(u64 base_rng_seed) {
    const u64 batch = 1;
    const u64 img_H = 28;
    const u64 img_W = 28;
    const u64 C = 1;
    const u64 vae_scale_factor = 1;
    const u64 out_H = 28;
    const u64 out_W = 28;
    const u64 H = out_H / vae_scale_factor;
    const u64 W = out_W / vae_scale_factor;
    const u64 latent_C = 4;
    const u64 seq_len = 1;
    const u64 hidden = 1024;
    const u64 ctx_dim = hidden;
    const u64 img_hidden = 1024;
    const u64 img_emb_dim = 768;
    const u64 vocab = 49408;
    const u64 patch_size = 14;
    const u64 time_in_dim = 320;
    const u64 temb_dim = 1280;
    const u64 class_labels_dim = ctx_dim * 2;
    const u64 noise_level = 10;
    const int num_inference_steps = 12;
    const double guidance_scale = 7.5;
    const u64 text_layers = 2;
    const u64 vision_layers = 2;
    const u64 feature_layers = 1;
    const u64 unet_layers_per_block = 1;
    const u64 superres_layers = 1;
    const u64 cfg_copies = 2;
    // Restore the non-minimal small benchmark topology: 28x28 IO, CLIP-style
    // text/image towers, a two-stage UNet, full VAE decode, and optional super-res.
    const std::vector<u64> unet_channels = {latent_C, latent_C * 2};
    const std::vector<u64> vae_channels = {latent_C};

    keygen_configure_progress(num_inference_steps);
    keygen_ckpt(0, 1, "start:complete_arch");

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

    span<u64> text_token_embed(vocab * hidden);
    span<u64> text_pos_embed(seq_len * hidden);

    span<u64> img_patch_w(img_hidden * C * patch_size * patch_size);
    span<u64> img_patch_b(img_hidden);
    span<u64> img_cls(img_hidden);
    span<u64> img_pos_embed((num_patches + 1) * img_hidden);
    span<u64> img_proj_w(img_hidden * img_emb_dim);
    span<u64> img_proj_b(img_emb_dim);

    std::vector<TransformerWeights> text_blocks;
    std::vector<TransformerWeights> vision_blocks;
    for (u64 i = 0; i < text_layers; ++i) {
        text_blocks.push_back(make_transformer_weights(hidden, hidden, hidden * 4, false, false));
    }
    for (u64 i = 0; i < vision_layers; ++i) {
        vision_blocks.push_back(make_transformer_weights(img_hidden, img_hidden, img_hidden * 4, false, false));
    }
    auto text_out_norm = make_norm_affine_weights(hidden);
    auto image_pre_norm = make_norm_affine_weights(img_hidden);
    auto image_post_norm = make_norm_affine_weights(img_hidden);

    auto feature_upscaler = make_feature_upscaler_weights(img_emb_dim, ctx_dim, ctx_dim, feature_layers);

    span<u64> time_w1(time_in_dim * temb_dim);
    span<u64> time_b1(temb_dim);
    span<u64> time_w2(temb_dim * temb_dim);
    span<u64> time_b2(temb_dim);
    span<u64> class_proj_w(class_labels_dim * temb_dim);
    span<u64> class_proj_b(temb_dim);

    CompleteUNetWeights unet_full;
    unet_full.in_ch = latent_C;
    unet_full.out_ch = latent_C;
    unet_full.temb_dim = temb_dim;
    unet_full.out_norm = make_norm_affine_weights(unet_channels.front());
    unet_full.conv_in_w = span<u64>(unet_channels.front() * latent_C * 3 * 3);
    unet_full.conv_in_b_plain = span<u64>(unet_channels.front());
    unet_full.conv_out_w = span<u64>(latent_C * unet_channels.front() * 3 * 3);
    unet_full.conv_out_b_plain = span<u64>(latent_C);

    u64 curr_ch = unet_channels.front();
    std::vector<u64> skip_channels{curr_ch};
    for (u64 i = 0; i < unet_channels.size(); ++i) {
        DownBlockWeights block;
        block.in_ch = curr_ch;
        block.out_ch = unet_channels[i];
        block.has_cross_attn = true;
        block.has_downsample = (i + 1 < unet_channels.size());
        u64 block_ch = curr_ch;
        for (u64 layer = 0; layer < unet_layers_per_block; ++layer) {
            block.resblocks.push_back(make_resblock2d_weights(block_ch, block.out_ch, temb_dim, 1));
            block.attn.push_back(make_transformer2d_weights(block.out_ch, block.out_ch, ctx_dim, 1, 1));
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

    unet_full.mid_res1 = make_resblock2d_weights(curr_ch, curr_ch, temb_dim, 1);
    unet_full.mid_attn = make_transformer2d_weights(curr_ch, curr_ch, ctx_dim, 1, 1);
    unet_full.mid_res2 = make_resblock2d_weights(curr_ch, curr_ch, temb_dim, 1);

    auto rev_channels = unet_channels;
    std::reverse(rev_channels.begin(), rev_channels.end());
    u64 up_ch = curr_ch;
    for (u64 i = 0; i < rev_channels.size(); ++i) {
        UpBlockWeights block;
        block.in_ch = up_ch;
        block.out_ch = rev_channels[i];
        block.has_cross_attn = true;
        block.has_upsample = (i + 1 < rev_channels.size());
        u64 num_res = unet_layers_per_block + 1;
        for (u64 j = 0; j < num_res; ++j) {
            always_assert(!skip_channels.empty());
            u64 skip_ch = skip_channels.back();
            skip_channels.pop_back();
            block.skip_channels.push_back(skip_ch);
            block.resblocks.push_back(make_resblock2d_weights(up_ch + skip_ch, block.out_ch, temb_dim, 1));
            block.attn.push_back(make_transformer2d_weights(block.out_ch, block.out_ch, ctx_dim, 1, 1));
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
    vae_full.mid_ch = vae_channels.front();
    vae_full.mid_attn_gn = make_norm_affine_weights(vae_full.mid_ch);
    vae_full.post_quant_w = span<u64>(vae_full.mid_ch * latent_C);
    vae_full.post_quant_b_plain = span<u64>(vae_full.mid_ch);
    vae_full.mid_res1 = make_resblock2d_weights(vae_full.mid_ch, vae_full.mid_ch, 1, 1);
    vae_full.mid_res2 = make_resblock2d_weights(vae_full.mid_ch, vae_full.mid_ch, 1, 1);
    vae_full.mid_attn = make_transformer_weights(vae_full.mid_ch, vae_full.mid_ch, vae_full.mid_ch * 4, false, false);
    u64 vae_ch = vae_full.mid_ch;
    for (u64 i = 0; i < vae_channels.size(); ++i) {
        VAEUpBlockWeights block;
        block.in_ch = vae_ch;
        block.out_ch = vae_channels[i];
        block.has_upsample = (i == 0) && (vae_channels.size() > 1);
        if (block.has_upsample) {
            block.up_w = span<u64>(block.out_ch * block.in_ch * 3 * 3);
            block.up_b_plain = span<u64>(block.out_ch);
        }
        block.resblocks.push_back(make_resblock2d_weights(block.has_upsample ? block.out_ch : block.in_ch,
                                                          block.out_ch, 1, 1));
        vae_ch = block.out_ch;
        vae_full.up_blocks.push_back(std::move(block));
    }
    vae_full.out_norm = make_norm_affine_weights(vae_ch);
    vae_full.conv_out_w = span<u64>(C * vae_ch * 3 * 3);
    vae_full.conv_out_b_plain = span<u64>(C);

    SuperResWeights superres_w;
    superres_w.in_ch = C;
    superres_w.hidden_ch = latent_C;
    superres_w.out_ch = C;
    superres_w.out_norm = make_norm_affine_weights(superres_w.hidden_ch);
    superres_w.conv_in_w = span<u64>(superres_w.hidden_ch * C * 3 * 3);
    superres_w.conv_in_b_plain = span<u64>(superres_w.hidden_ch);
    superres_w.conv_mid_w = span<u64>(superres_w.hidden_ch * superres_w.hidden_ch * 3 * 3);
    superres_w.conv_mid_b_plain = span<u64>(superres_w.hidden_ch);
    superres_w.conv_out_w = span<u64>(C * superres_w.hidden_ch * 3 * 3);
    superres_w.conv_out_b_plain = span<u64>(C);
    for (u64 i = 0; i < superres_layers; ++i) {
        superres_w.resblocks.push_back(make_resblock2d_weights(superres_w.hidden_ch, superres_w.hidden_ch, 1, 1));
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
    auto fill_dense = [&](DenseBlockWeights &w, u64 in_dim) {
        fill_kaiming_uniform(w.w1, in_dim);
        fill_bias_uniform(w.b1_plain, in_dim);
        fill_kaiming_uniform(w.w2, w.hidden_dim);
        fill_bias_uniform(w.b2_plain, w.hidden_dim);
    };
    auto fill_resblock = [&](ResBlock2DWeights &w) {
        fill_norm_affine_identity(w.gn1, f);
        fill_norm_affine_identity(w.gn2, f);
        fill_kaiming_uniform(w.conv1_w, w.in_ch * 3 * 3);
        fill_bias_uniform(w.conv1_b_plain, w.in_ch * 3 * 3);
        fill_kaiming_uniform(w.conv2_w, w.out_ch * 3 * 3);
        fill_bias_uniform(w.conv2_b_plain, w.out_ch * 3 * 3);
        fill_kaiming_uniform(w.temb_w, std::max<u64>(1, w.temb_dim));
        fill_bias_uniform(w.temb_b_plain, std::max<u64>(1, w.temb_dim));
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

    if (party == SERVER || party == DEALER) {
        current_rng_seed = base_rng_seed;
        fill_kaiming_uniform(text_token_embed, vocab);
        fill_kaiming_uniform(text_pos_embed, seq_len);
        fill_kaiming_uniform(img_patch_w, C * patch_size * patch_size);
        fill_bias_uniform(img_patch_b, C * patch_size * patch_size);
        fill_bias_uniform(img_cls, img_hidden);
        fill_kaiming_uniform(img_pos_embed, num_patches + 1);
        fill_kaiming_uniform(img_proj_w, img_hidden);
        fill_bias_uniform(img_proj_b, img_hidden);
        fill_norm_affine_identity(text_out_norm, f);
        fill_norm_affine_identity(image_pre_norm, f);
        fill_norm_affine_identity(image_post_norm, f);
        for (auto &blk : text_blocks) fill_transformer(blk, hidden, hidden, false);
        for (auto &blk : vision_blocks) fill_transformer(blk, img_hidden, img_hidden, false);
        fill_kaiming_uniform(feature_upscaler.proj_in_w, img_emb_dim);
        fill_bias_uniform(feature_upscaler.proj_in_b_plain, img_emb_dim);
        fill_kaiming_uniform(feature_upscaler.proj_out_w, ctx_dim);
        fill_bias_uniform(feature_upscaler.proj_out_b_plain, ctx_dim);
        for (auto &layer : feature_upscaler.layers) fill_dense(layer, ctx_dim);
        fill_kaiming_uniform(time_w1, time_in_dim);
        fill_bias_uniform(time_b1, time_in_dim);
        fill_kaiming_uniform(time_w2, temb_dim);
        fill_bias_uniform(time_b2, temb_dim);
        fill_kaiming_uniform(class_proj_w, class_labels_dim);
        fill_bias_uniform(class_proj_b, class_labels_dim);
        fill_kaiming_uniform(unet_full.conv_in_w, latent_C * 3 * 3);
        fill_bias_uniform(unet_full.conv_in_b_plain, latent_C * 3 * 3);
        fill_kaiming_uniform(unet_full.conv_out_w, unet_channels.front() * 3 * 3);
        fill_bias_uniform(unet_full.conv_out_b_plain, unet_channels.front() * 3 * 3);
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
        fill_resblock(vae_full.mid_res1);
        fill_norm_affine_identity(vae_full.mid_attn_gn, f);
        fill_transformer(vae_full.mid_attn, vae_full.mid_ch, vae_full.mid_ch, false);
        fill_resblock(vae_full.mid_res2);
        for (auto &blk : vae_full.up_blocks) {
            if (blk.has_upsample) {
                fill_kaiming_uniform(blk.up_w, blk.in_ch * 3 * 3);
                fill_bias_uniform(blk.up_b_plain, blk.in_ch * 3 * 3);
            }
            for (auto &res : blk.resblocks) fill_resblock(res);
        }
        fill_norm_affine_identity(vae_full.out_norm, f);
        fill_kaiming_uniform(vae_full.conv_out_w, vae_ch * 3 * 3);
        fill_bias_uniform(vae_full.conv_out_b_plain, vae_ch * 3 * 3);
        fill_kaiming_uniform(superres_w.conv_in_w, C * 3 * 3);
        fill_bias_uniform(superres_w.conv_in_b_plain, C * 3 * 3);
        fill_kaiming_uniform(superres_w.conv_mid_w, superres_w.hidden_ch * 3 * 3);
        fill_bias_uniform(superres_w.conv_mid_b_plain, superres_w.hidden_ch * 3 * 3);
        fill_kaiming_uniform(superres_w.conv_out_w, superres_w.hidden_ch * 3 * 3);
        fill_bias_uniform(superres_w.conv_out_b_plain, superres_w.hidden_ch * 3 * 3);
        fill_norm_affine_identity(superres_w.out_norm, f);
        for (auto &res : superres_w.resblocks) fill_resblock(res);
    }

    auto io_attn = [&](AttnWeights &w, bool enabled) {
        if (!enabled) return;
        input::call(w.wq, SERVER);
        input::call(w.bq_plain, SERVER);
        input::call(w.wk, SERVER);
        input::call(w.bk_plain, SERVER);
        input::call(w.wv, SERVER);
        input::call(w.bv_plain, SERVER);
        input::call(w.wout, SERVER);
        input::call(w.bout_plain, SERVER);
    };
    auto io_ff = [&](FFWeights &w) {
        input::call(w.w_up, SERVER);
        input::call(w.b_up_plain, SERVER);
        input::call(w.w_down, SERVER);
        input::call(w.b_down_plain, SERVER);
    };
    auto io_transformer = [&](TransformerWeights &w, bool with_cross) {
        io_norm_affine(w.norm1);
        io_norm_affine(w.norm2);
        io_norm_affine(w.norm3);
        io_attn(w.self_attn, true);
        io_attn(w.cross_attn, with_cross);
        io_ff(w.ff);
    };
    auto io_dense = [&](DenseBlockWeights &w) {
        input::call(w.w1, SERVER);
        input::call(w.b1_plain, SERVER);
        input::call(w.w2, SERVER);
        input::call(w.b2_plain, SERVER);
    };
    auto io_resblock = [&](ResBlock2DWeights &w) {
        io_norm_affine(w.gn1);
        io_norm_affine(w.gn2);
        input::call(w.conv1_w, SERVER);
        input::call(w.conv1_b_plain, SERVER);
        input::call(w.conv2_w, SERVER);
        input::call(w.conv2_b_plain, SERVER);
        input::call(w.temb_w, SERVER);
        input::call(w.temb_b_plain, SERVER);
        if (w.has_skip) {
            input::call(w.skip_w, SERVER);
            input::call(w.skip_b_plain, SERVER);
        }
    };
    auto io_transformer2d = [&](Transformer2DWeights &w) {
        io_norm_affine(w.gn);
        input::call(w.proj_in_w, SERVER);
        input::call(w.proj_in_b_plain, SERVER);
        input::call(w.proj_out_w, SERVER);
        input::call(w.proj_out_b_plain, SERVER);
        for (auto &blk : w.blocks) io_transformer(blk, true);
    };

    input::call(text_token_embed, SERVER);
    input::call(text_pos_embed, SERVER);
    input::call(img_patch_w, SERVER);
    input::call(img_patch_b, SERVER);
    input::call(img_cls, SERVER);
    input::call(img_pos_embed, SERVER);
    input::call(img_proj_w, SERVER);
    input::call(img_proj_b, SERVER);
    io_norm_affine(text_out_norm);
    io_norm_affine(image_pre_norm);
    io_norm_affine(image_post_norm);
    for (auto &blk : text_blocks) io_transformer(blk, false);
    for (auto &blk : vision_blocks) io_transformer(blk, false);
    input::call(feature_upscaler.proj_in_w, SERVER);
    input::call(feature_upscaler.proj_in_b_plain, SERVER);
    input::call(feature_upscaler.proj_out_w, SERVER);
    input::call(feature_upscaler.proj_out_b_plain, SERVER);
    for (auto &layer : feature_upscaler.layers) io_dense(layer);
    input::call(time_w1, SERVER);
    input::call(time_b1, SERVER);
    input::call(time_w2, SERVER);
    input::call(time_b2, SERVER);
    input::call(class_proj_w, SERVER);
    input::call(class_proj_b, SERVER);
    input::call(unet_full.conv_in_w, SERVER);
    input::call(unet_full.conv_in_b_plain, SERVER);
    input::call(unet_full.conv_out_w, SERVER);
    input::call(unet_full.conv_out_b_plain, SERVER);
    io_norm_affine(unet_full.out_norm);
    for (auto &blk : unet_full.down_blocks) {
        for (auto &res : blk.resblocks) io_resblock(res);
        for (auto &attn : blk.attn) io_transformer2d(attn);
        if (blk.has_downsample) {
            input::call(blk.down_w, SERVER);
            input::call(blk.down_b_plain, SERVER);
        }
    }
    io_resblock(unet_full.mid_res1);
    io_transformer2d(unet_full.mid_attn);
    io_resblock(unet_full.mid_res2);
    for (auto &blk : unet_full.up_blocks) {
        for (auto &res : blk.resblocks) io_resblock(res);
        for (auto &attn : blk.attn) io_transformer2d(attn);
        if (blk.has_upsample) {
            input::call(blk.up_w, SERVER);
            input::call(blk.up_b_plain, SERVER);
        }
    }
    input::call(vae_full.post_quant_w, SERVER);
    input::call(vae_full.post_quant_b_plain, SERVER);
    io_resblock(vae_full.mid_res1);
    io_norm_affine(vae_full.mid_attn_gn);
    io_transformer(vae_full.mid_attn, false);
    io_resblock(vae_full.mid_res2);
    for (auto &blk : vae_full.up_blocks) {
        if (blk.has_upsample) {
            input::call(blk.up_w, SERVER);
            input::call(blk.up_b_plain, SERVER);
        }
        for (auto &res : blk.resblocks) io_resblock(res);
    }
    io_norm_affine(vae_full.out_norm);
    input::call(vae_full.conv_out_w, SERVER);
    input::call(vae_full.conv_out_b_plain, SERVER);
    input::call(superres_w.conv_in_w, SERVER);
    input::call(superres_w.conv_in_b_plain, SERVER);
    input::call(superres_w.conv_mid_w, SERVER);
    input::call(superres_w.conv_mid_b_plain, SERVER);
    input::call(superres_w.conv_out_w, SERVER);
    input::call(superres_w.conv_out_b_plain, SERVER);
    io_norm_affine(superres_w.out_norm);
    for (auto &res : superres_w.resblocks) io_resblock(res);
    keygen_ckpt(1, 1, "auth_weights:complete_arch");

    auto auth_attn = [&](AttnWeights &w, bool enabled) {
        if (!enabled) return;
        w.bq = auth_from_plain_open(w.bq_plain);
        w.bk = auth_from_plain_open(w.bk_plain);
        w.bv = auth_from_plain_open(w.bv_plain);
        w.bout = auth_from_plain_open(w.bout_plain);
    };
    auto auth_ff = [&](FFWeights &w) {
        w.b_up = auth_from_plain_open(w.b_up_plain);
        w.b_down = auth_from_plain_open(w.b_down_plain);
    };
    auto auth_transformer = [&](TransformerWeights &w, bool with_cross) {
        auth_norm_affine(w.norm1);
        auth_norm_affine(w.norm2);
        auth_norm_affine(w.norm3);
        auth_attn(w.self_attn, true);
        auth_attn(w.cross_attn, with_cross);
        auth_ff(w.ff);
    };
    auto auth_dense = [&](DenseBlockWeights &w) {
        w.b1 = auth_from_plain_open(w.b1_plain);
        w.b2 = auth_from_plain_open(w.b2_plain);
    };
    auto auth_resblock = [&](ResBlock2DWeights &w) {
        auth_norm_affine(w.gn1);
        auth_norm_affine(w.gn2);
        w.conv1_b = auth_from_plain_open(w.conv1_b_plain);
        w.conv2_b = auth_from_plain_open(w.conv2_b_plain);
        w.temb_b = auth_from_plain_open(w.temb_b_plain);
        if (w.has_skip) w.skip_b = auth_from_plain_open(w.skip_b_plain);
    };
    auto auth_transformer2d = [&](Transformer2DWeights &w) {
        auth_norm_affine(w.gn);
        w.proj_in_b = auth_from_plain_open(w.proj_in_b_plain);
        w.proj_out_b = auth_from_plain_open(w.proj_out_b_plain);
        for (auto &blk : w.blocks) auth_transformer(blk, true);
    };

    auto img_patch_b_a = auth_from_plain_open(img_patch_b);
    auto img_cls_a = auth_from_plain_open(img_cls);
    auto img_pos_embed_a = auth_from_plain_open(img_pos_embed);
    auto img_proj_b_a = auth_from_plain_open(img_proj_b);
    auth_norm_affine(text_out_norm);
    auth_norm_affine(image_pre_norm);
    auth_norm_affine(image_post_norm);
    for (auto &blk : text_blocks) auth_transformer(blk, false);
    for (auto &blk : vision_blocks) auth_transformer(blk, false);
    feature_upscaler.proj_in_b = auth_from_plain_open(feature_upscaler.proj_in_b_plain);
    feature_upscaler.proj_out_b = auth_from_plain_open(feature_upscaler.proj_out_b_plain);
    for (auto &layer : feature_upscaler.layers) auth_dense(layer);
    auto time_b1_a = auth_from_plain_open(time_b1);
    auto time_b2_a = auth_from_plain_open(time_b2);
    auto class_proj_b_a = auth_from_plain_open(class_proj_b);
    unet_full.conv_in_b = auth_from_plain_open(unet_full.conv_in_b_plain);
    unet_full.conv_out_b = auth_from_plain_open(unet_full.conv_out_b_plain);
    auth_norm_affine(unet_full.out_norm);
    for (auto &blk : unet_full.down_blocks) {
        for (auto &res : blk.resblocks) auth_resblock(res);
        for (auto &attn : blk.attn) auth_transformer2d(attn);
        if (blk.has_downsample) blk.down_b = auth_from_plain_open(blk.down_b_plain);
    }
    auth_resblock(unet_full.mid_res1);
    auth_transformer2d(unet_full.mid_attn);
    auth_resblock(unet_full.mid_res2);
    for (auto &blk : unet_full.up_blocks) {
        for (auto &res : blk.resblocks) auth_resblock(res);
        for (auto &attn : blk.attn) auth_transformer2d(attn);
        if (blk.has_upsample) blk.up_b = auth_from_plain_open(blk.up_b_plain);
    }
    vae_full.post_quant_b = auth_from_plain_open(vae_full.post_quant_b_plain);
    auth_resblock(vae_full.mid_res1);
    auth_norm_affine(vae_full.mid_attn_gn);
    auth_transformer(vae_full.mid_attn, false);
    auth_resblock(vae_full.mid_res2);
    for (auto &blk : vae_full.up_blocks) {
        if (blk.has_upsample) blk.up_b = auth_from_plain_open(blk.up_b_plain);
        for (auto &res : blk.resblocks) auth_resblock(res);
    }
    auth_norm_affine(vae_full.out_norm);
    vae_full.conv_out_b = auth_from_plain_open(vae_full.conv_out_b_plain);
    superres_w.conv_in_b = auth_from_plain_open(superres_w.conv_in_b_plain);
    superres_w.conv_mid_b = auth_from_plain_open(superres_w.conv_mid_b_plain);
    superres_w.conv_out_b = auth_from_plain_open(superres_w.conv_out_b_plain);
    auth_norm_affine(superres_w.out_norm);
    for (auto &res : superres_w.resblocks) auth_resblock(res);

    shark::utils::start_timer("input");

    // -----------------------------
    // Inputs: prompt + image
    // -----------------------------
    const char *prompt_env = std::getenv("UNCLIP_PROMPT");
    std::string prompt = prompt_env && *prompt_env ? std::string(prompt_env) : std::string("a snowy mountain landscape");
    auto prompt_ids = tokenize_prompt(prompt, vocab, seq_len);
    auto neg_prompt_ids = tokenize_prompt("", vocab, seq_len);

    span<u64> prompt_cfg_plain(cfg_copies * batch * seq_len * hidden);
    if (party == SERVER || party == DEALER) {
        auto pe = build_prompt_embeds(prompt_ids, text_token_embed, text_pos_embed, hidden, seq_len);
        auto ne = build_prompt_embeds(neg_prompt_ids, text_token_embed, text_pos_embed, hidden, seq_len);
        for (u64 b = 0; b < batch; ++b) {
            u64 neg_base = b * seq_len * hidden;
            u64 pos_base = (batch + b) * seq_len * hidden;
            for (u64 i = 0; i < seq_len * hidden; ++i) {
                prompt_cfg_plain[neg_base + i] = ne[i];
                prompt_cfg_plain[pos_base + i] = pe[i];
            }
        }
    }
    input::call(prompt_cfg_plain, SERVER);
    keygen_ckpt(2, 1, "input:prompt_embeds");
    clip_batch_check("debug:after prompt_embeds");
    auto prompt_cfg = auth_from_plain_open(prompt_cfg_plain);
    auto prompt_ctx = apply_batched_self_transformer_stack(cfg_copies * batch, seq_len, hidden,
                                                           prompt_cfg, text_blocks, "text_encoder");
    prompt_ctx = timed_layernorm_rows("layernorm:text_encoder.out",
                                      cfg_copies * batch * seq_len, hidden, prompt_ctx,
                                      &text_out_norm.weight, &text_out_norm.bias);

    span<u64> image_input(batch * img_H * img_W * C);
    if (party == CLIENT || party == DEALER) {
        current_rng_seed = base_rng_seed;
        const size_t needed = (size_t)(img_H * img_W * C);
        std::vector<u64> image_values(needed);
        for (size_t i = 0; i < needed; ++i) {
            image_values[i] = qrand_uniform(f);
        }
        for (u64 h = 0; h < img_H; ++h) {
            for (u64 widx = 0; widx < img_W; ++widx) {
                for (u64 c = 0; c < C; ++c) {
                    size_t src = ((size_t)h * img_W + widx) * C + c;
                    size_t dst = ((size_t)h * img_W + widx) * C + c;
                    image_input[dst] = image_values[src];
                }
            }
        }
    }
    input::call(image_input, CLIENT);
    keygen_ckpt(2, 2, "input:image");
    clip_batch_check("debug:after image_input");
    if (std::getenv("UNCLIP_APPLY_CLIP_IMAGE_NORM")) {
        image_input = normalize_image_input(image_input, img_H, img_W, C);
        clip_batch_check("debug:after image_norm");
    }
    auto image_input_a = auth_from_plain_open(image_input);

    // -----------------------------
    // Image encoder (patch conv + transformer + proj)
    // -----------------------------
    auto patch = conv2d_apply(batch, img_H, img_W, C, img_hidden, patch_size, patch_size, 0,
                              image_input_a, img_patch_w, &img_patch_b_a);
    keygen_ckpt(3, 1, "image_encoder:patch_conv");
    const u64 N = num_patches;
    AuthShare img_tokens = auth_alloc(batch * (N + 1) * img_hidden);
    for (u64 n = 0; n < batch; ++n) {
        u64 token_base = n * (N + 1) * img_hidden;
        u64 patch_base = n * N * img_hidden;
        for (u64 c = 0; c < img_hidden; ++c) {
            img_tokens.share[token_base + c] = img_cls_a.share[c] + img_pos_embed_a.share[c];
            img_tokens.tag[token_base + c] = img_cls_a.tag[c] + img_pos_embed_a.tag[c];
        }
        for (u64 i = 0; i < N; ++i) {
            for (u64 c = 0; c < img_hidden; ++c) {
                u64 dst = token_base + (i + 1) * img_hidden + c;
                u64 src = patch_base + i * img_hidden + c;
                u64 pos = (i + 1) * img_hidden + c;
                img_tokens.share[dst] = patch.share[src] + img_pos_embed_a.share[pos];
                img_tokens.tag[dst] = patch.tag[src] + img_pos_embed_a.tag[pos];
            }
        }
    }
    auto img_tokens_pre = timed_layernorm_rows("layernorm:image_encoder.tokens",
                                               batch * (N + 1), img_hidden, img_tokens,
                                               &image_pre_norm.weight, &image_pre_norm.bias);
    auto img_tokens_vit = apply_batched_self_transformer_stack(batch, N + 1, img_hidden,
                                                               img_tokens_pre, vision_blocks,
                                                               "image_encoder.transformer");
    auto img_tokens_post = timed_layernorm_rows("layernorm:image_encoder.post",
                                                batch * (N + 1), img_hidden, img_tokens_vit,
                                                &image_post_norm.weight, &image_post_norm.bias);
    keygen_ckpt(3, 2, "image_encoder:token_ln");
    auto cls_tokens = extract_cls_tokens(batch, N + 1, img_hidden, img_tokens_post);
    auto img_embed = linear_mat(batch, img_hidden, img_emb_dim, cls_tokens, img_proj_w, &img_proj_b_a);
    keygen_ckpt(3, 3, "image_encoder:cls_proj");

    // -----------------------------
    // Image noising + feature upscaler
    // -----------------------------
    auto alphas_cumprod_ddpm = build_alphas_cumprod_cos(1000);
    double alpha_noise = alphas_cumprod_ddpm[(int)noise_level];
    span<u64> noise(img_embed.share.size());
    if (party == CLIENT || party == DEALER) {
        current_rng_seed = base_rng_seed;
        for (u64 i = 0; i < noise.size(); ++i) {
            noise[i] = qrand_normal(f, 1.0);
        }
    }
    input::call(noise, CLIENT);
    keygen_ckpt(2, 3, "input:noise");
    auto noise_a = auth_from_plain_open(noise);
    auto img_embed_noised = add_noise_ddpm(img_embed, noise_a, alpha_noise);
    auto image_cond = feature_upscaler_apply(img_embed_noised, batch, feature_upscaler);

    auto noise_level_embed = make_timestep_embedding((int)noise_level, ctx_dim, f, true);
    auto noise_level_embed_a = auth_from_plain_open(noise_level_embed);
    AuthShare class_labels = auth_alloc(batch * class_labels_dim);
    for (u64 n = 0; n < batch; ++n) {
        u64 cond_base = n * ctx_dim;
        u64 dst_base = n * class_labels_dim;
        for (u64 i = 0; i < ctx_dim; ++i) {
            class_labels.share[dst_base + i] = image_cond.share[cond_base + i];
            class_labels.tag[dst_base + i] = image_cond.tag[cond_base + i];
            class_labels.share[dst_base + ctx_dim + i] = noise_level_embed_a.share[i];
            class_labels.tag[dst_base + ctx_dim + i] = noise_level_embed_a.tag[i];
        }
    }

    AuthShare class_labels_cfg = auth_alloc(cfg_copies * batch * class_labels_dim);
    for (u64 i = 0; i < batch * class_labels_dim; ++i) {
        class_labels_cfg.share[i] = 0;
        class_labels_cfg.tag[i] = 0;
        class_labels_cfg.share[batch * class_labels_dim + i] = class_labels.share[i];
        class_labels_cfg.tag[batch * class_labels_dim + i] = class_labels.tag[i];
    }

    // -----------------------------
    // Latents
    // -----------------------------
    span<u64> latents(batch * H * W * latent_C);
    if (party == CLIENT || party == DEALER) {
        current_rng_seed = base_rng_seed;
        for (u64 i = 0; i < latents.size(); ++i) {
            latents[i] = qrand_normal(f, 1.0);
        }
    }
    input::call(latents, CLIENT);
    keygen_ckpt(2, 4, "input:latents");
    shark::utils::stop_timer("input");
    auto latents_a = auth_from_plain_open(latents);

    auto alphas_cumprod = build_alphas_cumprod(1000, 0.00085, 0.012);
    std::vector<int> timesteps(num_inference_steps);
    if (num_inference_steps == 1) {
        timesteps[0] = 999;
    } else {
        for (int i = 0; i < num_inference_steps; ++i) {
            double t = (double)(1000 - 1) * (1.0 - (double)i / (double)(num_inference_steps - 1));
            timesteps[i] = (int)std::llround(t);
        }
    }

    // -----------------------------
    // UNet forward (complete small arch)
    // -----------------------------
    auto unet_forward = [&](const AuthShare &lat_in, const AuthShare &prompt_ctx_in,
                            const AuthShare &cls_labels, int timestep,
                            int diffusion_step, int diffusion_total) -> AuthShare {
        const u64 Bcfg = cfg_copies * batch;

        auto unet_step_ckpt = [&](int component, const char *op_name) {
            std::ostringstream oss;
            oss << "unet:" << op_name
                << " step " << diffusion_step << "/" << diffusion_total
                << " (t=" << timestep << ")";
            keygen_ckpt(5, component, oss.str());
        };

        auto t_base = make_timestep_embedding(timestep, time_in_dim, f, true);
        span<u64> t_embed_plain(Bcfg * time_in_dim);
        if (party == SERVER || party == DEALER) {
            for (u64 n = 0; n < Bcfg; ++n) {
                for (u64 i = 0; i < time_in_dim; ++i) {
                    t_embed_plain[n * time_in_dim + i] = t_base[i];
                }
            }
        }
        auto t_embed = auth_from_plain_open(t_embed_plain);

        shark::utils::start_timer("linear1_mlp");
        auto t1 = linear_mat(Bcfg, time_in_dim, temb_dim, t_embed, time_w1, &time_b1_a);
        shark::utils::stop_timer("linear1_mlp");
        shark::utils::start_timer("gelu:unet.time_mlp");
        shark::utils::start_timer("exp_mlp");
        auto t1_act = timed_silu("silu:unet.time_mlp", t1);
        shark::utils::stop_timer("exp_mlp");
        shark::utils::stop_timer("gelu:unet.time_mlp");
        shark::utils::start_timer("linear3_mlp");
        auto temb = linear_mat(Bcfg, temb_dim, temb_dim, t1_act, time_w2, &time_b2_a);
        shark::utils::stop_timer("linear3_mlp");
        auto class_proj = linear_mat(Bcfg, class_labels_dim, temb_dim, cls_labels, class_proj_w, &class_proj_b_a);
        temb = ADD_CALL(temb, class_proj);
        unet_step_ckpt(1, "time_mlp");

        auto h0 = conv2d_apply_k3_same(Bcfg, H, W, latent_C, unet_channels.front(), 1,
                                       lat_in, unet_full.conv_in_w, &unet_full.conv_in_b);
        unet_step_ckpt(2, "conv_in");

        AuthTensor4D x{std::move(h0), Bcfg, H, W, unet_channels.front()};
        std::vector<AuthTensor4D> skips;
        skips.reserve(16);
        skips.push_back(x);
        for (u64 i = 0; i < unet_full.down_blocks.size(); ++i) {
            x = down_block_apply(x, temb, prompt_ctx_in, seq_len, ctx_dim,
                                 unet_full.down_blocks[i], skips,
                                 "unet.down" + std::to_string(i));
        }
        unet_step_ckpt(3, "down_path");

        x = resblock2d_apply(x, temb, unet_full.mid_res1, "unet.mid.res1");
        x = transformer2d_apply(x, prompt_ctx_in, seq_len, ctx_dim,
                                unet_full.mid_attn, "unet.mid.attn");
        x = resblock2d_apply(x, temb, unet_full.mid_res2, "unet.mid.res2");
        unet_step_ckpt(4, "mid");

        for (u64 i = 0; i < unet_full.up_blocks.size(); ++i) {
            x = up_block_apply(x, skips, temb, prompt_ctx_in, seq_len, ctx_dim,
                               unet_full.up_blocks[i], "unet.up" + std::to_string(i));
        }
        always_assert(skips.empty());
        unet_step_ckpt(5, "up_path");

        auto out_norm = timed_groupnorm_apply("layernorm:unet.out", x.B, x.H, x.W, x.C, x.data,
                                              &unet_full.out_norm.weight, &unet_full.out_norm.bias);
        auto out_act = timed_silu("silu:unet.out", out_norm);
        auto out = conv2d_apply_k3_same(x.B, x.H, x.W, x.C, unet_full.out_ch, 1,
                                        out_act, unet_full.conv_out_w, &unet_full.conv_out_b);
        unet_step_ckpt(6, "conv_out");
        return out;
    };

    // Diffusion loop.
    for (int step = 0; step < num_inference_steps; ++step) {
        AuthShare latents_cfg = auth_alloc(cfg_copies * latents_a.share.size());
        for (u64 i = 0; i < latents_a.share.size(); ++i) {
            latents_cfg.share[i] = latents_a.share[i];
            latents_cfg.tag[i] = latents_a.tag[i];
            latents_cfg.share[latents_a.share.size() + i] = latents_a.share[i];
            latents_cfg.tag[latents_a.share.size() + i] = latents_a.tag[i];
        }

        int t = timesteps[step];
        auto noise_pred = unet_forward(latents_cfg, prompt_ctx, class_labels_cfg,
                                       t, step + 1, num_inference_steps);

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

        int t_prev = std::max(t - 1, 0);
        double alpha_t = alphas_cumprod[t];
        double alpha_prev = alphas_cumprod[t_prev];
        latents_a = scheduler_step_linear(latents_a, guided, alpha_t, alpha_prev);

        std::ostringstream step_label;
        step_label << "diffusion:step " << (step + 1) << "/" << num_inference_steps
                   << " (t=" << t << ")";
        keygen_ckpt(4, step + 1, step_label.str());
        if (profile_step_summary_enabled() && party != DEALER) {
            std::cout << "[PROFILE_STEP] " << step_label.str() << std::endl;
            if (shark::protocols::peer) {
                u64 total_comm = shark::protocols::peer->bytesReceived() + shark::protocols::peer->bytesSent();
                u64 total_rounds = shark::protocols::peer->roundsReceived() + shark::protocols::peer->roundsSent();
                std::cout << "[PROFILE_STEP] total_comm: " << (double)total_comm / 1024.0 << " KB (" << total_comm
                          << " bytes), total_rounds: " << total_rounds << std::endl;
            }
            print_profile_timers();
        }
    }

    // -----------------------------
    // VAE decode (+ optional superres)
    // -----------------------------
    double scaling_factor = 0.18215;
    auto inv_scale = make_public_const(latents_a.share.size(), 1.0 / scaling_factor, f);
    auto latents_scaled = LRS_CALL(MUL_CALL(latents_a, inv_scale), f);
    AuthTensor4D latent_tensor{std::move(latents_scaled), batch, H, W, latent_C};
    auto decoded = vae_decoder_apply(latent_tensor, vae_full);
    keygen_ckpt(6, 1, "vae:decode");

    const bool enable_superres = std::getenv("UNCLIP_ENABLE_SUPERRES") != nullptr;
    AuthTensor4D final_img = std::move(decoded);
    if (enable_superres) {
        final_img = superres_apply(final_img, superres_w);
        keygen_ckpt(6, 2, "superres:decode");
    }

    if (party != DEALER) {
        shark::utils::start_timer("reconstruct");
        auto out_plain = authenticated_reconstruct(final_img.data.share, final_img.data.tag);
        shark::utils::stop_timer("reconstruct");
        clip_batch_check("output");
        for (u64 i = 0; i < out_plain.size(); ++i) {
            double v = (double)(int64_t)out_plain[i] / (double)(1ULL << f);
            int64_t q = (int64_t)std::llround(std::tanh(v) * (double)(1ULL << f));
            out_plain[i] = (u64)q;
        }
        std::cout << "[UNCLIP] output first 8 values: ";
        for (u64 i = 0; i < std::min<u64>(8, out_plain.size()); ++i) {
            std::cout << (int64_t)out_plain[i] << " ";
        }
        std::cout << std::endl;
        if (party == CLIENT) {
            if (write_jpg_from_fixed(out_plain, final_img.H, final_img.W, final_img.C, f, "unclip_out.jpg")) {
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
            std::cout << "[PROFILE] total_comm: " << (double)total_comm / 1024.0 << " KB (" << total_comm << " bytes)" << std::endl;
        }
        print_profile_timers();
        print_profile_components_table();
        print_legacy_profile_lines();
    }

    finalize::call();
    return 0;
}

int main(int argc, char **argv) {
    init::from_args(argc, argv);
    mpspdz_32bit_compaison = false;

    print_key_file_sizes();
    if (party != DEALER) {
        shark::utils::start_timer("total_eval");
    }

    const char *seed_env = std::getenv("UNCLIP_SEED");
    if (seed_env && *seed_env) {
        current_rng_seed = std::strtoull(seed_env, nullptr, 0);
    }
    const u64 base_rng_seed = current_rng_seed;

    return run_complete_arch(base_rng_seed);

#if 0
    keygen_configure_progress(num_inference_steps);
    keygen_ckpt(0, 1, "start");

    // -----------------------------
    // Model weights (SERVER-owned)
    // -----------------------------
    const u64 img_grid_h = img_H / patch_size;
    const u64 img_grid_w = img_W / patch_size;
    const u64 num_patches = img_grid_h * img_grid_w;

    // Text embeddings.
    span<u64> text_token_embed(vocab * hidden);
    span<u64> text_pos_embed(seq_len * hidden);

    // Image encoder weights.
    span<u64> img_patch_w(img_hidden * C * patch_size * patch_size);
    span<u64> img_patch_b(img_hidden);
    span<u64> img_cls(img_hidden);
    span<u64> img_pos_embed((num_patches + 1) * img_hidden);
    span<u64> img_proj_w(img_hidden * img_emb_dim);
    span<u64> img_proj_b(img_emb_dim);

    // UNet weights.
    span<u64> unet_conv_in_w(latent_C * latent_C * 3 * 3);
    span<u64> unet_conv_in_b(latent_C);
    span<u64> unet_conv_out_w(latent_C * latent_C * 3 * 3);
    span<u64> unet_conv_out_b(latent_C);

    // ResBlock weights (two blocks).
    span<u64> rb1_conv1_w(latent_C * latent_C * 3 * 3);
    span<u64> rb1_conv1_b(latent_C);
    span<u64> rb1_conv2_w(latent_C * latent_C * 3 * 3);
    span<u64> rb1_conv2_b(latent_C);
    span<u64> rb1_temb_w(temb_dim * latent_C);
    span<u64> rb1_temb_b(latent_C);

    span<u64> rb2_conv1_w(latent_C * latent_C * 3 * 3);
    span<u64> rb2_conv1_b(latent_C);
    span<u64> rb2_conv2_w(latent_C * latent_C * 3 * 3);
    span<u64> rb2_conv2_b(latent_C);
    span<u64> rb2_temb_w(temb_dim * latent_C);
    span<u64> rb2_temb_b(latent_C);

    // Time MLP and image projection.
    span<u64> time_w1(time_in_dim * temb_dim);
    span<u64> time_b1(temb_dim);
    span<u64> time_w2(temb_dim * temb_dim);
    span<u64> time_b2(temb_dim);
    span<u64> img_proj_w2(class_labels_dim * temb_dim);
    span<u64> img_proj_b2(temb_dim);

    // Transformer2D weights (mid attention).
    TransformerWeights mid_attn_w;
    mid_attn_w.ff.inner_dim = latent_C * 4;
    span<u64> mid_self_bq(latent_C), mid_self_bk(latent_C), mid_self_bv(latent_C), mid_self_bout(latent_C);
    span<u64> mid_cross_bq(latent_C), mid_cross_bk(latent_C), mid_cross_bv(latent_C), mid_cross_bout(latent_C);
    span<u64> mid_ff_b_up(mid_attn_w.ff.inner_dim * 2), mid_ff_b_down(latent_C);

    mid_attn_w.self_attn.wq = span<u64>(latent_C * latent_C);
    mid_attn_w.self_attn.wk = span<u64>(latent_C * latent_C);
    mid_attn_w.self_attn.wv = span<u64>(latent_C * latent_C);
    mid_attn_w.self_attn.wout = span<u64>(latent_C * latent_C);

    mid_attn_w.cross_attn.wq = span<u64>(latent_C * latent_C);
    mid_attn_w.cross_attn.wk = span<u64>(ctx_dim * latent_C);
    mid_attn_w.cross_attn.wv = span<u64>(ctx_dim * latent_C);
    mid_attn_w.cross_attn.wout = span<u64>(latent_C * latent_C);

    mid_attn_w.ff.w_up = span<u64>(latent_C * (mid_attn_w.ff.inner_dim * 2));
    mid_attn_w.ff.w_down = span<u64>(mid_attn_w.ff.inner_dim * latent_C);

    // VAE decode weights (simplified).
    span<u64> vae_post_w(latent_C * latent_C);
    span<u64> vae_post_b(latent_C);
    span<u64> vae_res1_c1_w(latent_C * latent_C * 3 * 3);
    span<u64> vae_res1_c1_b(latent_C);
    span<u64> vae_res1_c2_w(latent_C * latent_C * 3 * 3);
    span<u64> vae_res1_c2_b(latent_C);
    span<u64> vae_res2_c1_w(latent_C * latent_C * 3 * 3);
    span<u64> vae_res2_c1_b(latent_C);
    span<u64> vae_res2_c2_w(latent_C * latent_C * 3 * 3);
    span<u64> vae_res2_c2_b(latent_C);

    AttnWeights vae_attn_w;
    span<u64> vae_attn_bq(latent_C), vae_attn_bk(latent_C), vae_attn_bv(latent_C), vae_attn_bout(latent_C);
    vae_attn_w.wq = span<u64>(latent_C * latent_C);
    vae_attn_w.wk = span<u64>(latent_C * latent_C);
    vae_attn_w.wv = span<u64>(latent_C * latent_C);
    vae_attn_w.wout = span<u64>(latent_C * latent_C);

    span<u64> vae_out_w(C * latent_C * 3 * 3);
    span<u64> vae_out_b(C);

    auto fill_normal = [&](span<u64> &buf, double stddev) {
        for (u64 i = 0; i < buf.size(); ++i) buf[i] = qrand_normal(f, stddev);
    };
    auto fill_kaiming_uniform = [&](span<u64> &buf, u64 fan_in) {
        // Match PyTorch nn.Linear/nn.Conv default reset_parameters:
        // kaiming_uniform_(a=sqrt(5)) => bound = 1/sqrt(fan_in)
        double bound = (fan_in > 0) ? (1.0 / std::sqrt((double)fan_in)) : 0.0;
        for (u64 i = 0; i < buf.size(); ++i) buf[i] = qrand_uniform_symmetric(f, bound);
    };
    auto fill_bias_uniform = [&](span<u64> &buf, u64 fan_in) {
        double bound = (fan_in > 0) ? (1.0 / std::sqrt((double)fan_in)) : 0.0;
        for (u64 i = 0; i < buf.size(); ++i) buf[i] = qrand_uniform_symmetric(f, bound);
    };

    if (party == SERVER || party == DEALER) {
        current_rng_seed = base_rng_seed;
        // Keep these three close to CLIP init in the Python reference.
        fill_normal(text_token_embed, 1.0);   // nn.Embedding default
        fill_normal(text_pos_embed, 0.02);    // explicit normal_(std=0.02)
        fill_normal(img_cls, 0.02);           // explicit normal_(std=0.02)
        fill_normal(img_pos_embed, 0.02);     // explicit normal_(std=0.02)

        // Conv / Linear defaults in Python: kaiming_uniform_(a=sqrt(5)) + bias uniform.
        fill_kaiming_uniform(img_patch_w, C * patch_size * patch_size);
        fill_bias_uniform(img_patch_b, C * patch_size * patch_size);
        fill_kaiming_uniform(img_proj_w, img_hidden);
        fill_bias_uniform(img_proj_b, img_hidden);

        fill_kaiming_uniform(unet_conv_in_w, latent_C * 3 * 3);
        fill_bias_uniform(unet_conv_in_b, latent_C * 3 * 3);
        fill_kaiming_uniform(unet_conv_out_w, latent_C * 3 * 3);
        fill_bias_uniform(unet_conv_out_b, latent_C * 3 * 3);

        fill_kaiming_uniform(rb1_conv1_w, latent_C * 3 * 3);
        fill_bias_uniform(rb1_conv1_b, latent_C * 3 * 3);
        fill_kaiming_uniform(rb1_conv2_w, latent_C * 3 * 3);
        fill_bias_uniform(rb1_conv2_b, latent_C * 3 * 3);
        fill_kaiming_uniform(rb1_temb_w, temb_dim);
        fill_bias_uniform(rb1_temb_b, temb_dim);

        fill_kaiming_uniform(rb2_conv1_w, latent_C * 3 * 3);
        fill_bias_uniform(rb2_conv1_b, latent_C * 3 * 3);
        fill_kaiming_uniform(rb2_conv2_w, latent_C * 3 * 3);
        fill_bias_uniform(rb2_conv2_b, latent_C * 3 * 3);
        fill_kaiming_uniform(rb2_temb_w, temb_dim);
        fill_bias_uniform(rb2_temb_b, temb_dim);

        fill_kaiming_uniform(time_w1, time_in_dim);
        fill_bias_uniform(time_b1, time_in_dim);
        fill_kaiming_uniform(time_w2, temb_dim);
        fill_bias_uniform(time_b2, temb_dim);
        fill_kaiming_uniform(img_proj_w2, class_labels_dim);
        fill_bias_uniform(img_proj_b2, class_labels_dim);

        fill_kaiming_uniform(mid_attn_w.self_attn.wq, latent_C);
        fill_bias_uniform(mid_self_bq, latent_C);
        fill_kaiming_uniform(mid_attn_w.self_attn.wk, latent_C);
        fill_bias_uniform(mid_self_bk, latent_C);
        fill_kaiming_uniform(mid_attn_w.self_attn.wv, latent_C);
        fill_bias_uniform(mid_self_bv, latent_C);
        fill_kaiming_uniform(mid_attn_w.self_attn.wout, latent_C);
        fill_bias_uniform(mid_self_bout, latent_C);

        fill_kaiming_uniform(mid_attn_w.cross_attn.wq, latent_C);
        fill_bias_uniform(mid_cross_bq, latent_C);
        fill_kaiming_uniform(mid_attn_w.cross_attn.wk, ctx_dim);
        fill_bias_uniform(mid_cross_bk, ctx_dim);
        fill_kaiming_uniform(mid_attn_w.cross_attn.wv, ctx_dim);
        fill_bias_uniform(mid_cross_bv, ctx_dim);
        fill_kaiming_uniform(mid_attn_w.cross_attn.wout, latent_C);
        fill_bias_uniform(mid_cross_bout, latent_C);

        fill_kaiming_uniform(mid_attn_w.ff.w_up, latent_C);
        fill_bias_uniform(mid_ff_b_up, latent_C);
        fill_kaiming_uniform(mid_attn_w.ff.w_down, mid_attn_w.ff.inner_dim);
        fill_bias_uniform(mid_ff_b_down, mid_attn_w.ff.inner_dim);

        fill_kaiming_uniform(vae_post_w, latent_C);
        fill_bias_uniform(vae_post_b, latent_C);
        fill_kaiming_uniform(vae_res1_c1_w, latent_C * 3 * 3);
        fill_bias_uniform(vae_res1_c1_b, latent_C * 3 * 3);
        fill_kaiming_uniform(vae_res1_c2_w, latent_C * 3 * 3);
        fill_bias_uniform(vae_res1_c2_b, latent_C * 3 * 3);
        fill_kaiming_uniform(vae_res2_c1_w, latent_C * 3 * 3);
        fill_bias_uniform(vae_res2_c1_b, latent_C * 3 * 3);
        fill_kaiming_uniform(vae_res2_c2_w, latent_C * 3 * 3);
        fill_bias_uniform(vae_res2_c2_b, latent_C * 3 * 3);

        fill_kaiming_uniform(vae_attn_w.wq, latent_C);
        fill_bias_uniform(vae_attn_bq, latent_C);
        fill_kaiming_uniform(vae_attn_w.wk, latent_C);
        fill_bias_uniform(vae_attn_bk, latent_C);
        fill_kaiming_uniform(vae_attn_w.wv, latent_C);
        fill_bias_uniform(vae_attn_bv, latent_C);
        fill_kaiming_uniform(vae_attn_w.wout, latent_C);
        fill_bias_uniform(vae_attn_bout, latent_C);
        fill_kaiming_uniform(vae_out_w, latent_C * 3 * 3);
        fill_bias_uniform(vae_out_b, latent_C * 3 * 3);
    }

    // Share model weights.
    shark::utils::start_timer("input");
    input::call(text_token_embed, SERVER);
    input::call(text_pos_embed, SERVER);
    input::call(img_patch_w, SERVER);
    input::call(img_patch_b, SERVER);
    input::call(img_cls, SERVER);
    input::call(img_pos_embed, SERVER);
    input::call(img_proj_w, SERVER);
    input::call(img_proj_b, SERVER);
    keygen_ckpt(1, 1, "auth_weights:image_encoder");
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
    keygen_ckpt(1, 2, "auth_weights:unet_core");
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
    keygen_ckpt(1, 3, "auth_weights:unet_mid_attn");
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
    keygen_ckpt(1, 4, "auth_weights:vae");

    // Authenticate biases and additive parameters once.
    auto img_patch_b_a = auth_from_plain_open(img_patch_b);
    auto img_cls_a = auth_from_plain_open(img_cls);
    auto img_pos_embed_a = auth_from_plain_open(img_pos_embed);
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
    const char *prompt_env = std::getenv("UNCLIP_PROMPT");
    std::string prompt = prompt_env && *prompt_env ? std::string(prompt_env) : std::string("a snowy mountain landscape");
    auto prompt_ids = tokenize_prompt(prompt, vocab, seq_len);
    auto neg_prompt_ids = tokenize_prompt("", vocab, seq_len);

    span<u64> prompt_embeds(batch * seq_len * hidden);
    span<u64> neg_prompt_embeds(batch * seq_len * hidden);
    if (party == SERVER || party == DEALER) {
        auto pe = build_prompt_embeds(prompt_ids, text_token_embed, text_pos_embed, hidden, seq_len);
        auto ne = build_prompt_embeds(neg_prompt_ids, text_token_embed, text_pos_embed, hidden, seq_len);
        for (u64 b = 0; b < batch; ++b) {
            u64 base = b * seq_len * hidden;
            for (u64 i = 0; i < seq_len * hidden; ++i) {
                prompt_embeds[base + i] = pe[i];
                neg_prompt_embeds[base + i] = ne[i];
            }
        }
    }
    input::call(prompt_embeds, SERVER);
    input::call(neg_prompt_embeds, SERVER);
    keygen_ckpt(2, 1, "input:prompt_embeds");
    clip_batch_check("debug:after prompt_embeds");

    auto prompt_embeds_a = auth_from_plain_open(prompt_embeds);
    auto neg_prompt_embeds_a = auth_from_plain_open(neg_prompt_embeds);

    // Concatenate for classifier-free guidance.
    AuthShare prompt_embeds_cfg = auth_alloc(2 * batch * seq_len * hidden);
    for (u64 i = 0; i < batch * seq_len * hidden; ++i) {
        prompt_embeds_cfg.share[i] = neg_prompt_embeds_a.share[i];
        prompt_embeds_cfg.tag[i] = neg_prompt_embeds_a.tag[i];
        prompt_embeds_cfg.share[batch * seq_len * hidden + i] = prompt_embeds_a.share[i];
        prompt_embeds_cfg.tag[batch * seq_len * hidden + i] = prompt_embeds_a.tag[i];
    }

    // Image input: [B, H, W, C] in HWC layout.
    // Python demo uses torch.rand([1,3,224,224]) in [0,1), so keep that by default.
    // Set UNCLIP_APPLY_CLIP_IMAGE_NORM=1 to force CLIP mean/std normalization.
    span<u64> image_input(batch * img_H * img_W * C);
    if (party == CLIENT || party == DEALER) {
        current_rng_seed = base_rng_seed;
        const size_t needed = (size_t)(img_H * img_W * C);
        std::vector<u64> image_values(needed);
        for (size_t i = 0; i < needed; ++i) image_values[i] = qrand_uniform(f);
        for (u64 h = 0; h < img_H; ++h) {
            for (u64 widx = 0; widx < img_W; ++widx) {
                for (u64 c = 0; c < C; ++c) {
                    size_t src = ((size_t)h * img_W + widx) * C + c;
                    size_t dst = ((size_t)h * img_W + widx) * C + c;
                    image_input[dst] = image_values[src];
                }
            }
        }
    }
    input::call(image_input, CLIENT);
    keygen_ckpt(2, 2, "input:image");
    clip_batch_check("debug:after image_input");
    if (std::getenv("UNCLIP_APPLY_CLIP_IMAGE_NORM")) {
        image_input = normalize_image_input(image_input, img_H, img_W, C);
        clip_batch_check("debug:after image_norm");
    }
    auto image_input_a = auth_from_plain_open(image_input);

    // -----------------------------
    // Image encoder (patch conv + proj)
    // -----------------------------
    // Patch conv (k=patch_size, stride=patch_size) => output [B, H', W', img_hidden]
    auto patch = conv2d_apply(batch, img_H, img_W, C, img_hidden, patch_size, patch_size, 0,
                              image_input_a, img_patch_w, &img_patch_b_a);
    keygen_ckpt(3, 1, "image_encoder:patch_conv");
    clip_batch_check("debug:after patch conv");

    // Flatten patches to [N, img_hidden], add cls token + pos embeds.
    const u64 N = num_patches;
    AuthShare img_tokens = auth_alloc((N + 1) * img_hidden);
    for (u64 c = 0; c < img_hidden; ++c) {
        img_tokens.share[c] = img_cls_a.share[c] + img_pos_embed_a.share[c];
        img_tokens.tag[c] = img_cls_a.tag[c] + img_pos_embed_a.tag[c];
    }
    for (u64 i = 0; i < N; ++i) {
        for (u64 c = 0; c < img_hidden; ++c) {
            u64 dst = (i + 1) * img_hidden + c;
            u64 src = i * img_hidden + c;
            img_tokens.share[dst] = patch.share[src] + img_pos_embed_a.share[dst];
            img_tokens.tag[dst] = patch.tag[src] + img_pos_embed_a.tag[dst];
        }
    }
    // Layernorm across tokens (approx).
    auto img_tokens_ln = timed_layernorm_rows("layernorm:image_encoder.tokens", N + 1, img_hidden, img_tokens);
    keygen_ckpt(3, 2, "image_encoder:token_ln");
    // CLS token -> projection
    AuthShare cls_token = auth_alloc(img_hidden);
    for (u64 i = 0; i < img_hidden; ++i) {
        cls_token.share[i] = img_tokens_ln.share[i];
        cls_token.tag[i] = img_tokens_ln.tag[i];
    }
    auto img_embed = linear_mat(1, img_hidden, img_emb_dim, cls_token, img_proj_w, &img_proj_b_a);
    keygen_ckpt(3, 3, "image_encoder:cls_proj");

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
    keygen_ckpt(2, 3, "input:noise");
    auto noise_a = auth_from_plain_open(noise);
    auto img_embed_noised = add_noise_ddpm(img_embed, noise_a, alpha_noise);

    // Noise-level embedding (sin/cos) and concatenate with image embeds.
    auto noise_level_embed = make_timestep_embedding((int)noise_level, img_emb_dim, f, true);
    auto noise_level_embed_a = auth_from_plain_open(noise_level_embed);

    AuthShare class_labels = auth_alloc(2 * img_emb_dim);
    for (u64 i = 0; i < img_emb_dim; ++i) {
        class_labels.share[i] = img_embed_noised.share[i];
        class_labels.tag[i] = img_embed_noised.tag[i];
        class_labels.share[img_emb_dim + i] = noise_level_embed_a.share[i];
        class_labels.tag[img_emb_dim + i] = noise_level_embed_a.tag[i];
    }

    // CFG: prepend zeros.
    AuthShare class_labels_cfg = auth_alloc(2 * batch * class_labels.share.size());
    for (u64 i = 0; i < batch * class_labels.share.size(); ++i) {
        class_labels_cfg.share[i] = 0;
        class_labels_cfg.tag[i] = 0;
        class_labels_cfg.share[batch * class_labels.share.size() + i] = class_labels.share[i % class_labels.share.size()];
        class_labels_cfg.tag[batch * class_labels.share.size() + i] = class_labels.tag[i % class_labels.share.size()];
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
    keygen_ckpt(2, 4, "input:latents");
    shark::utils::stop_timer("input");
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
    auto unet_forward = [&](const AuthShare &lat_in, const AuthShare &prompt_ctx,
                            const AuthShare &cls_labels, int timestep,
                            int diffusion_step, int diffusion_total) -> AuthShare {
        u64 B = 2 * batch;
        u64 Ht = H;
        u64 Wt = W;
        auto unet_step_ckpt = [&](int component, const char *op_name) {
            std::ostringstream oss;
            oss << "unet:" << op_name
                << " step " << diffusion_step << "/" << diffusion_total
                << " (t=" << timestep << ")";
            keygen_ckpt(5, component, oss.str());
        };
        auto t_base = make_timestep_embedding(timestep, time_in_dim, f, true);
        span<u64> t_embed_plain(B * time_in_dim);
        if (party == SERVER || party == DEALER) {
            for (u64 n = 0; n < B; ++n) {
                for (u64 i = 0; i < time_in_dim; ++i) {
                    t_embed_plain[n * time_in_dim + i] = t_base[i];
                }
            }
        } else {
            for (u64 i = 0; i < t_embed_plain.size(); ++i) t_embed_plain[i] = 0;
        }
        auto t_embed = auth_from_plain_open(t_embed_plain);

        shark::utils::start_timer("linear1_mlp");
        auto t1 = linear_mat(B, time_in_dim, temb_dim, t_embed, time_w1, &time_b1_a);
        shark::utils::stop_timer("linear1_mlp");
        shark::utils::start_timer("gelu:unet.time_mlp");
        shark::utils::start_timer("exp_mlp");
        auto t1_act = timed_silu("silu:unet.time_mlp", t1);
        shark::utils::stop_timer("exp_mlp");
        shark::utils::stop_timer("gelu:unet.time_mlp");
        unet_step_ckpt(1, "time_mlp");
        shark::utils::start_timer("linear3_mlp");
        auto temb = linear_mat(B, temb_dim, temb_dim, t1_act, time_w2, &time_b2_a);
        shark::utils::stop_timer("linear3_mlp");

        auto imgp = linear_mat(B, class_labels_dim, temb_dim, cls_labels, img_proj_w2, &img_proj_b2_a);
        temb = ADD_CALL(temb, imgp);

        // conv_in
        auto h0 = conv2d_apply_k3_same(B, Ht, Wt, latent_C, latent_C, 1, lat_in, unet_conv_in_w, &unet_conv_in_b_a);
        unet_step_ckpt(2, "conv_in");

        // ResBlock1
        auto h1n = timed_groupnorm_apply("layernorm:unet.rb1.gn1", B, Ht, Wt, latent_C, h0);
        auto h1a = timed_silu("silu:unet.rb1.act1", h1n);
        auto h1c1 = conv2d_apply_k3_same(B, Ht, Wt, latent_C, latent_C, 1, h1a, rb1_conv1_w, &rb1_conv1_b_a);
        // temb broadcast
        auto temb1 = linear_mat(B, temb_dim, latent_C, temb, rb1_temb_w, &rb1_temb_b_a);
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
        auto h1c2 = conv2d_apply_k3_same(B, Ht, Wt, latent_C, latent_C, 1, h1a2, rb1_conv2_w, &rb1_conv2_b_a);
        auto h1 = ADD_CALL(h1c2, h0);
        unet_step_ckpt(3, "resblock1");

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
            auto x_b_out = basic_transformer_block(Nseq, latent_C, seq_len, ctx_dim, x_b, ctx_b, mid_attn_w);
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
        unet_step_ckpt(4, "mid_attn");

        // ResBlock2
        auto h2n = timed_groupnorm_apply("layernorm:unet.rb2.gn1", B, Ht, Wt, latent_C, h_attn);
        auto h2a = timed_silu("silu:unet.rb2.act1", h2n);
        auto h2c1 = conv2d_apply_k3_same(B, Ht, Wt, latent_C, latent_C, 1, h2a, rb2_conv1_w, &rb2_conv1_b_a);
        auto temb2 = linear_mat(B, temb_dim, latent_C, temb, rb2_temb_w, &rb2_temb_b_a);
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
        auto h2c2 = conv2d_apply_k3_same(B, Ht, Wt, latent_C, latent_C, 1, h2a2, rb2_conv2_w, &rb2_conv2_b_a);
        auto h2 = ADD_CALL(h2c2, h_attn);
        unet_step_ckpt(5, "resblock2");

        auto out = conv2d_apply_k3_same(B, Ht, Wt, latent_C, latent_C, 1, h2, unet_conv_out_w, &unet_conv_out_b_a);
        unet_step_ckpt(6, "conv_out");
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

        int t = timesteps[step];
        auto noise_pred = unet_forward(latents_cfg, prompt_embeds_cfg, class_labels_cfg, t, step + 1, num_inference_steps);

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

        int t_prev = std::max(t - 1, 0);
        double alpha_t = alphas_cumprod[t];
        double alpha_prev = alphas_cumprod[t_prev];
        latents_a = scheduler_step_linear(latents_a, guided, alpha_t, alpha_prev);
        std::ostringstream step_label;
        step_label << "diffusion:step " << (step + 1) << "/" << num_inference_steps << " (t=" << t << ")";
        keygen_ckpt(4, step + 1, step_label.str());
        if (profile_step_summary_enabled() && party != DEALER) {
            std::cout << "[PROFILE_STEP] " << step_label.str() << std::endl;
            if (shark::protocols::peer) {
                u64 total_comm = shark::protocols::peer->bytesReceived() + shark::protocols::peer->bytesSent();
                u64 total_rounds = shark::protocols::peer->roundsReceived() + shark::protocols::peer->roundsSent();
                std::cout << "[PROFILE_STEP] total_comm: " << (double)total_comm / 1024.0 << " KB (" << total_comm
                          << " bytes), total_rounds: " << total_rounds << std::endl;
            }
            print_profile_timers();
        }
    }

    // -----------------------------
    // VAE decode (lightweight but uses secure ops)
    // -----------------------------
    double scaling_factor = 0.18215;
    auto inv_scale = make_public_const(latents_a.share.size(), 1.0 / scaling_factor, f);
    auto latents_scaled = LRS_CALL(MUL_CALL(latents_a, inv_scale), f);

    auto z = conv2d_apply(batch, H, W, latent_C, latent_C, 1, 1, 0, latents_scaled, vae_post_w, &vae_post_b_a);
    keygen_ckpt(6, 1, "vae:post");

    auto r1n = timed_groupnorm_apply("layernorm:vae.res1.gn1", batch, H, W, latent_C, z);
    auto r1a = timed_silu("silu:vae.res1.act1", r1n);
    auto r1c1 = conv2d_apply_k3_same(batch, H, W, latent_C, latent_C, 1, r1a, vae_res1_c1_w, &vae_res1_c1_b_a);
    auto r1n2 = timed_groupnorm_apply("layernorm:vae.res1.gn2", batch, H, W, latent_C, r1c1);
    auto r1a2 = timed_silu("silu:vae.res1.act2", r1n2);
    auto r1c2 = conv2d_apply_k3_same(batch, H, W, latent_C, latent_C, 1, r1a2, vae_res1_c2_w, &vae_res1_c2_b_a);
    auto r1 = ADD_CALL(r1c2, z);
    keygen_ckpt(6, 2, "vae:res1");

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
        auto x_b_out = apply_attn(x_b, Nseq, latent_C, x_b, Nseq, latent_C, vae_attn_w, "softmax:vae.self_attn");
        for (u64 i = 0; i < Nseq * latent_C; ++i) {
            attn_out.share[offset + i] = x_b_out.share[i];
            attn_out.tag[offset + i] = x_b_out.tag[i];
        }
    }
    keygen_ckpt(6, 3, "vae:attn");
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
    auto r2c1 = conv2d_apply_k3_same(batch, H, W, latent_C, latent_C, 1, r2a, vae_res2_c1_w, &vae_res2_c1_b_a);
    auto r2n2 = timed_groupnorm_apply("layernorm:vae.res2.gn2", batch, H, W, latent_C, r2c1);
    auto r2a2 = timed_silu("silu:vae.res2.act2", r2n2);
    auto r2c2 = conv2d_apply_k3_same(batch, H, W, latent_C, latent_C, 1, r2a2, vae_res2_c2_w, &vae_res2_c2_b_a);
    auto r2 = ADD_CALL(r2c2, r1_attn);
    keygen_ckpt(6, 4, "vae:res2");

    auto out_img = conv2d_apply_k3_same(batch, H, W, latent_C, C, 1, r2, vae_out_w, &vae_out_b_a);
    keygen_ckpt(6, 5, "vae:out");

    // Reveal first few pixels for debug.
    if (party != DEALER) {
        auto out_plain = authenticated_reconstruct(out_img.share, out_img.tag);
        clip_batch_check("output");
        // Final activation is on revealed output, so apply tanh in clear to avoid
        // re-authentication and extra MAC state in the secure pipeline tail.
        for (u64 i = 0; i < out_plain.size(); ++i) {
            double v = (double)(int64_t)out_plain[i] / (double)(1ULL << f);
            int64_t q = (int64_t)std::llround(std::tanh(v) * (double)(1ULL << f));
            out_plain[i] = (u64)q;
        }
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
            std::cout << "[PROFILE] total_comm: " << (double)total_comm / 1024.0 << " KB (" << total_comm << " bytes)" << std::endl;
        }
        print_profile_timers();
        print_profile_components_table();
        print_legacy_profile_lines();
    }

    finalize::call();
    return 0;
#endif
}
