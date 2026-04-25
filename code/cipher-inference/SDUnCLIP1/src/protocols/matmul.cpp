#include <shark/protocols/matmul.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>
#include <shark/utils/eigen.hpp>

using namespace shark::matrix;

namespace shark {
    namespace protocols {
        namespace matmul {
            namespace {
                static void gen_secret_full(u64 a, u64 b, u64 c)
                {
                    shark::span<u128> r_X(a * b);
                    shark::span<u128> r_Y(b * c);
                    shark::span<u128> r_Z(a * c);
                    randomize_full(r_X);
                    randomize_full(r_Y);
                    randomize_full(r_Z);

                    auto mat_r_X = getMat(a, b, r_X);
                    auto mat_r_Y = getMat(b, c, r_Y);
                    auto mat_r_Z = getMat(a, c, r_Z);

                    shark::span<u128> r_C(a * c);
                    auto mat_r_C = getMat(a, c, r_C);
                    mat_r_C = mat_r_X * mat_r_Y + mat_r_Z;

                    send_authenticated_ashare_full(r_X);
                    send_authenticated_ashare_full(r_Y);
                    send_authenticated_ashare_full(r_C);
                }
            }

            void gen(u64 a, u64 b, u64 c, const shark::span<u64> &r_X, const shark::span<u64> &r_Y, shark::span<u64> &r_Z)
            {
                always_assert(r_X.size() == a * b);
                always_assert(r_Y.size() == b * c);
                always_assert(r_Z.size() == a * c);

                randomize(r_Z);
                auto mat_r_X = getMat(a, b, r_X);
                auto mat_r_Y = getMat(b, c, r_Y);
                auto mat_r_Z = getMat(a, c, r_Z);

                shark::span<u64> r_C(a * c);
                auto mat_r_C = getMat(a, c, r_C);
                // r_C = r_X @ r_Y + r_Z
                // shark::utils::matmuladd(a, b, c, r_X, r_Y, r_Z, r_C);
                mat_r_C = mat_r_X * mat_r_Y + mat_r_Z;

                send_authenticated_ashare(r_X);
                send_authenticated_ashare(r_Y);
                send_authenticated_ashare(r_C);
            }

            void eval(u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                shark::utils::start_timer("key_read");
                auto [r_X, r_X_tag] = recv_authenticated_ashare(a * b);
                auto [r_Y, r_Y_tag] = recv_authenticated_ashare(b * c);
                auto [r_Z, r_Z_tag] = recv_authenticated_ashare(a * c);
                shark::utils::stop_timer("key_read");

                auto mat_X = getMat(a, b, X).cast<u128>();
                auto mat_Y = getMat(b, c, Y).cast<u128>();

                shark::span<u64> Z_share(a * c);
                shark::span<u128> Z_raw(a * c);
                shark::span<u128> Z_tag(a * c);
                auto mat_Z_raw = getMat(a, c, Z_raw);
                auto mat_Z_tag = getMat(a, c, Z_tag);

                auto mat_r_X = getMat(a, b, r_X).cast<u128>();
                auto mat_r_X_tag = getMat(a, b, r_X_tag);
                auto mat_r_Y = getMat(b, c, r_Y).cast<u128>();
                auto mat_r_Y_tag = getMat(b, c, r_Y_tag);
                auto mat_r_Z = getMat(a, c, r_Z).cast<u128>();
                auto mat_r_Z_tag = getMat(a, c, r_Z_tag);

                // Z = r_Z + X @ Y - r_X @ Y - X @ r_Y
                mat_Z_raw = mat_r_Z + (mat_X * u128(u64(party)) - mat_r_X) * mat_Y;
                mat_Z_raw -= mat_X * mat_r_Y;

                mat_Z_tag = mat_r_Z_tag + (mat_X * ring_key - mat_r_X_tag) * mat_Y;
                mat_Z_tag -= mat_X * mat_r_Y_tag;
                #pragma omp parallel for
                for (u64 i = 0; i < Z_raw.size(); ++i)
                {
                    Z_share[i] = getLow(Z_raw[i]);
                    Z_tag[i] = mac_sub_u128(Z_tag[i], mac_wrap_u64(getHigh(Z_raw[i])));
                }

                Z = authenticated_reconstruct(Z_share, Z_tag);
            }

            void eval_share(u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z_share, shark::span<u128> &Z_tag)
            {
                shark::utils::start_timer("key_read");
                auto [r_X, r_X_tag] = recv_authenticated_ashare(a * b);
                auto [r_Y, r_Y_tag] = recv_authenticated_ashare(b * c);
                auto [r_Z, r_Z_tag] = recv_authenticated_ashare(a * c);
                shark::utils::stop_timer("key_read");

                shark::span<u64> r_X_low(a * b);
                shark::span<u64> r_Y_low(b * c);
                shark::span<u64> r_Z_low(a * c);
                #pragma omp parallel for
                for (u64 i = 0; i < a * b; ++i) r_X_low[i] = r_X[i];
                #pragma omp parallel for
                for (u64 i = 0; i < b * c; ++i) r_Y_low[i] = r_Y[i];
                #pragma omp parallel for
                for (u64 i = 0; i < a * c; ++i) r_Z_low[i] = r_Z[i];

                auto mat_X = getMat(a, b, X);
                auto mat_Y = getMat(b, c, Y);
                auto mat_X_u128 = getMat(a, b, X).cast<u128>();
                auto mat_Y_u128 = getMat(b, c, Y).cast<u128>();

                shark::span<u128> Z_raw(a * c);
                auto mat_Z_raw = getMat(a, c, Z_raw);
                auto mat_Z_tag = getMat(a, c, Z_tag);

                auto mat_r_X = getMat(a, b, r_X_low);
                auto mat_r_X_tag = getMat(a, b, r_X_tag);
                auto mat_r_Y = getMat(b, c, r_Y_low);
                auto mat_r_Y_tag = getMat(b, c, r_Y_tag);
                auto mat_r_Z = getMat(a, c, r_Z_low);
                auto mat_r_Z_tag = getMat(a, c, r_Z_tag);

                mat_Z_raw = mat_r_Z.cast<u128>()
                          + (mat_X_u128 * u128(u64(party)) - mat_r_X.cast<u128>()) * mat_Y_u128;
                mat_Z_raw -= mat_X_u128 * mat_r_Y.cast<u128>();

                mat_Z_tag = mat_r_Z_tag + (mat_X_u128 * ring_key - mat_r_X_tag) * mat_Y_u128;
                mat_Z_tag -= mat_X_u128 * mat_r_Y_tag;
                #pragma omp parallel for
                for (u64 i = 0; i < Z_raw.size(); ++i)
                {
                    Z_share[i] = getLow(Z_raw[i]);
                    Z_tag[i] = mac_sub_u128(Z_tag[i], mac_wrap_u64(getHigh(Z_raw[i])));
                }
            }

            void call(u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                if (party == DEALER)
                {
                    gen(a, b, c, X, Y, Z);
                }
                else
                {
                    eval(a, b, c, X, Y, Z);
                }
            }

            void call_share(u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z_share, shark::span<u128> &Z_tag)
            {
                if (party == DEALER)
                {
                    shark::span<u64> Z(a * c);
                    gen(a, b, c, X, Y, Z);
                    #pragma omp parallel for
                    for (u64 i = 0; i < Z.size(); i++)
                    {
                        Z_share[i] = Z[i];
                        Z_tag[i] = mac_mul_u64(Z[i]);
                    }
                }
                else
                {
                    eval_share(a, b, c, X, Y, Z_share, Z_tag);
                }
            }

            void call_share_secret_full(u64 a, u64 b, u64 c, const shark::span<u128> &X_share, const shark::span<u128> &X_tag,
                                        const shark::span<u128> &Y_share, const shark::span<u128> &Y_tag,
                                        shark::span<u128> &Z_share, shark::span<u128> &Z_tag)
            {
                always_assert(X_share.size() == a * b);
                always_assert(X_tag.size() == a * b);
                always_assert(Y_share.size() == b * c);
                always_assert(Y_tag.size() == b * c);
                always_assert(Z_share.size() == a * c);
                always_assert(Z_tag.size() == a * c);

                if (party == DEALER)
                {
                    gen_secret_full(a, b, c);
                    auto mat_X = getMat(a, b, X_share);
                    auto mat_Y = getMat(b, c, Y_share);
                    auto mat_Z = getMat(a, c, Z_share);
                    mat_Z = mat_X * mat_Y;
                    #pragma omp parallel for
                    for (u64 i = 0; i < Z_share.size(); i++)
                    {
                        Z_tag[i] = mac_mul_u128(Z_share[i]);
                    }
                    return;
                }

                shark::utils::start_timer("key_read");
                auto [r_X, r_X_tag] = recv_authenticated_ashare_full(a * b);
                auto [r_Y, r_Y_tag] = recv_authenticated_ashare_full(b * c);
                auto [r_C, r_C_tag] = recv_authenticated_ashare_full(a * c);
                shark::utils::stop_timer("key_read");

                shark::span<u128> d_share(a * b);
                shark::span<u128> d_tag(a * b);
                #pragma omp parallel for
                for (u64 i = 0; i < a * b; i++)
                {
                    d_share[i] = X_share[i] - r_X[i];
                    d_tag[i] = X_tag[i] - r_X_tag[i];
                }
                auto D = authenticated_reconstruct_full(d_share, d_tag);

                shark::span<u128> e_share(b * c);
                shark::span<u128> e_tag(b * c);
                #pragma omp parallel for
                for (u64 i = 0; i < b * c; i++)
                {
                    e_share[i] = Y_share[i] - r_Y[i];
                    e_tag[i] = Y_tag[i] - r_Y_tag[i];
                }
                auto E = authenticated_reconstruct_full(e_share, e_tag);

                auto mat_D = getMat(a, b, D);
                auto mat_E = getMat(b, c, E);
                auto mat_A = getMat(a, b, r_X);
                auto mat_B = getMat(b, c, r_Y);
                auto mat_A_tag = getMat(a, b, r_X_tag);
                auto mat_B_tag = getMat(b, c, r_Y_tag);
                auto mat_C = getMat(a, c, r_C);
                auto mat_C_tag = getMat(a, c, r_C_tag);

                auto mat_Z_share = getMat(a, c, Z_share);
                auto mat_Z_tag = getMat(a, c, Z_tag);

                auto mat_DE = mat_D * mat_E;
                mat_Z_share = mat_C + mat_D * mat_B + mat_A * mat_E + mat_DE * u128(party);
                mat_Z_tag = mat_C_tag + mat_D * mat_B_tag + mat_A_tag * mat_E + mat_DE * ring_key;
            }
            shark::span<u64> call(u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y)
            {
                shark::span<u64> Z(a * c);
                call(a, b, c, X, Y, Z);
                return Z;
            }

            AuthShare call_share(u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y)
            {
                AuthShare out{
                    shark::span<u64>(a * c),
                    shark::span<u128>(a * c)
                };
                call_share(a, b, c, X, Y, out.share, out.tag);
                return out;
            }

            AuthShareFull call_share_secret_full(u64 a, u64 b, u64 c, const shark::span<u128> &X_share, const shark::span<u128> &X_tag,
                                                 const shark::span<u128> &Y_share, const shark::span<u128> &Y_tag)
            {
                AuthShareFull out{
                    shark::span<u128>(a * c),
                    shark::span<u128>(a * c)
                };
                call_share_secret_full(a, b, c, X_share, X_tag, Y_share, Y_tag, out.share, out.tag);
                return out;
            }

            shark::span<u64> emul(u64 a, u64 b, u64 c, const shark::span<u64> &X, const shark::span<u64> &Y)
            {
                shark::span<u64> Z(a * c);
                auto X_mat = getMat(a, b, X);
                auto Y_mat = getMat(b, c, Y);
                auto Z_mat = getMat(a, c, Z);
                Z_mat = X_mat * Y_mat;
                return Z;
            }
        }
    }
}
