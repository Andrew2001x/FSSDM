#include <shark/protocols/mul.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>

#include <cstdlib>

namespace shark {
    namespace protocols {
        namespace mul {

            static bool public_alpha_enabled()
            {
                static int enabled = -1;
                if (enabled < 0)
                {
                    enabled = (std::getenv("SHARK_PUBLIC_ALPHA") != nullptr) ? 1 : 0;
                }
                return enabled == 1;
            }

            static void gen_secret_full(u64 n)
            {
                shark::span<u128> r_X(n);
                shark::span<u128> r_Y(n);
                shark::span<u128> r_Z(n);
                randomize_full(r_X);
                randomize_full(r_Y);
                randomize_full(r_Z);

                shark::span<u128> r_C(n);
                #pragma omp parallel for
                for (u64 i = 0; i < n; i++)
                {
                    r_C[i] = r_X[i] * r_Y[i] + r_Z[i];
                }

                send_authenticated_ashare_full(r_X);
                send_authenticated_ashare_full(r_Y);
                send_authenticated_ashare_full(r_C);
            }

            void gen(const shark::span<u64> &r_X, const shark::span<u64> &r_Y, shark::span<u64> &r_Z)
            {
                u64 n = r_X.size();
                always_assert(r_Y.size() == n);
                always_assert(r_Z.size() == n);

                randomize(r_Z);

                shark::span<u64> r_C(n);
                // r_C = r_X @ r_Y + r_Z
                #pragma omp parallel for
                for (u64 i = 0; i < n; i++)
                {
                    r_C[i] = r_X[i] * r_Y[i] + r_Z[i];
                }

                send_authenticated_ashare(r_X);
                send_authenticated_ashare(r_Y);
                send_authenticated_ashare(r_C);
            }

            void eval(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                u64 n = X.size();
                always_assert(Y.size() == n);
                always_assert(Z.size() == n);

                shark::utils::start_timer("key_read");
                auto [r_X, r_X_tag] = recv_authenticated_ashare(n);
                auto [r_Y, r_Y_tag] = recv_authenticated_ashare(n);
                auto [r_Z, r_Z_tag] = recv_authenticated_ashare(n);
                shark::utils::stop_timer("key_read");


                shark::span<u64> Z_share(n);
                shark::span<u128> Z_tag(n);

                // Z = r_Z + X @ Y - r_X @ Y - X @ r_Y
                #pragma omp parallel for
                for (u64 i = 0; i < n; i++)
                {
                    const u128 raw = u128(r_Z[i])
                                   + (u128(u64(party)) * u128(X[i]) - u128(r_X[i])) * u128(Y[i])
                                   - u128(r_Y[i]) * u128(X[i]);
                    Z_share[i] = getLow(raw);
                    if (public_alpha_enabled())
                    {
                        Z_tag[i] = mac_sub_u128(mac_mul_u128(raw), mac_wrap_u64(getHigh(raw)));
                    }
                    else
                    {
                        u128 t = mac_sub(mac_mul_u64(X[i]), r_X_tag[i]);
                        u128 term1 = mac_mul_tag(t, Y[i]);
                        u128 term2 = mac_mul_tag(r_Y_tag[i], X[i]);
                        Z_tag[i] = mac_sub_u128(mac_add(r_Z_tag[i], mac_sub(term1, term2)),
                                                mac_wrap_u64(getHigh(raw)));
                    }
                }

                Z = authenticated_reconstruct(Z_share, Z_tag);
            }

            void eval_share(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z_share, shark::span<u128> &Z_tag)
            {
                u64 n = X.size();
                always_assert(Y.size() == n);
                always_assert(Z_share.size() == n);
                always_assert(Z_tag.size() == n);

                shark::utils::start_timer("key_read");
                auto [r_X, r_X_tag] = recv_authenticated_ashare(n);
                auto [r_Y, r_Y_tag] = recv_authenticated_ashare(n);
                auto [r_Z, r_Z_tag] = recv_authenticated_ashare(n);
                shark::utils::stop_timer("key_read");

                #pragma omp parallel for
                for (u64 i = 0; i < n; i++)
                {
                    const u128 raw = u128(r_Z[i])
                                   + (u128(u64(party)) * u128(X[i]) - u128(r_X[i])) * u128(Y[i])
                                   - u128(r_Y[i]) * u128(X[i]);
                    Z_share[i] = getLow(raw);
                    if (public_alpha_enabled())
                    {
                        Z_tag[i] = mac_sub_u128(mac_mul_u128(raw), mac_wrap_u64(getHigh(raw)));
                    }
                    else
                    {
                        u128 t = mac_sub(mac_mul_u64(X[i]), r_X_tag[i]);
                        u128 term1 = mac_mul_tag(t, Y[i]);
                        u128 term2 = mac_mul_tag(r_Y_tag[i], X[i]);
                        Z_tag[i] = mac_sub_u128(mac_add(r_Z_tag[i], mac_sub(term1, term2)),
                                                mac_wrap_u64(getHigh(raw)));
                    }
                }
            }

            void call(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                if (party == DEALER)
                {
                    gen(X, Y, Z);
                }
                else
                {
                    eval(X, Y, Z);
                }
            }

            void call_share(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z_share, shark::span<u128> &Z_tag)
            {
                if (party == DEALER)
                {
                    shark::span<u64> Z(X.size());
                    gen(X, Y, Z);
                    #pragma omp parallel for
                    for (u64 i = 0; i < Z.size(); i++)
                    {
                        Z_share[i] = Z[i];
                        Z_tag[i] = mac_mul_u64(Z[i]);
                    }
                }
                else
                {
                    eval_share(X, Y, Z_share, Z_tag);
                }
            }

            void call_share_secret_full(const shark::span<u128> &X_share, const shark::span<u128> &X_tag,
                                        const shark::span<u128> &Y_share, const shark::span<u128> &Y_tag,
                                        shark::span<u128> &Z_share, shark::span<u128> &Z_tag)
            {
                u64 n = X_share.size();
                always_assert(X_tag.size() == n);
                always_assert(Y_share.size() == n);
                always_assert(Y_tag.size() == n);
                always_assert(Z_share.size() == n);
                always_assert(Z_tag.size() == n);

                if (party == DEALER)
                {
                    gen_secret_full(n);
                    #pragma omp parallel for
                    for (u64 i = 0; i < n; i++)
                    {
                        u128 z = X_share[i] * Y_share[i];
                        Z_share[i] = z;
                        Z_tag[i] = mac_mul_u128(z);
                    }
                    return;
                }

                shark::utils::start_timer("key_read");
                auto [r_X, r_X_tag] = recv_authenticated_ashare_full(n);
                auto [r_Y, r_Y_tag] = recv_authenticated_ashare_full(n);
                auto [r_Z, r_Z_tag] = recv_authenticated_ashare_full(n);
                shark::utils::stop_timer("key_read");

                shark::span<u128> d_share(n);
                shark::span<u128> d_tag(n);
                #pragma omp parallel for
                for (u64 i = 0; i < n; i++)
                {
                    d_share[i] = X_share[i] - r_X[i];
                    d_tag[i] = X_tag[i] - r_X_tag[i];
                }
                auto D = authenticated_reconstruct_full(d_share, d_tag);

                shark::span<u128> e_share(n);
                shark::span<u128> e_tag(n);
                #pragma omp parallel for
                for (u64 i = 0; i < n; i++)
                {
                    e_share[i] = Y_share[i] - r_Y[i];
                    e_tag[i] = Y_tag[i] - r_Y_tag[i];
                }
                auto E = authenticated_reconstruct_full(e_share, e_tag);

                #pragma omp parallel for
                for (u64 i = 0; i < n; i++)
                {
                    u128 de = D[i] * E[i];
                    Z_share[i] = r_Z[i] + D[i] * r_Y[i] + r_X[i] * E[i] + de * u128(party);
                    Z_tag[i] = r_Z_tag[i] + D[i] * r_Y_tag[i] + r_X_tag[i] * E[i] + de * ring_key;
                }
            }

            shark::span<u64> call(const shark::span<u64> &X, const shark::span<u64> &Y)
            {
                shark::span<u64> Z(X.size());
                call(X, Y, Z);
                return Z;
            }

            AuthShare call_share(const shark::span<u64> &X, const shark::span<u64> &Y)
            {
                AuthShare out{
                    shark::span<u64>(X.size()),
                    shark::span<u128>(X.size())
                };
                call_share(X, Y, out.share, out.tag);
                return out;
            }

            AuthShareFull call_share_secret_full(const shark::span<u128> &X_share, const shark::span<u128> &X_tag,
                                                 const shark::span<u128> &Y_share, const shark::span<u128> &Y_tag)
            {
                AuthShareFull out{
                    shark::span<u128>(X_share.size()),
                    shark::span<u128>(X_share.size())
                };
                call_share_secret_full(X_share, X_tag, Y_share, Y_tag, out.share, out.tag);
                return out;
            }
        }
    }
}
