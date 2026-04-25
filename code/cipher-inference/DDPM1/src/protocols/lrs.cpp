#include <shark/protocols/lrs.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>

namespace shark
{
    namespace protocols
    {
        namespace lrs
        {
            void gen(const shark::span<u64> &X, shark::span<u64> &Y, int f)
            {
                always_assert(X.size() == Y.size());
                always_assert(f > 0 && f < 64);

                randomize(Y);

                send_dcfbit(X, 64);
                send_dcfbit(X, f);

                shark::span<u8> r_w(X.size());
                shark::span<u8> r_t(X.size());

                randomize(r_w);
                randomize(r_t);

                send_authenticated_bshare(r_w);
                send_authenticated_bshare(r_t);

                shark::span<u64> T(X.size() * 4);
                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); ++i)
                {
                    for (u64 j = 0; j < 4; ++j)
                    {
                        const u8 w = (j / 1) % 2;
                        const u8 t = (j / 2) % 2;
                        T[i * 4 + j] =
                            ((u64(1) << (64 - f)) * u64(w ^ r_w[i]))
                            - u64(t ^ r_t[i])
                            - (X[i] >> f)
                            + Y[i];
                    }
                }
                send_authenticated_ashare(T);
            }

            void eval(const shark::span<u64> &X, shark::span<u64> &Y, int f)
            {
                always_assert(X.size() == Y.size());
                always_assert(f > 0 && f < 64);

                shark::utils::start_timer("key_read");
                auto dcfkeysN = recv_dcfbit(X.size(), 64);
                auto dcfkeysF = recv_dcfbit(X.size(), f);
                auto r_w = recv_authenticated_bshare(X.size());
                auto r_t = recv_authenticated_bshare(X.size());
                auto [T, T_tag] = recv_authenticated_ashare(X.size() * 4);
                shark::utils::stop_timer("key_read");

                shark::span<FKOS> w(X.size());
                shark::span<FKOS> t(X.size());
                shark::span<u64> Y_share(X.size());
                shark::span<u128> Y_tag(X.size());

                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); ++i)
                {
                    w[i] = dcfbit_eval(dcfkeysN[i], u128(X[i]));
                    t[i] = dcfbit_eval(dcfkeysF[i], u128(X[i]));

                    w[i] = xor_fkos(w[i], r_w[i]);
                    t[i] = xor_fkos(t[i], r_t[i]);
                }

                auto w_cap = authenticated_reconstruct(w);
                auto t_cap = authenticated_reconstruct(t);

                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); ++i)
                {
                    const u64 idx = 2 * u64(t_cap[i]) + u64(w_cap[i]);
                    auth_local_add_public_u64(
                        T[i * 4 + idx], T_tag[i * 4 + idx], X[i] >> f,
                        party == CLIENT, Y_share[i], Y_tag[i]);
                }

                Y = authenticated_reconstruct(Y_share, Y_tag);
            }

            void eval_share(const shark::span<u64> &X, shark::span<u64> &Y_share, shark::span<u128> &Y_tag, int f)
            {
                always_assert(X.size() == Y_share.size());
                always_assert(X.size() == Y_tag.size());
                always_assert(f > 0 && f < 64);

                shark::utils::start_timer("key_read");
                auto dcfkeysN = recv_dcfbit(X.size(), 64);
                auto dcfkeysF = recv_dcfbit(X.size(), f);
                auto r_w = recv_authenticated_bshare(X.size());
                auto r_t = recv_authenticated_bshare(X.size());
                auto [T, T_tag] = recv_authenticated_ashare(X.size() * 4);
                shark::utils::stop_timer("key_read");

                shark::span<FKOS> w(X.size());
                shark::span<FKOS> t(X.size());

                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); ++i)
                {
                    w[i] = dcfbit_eval(dcfkeysN[i], u128(X[i]));
                    t[i] = dcfbit_eval(dcfkeysF[i], u128(X[i]));

                    w[i] = xor_fkos(w[i], r_w[i]);
                    t[i] = xor_fkos(t[i], r_t[i]);
                }

                auto w_cap = authenticated_reconstruct(w);
                auto t_cap = authenticated_reconstruct(t);

                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); ++i)
                {
                    const u64 idx = 2 * u64(t_cap[i]) + u64(w_cap[i]);
                    auth_local_add_public_u64(
                        T[i * 4 + idx], T_tag[i * 4 + idx], X[i] >> f,
                        party == CLIENT, Y_share[i], Y_tag[i]);
                }
            }

            void call(const shark::span<u64> &X, shark::span<u64> &Y, int f)
            {
                if (party == DEALER)
                {
                    gen(X, Y, f);
                }
                else
                {
                    eval(X, Y, f);
                }
            }

            void call_share(const shark::span<u64> &X, shark::span<u64> &Y_share, shark::span<u128> &Y_tag, int f)
            {
                if (party == DEALER)
                {
                    shark::span<u64> Y(X.size());
                    gen(X, Y, f);
                    #pragma omp parallel for
                    for (u64 i = 0; i < Y.size(); ++i)
                    {
                        Y_share[i] = Y[i];
                        Y_tag[i] = mac_mul_u64(Y[i]);
                    }
                }
                else
                {
                    eval_share(X, Y_share, Y_tag, f);
                }
            }

            void call_share_secret_full(const shark::span<u128> &X_share, const shark::span<u128> &X_tag,
                                        shark::span<u128> &Y_share, shark::span<u128> &Y_tag, int f)
            {
                always_assert(X_share.size() == X_tag.size());
                always_assert(X_share.size() == Y_share.size());
                always_assert(X_share.size() == Y_tag.size());
                always_assert(f > 0 && f < 64);

                if (party == DEALER)
                {
                    #pragma omp parallel for
                    for (u64 i = 0; i < X_share.size(); ++i)
                    {
                        Y_share[i] = u128(getLow(X_share[i] >> f));
                        Y_tag[i] = mac_mul_u128(Y_share[i]);
                    }

                    shark::span<u128> r(X_share.size());
                    randomize_full(r);
                    send_authenticated_ashare_full(r);

                    shark::span<u64> r_lo(X_share.size());
                    const u64 mask_f = (u64(1) << f) - 1;
                    #pragma omp parallel for
                    for (u64 i = 0; i < X_share.size(); ++i)
                    {
                        r_lo[i] = getLow(r[i]) & mask_f;
                    }
                    send_dcfbit(r_lo, f);

                    shark::span<u8> r_t(X_share.size());
                    randomize(r_t);

                    send_authenticated_bshare(r_t);

                    shark::span<u128> T(X_share.size() * 2);
                    // Plaintexts are 64-bit fixed-point values; the full path keeps authenticated shares/tags
                    // in Z_{2^128}, but only the low-64 payload semantics survive truncation. The 128-bit wrap bit
                    // has no contribution to low64(x >> f), so only the low-f correction bit t remains.
                    #pragma omp parallel for
                    for (u64 i = 0; i < X_share.size(); ++i)
                    {
                        const u128 r_hi = u128(getLow(r[i] >> f));
                        for (u64 j = 0; j < 2; ++j)
                        {
                            const u8 t = u8(j);
                            T[i * 2 + j] =
                                - u128(t ^ r_t[i])
                                - r_hi;
                        }
                    }
                    send_authenticated_ashare_full(T);
                    return;
                }

                shark::utils::start_timer("key_read");
                auto [r_share, r_tag] = recv_authenticated_ashare_full(X_share.size());
                auto dcfkeysF = recv_dcfbit(X_share.size(), f);
                auto r_t = recv_authenticated_bshare(X_share.size());
                auto [T, T_tag] = recv_authenticated_ashare_full(X_share.size() * 2);
                shark::utils::stop_timer("key_read");

                // Full-width path: keep authenticated shares/tags in Z_{2^128} and reconstruct masked c = x + r.
                shark::span<u128> c_auth_share(X_share.size());
                shark::span<u128> c_auth_tag(X_share.size());
                #pragma omp parallel for
                for (u64 i = 0; i < X_share.size(); ++i)
                {
                    auth_local_add_u128(X_share[i], X_tag[i], r_share[i], r_tag[i],
                                        c_auth_share[i], c_auth_tag[i]);
                }
                auto c = authenticated_reconstruct_full(c_auth_share, c_auth_tag);

                shark::span<FKOS> t(X_share.size());
                const u64 mask_f = (u64(1) << f) - 1;

                #pragma omp parallel for
                for (u64 i = 0; i < X_share.size(); ++i)
                {
                    t[i] = dcfbit_eval(dcfkeysF[i], u128(getLow(c[i]) & mask_f));
                    t[i] = xor_fkos(t[i], r_t[i]);
                }

                auto t_cap = authenticated_reconstruct(t);

                #pragma omp parallel for
                for (u64 i = 0; i < X_share.size(); ++i)
                {
                    const u64 idx = u64(t_cap[i]);
                    const u128 c_hi = u128(getLow(c[i] >> f));
                    auth_local_add_public_u128(
                        T[i * 2 + idx], T_tag[i * 2 + idx], c_hi,
                        party == CLIENT, Y_share[i], Y_tag[i]);
                }
            }

            shark::span<u64> call(const shark::span<u64> &X, int f)
            {
                shark::span<u64> Y(X.size());
                call(X, Y, f);
                return Y;
            }

            AuthShare call_share(const shark::span<u64> &X, int f)
            {
                AuthShare out{
                    shark::span<u64>(X.size()),
                    shark::span<u128>(X.size())
                };
                call_share(X, out.share, out.tag, f);
                return out;
            }

            AuthShareFull call_share_secret_full(const shark::span<u128> &X_share, const shark::span<u128> &X_tag, int f)
            {
                AuthShareFull out{
                    shark::span<u128>(X_share.size()),
                    shark::span<u128>(X_share.size())
                };
                call_share_secret_full(X_share, X_tag, out.share, out.tag, f);
                return out;
            }
        }
    }
}
