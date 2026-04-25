
#include <shark/protocols/common.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/spline.hpp>
#include <shark/utils/assert.hpp>

namespace shark
{
    namespace protocols
    {
        namespace spline
        {
            u64 nCr(u64 n, u64 r)
            {
                if (r > n)
                    return 0;
                if (r == 0 || r == n)
                    return 1;
                return nCr(n - 1, r - 1) + nCr(n - 1, r);
            }

            u64 pow(u64 x, int p)
            {
                if (p < 0)
                    return u64(0);
                if (p == 0)
                    return u64(1);
                u64 res = pow(x, p / 2);
                res = res * res;
                if (p % 2)
                    res = res * x;
                return res;
            }

            static inline u64 mask_bits(int bin)
            {
                if (bin >= 64)
                    return ~0ULL;
                return (1ULL << bin) - 1;
            }

            static inline u128 mask_bits_u128(int bin)
            {
                if (bin >= 128)
                    return ~u128(0);
                return (u128(1) << bin) - 1;
            }

            static inline void open_plain(const shark::span<u64> &share, shark::span<u64> &out)
            {
                if (party == DEALER)
                {
                    return;
                }

                shark::span<u64> tmp(share.size());
                if (party == SERVER)
                {
                    peer->send_array(share);
                    peer->recv_array(tmp);
                }
                else
                {
                    peer->recv_array(tmp);
                    peer->send_array(share);
                }

                #pragma omp parallel for
                for (u64 i = 0; i < share.size(); ++i)
                {
                    out[i] = share[i] + tmp[i];
                }
            }

            static inline void open_plain_full(const shark::span<u128> &share, shark::span<u128> &out)
            {
                if (party == DEALER)
                {
                    return;
                }

                shark::span<u128> tmp(share.size());
                if (party == SERVER)
                {
                    peer->send_array(share);
                    peer->recv_array(tmp);
                }
                else
                {
                    peer->recv_array(tmp);
                    peer->send_array(share);
                }

                #pragma omp parallel for
                for (u64 i = 0; i < share.size(); ++i)
                {
                    out[i] = share[i] + tmp[i];
                }
            }

            void gen(int bin, int degree, const std::vector<u64> &knots, const shark::span<u64> &X, shark::span<u64> &Y)
            {
                always_assert(X.size() == Y.size());

                const u64 m = knots.size();
                for (u64 i = 1; i < m; i++)
                {
                    always_assert(knots[i - 1] < knots[i]);
                }
                if (bin != 64 && m > 0)
                {
                    always_assert(knots[m - 1] < (1ull << bin));
                }

                const u64 mask = mask_bits(bin);
                const u128 mask_u128 = mask_bits_u128(bin);

                // Random mask for comparison inputs.
                shark::span<u128> r(X.size());
                randomize_full(r);
                if (bin < 64)
                {
                    #pragma omp parallel for
                    for (u64 i = 0; i < X.size(); ++i)
                    {
                        r[i] &= mask_u128;
                    }
                }
                send_authenticated_ashare_full(r);

                // DCF key for alpha = r (carry for c < r).
                {
                    shark::span<u128> alpha_r(X.size());
                    #pragma omp parallel for
                    for (u64 i = 0; i < X.size(); ++i)
                    {
                        alpha_r[i] = r[i] & mask_u128;
                    }
                    send_dcfring(alpha_r, bin);
                }

                // DCF keys for alpha = r + knots[j], and send overflow bits t3.
                for (u64 j = 0; j < m; ++j)
                {
                    shark::span<u128> alpha(X.size());
                    shark::span<u8> t3(X.size());
                    #pragma omp parallel for
                    for (u64 i = 0; i < X.size(); ++i)
                    {
                        u128 sum_full = r[i] + u128(knots[j]);
                        alpha[i] = sum_full & mask_u128;
                        if (bin >= 64)
                        {
                            t3[i] = (u8)((sum_full >> 64) & 1);
                        }
                        else
                        {
                            t3[i] = (u8)((sum_full >> bin) & 1);
                        }
                    }
                    send_dcfring(alpha, bin);
                    server->send_array(t3);
                    client->send_array(t3);
                }

                shark::span<u64> r_selected_poly((degree + 1) * X.size());
                randomize(r_selected_poly);
                send_authenticated_ashare(r_selected_poly);

                shark::span<u64> S(X.size() * (degree + 1));
                shark::span<u64> T(X.size() * (degree + 1));
                randomize(Y);

                for (u64 i = 0; i < X.size(); ++i)
                {
                    for (u64 k = 0; k < degree + 1; ++k)
                    {
                        S[i * (degree + 1) + k] = pow(-X[i], k);
                    }
                }

                for (u64 i = 0; i < X.size(); ++i)
                {
                    for (u64 k = 0; k < degree + 1; ++k)
                    {
                        u64 coeff = 0;
                        for (u64 j = k; j < degree + 1; ++j)
                        {
                            coeff = coeff - nCr(j, k) * r_selected_poly[i * (degree + 1) + j] * pow(-X[i], j - k);
                        }
                        T[i * (degree + 1) + k] = coeff;
                    }

                    T[i * (degree + 1)] = T[i * (degree + 1)] + Y[i];
                }

                send_authenticated_ashare(S);
                send_authenticated_ashare(T);

            }

            void eval(int bin, int degree, const std::vector<u64> &knots, const std::vector<u64> &polynomials, const shark::span<u64> &X, shark::span<u64> &Y)
            {
                u64 m = knots.size();
                u64 p = polynomials.size();
                always_assert((m + 1) * (degree + 1) == p);
                always_assert(X.size() == Y.size());
                always_assert(bin <= 64);

                // check if knots are sorted
                for (u64 i = 1; i < m; i++)
                {
                    always_assert(knots[i - 1] < knots[i]);
                }
                if (bin != 64 && m > 0)
                {
                    always_assert(knots[m - 1] < (1ull << bin));
                }

                const u64 mask = mask_bits(bin);
                const u128 mask_u128 = mask_bits_u128(bin);
                const u128 one_share = (party == CLIENT) ? u128(1) : u128(0);
                // MAC share for public constant 1 must use each party's MAC key share.
                const u128 one_tag = mac_mul_u128(u128(1));

                auto [r_share, r_tag] = recv_authenticated_ashare_full(X.size());
                auto dcfkey_r = recv_dcfring(X.size(), bin);

                std::vector<shark::span<crypto::DCFRingKey>> dcfkeys_knots(m);
                std::vector<shark::span<u8>> t3_knots(m);
                for (u64 j = 0; j < m; ++j)
                {
                    auto key = recv_dcfring(X.size(), bin);
                    auto t3 = dealer->recv_array<u8>(X.size());
                    dcfkeys_knots[j] = std::move(key);
                    t3_knots[j] = std::move(t3);
                }

                auto [r_selected_poly, r_selected_poly_tag] = recv_authenticated_ashare((degree + 1) * X.size());
                auto [S_share, S_tag] = recv_authenticated_ashare(X.size() * (degree + 1));
                auto [T_share, T_tag] = recv_authenticated_ashare(X.size() * (degree + 1));

                // open masked input c = X + r
                shark::span<u128> c_share(X.size());
                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); ++i)
                {
                    c_share[i] = u128(X[i]) + r_share[i];
                    if (bin < 64)
                    {
                        c_share[i] &= mask_u128;
                    }
                }
                shark::span<u128> c(X.size());
                open_plain_full(c_share, c);
                if (bin < 64)
                {
                    #pragma omp parallel for
                    for (u64 i = 0; i < X.size(); ++i)
                    {
                        c[i] &= mask_u128;
                    }
                }

                // t1 = [c < r] as arithmetic shares
                shark::span<u128> t1_share(X.size());
                shark::span<u128> t1_tag(X.size());
                shark::span<u64> t1_low(X.size());
                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); ++i)
                {
                    auto t = dcfring_eval(party, dcfkey_r[i], c[i]);
                    t1_share[i] = std::get<0>(t);
                    t1_tag[i] = std::get<1>(t);
                    t1_low[i] = getLow(t1_share[i]);
                }

                shark::span<u128> selected_poly_share((degree + 1) * X.size());
                shark::span<u128> selected_poly_tag((degree + 1) * X.size());
                for (u64 i = 0; i < (degree + 1) * X.size(); ++i)
                {
                    selected_poly_share[i] = r_selected_poly[i];
                    selected_poly_tag[i] = r_selected_poly_tag[i];
                }

                shark::span<u128> ge_prev_share(X.size());
                shark::span<u128> ge_prev_tag(X.size());

                for (u64 j = 0; j < m; ++j)
                {
                    auto &dcfkey = dcfkeys_knots[j];
                    auto &t3 = t3_knots[j];

                    shark::span<u128> t2_share(X.size());
                    shark::span<u128> t2_tag(X.size());
                    shark::span<u64> t2_low(X.size());
                    #pragma omp parallel for
                    for (u64 i = 0; i < X.size(); ++i)
                    {
                        auto t = dcfring_eval(party, dcfkey[i], c[i]);
                        t2_share[i] = std::get<0>(t);
                        t2_tag[i] = std::get<1>(t);
                        t2_low[i] = getLow(t2_share[i]);
                    }

                    shark::span<u64> prod_share(X.size());
                    shark::span<u128> prod_tag(X.size());
                    mul::call_share(t1_low, t2_low, prod_share, prod_tag);

                    shark::span<u128> lt_share(X.size());
                    shark::span<u128> lt_tag(X.size());
                    #pragma omp parallel for
                    for (u64 i = 0; i < X.size(); ++i)
                    {
                        if (t3[i] == 0)
                        {
                            lt_share[i] = t1_share[i] + t2_share[i] - (prod_share[i] * 2);
                            lt_tag[i] = t1_tag[i] + t2_tag[i] - (prod_tag[i] * 2);
                        }
                        else
                        {
                            lt_share[i] = one_share - t1_share[i] - t2_share[i] + (prod_share[i] * 2);
                            lt_tag[i] = one_tag - t1_tag[i] - t2_tag[i] + (prod_tag[i] * 2);
                        }
                    }

                    shark::span<u128> ge_share(X.size());
                    shark::span<u128> ge_tag(X.size());
                    #pragma omp parallel for
                    for (u64 i = 0; i < X.size(); ++i)
                    {
                        ge_share[i] = one_share - lt_share[i];
                        ge_tag[i] = one_tag - lt_tag[i];
                    }

                    shark::span<u128> seg_share(X.size());
                    shark::span<u128> seg_tag(X.size());
                    if (j == 0)
                    {
                        #pragma omp parallel for
                        for (u64 i = 0; i < X.size(); ++i)
                        {
                            seg_share[i] = lt_share[i];
                            seg_tag[i] = lt_tag[i];
                        }
                    }
                    else
                    {
                        #pragma omp parallel for
                        for (u64 i = 0; i < X.size(); ++i)
                        {
                            seg_share[i] = ge_prev_share[i] - ge_share[i];
                            seg_tag[i] = ge_prev_tag[i] - ge_tag[i];
                        }
                    }

                    #pragma omp parallel for
                    for (u64 i = 0; i < X.size(); ++i)
                    {
                        for (u64 k = 0; k < degree + 1; ++k)
                        {
                            selected_poly_share[i * (degree + 1) + k] += seg_share[i] * polynomials[j * (degree + 1) + k];
                            selected_poly_tag[i * (degree + 1) + k] += seg_tag[i] * polynomials[j * (degree + 1) + k];
                        }
                    }

                    ge_prev_share = std::move(ge_share);
                    ge_prev_tag = std::move(ge_tag);
                }

                // last segment: X >= last knot
                if (m > 0)
                {
                    #pragma omp parallel for
                    for (u64 i = 0; i < X.size(); ++i)
                    {
                        for (u64 k = 0; k < degree + 1; ++k)
                        {
                            selected_poly_share[i * (degree + 1) + k] += ge_prev_share[i] * polynomials[m * (degree + 1) + k];
                            selected_poly_tag[i * (degree + 1) + k] += ge_prev_tag[i] * polynomials[m * (degree + 1) + k];
                        }
                    }
                }

                auto selected_poly = authenticated_reconstruct(selected_poly_share, selected_poly_tag);

                shark::span<u128> Y_share(X.size());
                shark::span<u128> Y_tag(X.size());
                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); i++)
                {
                    u128 y_share = 0;
                    u128 y_tag = 0;

                    for (u64 k = 0; k < degree + 1; ++k)
                    {
                        y_share += T_share[i * (degree + 1) + k] * pow(X[i], k);
                        y_tag += T_tag[i * (degree + 1) + k] * pow(X[i], k);

                        for (u64 j = 0; j < k + 1; j++)
                        {
                            y_share += nCr(k, j) * S_share[i * (degree + 1) + k - j] * selected_poly[i * (degree + 1) + k] * pow(X[i], j);
                            y_tag += nCr(k, j) * S_tag[i * (degree + 1) + k - j] * selected_poly[i * (degree + 1) + k] * pow(X[i], j);
                        }
                    }

                    Y_share[i] = y_share;
                    Y_tag[i] = y_tag;

                }

                Y = authenticated_reconstruct(Y_share, Y_tag);

            }

            void call(int bin, int degree, const std::vector<u64> &knots, const std::vector<u64> &polynomials, const shark::span<u64> &X, shark::span<u64> &Y)
            {
                if (party == DEALER)
                {
                    gen(bin, degree, knots, X, Y);
                }
                else
                {
                    eval(bin, degree, knots, polynomials, X, Y);
                }
            }

            shark::span<u64> call(int bin, int degree, const std::vector<u64> &knots, const std::vector<u64> &polynomials, const shark::span<u64> &X)
            {
                shark::span<u64> Y(X.size());
                call(bin, degree, knots, polynomials, X, Y);
                return Y;
            }
        }
    }
}
