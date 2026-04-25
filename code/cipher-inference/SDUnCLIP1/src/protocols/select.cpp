#include <shark/protocols/select.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>

namespace shark
{
    namespace protocols
    {
        namespace select
        {
            void gen(const shark::span<u8> &r_s, const shark::span<u64> &r_X, shark::span<u64> &r_Y)
            {
                always_assert(r_X.size() == r_s.size());
                always_assert(r_Y.size() == r_s.size());
                randomize(r_Y);

                shark::span<u64> u(r_X.size());
                shark::span<u64> v(r_X.size());
                shark::span<u64> w(r_X.size());
                shark::span<u64> z(r_X.size());

                for (u64 i = 0; i < r_X.size(); ++i)
                {
                    u[i] = r_s[i];
                    v[i] = r_X[i];
                    w[i] = u[i] * r_X[i] + r_Y[i];
                    z[i] = u[i] * r_X[i] * 2;
                }

                send_authenticated_ashare(u);
                send_authenticated_ashare(v);
                send_authenticated_ashare(w);
                send_authenticated_ashare(z);
            }

            void eval(const shark::span<u8> &s, const shark::span<u64> &X, shark::span<u64> &Y)
            {
                always_assert(X.size() == s.size());
                always_assert(Y.size() == s.size());
                shark::span<u64> Y_share(Y.size());
                shark::span<u128> Y_tag(Y.size());

                shark::utils::start_timer("key_read");
                auto [u, u_tag] = recv_authenticated_ashare(X.size());
                auto [v, v_tag] = recv_authenticated_ashare(X.size());
                auto [w, w_tag] = recv_authenticated_ashare(X.size());
                auto [z, z_tag] = recv_authenticated_ashare(X.size());
                shark::utils::stop_timer("key_read");

                // #pragma omp parallel for
                for (u64 i = 0; i < X.size(); i++)
                {
                    u128 raw = 0;
                    if (s[i] == 0)
                    {
                        raw = u128(u[i]) * u128(X[i]) + u128(w[i]) - u128(z[i]);
                        Y_share[i] = getLow(raw);
                        Y_tag[i] = mac_sub_u128(u_tag[i] * u128(X[i]) + w_tag[i] - z_tag[i],
                                                mac_wrap_u64(getHigh(raw)));
                    }
                    else
                    {
                        raw = u128(w[i]) + u128(X[i]) * u128(u64(party)) - u128(u[i]) * u128(X[i]) - u128(v[i]);
                        Y_share[i] = getLow(raw);
                        u128 t0 = mac_mul_u64(X[i]);
                        u128 t1 = mac_mul_tag(u_tag[i], X[i]);
                        u128 tmp = mac_sub(t0, t1);
                        tmp = mac_sub(tmp, v_tag[i]);
                        Y_tag[i] = mac_sub_u128(mac_add(tmp, w_tag[i]), mac_wrap_u64(getHigh(raw)));
                    }
                }

                Y = authenticated_reconstruct(Y_share, Y_tag);
                
            }

            void call(const shark::span<u8> &s, const shark::span<u64> &X, shark::span<u64> &Y)
            {
                if (party == DEALER)
                {
                    gen(s, X, Y);
                }
                else
                {
                    eval(s, X, Y);
                }
            }

            shark::span<u64> call(const shark::span<u8> &s, const shark::span<u64> &X)
            {
                shark::span<u64> Y(X.size());
                call(s, X, Y);
                return Y;
            }
        }

    }
}
