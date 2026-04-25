#pragma once

#include <shark/types/u64.hpp>
#include <shark/types/u128.hpp>
#include <shark/types/span.hpp>
#include <shark/protocols/lrs.hpp>
#include <shark/utils/assert.hpp>

namespace shark
{
    namespace protocols
    {
        namespace ars
        {
            struct AuthShare
            {
                shark::span<u64> share;
                shark::span<u128> tag;
            };

            struct AuthShareFull
            {
                shark::span<u128> share;
                shark::span<u128> tag;
            };

            inline void call(const shark::span<u64> &X, shark::span<u64> &Y, int f)
            {
                always_assert(f > 0 && f < 64);
                if (party == DEALER)
                {
                    // Dealer should emulate arithmetic shift (not logical).
                    for (int i = 0; i < X.size(); ++i)
                    {
                        Y[i] = (u64)(int64_t(X[i]) >> f);
                    }
                }
                else
                {
                    shark::span<u64> X_temp(X.size());
                    for (int i = 0; i < X.size(); ++i)
                    {
                        X_temp[i] = X[i] + ((party == CLIENT) ? (1ull << 63) : 0);
                    }
                    lrs::call(X_temp, Y, f);
                    if (party == CLIENT)
                    {
                        for (int i = 0; i < X.size(); ++i)
                        {
                            Y[i] -= (1ull << (63 - f));
                        }
                    }
                }
            }

            inline shark::span<u64> call(const shark::span<u64> &X, int f)
            {
                shark::span<u64> Y(X.size());
                call(X, Y, f);
                return Y;
            }

            inline void call_share(const shark::span<u64> &X, shark::span<u64> &Y_share, shark::span<u128> &Y_tag, int f)
            {
                always_assert(f > 0 && f < 64);
                if (party == DEALER)
                {
                    // For dealer, mirror evaluator arithmetic shift:
                    // add bias, use LRS preproc, then subtract offset.
                    shark::span<u64> X_temp(X.size());
                    for (u64 i = 0; i < X.size(); ++i)
                    {
                        X_temp[i] = X[i] + (1ull << 63);
                    }

                    auto auth = lrs::call_share(X_temp, f);
                    always_assert(Y_share.size() == auth.share.size());
                    always_assert(Y_tag.size() == auth.tag.size());
                    #pragma omp parallel for
                    for (u64 i = 0; i < Y_share.size(); ++i)
                    {
                        Y_share[i] = auth.share[i];
                        Y_tag[i] = auth.tag[i];
                    }
                    u64 offset = (f >= 64) ? 0 : (1ull << (63 - f));
                    #pragma omp parallel for
                    for (u64 i = 0; i < X.size(); ++i)
                    {
                        auth_local_sub_public_u64(
                            Y_share[i], Y_tag[i], offset, true,
                            Y_share[i], Y_tag[i]);
                    }
                    return;
                }

                shark::span<u64> X_temp(X.size());
                for (u64 i = 0; i < X.size(); ++i)
                {
                    X_temp[i] = X[i] + ((party == CLIENT) ? (1ull << 63) : 0);
                }

                auto auth = lrs::call_share(X_temp, f);
                always_assert(Y_share.size() == auth.share.size());
                always_assert(Y_tag.size() == auth.tag.size());
                #pragma omp parallel for
                for (u64 i = 0; i < Y_share.size(); ++i)
                {
                    Y_share[i] = auth.share[i];
                    Y_tag[i] = auth.tag[i];
                }

                u64 offset = (f >= 64) ? 0 : (1ull << (63 - f));
                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); ++i)
                {
                    auth_local_sub_public_u64(
                        Y_share[i], Y_tag[i], offset, party == CLIENT,
                        Y_share[i], Y_tag[i]);
                }
            }

            inline void call_share_secret_full(const shark::span<u128> &X_share, const shark::span<u128> &X_tag,
                                               shark::span<u128> &Y_share, shark::span<u128> &Y_tag, int f)
            {
                always_assert(f > 0 && f < 64);
                always_assert(X_share.size() == X_tag.size());
                always_assert(X_share.size() == Y_share.size());
                always_assert(X_share.size() == Y_tag.size());

                const u64 n = X_share.size();
                const u128 bias = (f >= 64) ? u128(0) : (u128(1) << 63);

                shark::span<u128> X_temp_share(n);
                shark::span<u128> X_temp_tag(n);
                #pragma omp parallel for
                for (u64 i = 0; i < n; ++i)
                {
                    auth_local_add_public_u128(
                        X_share[i], X_tag[i], bias,
                        (party == CLIENT) || (party == DEALER),
                        X_temp_share[i], X_temp_tag[i]);
                }

                auto auth = lrs::call_share_secret_full(X_temp_share, X_temp_tag, f);
                always_assert(Y_share.size() == auth.share.size());
                always_assert(Y_tag.size() == auth.tag.size());
                #pragma omp parallel for
                for (u64 i = 0; i < Y_share.size(); ++i)
                {
                    Y_share[i] = auth.share[i];
                    Y_tag[i] = auth.tag[i];
                }

                u128 offset = (f >= 64) ? u128(0) : (u128(1) << (63 - f));
                #pragma omp parallel for
                for (u64 i = 0; i < n; ++i)
                {
                    auth_local_sub_public_u128(
                        Y_share[i], Y_tag[i], offset,
                        (party == CLIENT) || (party == DEALER),
                        Y_share[i], Y_tag[i]);
                }
            }

            inline AuthShare call_share(const shark::span<u64> &X, int f)
            {
                AuthShare out{
                    shark::span<u64>(X.size()),
                    shark::span<u128>(X.size())
                };
                call_share(X, out.share, out.tag, f);
                return out;
            }

            inline AuthShareFull call_share_secret_full(const shark::span<u128> &X_share, const shark::span<u128> &X_tag, int f)
            {
                AuthShareFull out{
                    shark::span<u128>(X_share.size()),
                    shark::span<u128>(X_share.size())
                };
                call_share_secret_full(X_share, X_tag, out.share, out.tag, f);
                return out;
            }

            inline shark::span<u64> emul(const shark::span<u64> &X, int f)
            {
                always_assert(f > 0 && f < 64);
                shark::span<u64> Y(X.size());
                for (int i = 0; i < X.size(); ++i)
                {
                    Y[i] = int64_t(X[i]) >> f;
                }
                return Y;
            }
        }
    }
}
