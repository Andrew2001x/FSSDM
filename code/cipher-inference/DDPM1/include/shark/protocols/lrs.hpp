#pragma once

#include <shark/types/u64.hpp>
#include <shark/types/u128.hpp>
#include <shark/types/span.hpp>

namespace shark
{
    namespace protocols
    {
        namespace lrs
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

            void gen(const shark::span<u64> &X, shark::span<u64> &Y, int f);
            void eval(const shark::span<u64> &X, shark::span<u64> &Y, int f);
            void eval_share(const shark::span<u64> &X, shark::span<u64> &Y_share, shark::span<u128> &Y_tag, int f);
            void call(const shark::span<u64> &X, shark::span<u64> &Y, int f);
            void call_share(const shark::span<u64> &X, shark::span<u64> &Y_share, shark::span<u128> &Y_tag, int f);
            void call_share_secret_full(const shark::span<u128> &X_share, const shark::span<u128> &X_tag,
                                        shark::span<u128> &Y_share, shark::span<u128> &Y_tag, int f);
            shark::span<u64> call(const shark::span<u64> &X, int f);
            AuthShare call_share(const shark::span<u64> &X, int f);
            AuthShareFull call_share_secret_full(const shark::span<u128> &X_share, const shark::span<u128> &X_tag, int f);
        }
    }
}
