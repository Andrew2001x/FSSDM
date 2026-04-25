#pragma once

#include <shark/types/u64.hpp>
#include <shark/types/u128.hpp>
#include <shark/types/span.hpp>

namespace shark {
    namespace protocols {
        namespace mul {
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

            void gen(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z);
            void eval(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z);
            void eval_share(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z_share, shark::span<u128> &Z_tag);
            void call(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z);
            void call_share(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z_share, shark::span<u128> &Z_tag);
            void call_share_secret_full(const shark::span<u128> &X_share, const shark::span<u128> &X_tag,
                                        const shark::span<u128> &Y_share, const shark::span<u128> &Y_tag,
                                        shark::span<u128> &Z_share, shark::span<u128> &Z_tag);
            shark::span<u64> call(const shark::span<u64> &X, const shark::span<u64> &Y);
            AuthShare call_share(const shark::span<u64> &X, const shark::span<u64> &Y);
            AuthShareFull call_share_secret_full(const shark::span<u128> &X_share, const shark::span<u128> &X_tag,
                                                 const shark::span<u128> &Y_share, const shark::span<u128> &Y_tag);
        }
    }
}
