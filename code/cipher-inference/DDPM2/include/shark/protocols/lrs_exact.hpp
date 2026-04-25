#pragma once

#include <shark/types/u64.hpp>
#include <shark/types/u128.hpp>
#include <shark/types/span.hpp>

namespace shark
{
    namespace protocols
    {
        namespace lrs_exact
        {
            struct AuthShare
            {
                shark::span<u64> share;
                shark::span<u128> tag;
            };
            /**
             * Exact truncation protocol using DCF for wrap detection.
             *
             * Unlike the probabilistic lrs protocol, this protocol computes
             * Y = X >> f exactly by detecting and correcting for carry
             * propagation in the lower f bits.
             *
             * The key insight is that for masked truncation:
             *   c = X + r (revealed)
             *   naive: Y = (c >> f) - (r >> f) has error when lower bits wrap
             *   exact: Y = (c >> f) - (r >> f) - t, where t detects the wrap
             *
             * The wrap bit t = 1 iff (c mod 2^f) < (r mod 2^f), which is
             * computed using DCF (Distributed Comparison Function).
             */
            void gen(const shark::span<u64> &X, shark::span<u64> &Y, int f);
            void eval(const shark::span<u64> &X, shark::span<u64> &Y, int f);
            void eval_share(const shark::span<u64> &X, shark::span<u64> &Y_share, shark::span<u128> &Y_tag, int f);
            void call(const shark::span<u64> &X, shark::span<u64> &Y, int f);
            void call_share(const shark::span<u64> &X, shark::span<u64> &Y_share, shark::span<u128> &Y_tag, int f);
            shark::span<u64> call(const shark::span<u64> &X, int f);
            AuthShare call_share(const shark::span<u64> &X, int f);
        }
    }
}
