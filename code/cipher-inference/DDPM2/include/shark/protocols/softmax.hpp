#pragma once
#include <shark/types/u64.hpp>
#include <shark/types/span.hpp>

namespace shark
{
    namespace protocols
    {
        namespace softmax
        {
            // Compute row-wise softmax over a flattened matrix with s1 rows and s2 columns.
            // X has length s1 * s2 and is fixed-point with f fractional bits.
            // Y will have the same length. Output fractional bits equal to f_out (default: f).
            void call(u64 s1, u64 s2, int f, const shark::span<u64> &X, shark::span<u64> &Y, int f_out = -1);
            shark::span<u64> call(u64 s1, u64 s2, int f, const shark::span<u64> &X, int f_out = -1);
        }
    }
}