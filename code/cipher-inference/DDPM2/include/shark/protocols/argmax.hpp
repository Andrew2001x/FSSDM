#pragma once

#include <shark/protocols/common.hpp>
#include <shark/protocols/drelu.hpp>
#include <shark/protocols/select.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/add.hpp>
#include <shark/protocols/lrs_exact.hpp>
#include <shark/utils/assert.hpp>
#include <vector>

namespace shark
{
    namespace protocols
    {
        namespace argmax
        {
            // Secure argmax: returns the index of the maximum element
            // Uses pairwise comparisons with drelu and select
            //
            // For a vector of n elements, we perform a tournament-style comparison:
            // 1. Compare pairs of elements
            // 2. Track both the max value and its index
            // 3. Recursively reduce until we have the global max and its index
            //
            // The index is represented as a one-hot encoding initially,
            // then converted to an integer index at the end

            // Helper: compare two values and return (max_value, is_first_larger)
            // is_first_larger = 1 if X > Y, 0 otherwise
            inline void _compare_pair(
                const shark::span<u64> &X,
                const shark::span<u64> &Y,
                shark::span<u64> &max_val,
                shark::span<u8> &is_X_larger)
            {
                always_assert(X.size() == Y.size());
                always_assert(max_val.size() == X.size());
                always_assert(is_X_larger.size() == X.size());

                // Compute X - Y and check sign
                shark::span<u64> diff(X.size());
                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); ++i)
                {
                    diff[i] = X[i] - Y[i];
                }

                // drelu returns 1 if diff >= 0 (X >= Y), 0 otherwise
                auto ge_result = drelu::call(diff);

                // Copy to output
                for (u64 i = 0; i < X.size(); ++i)
                {
                    is_X_larger[i] = ge_result[i];
                }

                // Select max value: if X >= Y, take X; else take Y
                // max_val = X * is_X_larger + Y * (1 - is_X_larger)
                // Using select: select(cond, val) returns val if cond=1, else 0
                // So: max_val = select(is_X_larger, X-Y) + Y

                shark::span<u64> diff_selected(X.size());
                select::call(is_X_larger, diff, diff_selected);

                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); ++i)
                {
                    max_val[i] = diff_selected[i] + Y[i];
                }
            }

            // Secure argmax for a single row of n elements
            // Returns the index (0 to n-1) of the maximum element
            // Uses tournament-style reduction
            inline void call(
                u64 n,  // number of elements
                const shark::span<u64> &X,  // input values [n]
                shark::span<u64> &index_out  // output index [1]
            )
            {
                always_assert(X.size() == n);
                always_assert(index_out.size() == 1);
                always_assert(n > 0);

                if (n == 1)
                {
                    // Only one element, index is 0
                    index_out[0] = 0;
                    return;
                }

                // Initialize: each element has index = its position
                // We track (value, index) pairs and reduce
                shark::span<u64> current_vals(n);
                shark::span<u64> current_indices(n);

                for (u64 i = 0; i < n; ++i)
                {
                    current_vals[i] = X[i];
                    current_indices[i] = i;  // Plaintext indices (public)
                }

                // Tournament reduction
                u64 curr_size = n;
                while (curr_size > 1)
                {
                    u64 pairs = curr_size / 2;
                    u64 next_size = (curr_size + 1) / 2;  // ceil(curr_size / 2)

                    shark::span<u64> next_vals(next_size);
                    shark::span<u64> next_indices(next_size);

                    // Compare pairs
                    shark::span<u64> left_vals(pairs);
                    shark::span<u64> right_vals(pairs);

                    for (u64 i = 0; i < pairs; ++i)
                    {
                        left_vals[i] = current_vals[2*i];
                        right_vals[i] = current_vals[2*i + 1];
                    }

                    shark::span<u64> max_vals(pairs);
                    shark::span<u8> left_wins(pairs);

                    _compare_pair(left_vals, right_vals, max_vals, left_wins);

                    // Select winner indices
                    // winner_index = left_wins ? left_index : right_index
                    for (u64 i = 0; i < pairs; ++i)
                    {
                        next_vals[i] = max_vals[i];

                        // Index selection: we need to do this securely
                        // left_index = current_indices[2*i]
                        // right_index = current_indices[2*i + 1]
                        // winner = left_wins[i] ? left_index : right_index

                        u64 left_idx = current_indices[2*i];
                        u64 right_idx = current_indices[2*i + 1];
                        u64 idx_diff = left_idx - right_idx;

                        // Securely select: result = right_idx + left_wins * (left_idx - right_idx)
                        shark::span<u64> idx_diff_span(1);
                        shark::span<u8> left_wins_span(1);
                        shark::span<u64> selected(1);

                        idx_diff_span[0] = idx_diff;
                        left_wins_span[0] = left_wins[i];

                        select::call(left_wins_span, idx_diff_span, selected);

                        next_indices[i] = selected[0] + right_idx;
                    }

                    // Handle odd element (if any)
                    if (curr_size % 2 == 1)
                    {
                        next_vals[next_size - 1] = current_vals[curr_size - 1];
                        next_indices[next_size - 1] = current_indices[curr_size - 1];
                    }

                    // Move to next round
                    current_vals = std::move(next_vals);
                    current_indices = std::move(next_indices);
                    curr_size = next_size;
                }

                // Final result
                index_out[0] = current_indices[0];
            }

            inline shark::span<u64> call(u64 n, const shark::span<u64> &X)
            {
                shark::span<u64> index_out(1);
                call(n, X, index_out);
                return index_out;
            }

            // Batch version: argmax for each row in a 2D array
            // X is [rows, cols], returns indices [rows]
            inline void call_batch(
                u64 rows,
                u64 cols,
                const shark::span<u64> &X,
                shark::span<u64> &indices_out
            )
            {
                always_assert(X.size() == rows * cols);
                always_assert(indices_out.size() == rows);

                for (u64 r = 0; r < rows; ++r)
                {
                    shark::span<u64> row_data(cols);
                    for (u64 c = 0; c < cols; ++c)
                    {
                        row_data[c] = X[r * cols + c];
                    }

                    shark::span<u64> row_idx(1);
                    call(cols, row_data, row_idx);

                    indices_out[r] = row_idx[0];
                }
            }

            inline shark::span<u64> call_batch(u64 rows, u64 cols, const shark::span<u64> &X)
            {
                shark::span<u64> indices_out(rows);
                call_batch(rows, cols, X, indices_out);
                return indices_out;
            }
        }
    }
}
