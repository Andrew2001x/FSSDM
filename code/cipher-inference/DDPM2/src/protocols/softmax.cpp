#include <shark/protocols/softmax.hpp>
#include <shark/protocols/relu.hpp>
#include <shark/protocols/common.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/lrs.hpp>
#include <shark/protocols/ars.hpp>
#include <shark/protocols/reciprocal.hpp>
#include <shark/protocols/sigmoid.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>

namespace shark {
    namespace protocols {
        namespace softmax {

            static void _max_pair(const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &res) {
                always_assert(X.size() == Y.size());
                always_assert(X.size() == res.size());

                shark::span<u64> tmp(X.size());

                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); ++i) {
                    tmp[i] = X[i] - Y[i];
                }

                // max(X, Y) = Y + ReLU(X - Y)
                auto reluout = relu::call(tmp);

                #pragma omp parallel for
                for (u64 i = 0; i < X.size(); ++i) {
                    res[i] = reluout[i] + Y[i];
                }
            }

            static void _rowmax(u64 s1, u64 s2, const shark::span<u64> &X, shark::span<u64> &Z) {
                always_assert(X.size() == s1 * s2);
                always_assert(Z.size() == s1);

                shark::span<u64> res(s1 * s2);

                #pragma omp parallel for collapse(2)
                for (u64 i = 0; i < s1; ++i) {
                    for (u64 j = 0; j < s2; ++j) {
                        res[i * s2 + j] = X[i * s2 + j];
                    }
                }

                u64 curr = s2;
                while (curr != 1) {
                    u64 curr2 = curr / 2;

                    shark::span<u64> left(s1 * curr2);
                    shark::span<u64> right(s1 * curr2);

                    #pragma omp parallel for collapse(2)
                    for (u64 i = 0; i < s1; ++i) {
                        for (u64 j = 0; j < curr2; ++j) {
                            left[i * curr2 + j] = res[i * curr + 2 * j];
                            right[i * curr2 + j] = res[i * curr + 2 * j + 1];
                        }
                    }

                    _max_pair(left, right, left);

                    u64 currNext = (curr % 2 == 0) ? (curr / 2) : (curr / 2 + 1);
                    shark::span<u64> resNext(s1 * currNext);

                    if (curr % 2 == 1) {
                        #pragma omp parallel for
                        for (u64 i = 0; i < s1; ++i) {
                            resNext[i * currNext + currNext - 1] = res[i * curr + curr - 1];
                        }
                    }

                    #pragma omp parallel for collapse(2)
                    for (u64 i = 0; i < s1; ++i) {
                        for (u64 j = 0; j < curr2; ++j) {
                            resNext[i * currNext + j] = left[i * curr2 + j];
                        }
                    }

                    res = resNext;
                    curr = currNext;
                }

                #pragma omp parallel for
                for (u64 i = 0; i < s1; ++i) {
                    Z[i] = res[i];
                }
            }

            void call(u64 s1, u64 s2, int f, const shark::span<u64> &X, shark::span<u64> &Y, int f_out) {
                always_assert(X.size() == s1 * s2);
                always_assert(Y.size() == s1 * s2);
                always_assert(f >= 12);

                if (f_out < 0) f_out = f;

                // 1) Row-wise max for numerical stability
                shark::utils::start_timer("softmax_rowmax");
                shark::span<u64> M(s1);
                _rowmax(s1, s2, X, M);
                shark::utils::stop_timer("softmax_rowmax");
                debug_batch_check("softmax:after rowmax");

                // 2) Shift inputs by max: D = X - broadcast(M)
                shark::utils::start_timer("softmax_shift");
                shark::span<u64> D(s1 * s2);
                #pragma omp parallel for collapse(2)
                for (u64 i = 0; i < s1; ++i) {
                    for (u64 j = 0; j < s2; ++j) {
                        D[i * s2 + j] = X[i * s2 + j] - M[i];
                    }
                }
                shark::utils::stop_timer("softmax_shift");
                debug_batch_check("softmax:after shift");

                // 3) Directly output shifted logits (X - rowmax) as softmax output
                shark::utils::start_timer("softmax_output_shifted");
                if (f_out <= f) {
                    // Adjust scale via right shift if needed
                    if (f_out < f) {
                        auto Out = shark::protocols::ars::call(D, f - f_out);
                        #pragma omp parallel for
                        for (u64 k = 0; k < s1 * s2; ++k) {
                            Y[k] = Out[k];
                        }
                    } else {
                        #pragma omp parallel for
                        for (u64 k = 0; k < s1 * s2; ++k) {
                            Y[k] = D[k];
                        }
                    }
                } else {
                    // If higher precision is requested, left-shift to increase fractional bits
                    u64 shift = static_cast<u64>(f_out - f);
                    #pragma omp parallel for
                    for (u64 k = 0; k < s1 * s2; ++k) {
                        Y[k] = D[k] << shift;
                    }
                }
                shark::utils::stop_timer("softmax_output_shifted");
            }

            shark::span<u64> call(u64 s1, u64 s2, int f, const shark::span<u64> &X, int f_out) {
                shark::span<u64> Y(s1 * s2);
                call(s1, s2, f, X, Y, f_out);
                return Y;
            }
        }
    }
}
