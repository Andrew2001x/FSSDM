#include <shark/protocols/lrs_exact.hpp>
#include <shark/protocols/lrs.hpp>

namespace shark
{
    namespace protocols
    {
        namespace lrs_exact
        {
            void gen(const shark::span<u64> &X, shark::span<u64> &Y, int f)
            {
                lrs::gen(X, Y, f);
            }

            void eval(const shark::span<u64> &X, shark::span<u64> &Y, int f)
            {
                lrs::eval(X, Y, f);
            }

            void eval_share(const shark::span<u64> &X, shark::span<u64> &Y_share, shark::span<u128> &Y_tag, int f)
            {
                lrs::eval_share(X, Y_share, Y_tag, f);
            }

            void call(const shark::span<u64> &X, shark::span<u64> &Y, int f)
            {
                lrs::call(X, Y, f);
            }

            void call_share(const shark::span<u64> &X, shark::span<u64> &Y_share, shark::span<u128> &Y_tag, int f)
            {
                lrs::call_share(X, Y_share, Y_tag, f);
            }

            shark::span<u64> call(const shark::span<u64> &X, int f)
            {
                return lrs::call(X, f);
            }

            AuthShare call_share(const shark::span<u64> &X, int f)
            {
                auto tmp = lrs::call_share(X, f);
                AuthShare out{
                    std::move(tmp.share),
                    std::move(tmp.tag)
                };
                return out;
            }
        }
    }
}
