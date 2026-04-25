#include <shark/protocols/output.hpp>
#include <shark/utils/assert.hpp>

namespace shark
{
    namespace protocols
    {
        namespace output
        {
            void gen(const shark::span<u64> &r_X)
            {
                (void)r_X;
                always_assert(false && "output::gen(u64) is disabled; use an explicit owner-directed reveal path");
            }

            void gen(const shark::span<u8> &r_X)
            {
                (void)r_X;
                always_assert(false && "output::gen(u8) is disabled; use an explicit owner-directed reveal path");
            }

            void eval(shark::span<u64> &X)
            {
                (void)X;
                always_assert(false && "output::eval(u64) is disabled; use an explicit owner-directed reveal path");
            }

            void eval(shark::span<u8> &X)
            {
                (void)X;
                always_assert(false && "output::eval(u8) is disabled; use an explicit owner-directed reveal path");
            }

            void call(shark::span<u64> &X)
            {
                (void)X;
                always_assert(false && "output::call(u64) is disabled; use an explicit owner-directed reveal path");
            }

            void call(shark::span<u8> &X)
            {
                (void)X;
                always_assert(false && "output::call(u8) is disabled; use an explicit owner-directed reveal path");
            }
        }
    }
}
