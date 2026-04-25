#pragma once

#include <shark/types/u64.hpp>
#include <shark/types/u128.hpp>
#include <shark/types/span.hpp>

namespace shark {
    namespace protocols {
        namespace conv {
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

            void  gen(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter, shark::span<u64> &Z);
            void eval(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter, shark::span<u64> &Z);
            void eval_share(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter, shark::span<u64> &Z_share, shark::span<u128> &Z_tag);
            void call(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter, shark::span<u64> &Z);
            void call_share(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter, shark::span<u64> &Z_share, shark::span<u128> &Z_tag);
            void call_share_secret_full(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW,
                                        const shark::span<u128> &Img_share, const shark::span<u128> &Img_tag,
                                        const shark::span<u128> &Filter_share, const shark::span<u128> &Filter_tag,
                                        shark::span<u128> &Z_share, shark::span<u128> &Z_tag);
            shark::span<u64> call(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter);
            AuthShare call_share(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter);
            AuthShareFull call_share_secret_full(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW,
                                                 const shark::span<u128> &Img_share, const shark::span<u128> &Img_tag,
                                                 const shark::span<u128> &Filter_share, const shark::span<u128> &Filter_tag);
            shark::span<u64> emul(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter);

        }
    }
}
