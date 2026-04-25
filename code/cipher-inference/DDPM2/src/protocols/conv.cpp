#include <shark/types/u128.hpp>
#include <shark/protocols/conv.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>
#include <shark/utils/eigen.hpp>
#include <cstdlib>
#include <iostream>

using namespace shark::matrix;

namespace shark {
    namespace protocols {
        namespace conv {
            template <typename T>
            shark::span<T> getReshapedImage(u64 F, u64 padding, u64 stride, u64 CI, u64 inH, u64 inW, const shark::span<T> &Img);

            template <typename T>
            shark::span<T> getReshapedOutput(u64 BS, u64 H, u64 W, u64 C, const shark::span<T> &Z);

            namespace {
                static void gen_secret_full(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, u64 bs)
                {
                    u64 outH = (inH - f + 2 * padding) / stride + 1;
                    u64 outW = (inW - f + 2 * padding) / stride + 1;
                    shark::span<u128> r_Img(bs * inH * inW * ci);
                    shark::span<u128> r_Filter(co * f * f * ci);
                    shark::span<u128> r_Z(bs * outH * outW * co);
                    randomize_full(r_Img);
                    randomize_full(r_Filter);
                    randomize_full(r_Z);

                    auto mat_r_Filter = getMat(co, f * f * ci, r_Filter);
                    auto reshaped_r_Img = getReshapedImage(f, padding, stride, ci, inH, inW, r_Img);
                    auto mat_r_Img = getMat(f * f * ci, bs * outH * outW, reshaped_r_Img);
                    auto reshaped_r_Z = getReshapedOutput(bs, outH, outW, co, r_Z);
                    auto mat_r_Z = getMat(co, bs * outH * outW, reshaped_r_Z);

                    shark::span<u128> r_C(co * bs * outH * outW);
                    auto mat_r_C = getMat(co, bs * outH * outW, r_C);
                    mat_r_C = mat_r_Filter * mat_r_Img + mat_r_Z;

                    send_authenticated_ashare_full(r_Img);
                    send_authenticated_ashare_full(r_Filter);
                    send_authenticated_ashare_full(r_C);
                }
            }

            template <typename T>
            shark::span<T> getReshapedImage(u64 F, u64 padding, u64 stride, u64 CI, u64 inH, u64 inW, const shark::span<T> &Img)
            {
                // shape of Img is [bs, inH, inW, ci]
                always_assert(Img.size() % (inH * inW * CI) == 0);
                u64 BS = Img.size() / (inH * inW * CI);
                u64 outH = (inH - F + 2 * padding) / stride + 1;
                u64 outW = (inW - F + 2 * padding) / stride + 1;
                u64 reshapedImgCols = BS * outH * outW;
                u64 reshapedImgRows = F * F * CI;
                shark::span<T> reshapedImg(reshapedImgRows * reshapedImgCols);

                using i64 = int64_t;
                u64 linIdxFilterMult = 0;
                for (i64 n = 0; n < BS; n++){
		            i64 leftTopCornerH = 0 - padding;
		            i64 extremeRightBottomCornerH = inH - 1 + padding;
		            while((leftTopCornerH + F - 1) <= extremeRightBottomCornerH)
                    { 
			            i64 leftTopCornerW = 0 - padding;
			            i64 extremeRightBottomCornerW = inW - 1 + padding;
			            while((leftTopCornerW + F - 1) <= extremeRightBottomCornerW)
                        {

				            for (i64 fh = 0; fh < F; fh++)
                            {
					            for (i64 fw = 0; fw < F; fw++)
                                {
						            i64 curPosH = leftTopCornerH + fh;
						            i64 curPosW = leftTopCornerW + fw;
						            for (i64 _ci = 0; _ci < CI; _ci++)
                                    {
                                        u64 rowIdx = (fh*F*CI) + (fw*CI) + _ci;
							            if ((((curPosH < 0) || (curPosH >= inH)) || ((curPosW < 0) || (curPosW >= inW))))
                                        {
								            // reshapedImg(linIdxFilterMult, rowIdx) = 0L;
                                            reshapedImg[rowIdx * reshapedImgCols + linIdxFilterMult] = 0L;
							            }
							            else
                                        {
								            // reshapedImg(linIdxFilterMult, rowIdx) = input(n, curPosH, curPosW, _ci);
                                            reshapedImg[rowIdx * reshapedImgCols + linIdxFilterMult] = Img[n * inH * inW * CI + curPosH * inW * CI + curPosW * CI + _ci];

							            }
						            }
					            }
				            }

				            linIdxFilterMult = linIdxFilterMult + 1;
				            leftTopCornerW = leftTopCornerW + stride;
			            }

            			leftTopCornerH = leftTopCornerH + stride;
		            }
            	}

                return reshapedImg;
                // return getMat(reshapedImgRows, reshapedImgCols, reshapedImg);
            }

            template <typename T>
            shark::span<T> getReshapedOutput(u64 BS, u64 H, u64 W, u64 C, const shark::span<T> &Z)
            {
                // shape of Z is [bs, h, w, c]
                // reshape to [c, bs * h * w]
                always_assert(Z.size() == BS * H * W * C);
                shark::span<T> reshapedZ(C * BS * H * W);
                for (u64 i = 0; i < BS; i++)
                {
                    for (u64 j = 0; j < H; j++)
                    {
                        for (u64 k = 0; k < W; k++)
                        {
                            for (u64 l = 0; l < C; l++)
                            {
                                reshapedZ[l * BS * H * W + i * H * W + j * W + k] = Z[i * H * W * C + j * W * C + k * C + l];
                            }
                        }
                    }
                }
                return reshapedZ;
            }

            template <typename T>
            void getReshapedOutputReversed(u64 BS, u64 H, u64 W, u64 C, const shark::span<T> &reshapedZ, shark::span<T> &Z)
            {
                // shape of Z is [bs, h, w, c]
                // reshape to [c, bs * h * w]
                always_assert(reshapedZ.size() == BS * H * W * C);
                always_assert(Z.size() == BS * H * W * C);

                for (u64 i = 0; i < BS; i++)
                {
                    for (u64 j = 0; j < H; j++)
                    {
                        for (u64 k = 0; k < W; k++)
                        {
                            for (u64 l = 0; l < C; l++)
                            {
                                Z[i * H * W * C + j * W * C + k * C + l] = reshapedZ[l * BS * H * W + i * H * W + j * W + k];
                            }
                        }
                    }
                }
            }

            void gen(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &r_Img, const shark::span<u64> &r_Filter, shark::span<u64> &r_Z)
            {
                always_assert(r_Img.size() % (inH * inW * ci) == 0);
                u64 bs = r_Img.size() / (inH * inW * ci);
                always_assert(r_Filter.size() == co * f * f * ci);
                u64 outH = (inH - f + 2 * padding) / stride + 1;
                u64 outW = (inW - f + 2 * padding) / stride + 1;
                always_assert(r_Z.size() == bs * outH * outW * co);

                randomize(r_Z);
                auto mat_r_Filter = getMat(co, f * f * ci, r_Filter);
                auto reshaped_r_Img = getReshapedImage(f, padding, stride, ci, inH, inW, r_Img);
                auto mat_r_Img = getMat(f * f * ci, bs * outH * outW, reshaped_r_Img);
                auto reshaped_r_Z = getReshapedOutput(bs, outH, outW, co, r_Z);
                auto mat_r_Z = getMat(co, bs * outH * outW, reshaped_r_Z);

                shark::span<u64> r_C(co * bs * outH * outW);
                auto mat_r_C = getMat(co, bs * outH * outW, r_C);
                // r_C = r_X @ r_Y + r_Z
                // shark::utils::matmuladd(a, b, c, r_X, r_Y, r_Z, r_C);
                mat_r_C = mat_r_Filter * mat_r_Img + mat_r_Z;

                send_authenticated_ashare(r_Img);
                send_authenticated_ashare(r_Filter);
                send_authenticated_ashare(r_C);
            }

            void eval(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter, shark::span<u64> &Z)
            {
                always_assert(Img.size() % (inH * inW * ci) == 0);
                u64 bs = Img.size() / (inH * inW * ci);
                always_assert(Filter.size() == co * f * f * ci);
                u64 outH = (inH - f + 2 * padding) / stride + 1;
                u64 outW = (inW - f + 2 * padding) / stride + 1;
                always_assert(Z.size() == bs * outH * outW * co);

                shark::utils::start_timer("key_read");
                auto [r_Img, r_Img_tag] = recv_authenticated_ashare(bs * inH * inW * ci);
                auto [r_Filter, r_Filter_tag] = recv_authenticated_ashare(co * f * f * ci);
                auto [r_C, r_C_tag] = recv_authenticated_ashare(co * bs * outH * outW);
                shark::utils::stop_timer("key_read");
                if (std::getenv("SHARK_DEBUG_CONV_MASK_AUTH"))
                {
                    shark::span<u64> tmp_img(r_Img.size());
                    #pragma omp parallel for
                    for (u64 i = 0; i < r_Img.size(); ++i) tmp_img[i] = r_Img[i];
                    (void)authenticated_reconstruct(tmp_img, r_Img_tag);
                    debug_batch_check("conv:mask_rimg");

                    shark::span<u64> tmp_f(r_Filter.size());
                    #pragma omp parallel for
                    for (u64 i = 0; i < r_Filter.size(); ++i) tmp_f[i] = r_Filter[i];
                    (void)authenticated_reconstruct(tmp_f, r_Filter_tag);
                    debug_batch_check("conv:mask_rfilter");

                    shark::span<u64> tmp_c(r_C.size());
                    #pragma omp parallel for
                    for (u64 i = 0; i < r_C.size(); ++i) tmp_c[i] = r_C[i];
                    (void)authenticated_reconstruct(tmp_c, r_C_tag);
                    debug_batch_check("conv:mask_rc");
                }

                // reshapes and casts
                auto reshaped_Img = getReshapedImage(f, padding, stride, ci, inH, inW, Img);
                auto reshaped_r_Img = getReshapedImage(f, padding, stride, ci, inH, inW, r_Img);
                auto reshaped_r_Img_tag = getReshapedImage(f, padding, stride, ci, inH, inW, r_Img_tag);

                auto mat_Img = getMat(f * f * ci, bs * outH * outW, reshaped_Img).cast<u128>();
                auto mat_r_Img = getMat(f * f * ci, bs * outH * outW, reshaped_r_Img);
                auto mat_r_Img_tag = getMat(f * f * ci, bs * outH * outW, reshaped_r_Img_tag);

                shark::span<u64> reshaped_Z(co * bs * outH * outW);
                shark::span<u128> reshaped_Z_raw(co * bs * outH * outW);
                shark::span<u128> reshaped_Z_tag(co * bs * outH * outW);
                auto mat_reshaped_Z_raw = getMat(co, bs * outH * outW, reshaped_Z_raw);
                auto mat_reshaped_Z_tag = getMat(co, bs * outH * outW, reshaped_Z_tag);

                auto mat_Filter = getMat(co, f * f * ci, Filter).cast<u128>();
                auto mat_r_Filter = getMat(co, f * f * ci, r_Filter);
                auto mat_r_Filter_tag = getMat(co, f * f * ci, r_Filter_tag);
                
                auto mat_r_C = getMat(co, bs * outH * outW, r_C).cast<u128>();
                auto mat_r_C_tag = getMat(co, bs * outH * outW, r_C_tag);

                // real computation
                // Z = r_Z + X @ Y - r_X @ Y - X @ r_Y
                mat_reshaped_Z_raw = mat_r_C + (mat_Filter * u128(u64(party)) - mat_r_Filter.cast<u128>()) * mat_Img;
                mat_reshaped_Z_raw -= mat_Filter * mat_r_Img.cast<u128>();

                mat_reshaped_Z_tag = mat_r_C_tag + (mat_Filter * ring_key - mat_r_Filter_tag) * mat_Img;
                mat_reshaped_Z_tag -= mat_Filter * mat_r_Img_tag;
                #pragma omp parallel for
                for (u64 i = 0; i < reshaped_Z.size(); ++i)
                {
                    reshaped_Z[i] = getLow(reshaped_Z_raw[i]);
                    reshaped_Z_tag[i] = mac_sub_u128(reshaped_Z_tag[i], mac_wrap_u64(getHigh(reshaped_Z_raw[i])));
                }

                auto Z_reconstructed = authenticated_reconstruct(reshaped_Z, reshaped_Z_tag);
                getReshapedOutputReversed(bs, outH, outW, co, Z_reconstructed, Z);
            }

            void eval_share(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter, shark::span<u64> &Z_share, shark::span<u128> &Z_tag)
            {
                always_assert(Img.size() % (inH * inW * ci) == 0);
                u64 bs = Img.size() / (inH * inW * ci);
                always_assert(Filter.size() == co * f * f * ci);
                u64 outH = (inH - f + 2 * padding) / stride + 1;
                u64 outW = (inW - f + 2 * padding) / stride + 1;
                always_assert(Z_share.size() == bs * outH * outW * co);
                always_assert(Z_tag.size() == bs * outH * outW * co);

                shark::utils::start_timer("key_read");
                auto [r_Img, r_Img_tag] = recv_authenticated_ashare(bs * inH * inW * ci);
                auto [r_Filter, r_Filter_tag] = recv_authenticated_ashare(co * f * f * ci);
                auto [r_C, r_C_tag] = recv_authenticated_ashare(co * bs * outH * outW);
                shark::utils::stop_timer("key_read");

                shark::span<u64> r_Img_low(r_Img.size());
                shark::span<u64> r_Filter_low(r_Filter.size());
                shark::span<u64> r_C_low(r_C.size());
                #pragma omp parallel for
                for (u64 i = 0; i < r_Img.size(); ++i) r_Img_low[i] = r_Img[i];
                #pragma omp parallel for
                for (u64 i = 0; i < r_Filter.size(); ++i) r_Filter_low[i] = r_Filter[i];
                #pragma omp parallel for
                for (u64 i = 0; i < r_C.size(); ++i) r_C_low[i] = r_C[i];

                auto reshaped_Img = getReshapedImage(f, padding, stride, ci, inH, inW, Img);
                auto reshaped_r_Img = getReshapedImage(f, padding, stride, ci, inH, inW, r_Img_low);
                auto reshaped_r_Img_tag = getReshapedImage(f, padding, stride, ci, inH, inW, r_Img_tag);

                auto mat_Img = getMat(f * f * ci, bs * outH * outW, reshaped_Img);
                auto mat_Img_u128 = getMat(f * f * ci, bs * outH * outW, reshaped_Img).cast<u128>();
                auto mat_r_Img = getMat(f * f * ci, bs * outH * outW, reshaped_r_Img);
                auto mat_r_Img_tag = getMat(f * f * ci, bs * outH * outW, reshaped_r_Img_tag);

                shark::span<u64> reshaped_Z(co * bs * outH * outW);
                shark::span<u128> reshaped_Z_raw(co * bs * outH * outW);
                shark::span<u128> reshaped_Z_tag(co * bs * outH * outW);
                auto mat_reshaped_Z_raw = getMat(co, bs * outH * outW, reshaped_Z_raw);
                auto mat_reshaped_Z_tag = getMat(co, bs * outH * outW, reshaped_Z_tag);

                auto mat_Filter = getMat(co, f * f * ci, Filter);
                auto mat_Filter_u128 = getMat(co, f * f * ci, Filter).cast<u128>();
                auto mat_r_Filter = getMat(co, f * f * ci, r_Filter_low);
                auto mat_r_Filter_tag = getMat(co, f * f * ci, r_Filter_tag);

                auto mat_r_C = getMat(co, bs * outH * outW, r_C_low);
                auto mat_r_C_tag = getMat(co, bs * outH * outW, r_C_tag);

                mat_reshaped_Z_raw = mat_r_C.cast<u128>()
                                   + (mat_Filter_u128 * u128(u64(party)) - mat_r_Filter.cast<u128>()) * mat_Img_u128;
                mat_reshaped_Z_raw -= mat_Filter_u128 * mat_r_Img.cast<u128>();

                mat_reshaped_Z_tag = mat_r_C_tag + (mat_Filter_u128 * ring_key - mat_r_Filter_tag) * mat_Img_u128;
                mat_reshaped_Z_tag -= mat_Filter_u128 * mat_r_Img_tag;
                #pragma omp parallel for
                for (u64 i = 0; i < reshaped_Z.size(); ++i)
                {
                    reshaped_Z[i] = getLow(reshaped_Z_raw[i]);
                    reshaped_Z_tag[i] = mac_sub_u128(reshaped_Z_tag[i], mac_wrap_u64(getHigh(reshaped_Z_raw[i])));
                }

                getReshapedOutputReversed(bs, outH, outW, co, reshaped_Z, Z_share);
                getReshapedOutputReversed(bs, outH, outW, co, reshaped_Z_tag, Z_tag);
            }

            void call(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter, shark::span<u64> &Z)
            {
                if (party == DEALER)
                {
                    gen(f, padding, stride, ci, co, inH, inW, Img, Filter, Z);
                }
                else
                {
                    eval(f, padding, stride, ci, co, inH, inW, Img, Filter, Z);
                }
            }

            void call_share(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter, shark::span<u64> &Z_share, shark::span<u128> &Z_tag)
            {
                if (party == DEALER)
                {
                    shark::span<u64> Z(Z_share.size());
                    gen(f, padding, stride, ci, co, inH, inW, Img, Filter, Z);
                    #pragma omp parallel for
                    for (u64 i = 0; i < Z.size(); i++)
                    {
                        Z_share[i] = Z[i];
                        Z_tag[i] = mac_mul_u64(Z[i]);
                    }
                }
                else
                {
                    eval_share(f, padding, stride, ci, co, inH, inW, Img, Filter, Z_share, Z_tag);
                }
            }

            void call_share_secret_full(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW,
                                        const shark::span<u128> &Img_share, const shark::span<u128> &Img_tag,
                                        const shark::span<u128> &Filter_share, const shark::span<u128> &Filter_tag,
                                        shark::span<u128> &Z_share, shark::span<u128> &Z_tag)
            {
                always_assert(Img_share.size() % (inH * inW * ci) == 0);
                u64 bs = Img_share.size() / (inH * inW * ci);
                always_assert(Img_tag.size() == Img_share.size());
                always_assert(Filter_share.size() == co * f * f * ci);
                always_assert(Filter_tag.size() == co * f * f * ci);
                u64 outH = (inH - f + 2 * padding) / stride + 1;
                u64 outW = (inW - f + 2 * padding) / stride + 1;
                always_assert(Z_share.size() == bs * outH * outW * co);
                always_assert(Z_tag.size() == bs * outH * outW * co);

                if (party == DEALER)
                {
                    gen_secret_full(f, padding, stride, ci, co, inH, inW, bs);
                    auto reshaped_Img = getReshapedImage(f, padding, stride, ci, inH, inW, Img_share);
                    auto mat_Img = getMat(f * f * ci, bs * outH * outW, reshaped_Img);
                    auto mat_Filter = getMat(co, f * f * ci, Filter_share);
                    shark::span<u128> reshaped_Z(co * bs * outH * outW);
                    auto mat_Z = getMat(co, bs * outH * outW, reshaped_Z);
                    mat_Z = mat_Filter * mat_Img;
                    #pragma omp parallel for
                    for (u64 i = 0; i < reshaped_Z.size(); i++)
                    {
                        Z_tag[i] = 0;
                    }
                    getReshapedOutputReversed(bs, outH, outW, co, reshaped_Z, Z_share);
                    #pragma omp parallel for
                    for (u64 i = 0; i < Z_share.size(); i++)
                    {
                        Z_tag[i] = mac_mul_u128(Z_share[i]);
                    }
                    return;
                }

                shark::utils::start_timer("key_read");
                auto [r_Img, r_Img_tag] = recv_authenticated_ashare_full(bs * inH * inW * ci);
                auto [r_Filter, r_Filter_tag] = recv_authenticated_ashare_full(co * f * f * ci);
                auto [r_C, r_C_tag] = recv_authenticated_ashare_full(co * bs * outH * outW);
                shark::utils::stop_timer("key_read");

                auto reshaped_Img = getReshapedImage(f, padding, stride, ci, inH, inW, Img_share);
                auto reshaped_Img_tag = getReshapedImage(f, padding, stride, ci, inH, inW, Img_tag);
                auto reshaped_r_Img = getReshapedImage(f, padding, stride, ci, inH, inW, r_Img);
                auto reshaped_r_Img_tag = getReshapedImage(f, padding, stride, ci, inH, inW, r_Img_tag);

                shark::span<u128> d_share(reshaped_Img.size());
                shark::span<u128> d_tag(reshaped_Img.size());
                #pragma omp parallel for
                for (u64 i = 0; i < reshaped_Img.size(); i++)
                {
                    d_share[i] = reshaped_Img[i] - reshaped_r_Img[i];
                    d_tag[i] = reshaped_Img_tag[i] - reshaped_r_Img_tag[i];
                }
                auto D = authenticated_reconstruct_full(d_share, d_tag);

                shark::span<u128> e_share(Filter_share.size());
                shark::span<u128> e_tag(Filter_share.size());
                #pragma omp parallel for
                for (u64 i = 0; i < Filter_share.size(); i++)
                {
                    e_share[i] = Filter_share[i] - r_Filter[i];
                    e_tag[i] = Filter_tag[i] - r_Filter_tag[i];
                }
                auto E = authenticated_reconstruct_full(e_share, e_tag);

                auto mat_D = getMat(f * f * ci, bs * outH * outW, D);
                auto mat_E = getMat(co, f * f * ci, E);
                auto mat_A = getMat(f * f * ci, bs * outH * outW, reshaped_r_Img);
                auto mat_A_tag = getMat(f * f * ci, bs * outH * outW, reshaped_r_Img_tag);
                auto mat_B = getMat(co, f * f * ci, r_Filter);
                auto mat_B_tag = getMat(co, f * f * ci, r_Filter_tag);
                auto mat_C = getMat(co, bs * outH * outW, r_C);
                auto mat_C_tag = getMat(co, bs * outH * outW, r_C_tag);

                shark::span<u128> reshaped_Z(co * bs * outH * outW);
                shark::span<u128> reshaped_Z_tag(co * bs * outH * outW);
                auto mat_Z = getMat(co, bs * outH * outW, reshaped_Z);
                auto mat_Z_tag = getMat(co, bs * outH * outW, reshaped_Z_tag);

                auto mat_DE = mat_E * mat_D;
                mat_Z = mat_C + mat_B * mat_D + mat_E * mat_A + mat_DE * u128(party);
                mat_Z_tag = mat_C_tag + mat_B_tag * mat_D + mat_E * mat_A_tag + mat_DE * ring_key;

                getReshapedOutputReversed(bs, outH, outW, co, reshaped_Z, Z_share);
                getReshapedOutputReversed(bs, outH, outW, co, reshaped_Z_tag, Z_tag);
            }
 
            shark::span<u64> call(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter)
            {
                always_assert(Img.size() % (inH * inW * ci) == 0);
                u64 bs = Img.size() / (inH * inW * ci);
                always_assert(Filter.size() == f * f * ci * co);
                u64 outH = (inH - f + 2 * padding) / stride + 1;
                u64 outW = (inW - f + 2 * padding) / stride + 1;

                shark::span<u64> Z(bs * outH * outW * co);
                call(f, padding, stride, ci, co, inH, inW, Img, Filter, Z);
                return Z;
            }

            AuthShare call_share(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter)
            {
                always_assert(Img.size() % (inH * inW * ci) == 0);
                u64 bs = Img.size() / (inH * inW * ci);
                u64 outH = (inH - f + 2 * padding) / stride + 1;
                u64 outW = (inW - f + 2 * padding) / stride + 1;
                AuthShare out{
                    shark::span<u64>(bs * outH * outW * co),
                    shark::span<u128>(bs * outH * outW * co)
                };
                call_share(f, padding, stride, ci, co, inH, inW, Img, Filter, out.share, out.tag);
                return out;
            }

            AuthShareFull call_share_secret_full(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW,
                                                 const shark::span<u128> &Img_share, const shark::span<u128> &Img_tag,
                                                 const shark::span<u128> &Filter_share, const shark::span<u128> &Filter_tag)
            {
                always_assert(Img_share.size() % (inH * inW * ci) == 0);
                u64 bs = Img_share.size() / (inH * inW * ci);
                u64 outH = (inH - f + 2 * padding) / stride + 1;
                u64 outW = (inW - f + 2 * padding) / stride + 1;
                AuthShareFull out{
                    shark::span<u128>(bs * outH * outW * co),
                    shark::span<u128>(bs * outH * outW * co)
                };
                call_share_secret_full(f, padding, stride, ci, co, inH, inW, Img_share, Img_tag, Filter_share, Filter_tag, out.share, out.tag);
                return out;
            }

            shark::span<u64> emul(u64 f, u64 padding, u64 stride, u64 ci, u64 co, u64 inH, u64 inW, const shark::span<u64> &Img, const shark::span<u64> &Filter)
            {
                always_assert(Img.size() % (inH * inW * ci) == 0);
                u64 bs = Img.size() / (inH * inW * ci);
                always_assert(Filter.size() == f * f * ci * co);
                u64 outH = (inH - f + 2 * padding) / stride + 1;
                u64 outW = (inW - f + 2 * padding) / stride + 1;

                shark::span<u64> Z(bs * outH * outW * co);
                shark::span<u64> reshaped_Z(co * bs * outH * outW);
                auto reshaped_Img = getReshapedImage(f, padding, stride, ci, inH, inW, Img);
                auto mat_Img = getMat(f * f * ci, bs * outH * outW, reshaped_Img);
                auto mat_Filter = getMat(co, f * f * ci, Filter);
                auto mat_reshaped_Z = getMat(co, bs * outH * outW, reshaped_Z);
                mat_reshaped_Z = mat_Filter * mat_Img;
                getReshapedOutputReversed(bs, outH, outW, co, reshaped_Z, Z);
                return Z;
            }
        }
    }
}
