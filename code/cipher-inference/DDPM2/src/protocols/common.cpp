#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <cryptoTools/Crypto/RandomOracle.h>
#include <shark/utils/timer.hpp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace shark
{
    namespace protocols
    {
        osuCrypto::PRNG prngGlobal;
        thread_local osuCrypto::PRNG *prngOverride = nullptr;
        int party;

        Peer *server;
        Peer *client;

        Dealer *dealer;
        Peer *peer;

        u128 ring_key;
        u64 bit_key;

        std::vector<u128> batchCheckArithmBuffer;
        std::vector<u64>  batchCheckBoolBuffer;

        bool parallel_reconstruct = true;

        bool mpspdz_32bit_compaison = false;

        static inline bool debug_batchcheck_logs_enabled()
        {
            static int enabled = -1;
            if (enabled < 0)
            {
                enabled = (std::getenv("SHARK_DEBUG_BATCHCHECK") != nullptr) ? 1 : 0;
            }
            return enabled == 1;
        }

        static inline bool debug_keypos_enabled()
        {
            static int enabled = -1;
            if (enabled < 0)
            {
                enabled = (std::getenv("SHARK_DEBUG_KEYPOS") != nullptr) ? 1 : 0;
            }
            return enabled == 1;
        }

        static inline bool debug_break_on_mismatch_enabled()
        {
            static int enabled = -1;
            if (enabled < 0)
            {
                enabled = (std::getenv("SHARK_DEBUG_BREAK_ON_MISMATCH") != nullptr) ? 1 : 0;
            }
            return enabled == 1;
        }

        static inline bool debug_trap_enabled()
        {
            static int enabled = -1;
            if (enabled < 0)
            {
                enabled = (std::getenv("SHARK_DEBUG_TRAP") != nullptr) ? 1 : 0;
            }
            return enabled == 1;
        }

        static inline void debug_print_keypos(const char *label)
        {
            if (!debug_keypos_enabled() || party == DEALER)
            {
                return;
            }
            const char *safe_label = label ? label : "<null>";
            uint64_t dealer_bytes = dealer ? dealer->bytesReceived() : 0;
            uint64_t peer_tx = peer ? peer->bytesSent() : 0;
            uint64_t peer_rx = peer ? peer->bytesReceived() : 0;
            printf("[debug] keypos at %s | dealer_bytes=%llu peer_tx=%llu peer_rx=%llu\n",
                   safe_label,
                   (unsigned long long)dealer_bytes,
                   (unsigned long long)peer_tx,
                   (unsigned long long)peer_rx);
            fflush(stdout);
        }

        template <typename T>
        static inline void append_pod_bytes(std::vector<char> &buf, const T &value)
        {
            const size_t old_size = buf.size();
            buf.resize(old_size + sizeof(T));
            std::memcpy(buf.data() + old_size, &value, sizeof(T));
        }

        template <typename T>
        static inline void append_span_bytes(std::vector<char> &buf, const shark::span<T> &arr)
        {
            const size_t bytes = (size_t)arr.size() * sizeof(T);
            const size_t old_size = buf.size();
            buf.resize(old_size + bytes);
            if (bytes > 0)
            {
                std::memcpy(buf.data() + old_size, arr.data(), bytes);
            }
        }

        static inline void flush_serialized(Peer *dst, std::vector<char> &buf)
        {
            if (buf.empty()) return;
            dst->keyBuf->write(buf.data(), (u64)buf.size());
        }

        template <typename T>
        static inline void write_pod_bytes_at(std::vector<char> &buf, size_t &offset, const T &value)
        {
            std::memcpy(buf.data() + offset, &value, sizeof(T));
            offset += sizeof(T);
        }

        template <typename T>
        static inline void write_span_bytes_at(std::vector<char> &buf, size_t &offset, const shark::span<T> &arr)
        {
            const size_t bytes = (size_t)arr.size() * sizeof(T);
            if (bytes > 0)
            {
                std::memcpy(buf.data() + offset, arr.data(), bytes);
            }
            offset += bytes;
        }

        template <typename F>
        static inline void fill_with_parallel_prng(u64 size, F &&fn)
        {
            if (size == 0)
            {
                return;
            }
#ifdef _OPENMP
            if (size > 4096)
            {
                const int num_threads = omp_get_max_threads();
                std::vector<block> seeds((size_t)num_threads);
                for (int t = 0; t < num_threads; ++t)
                {
                    seeds[(size_t)t] = rand<block>();
                }
                #pragma omp parallel
                {
                    const int tid = omp_get_thread_num();
                    osuCrypto::PRNG local_prng(seeds[(size_t)tid]);
                    auto *prev_prng = prngOverride;
                    prngOverride = &local_prng;
                    #pragma omp for schedule(static)
                    for (long long i = 0; i < (long long)size; ++i)
                    {
                        fn((u64)i);
                    }
                    prngOverride = prev_prng;
                }
                return;
            }
#endif
            for (u64 i = 0; i < size; ++i)
            {
                fn(i);
            }
        }

        template <typename T>
        static void send_dcfbit_impl(const shark::span<T> &share, int bin)
        {
            const u64 size = share.size();
            const size_t per_key_bytes =
                (size_t)(bin + 1) * sizeof(block) +
                (size_t)bin * sizeof(u8) +
                (size_t)bin * sizeof(u64) +
                sizeof(u8) + sizeof(u64);

            std::vector<char> server_buf;
            std::vector<char> client_buf;
            server_buf.resize((size_t)size * per_key_bytes);
            client_buf.resize((size_t)size * per_key_bytes);

            std::vector<block> seeds(size);
            for (u64 i = 0; i < size; ++i) seeds[i] = rand<block>();

            #pragma omp parallel for if(size > 8)
            for (long long i = 0; i < (long long)size; ++i)
            {
                osuCrypto::PRNG local_prng(seeds[(size_t)i]);
                prngOverride = &local_prng;
                auto [k0, k1] = crypto::dcfbit_gen(bin, share[(size_t)i]);
                prngOverride = nullptr;

                size_t server_off = (size_t)i * per_key_bytes;
                size_t client_off = (size_t)i * per_key_bytes;
                write_span_bytes_at(server_buf, server_off, k0.k);
                write_span_bytes_at(server_buf, server_off, k0.v_bit);
                write_span_bytes_at(server_buf, server_off, k0.v_tag_1);
                write_pod_bytes_at(server_buf, server_off, k0.g_bit);
                write_pod_bytes_at(server_buf, server_off, k0.g_tag_1);

                write_span_bytes_at(client_buf, client_off, k1.k);
                write_span_bytes_at(client_buf, client_off, k1.v_bit);
                write_span_bytes_at(client_buf, client_off, k1.v_tag_1);
                write_pod_bytes_at(client_buf, client_off, k1.g_bit);
                write_pod_bytes_at(client_buf, client_off, k1.g_tag_1);
            }

            flush_serialized(server, server_buf);
            flush_serialized(client, client_buf);
        }

        template <typename T>
        static void send_dcfring_impl(const shark::span<T> &share, int bin)
        {
            const u64 size = share.size();
            const size_t per_key_bytes =
                (size_t)(bin + 1) * sizeof(block) +
                (size_t)bin * sizeof(u128) * 2 +
                sizeof(u128) * 2;

            std::vector<char> server_buf;
            std::vector<char> client_buf;
            server_buf.resize((size_t)size * per_key_bytes);
            client_buf.resize((size_t)size * per_key_bytes);

            std::vector<block> seeds(size);
            for (u64 i = 0; i < size; ++i) seeds[i] = rand<block>();

            #pragma omp parallel for if(size > 8)
            for (long long i = 0; i < (long long)size; ++i)
            {
                osuCrypto::PRNG local_prng(seeds[(size_t)i]);
                prngOverride = &local_prng;
                auto [k0, k1] = crypto::dcfring_gen(bin, share[(size_t)i]);
                prngOverride = nullptr;

                size_t server_off = (size_t)i * per_key_bytes;
                size_t client_off = (size_t)i * per_key_bytes;
                write_span_bytes_at(server_buf, server_off, k0.k);
                write_span_bytes_at(server_buf, server_off, k0.v_ring);
                write_span_bytes_at(server_buf, server_off, k0.v_tag);
                write_pod_bytes_at(server_buf, server_off, k0.g_ring);
                write_pod_bytes_at(server_buf, server_off, k0.g_tag);

                write_span_bytes_at(client_buf, client_off, k1.k);
                write_span_bytes_at(client_buf, client_off, k1.v_ring);
                write_span_bytes_at(client_buf, client_off, k1.v_tag);
                write_pod_bytes_at(client_buf, client_off, k1.g_ring);
                write_pod_bytes_at(client_buf, client_off, k1.g_tag);
            }

            flush_serialized(server, server_buf);
            flush_serialized(client, client_buf);
        }

        template <typename T>
        static void send_dpfring_impl(const shark::span<T> &share, int bin)
        {
            const u64 size = share.size();
            const size_t per_key_bytes =
                (size_t)(bin + 1) * sizeof(block) + sizeof(u128) * 2;

            std::vector<char> server_buf;
            std::vector<char> client_buf;
            server_buf.resize((size_t)size * per_key_bytes);
            client_buf.resize((size_t)size * per_key_bytes);

            std::vector<block> seeds(size);
            for (u64 i = 0; i < size; ++i) seeds[i] = rand<block>();

            #pragma omp parallel for if(size > 8)
            for (long long i = 0; i < (long long)size; ++i)
            {
                osuCrypto::PRNG local_prng(seeds[(size_t)i]);
                prngOverride = &local_prng;
                auto [k0, k1] = crypto::dpfring_gen(bin, share[(size_t)i]);
                prngOverride = nullptr;

                size_t server_off = (size_t)i * per_key_bytes;
                size_t client_off = (size_t)i * per_key_bytes;
                write_span_bytes_at(server_buf, server_off, k0.k);
                write_pod_bytes_at(server_buf, server_off, k0.g_ring);
                write_pod_bytes_at(server_buf, server_off, k0.g_tag);

                write_span_bytes_at(client_buf, client_off, k1.k);
                write_pod_bytes_at(client_buf, client_off, k1.g_ring);
                write_pod_bytes_at(client_buf, client_off, k1.g_tag);
            }

            flush_serialized(server, server_buf);
            flush_serialized(client, client_buf);
        }

        template block rand<block>();
        template u64 rand<u64>();
        template u128 rand<u128>();
        template std::array<block, 2> rand<std::array<block, 2>>();

        void randomize(shark::span<u128> &share)
        {
            fill_with_parallel_prng(share.size(), [&](u64 i)
            {
                share[i] = rand<u64>();
            });
        }

        void randomize_full(shark::span<u128> &share)
        {
            fill_with_parallel_prng(share.size(), [&](u64 i)
            {
                share[i] = rand<u128>();
            });
        }

        void randomize_high(shark::span<u128> &share)
        {
            fill_with_parallel_prng(share.size(), [&](u64 i)
            {
                setHigh(share[i], rand<u64>());
            });
        }

        void randomize(shark::span<u64> &share)
        {
            fill_with_parallel_prng(share.size(), [&](u64 i)
            {
                share[i] = rand<u64>();
            });
        }

        void randomize(shark::span<u8> &share)
        {
            const u64 blocks = (share.size() + 63) / 64;
            fill_with_parallel_prng(blocks, [&](u64 block_idx)
            {
                const u64 tmp = rand<u64>();
                const u64 begin = block_idx * 64;
                const u64 end = std::min<u64>(begin + 64, share.size());
                for (u64 i = begin; i < end; ++i)
                {
                    share[i] = (u8)((tmp >> (i - begin)) & 1ULL);
                }
            });
        }

        void send_authenticated_ashare(const shark::span<u64> &share)
        {
            u64 size = share.size();
            // TODO: packing, PRG optimization
            shark::span<u64> share_0(size);
            shark::span<u64> share_1(size);
            shark::span<u128> share_0_tag(size);
            shark::span<u128> share_1_tag(size);

            randomize(share_0);
            
            #pragma omp parallel for
            for (u64 i = 0; i < size; i++)
            {
                share_1[i] = share[i] - share_0[i];
            }

            if (public_alpha_enabled())
            {
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    share_0_tag[i] = mac_mul_u64(share_0[i]);
                    share_1_tag[i] = mac_mul_u64(share_1[i]);
                }
            }
            else
            {
                randomize_full(share_0_tag);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    u128 raw_sum = u128(share_0[i]) + u128(share_1[i]);
                    share_1_tag[i] = mac_sub_u128(mac_mul_u128(raw_sum), share_0_tag[i]);
                }
            }

            server->send_array(share_0);
            server->send_array(share_0_tag);
            client->send_array(share_1);
            client->send_array(share_1_tag);
        }

        void send_authenticated_ashare_full(const shark::span<u128> &share)
        {
            u64 size = share.size();
            shark::span<u128> share_0(size);
            shark::span<u128> share_1(size);
            shark::span<u128> share_0_tag(size);
            shark::span<u128> share_1_tag(size);

            randomize_full(share_0);

            #pragma omp parallel for
            for (u64 i = 0; i < size; i++)
            {
                share_1[i] = share[i] - share_0[i];
            }

            if (public_alpha_enabled())
            {
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    share_0_tag[i] = mac_mul_u128(share_0[i]);
                    share_1_tag[i] = mac_mul_u128(share_1[i]);
                }
            }
            else
            {
                randomize_full(share_0_tag);
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    share_1_tag[i] = mac_sub_u128(mac_mul_u128(share[i]), share_0_tag[i]);
                }
            }

            server->send_array(share_0);
            server->send_array(share_0_tag);
            client->send_array(share_1);
            client->send_array(share_1_tag);
        }

        std::pair<shark::span<u64>, shark::span<u128> >
        recv_authenticated_ashare(u64 size)
        {
            auto share = dealer->recv_array<u64>(size);
            auto share_tag = dealer->recv_array<u128>(size);
            if (public_alpha_enabled())
            {
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    share_tag[i] = mac_mul_u64(share[i]);
                }
            }
            auto p = std::make_pair(std::move(share), std::move(share_tag));
            return p;
        }

        std::pair<shark::span<u128>, shark::span<u128> >
        recv_authenticated_ashare_full(u64 size)
        {
            auto share = dealer->recv_array<u128>(size);
            auto share_tag = dealer->recv_array<u128>(size);
            if (public_alpha_enabled())
            {
                #pragma omp parallel for
                for (u64 i = 0; i < size; i++)
                {
                    share_tag[i] = mac_mul_u128(share[i]);
                }
            }
            auto p = std::make_pair(std::move(share), std::move(share_tag));
            return p;
        }

        void send_authenticated_bshare(const shark::span<u8> &share)
        {
            u64 size = share.size();
            // TODO: packing, PRG optimization
            shark::span<u8> share_0(size);
            shark::span<u64> share_0_tag(size);
            
            shark::span<u8> share_1(size);
            shark::span<u64> share_1_tag(size);

            randomize(share_0);
            randomize(share_0_tag);

            #pragma omp parallel for
            for (u64 i = 0; i < size; i++)
            {
                // shares of x
                share_1[i] = share[i] ^ share_0[i];
                // share of tag
                share_1_tag[i] = share_0_tag[i] ^ (share[i] * bit_key);
            }

            server->send_array(share_0);
            server->send_array(share_0_tag);
            client->send_array(share_1);
            client->send_array(share_1_tag);
        }

        shark::span<FKOS> recv_authenticated_bshare(u64 size)
        {
            shark::span<FKOS> share(size);

            auto share_bit = dealer->recv_array<u8>(size);
            auto share_tag = dealer->recv_array<u64>(size);

            for (u64 i = 0; i < size; i++)
            {
                share[i] = std::make_tuple(share_bit[i], share_tag[i]);
            }

            return share;
        }

        void send_dcfbit(const shark::span<u64> &share, int bin)
        {
            send_dcfbit_impl(share, bin);
        }

        void send_dcfbit(const shark::span<u128> &share, int bin)
        {
            send_dcfbit_impl(share, bin);
        }

        void send_dcfring(const shark::span<u64> &share, int bin)
        {
            send_dcfring_impl(share, bin);
        }

        void send_dcfring(const shark::span<u128> &share, int bin)
        {
            send_dcfring_impl(share, bin);
        }

        void send_dpfring(const shark::span<u64> &share, int bin)
        {
            send_dpfring_impl(share, bin);
        }

        void send_dpfring(const shark::span<u128> &share, int bin)
        {
            send_dpfring_impl(share, bin);
        }

        shark::span<crypto::DCFBitKey> recv_dcfbit(u64 size, int bin)
        {
            shark::span<crypto::DCFBitKey> keys(size);
            for (u64 i = 0; i < size; ++i)
            {
                auto k = dealer->recv_array<block>(bin + 1);
                auto v_bit = dealer->recv_array<u8>(bin);
                auto v_tag_1 = dealer->recv_array<u64>(bin);
                // auto v_tag_2 = dealer->recv_array<u64>(bin);
                // auto v_tag_2 = shark::span<u64>(bin);
                u8 g_bit = dealer->recv<u8>();
                u64 g_tag_1 = dealer->recv<u64>();
                // u64 g_tag_2 = dealer->recv<u64>();

                keys[i] = std::move(
                    crypto::DCFBitKey(
                        std::move(k), 
                        std::move(v_bit), 
                        std::move(v_tag_1), 
                        // std::move(v_tag_2), 
                        g_bit, g_tag_1, 0
                    )
                );
            }
            return keys;
        }

        shark::span<crypto::DCFRingKey> recv_dcfring(u64 size, int bin)
        {
            shark::span<crypto::DCFRingKey> keys(size);
            for (u64 i = 0; i < size; ++i)
            {
                auto k = dealer->recv_array<block>(bin + 1);
                auto v_ring = dealer->recv_array<u128>(bin);
                auto v_tag = dealer->recv_array<u128>(bin);
                u128 g_ring = dealer->recv<u128>();
                u128 g_tag = dealer->recv<u128>();

                keys[i] = std::move(
                    crypto::DCFRingKey(
                        std::move(k), 
                        std::move(v_ring), 
                        std::move(v_tag), 
                        g_ring, g_tag
                    )
                );
            }
            return keys;
        }

        shark::span<crypto::DPFRingKey> recv_dpfring(u64 size, int bin)
        {
            shark::span<crypto::DPFRingKey> keys(size);
            for (u64 i = 0; i < size; ++i)
            {
                auto k = dealer->recv_array<block>(bin + 1);
                u128 g_ring = dealer->recv<u128>();
                u128 g_tag = dealer->recv<u128>();

                keys[i] = std::move(
                    crypto::DPFRingKey(
                        std::move(k), 
                        g_ring, g_tag
                    )
                );
            }
            return keys;
        }

        void authenticated_reconstruct(shark::span<u64> &share, const shark::span<u128> &share_tag, shark::span<u64> &res)
        {
            shark::utils::start_timer("auth_reconstruct_internal");
            shark::span<u64> tmp(share.size());
            shark::span<u64> local(share.size());
            for (u64 i = 0; i < share.size(); i++)
            {
                local[i] = share[i];
            }

            if (parallel_reconstruct)
            {
                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        peer->send_array(share);
                    }

                    #pragma omp section
                    {
                        peer->recv_array(tmp);
                    }
                }
            }
            else
            {
                peer->send_array(share);
                peer->recv_array(tmp);
            }
            for (u64 i = 0; i < share.size(); i++)
            {
                share[i] += tmp[i];
            }

            for (u64 i = 0; i < share.size(); i++)
            {
                u128 raw_sum = u128(local[i]) + u128(tmp[i]);
                res[i] = share[i];
                if (public_alpha_enabled())
                {
                    continue;
                }
                batchCheckArithmBuffer.push_back(mac_sub_u128(mac_mul_u128(raw_sum), share_tag[i]));
            }
            shark::utils::stop_timer("auth_reconstruct_internal");
        }

        shark::span<u64> authenticated_reconstruct(shark::span<u64> &share, const shark::span<u128> &share_tag)
        {
            shark::span<u64> res(share.size());
            authenticated_reconstruct(share, share_tag, res);
            return res;
        }

        void authenticated_reconstruct(shark::span<u128> &share, const shark::span<u128> &share_tag, shark::span<u64> &res)
        {
            shark::span<u64> share_low(share.size());
            #pragma omp parallel for
            for (u64 i = 0; i < share.size(); i++)
            {
                share_low[i] = getLow(share[i]);
            }
            authenticated_reconstruct(share_low, share_tag, res);
            #pragma omp parallel for
            for (u64 i = 0; i < share.size(); i++)
            {
                share[i] = u128(share_low[i]);
            }
        }

        shark::span<u64> authenticated_reconstruct(shark::span<u128> &share, const shark::span<u128> &share_tag)
        {
            shark::span<u64> res(share.size());
            authenticated_reconstruct(share, share_tag, res);
            return res;
        }

        void authenticated_reconstruct_full(shark::span<u128> &share, const shark::span<u128> &share_tag, shark::span<u128> &res)
        {
            shark::utils::start_timer("auth_reconstruct_internal");
            shark::span<u128> tmp(share.size());
            shark::span<u128> local(share.size());
            for (u64 i = 0; i < share.size(); i++)
            {
                local[i] = share[i];
            }

            if (parallel_reconstruct)
            {
                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        peer->send_array(share);
                    }

                    #pragma omp section
                    {
                        peer->recv_array(tmp);
                    }
                }
            }
            else
            {
                peer->send_array(share);
                peer->recv_array(tmp);
            }
            for (u64 i = 0; i < share.size(); i++)
            {
                share[i] += tmp[i];
            }

            for (u64 i = 0; i < share.size(); i++)
            {
                res[i] = share[i];
                if (public_alpha_enabled())
                {
                    continue;
                }
                batchCheckArithmBuffer.push_back(mac_sub_u128(mac_mul_u128(res[i]), share_tag[i]));
            }
            shark::utils::stop_timer("auth_reconstruct_internal");
        }

        shark::span<u128> authenticated_reconstruct_full(shark::span<u128> &share, const shark::span<u128> &share_tag)
        {
            shark::span<u128> res(share.size());
            authenticated_reconstruct_full(share, share_tag, res);
            return res;
        }

        shark::span<u8> authenticated_reconstruct(shark::span<FKOS> &share)
        {
            shark::utils::start_timer("auth_reconstruct_internal");
            shark::span<u8> tmp_bit(share.size());
            // shark::span<u64> tmp_M(share.size());
            shark::span<u8> share_bit(share.size());
            // shark::span<u64> share_K(share.size());
            // shark::span<u64> share_M(share.size());

            // TODO: unnecesary copy
            for (u64 i = 0; i < share.size(); i++)
            {
                share_bit[i] = std::get<0>(share[i]);
                // share_K[i] = std::get<1>(share[i]);
                // share_M[i] = std::get<2>(share[i]);
            }

            if (parallel_reconstruct)
            {
                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        peer->send_array(share_bit);
                        // peer->send_array(share_M);
                    }

                    #pragma omp section
                    {
                        peer->recv_array(tmp_bit);
                        // peer->recv_array(tmp_M);
                    }
                }
            }
            else
            {
                peer->send_array(share_bit);
                // peer->send_array(share_M);
                peer->recv_array(tmp_bit);
                // peer->recv_array(tmp_M);
            }

            // TODO: batch check like ring shares
            for (u64 i = 0; i < share.size(); i++)
            {
                // always_assert(tmp_M[i] == share_K[i] ^ (tmp_bit[i] * bit_key[party]));
                share_bit[i] ^= tmp_bit[i];
                if (public_alpha_enabled())
                {
                    continue;
                }
                u64 z = std::get<1>(share[i]) ^ (share_bit[i] * bit_key);
                batchCheckBoolBuffer.push_back(z);
            }
            shark::utils::stop_timer("auth_reconstruct_internal");
            return share_bit;
        }

        template <typename T>
        shark::span<u8> compute_commitment(int party, const T &x, const block &r)
        {
            osuCrypto::RandomOracle H;
            shark::span<u8> commitment(osuCrypto::RandomOracle::HashSize);
            auto party_bytes = shark::wire::serialize_value(party);
            auto x_bytes = shark::wire::serialize_value(x);
            auto r_bytes = shark::wire::serialize_value(r);
            H.Update(reinterpret_cast<const u8 *>(party_bytes.data()), party_bytes.size());
            H.Update(reinterpret_cast<const u8 *>(x_bytes.data()), x_bytes.size());
            H.Update(reinterpret_cast<const u8 *>(r_bytes.data()), r_bytes.size());
            H.Final(commitment.data());
            return commitment;
        }

        template <typename T>
        T commit_and_exchange(const T &x)
        {
            static u64 commit_exchange_id = 0;
            u64 my_commit_id = commit_exchange_id++;
            block r = rand<block>();
            shark::span<u8> commitment = compute_commitment(party, x, r);
            peer->send_array(commitment);
            auto peer_commitment = peer->recv_array<u8>(osuCrypto::RandomOracle::HashSize);

            peer->send(x);
            peer->send(r);
            T peer_x = peer->recv<T>();
            block peer_r = peer->recv<block>();

            shark::span<u8> peer_commitment2 = compute_commitment(SERVER + CLIENT - party, peer_x, peer_r);

            for (int i = 0; i < osuCrypto::RandomOracle::HashSize; i++)
            {
                if (peer_commitment[i] != peer_commitment2[i])
                {
                    auto x_bytes = shark::wire::serialize_value(x);
                    auto peer_bytes = shark::wire::serialize_value(peer_x);
                    auto head_byte = [](const auto &bytes, size_t idx) -> unsigned int
                    {
                        return idx < bytes.size() ? static_cast<unsigned int>(static_cast<unsigned char>(bytes[idx])) : 0u;
                    };
                    printf("[commit_and_exchange] mismatch party=%d id=%llu idx=%d wire_size=%zu bytesSent=%llu bytesRecv=%llu\n",
                           party,
                           (unsigned long long)my_commit_id,
                           i,
                           x_bytes.size(),
                           (unsigned long long)peer->bytesSent(),
                           (unsigned long long)peer->bytesReceived());
                    printf("[commit_and_exchange] commitment peer=%02x%02x%02x%02x local=%02x%02x%02x%02x\n",
                           peer_commitment[0], peer_commitment[1], peer_commitment[2], peer_commitment[3],
                           peer_commitment2[0], peer_commitment2[1], peer_commitment2[2], peer_commitment2[3]);
                    printf("[commit_and_exchange] x_head=%02x%02x%02x%02x peer_x_head=%02x%02x%02x%02x\n",
                           head_byte(x_bytes, 0), head_byte(x_bytes, 1), head_byte(x_bytes, 2), head_byte(x_bytes, 3),
                           head_byte(peer_bytes, 0), head_byte(peer_bytes, 1), head_byte(peer_bytes, 2), head_byte(peer_bytes, 3));
                    always_assert(false);
                }
            }

            return peer_x;
        }

        void batch_check()
        {

            if ((batchCheckArithmBuffer.size() == 0) && (batchCheckBoolBuffer.size() == 0))
            {
                return;
            }

            osuCrypto::PRNG prngBatchCheck;
            block r_prng = rand<block>();
            block peer_r_prng = commit_and_exchange(r_prng);
            block prng_seed = r_prng ^ peer_r_prng;
            prngBatchCheck.SetSeed(prng_seed);

            u128 batchCheckAccumulated = 0;
            u64 batchCheckBitsAccumulated = 0;
            for (u64 i = 0; i < batchCheckArithmBuffer.size(); i++)
            {
                batchCheckAccumulated += u128(prngBatchCheck.get<u64>()) * batchCheckArithmBuffer[i];
            }

            for (u64 i = 0; i < batchCheckBoolBuffer.size(); i++)
            {
                batchCheckBitsAccumulated += prngBatchCheck.get<u64>() * batchCheckBoolBuffer[i];
            }

            // commit and open batchCheckAccumulated
            auto batchCheckPair = std::make_pair(batchCheckAccumulated, batchCheckBitsAccumulated);
            auto peer_batchCheckPair = commit_and_exchange(batchCheckPair);

            if (debug_break_on_mismatch_enabled())
            {
                if (batchCheckAccumulated + peer_batchCheckPair.first != 0 ||
                    batchCheckBitsAccumulated != peer_batchCheckPair.second)
                {
                    printf("[debug] batch_check mismatch | arithm_hi=%llu arithm_lo=%llu peer_hi=%llu peer_lo=%llu bool=%llu peer_bool=%llu\n",
                           (unsigned long long)getHigh(batchCheckAccumulated),
                           (unsigned long long)getLow(batchCheckAccumulated),
                           (unsigned long long)getHigh(peer_batchCheckPair.first),
                           (unsigned long long)getLow(peer_batchCheckPair.first),
                           (unsigned long long)batchCheckBitsAccumulated,
                           (unsigned long long)peer_batchCheckPair.second);
                    debug_print_keypos("batch_check_mismatch");
                    if (debug_trap_enabled())
                    {
                        raise(SIGTRAP);
                    }
                }
            }

            always_assert(batchCheckAccumulated + peer_batchCheckPair.first == 0);
            always_assert(batchCheckBitsAccumulated == peer_batchCheckPair.second);
            batchCheckArithmBuffer.clear();
            batchCheckBoolBuffer.clear();
        }

        void debug_batch_check(const char *label)
        {
            if (party == DEALER)
            {
                return;
            }

            if (!debug_batchcheck_logs_enabled())
            {
                batch_check();
                return;
            }

            static bool verbose = std::getenv("SHARK_DEBUG_BATCHCHECK_VERBOSE") != nullptr;
            static bool pair = std::getenv("SHARK_DEBUG_BATCHCHECK_PAIR") != nullptr;
            if (verbose)
            {
                const char *safe_label = label ? label : "<null>";
                printf("[debug] batch_check at %s | arithm=%zu bool=%zu\n",
                       safe_label, batchCheckArithmBuffer.size(), batchCheckBoolBuffer.size());
                fflush(stdout);
            }
            debug_print_keypos(label);
            static bool scan = std::getenv("SHARK_DEBUG_BATCHCHECK_SCAN") != nullptr;
            if (scan)
            {
                size_t nz_a = 0;
                size_t first_a = (size_t)-1;
                for (size_t i = 0; i < batchCheckArithmBuffer.size(); ++i)
                {
                    if (batchCheckArithmBuffer[i] != 0)
                    {
                        if (first_a == (size_t)-1) first_a = i;
                        if (nz_a < 8)
                        {
                            printf("[debug] batch_check nz_arithm[%zu]=%llu (low64)\n", i, (unsigned long long)getLow(batchCheckArithmBuffer[i]));
                        }
                        nz_a++;
                    }
                }
                size_t nz_b = 0;
                size_t first_b = (size_t)-1;
                for (size_t i = 0; i < batchCheckBoolBuffer.size(); ++i)
                {
                    if (batchCheckBoolBuffer[i] != 0)
                    {
                        if (first_b == (size_t)-1) first_b = i;
                        if (nz_b < 8)
                        {
                            printf("[debug] batch_check nz_bool[%zu]=%llu\n", i, (unsigned long long)batchCheckBoolBuffer[i]);
                        }
                        nz_b++;
                    }
                }
                printf("[debug] batch_check scan %s | arithm_nz=%zu first=%zu bool_nz=%zu first=%zu\n",
                       label ? label : "<null>", nz_a, first_a, nz_b, first_b);
                fflush(stdout);
            }

            if (pair)
            {
                // Exchange first few z's to see if they sum to zero across parties.
                size_t n_a = batchCheckArithmBuffer.size();
                size_t n_b = batchCheckBoolBuffer.size();
                size_t n = n_a < 8 ? n_a : 8;
                if (n > 0)
                {
                    shark::span<u64> local(n);
                    for (size_t i = 0; i < n; ++i) local[i] = getLow(batchCheckArithmBuffer[i]);
                    shark::span<u64> peer_local(n);
                    #pragma omp parallel sections
                    {
                        #pragma omp section
                        {
                            peer->send_array(local);
                        }
                        #pragma omp section
                        {
                            peer->recv_array(peer_local);
                }
            }

            static bool find = std::getenv("SHARK_DEBUG_BATCHCHECK_FIND") != nullptr;
            if (find)
            {
                size_t n_a = batchCheckArithmBuffer.size();
                size_t n_b = batchCheckBoolBuffer.size();

                // Exchange full buffers (low64) to locate first mismatch index.
                if (n_a > 0)
                {
                    shark::span<u64> local_a(n_a);
                    for (size_t i = 0; i < n_a; ++i) local_a[i] = getLow(batchCheckArithmBuffer[i]);
                    shark::span<u64> peer_a(n_a);
                    if (parallel_reconstruct)
                    {
                        #pragma omp parallel sections
                        {
                            #pragma omp section
                            {
                                peer->send_array(local_a);
                            }
                            #pragma omp section
                            {
                                peer->recv_array(peer_a);
                            }
                        }
                    }
                    else
                    {
                        peer->send_array(local_a);
                        peer->recv_array(peer_a);
                    }

                    size_t first = (size_t)-1;
                    size_t mismatches = 0;
                    for (size_t i = 0; i < n_a; ++i)
                    {
                        if ((u64)(local_a[i] + peer_a[i]) != 0)
                        {
                            if (first == (size_t)-1) first = i;
                            if (mismatches < 8)
                            {
                                printf("[debug] batch_check find arithm[%zu] local=%llu peer=%llu sum=%llu\n",
                                       i,
                                       (unsigned long long)local_a[i],
                                       (unsigned long long)peer_a[i],
                                       (unsigned long long)(local_a[i] + peer_a[i]));
                            }
                            mismatches++;
                        }
                    }
                    printf("[debug] batch_check find arithm | mismatches=%zu first=%zu\n", mismatches, first);
                }

                if (n_b > 0)
                {
                    shark::span<u64> local_b(n_b);
                    for (size_t i = 0; i < n_b; ++i) local_b[i] = batchCheckBoolBuffer[i];
                    shark::span<u64> peer_b(n_b);
                    if (parallel_reconstruct)
                    {
                        #pragma omp parallel sections
                        {
                            #pragma omp section
                            {
                                peer->send_array(local_b);
                            }
                            #pragma omp section
                            {
                                peer->recv_array(peer_b);
                            }
                        }
                    }
                    else
                    {
                        peer->send_array(local_b);
                        peer->recv_array(peer_b);
                    }

                    size_t first = (size_t)-1;
                    size_t mismatches = 0;
                    for (size_t i = 0; i < n_b; ++i)
                    {
                        if (local_b[i] != peer_b[i])
                        {
                            if (first == (size_t)-1) first = i;
                            if (mismatches < 8)
                            {
                                printf("[debug] batch_check find bool[%zu] local=%llu peer=%llu xor=%llu\n",
                                       i,
                                       (unsigned long long)local_b[i],
                                       (unsigned long long)peer_b[i],
                                       (unsigned long long)(local_b[i] ^ peer_b[i]));
                            }
                            mismatches++;
                        }
                    }
                    printf("[debug] batch_check find bool | mismatches=%zu first=%zu\n", mismatches, first);
                }
            }
                    for (size_t i = 0; i < n; ++i)
                    {
                        u64 sum = local[i] + peer_local[i];
                        printf("[debug] batch_check pair arithm[%zu] local=%llu peer=%llu sum=%llu\n",
                               i,
                               (unsigned long long)local[i],
                               (unsigned long long)peer_local[i],
                               (unsigned long long)sum);
                    }
                }
                n = n_b < 8 ? n_b : 8;
                if (n > 0)
                {
                    shark::span<u64> local(n);
                    for (size_t i = 0; i < n; ++i) local[i] = batchCheckBoolBuffer[i];
                    shark::span<u64> peer_local(n);
                    #pragma omp parallel sections
                    {
                        #pragma omp section
                        {
                            peer->send_array(local);
                        }
                        #pragma omp section
                        {
                            peer->recv_array(peer_local);
                        }
                    }
                    for (size_t i = 0; i < n; ++i)
                    {
                        u64 sum = local[i] ^ peer_local[i];
                        printf("[debug] batch_check pair bool[%zu] local=%llu peer=%llu xor=%llu\n",
                               i,
                               (unsigned long long)local[i],
                               (unsigned long long)peer_local[i],
                               (unsigned long long)sum);
                    }
                }
                fflush(stdout);
            }

            batch_check();
        }

        void debug_batch_check_force(const char *label)
        {
            if (party == DEALER)
            {
                return;
            }

            if (!debug_batchcheck_logs_enabled())
            {
                return;
            }

            const char *safe_label = label ? label : "<null>";
            printf("[debug] batch_check at %s | arithm=%zu bool=%zu\n",
                   safe_label, batchCheckArithmBuffer.size(), batchCheckBoolBuffer.size());
            fflush(stdout);
            debug_print_keypos(label);
        }

        // FKOS arithmetic
        FKOS xor_fkos(FKOS x, FKOS y)
        {
            return std::make_tuple(std::get<0>(x) ^ std::get<0>(y), std::get<1>(x) ^ std::get<1>(y));
        }

        FKOS not_fkos(FKOS x)
        {
            if (party == SERVER)
            {
                return std::make_tuple(std::get<0>(x) ^ 1, std::get<1>(x) ^ bit_key);
            }
            else
            {
                return std::make_tuple(std::get<0>(x), std::get<1>(x) ^ bit_key);
            }
        }
    }
}
