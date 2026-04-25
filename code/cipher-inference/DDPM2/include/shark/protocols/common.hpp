#pragma once

#include <cryptoTools/Crypto/PRNG.h>
#include <cstdlib>

#include <shark/types/u128.hpp>
#include <shark/utils/comm.hpp>
#include <shark/crypto/dcfbit.hpp>
#include <shark/crypto/dcfring.hpp>
#include <shark/crypto/dpfring.hpp>

namespace shark {
    namespace protocols {
        /// this key is used to generate the keys for the two parties
        /// same key to this PRNG generates exactly same keys
        /// evaluators also use this object to generate commitment randomness
        extern osuCrypto::PRNG prngGlobal;
        /// party identifier
        extern int party;
        /// legend to party identifier
        enum Party {
            SERVER = 0,
            CLIENT = 1,
            DEALER = 2,
            EMUL = 3,
        };
        /// filenames of the keyfile for the two parties
        const std::string filename[2] = {"server.dat", "client.dat"};
        /// communication structs used by the dealer
        extern Peer *server;
        extern Peer *client;
        /// communication structs used by evaluators
        extern Dealer *dealer;
        extern Peer *peer;

        /// MAC key for SPDZ_sk style shares. For dealer, this is clear value. For evaluators, this is secret share of the clear value.
        extern u128 ring_key;
        /// MAC key for FKOS style shares. For dealer, this is clear value. For evaluators, this is secret share of the clear value.
        extern u64 bit_key;

        inline bool public_alpha_enabled()
        {
            static int enabled = -1;
            if (enabled < 0)
            {
                enabled = (std::getenv("SHARK_PUBLIC_ALPHA") != nullptr) ? 1 : 0;
            }
            return enabled == 1;
        }

        // --- MAC helpers for l=64, s=64 over the extended 128-bit ring ---
        inline u64 mac_key_low() { return (u64)ring_key; }
        inline u128 mac_mul_u64(u64 x) { return ring_key * u128(x); }
        inline u128 mac_mul_share(u128 x) { return x * ring_key; }
        inline u128 mac_mul_tag(u128 tag, u64 x) { return tag * u128(x); }
        inline u128 mac_add(u128 a, u128 b) { return a + b; }
        inline u128 mac_sub(u128 a, u128 b) { return a - b; }
        inline u128 mac_norm(u128 a) { return a; }
        inline u128 mac_wrap_u64(u64 wraps) { return (ring_key << 64) * u128(wraps); }

        // --- Extended-ring helpers for arbitrary 128-bit masked values ---
        inline u128 mac_mul_u128(u128 x) { return x * ring_key; }
        inline u128 mac_add_u128(u128 a, u128 b) { return a + b; }
        inline u128 mac_sub_u128(u128 a, u128 b) { return a - b; }

        inline void auth_local_add_u64(u64 a_share, u128 a_tag,
                                       u64 b_share, u128 b_tag,
                                       u64 &out_share, u128 &out_tag)
        {
            const u128 raw = u128(a_share) + u128(b_share);
            out_share = getLow(raw);
            out_tag = mac_sub_u128(mac_add_u128(a_tag, b_tag), mac_wrap_u64(getHigh(raw)));
        }

        inline void auth_local_sub_u64(u64 a_share, u128 a_tag,
                                       u64 b_share, u128 b_tag,
                                       u64 &out_share, u128 &out_tag)
        {
            const u64 borrow = (a_share < b_share) ? u64(1) : u64(0);
            out_share = a_share - b_share;
            out_tag = mac_add_u128(mac_sub_u128(a_tag, b_tag), mac_wrap_u64(borrow));
        }

        inline void auth_local_mul_public_u64(u64 share, u128 tag, u64 c,
                                              u64 &out_share, u128 &out_tag)
        {
            const u128 raw = u128(share) * u128(c);
            out_share = getLow(raw);
            out_tag = mac_sub_u128(mac_mul_tag(tag, c), mac_wrap_u64(getHigh(raw)));
        }

        inline void auth_local_add_public_u64(u64 share, u128 tag, u64 public_value,
                                              bool add_to_local_share,
                                              u64 &out_share, u128 &out_tag)
        {
            const u128 raw = u128(share) + u128(add_to_local_share ? public_value : u64(0));
            out_share = getLow(raw);
            out_tag = mac_add_u128(tag, mac_mul_u64(public_value));
            out_tag = mac_sub_u128(out_tag, mac_wrap_u64(getHigh(raw)));
        }

        inline void auth_local_sub_public_u64(u64 share, u128 tag, u64 public_value,
                                              bool sub_from_local_share,
                                              u64 &out_share, u128 &out_tag)
        {
            const u64 local_sub = sub_from_local_share ? public_value : u64(0);
            const u64 borrow = (share < local_sub) ? u64(1) : u64(0);
            out_share = share - local_sub;
            out_tag = mac_sub_u128(tag, mac_mul_u64(public_value));
            out_tag = mac_add_u128(out_tag, mac_wrap_u64(borrow));
        }

        inline void auth_local_neg_u64(u64 share, u128 tag,
                                       u64 &out_share, u128 &out_tag)
        {
            out_share = u64(0) - share;
            out_tag = mac_add_u128(u128(0) - tag, mac_wrap_u64(share != u64(0) ? u64(1) : u64(0)));
        }

        inline void auth_local_add_u128(u128 a_share, u128 a_tag,
                                        u128 b_share, u128 b_tag,
                                        u128 &out_share, u128 &out_tag)
        {
            out_share = a_share + b_share;
            out_tag = mac_add_u128(a_tag, b_tag);
        }

        inline void auth_local_sub_u128(u128 a_share, u128 a_tag,
                                        u128 b_share, u128 b_tag,
                                        u128 &out_share, u128 &out_tag)
        {
            out_share = a_share - b_share;
            out_tag = mac_sub_u128(a_tag, b_tag);
        }

        inline void auth_local_mul_public_u128(u128 share, u128 tag, u128 c,
                                               u128 &out_share, u128 &out_tag)
        {
            out_share = share * c;
            out_tag = mac_mul_tag(tag, c);
        }

        inline void auth_local_add_public_u128(u128 share, u128 tag, u128 public_value,
                                               bool add_to_local_share,
                                               u128 &out_share, u128 &out_tag)
        {
            out_share = share + (add_to_local_share ? public_value : u128(0));
            out_tag = mac_add_u128(tag, mac_mul_u128(public_value));
        }

        inline void auth_local_sub_public_u128(u128 share, u128 tag, u128 public_value,
                                               bool sub_from_local_share,
                                               u128 &out_share, u128 &out_tag)
        {
            out_share = share - (sub_from_local_share ? public_value : u128(0));
            out_tag = mac_sub_u128(tag, mac_mul_u128(public_value));
        }

        inline void auth_local_neg_u128(u128 share, u128 tag,
                                        u128 &out_share, u128 &out_tag)
        {
            out_share = u128(0) - share;
            out_tag = u128(0) - tag;
        }

        inline void mac_normalize_span(shark::span<u128> &t)
        {
            for (u64 i = 0; i < t.size(); ++i)
            {
                t[i] = mac_norm(t[i]);
            }
        }

        /// MP-SPDZ does all comparisons, like in ReLU and MaxPool, in bitlength of 32. Setting this to true will make the protocols do the same.
        extern bool mpspdz_32bit_compaison;

        using FKOS = std::tuple<u8, u64>;
        
        /// methods used by the dealer
        extern thread_local osuCrypto::PRNG *prngOverride;

        template <typename T>
        T rand() { return prngOverride ? prngOverride->get<T>() : prngGlobal.get<T>(); }

        void randomize(shark::span<u128> &share);
        void randomize_full(shark::span<u128> &share);
        void randomize(shark::span<u64> &share);
        void randomize(shark::span<u8> &share);
        void send_authenticated_ashare(const shark::span<u64> &share);
        void send_authenticated_ashare_full(const shark::span<u128> &share);
        void send_authenticated_bshare(const shark::span<u8> &share);
        void send_dcfbit(const shark::span<u64> &share, int bin);
        void send_dcfbit(const shark::span<u128> &share, int bin);
        void send_dcfring(const shark::span<u64> &share, int bin);
        void send_dcfring(const shark::span<u128> &share, int bin);
        void send_dpfring(const shark::span<u64> &share, int bin);
        void send_dpfring(const shark::span<u128> &share, int bin);

        /// methods used by the evaluators
        std::pair<shark::span<u64>, shark::span<u128>> recv_authenticated_ashare(u64 size);
        std::pair<shark::span<u128>, shark::span<u128>> recv_authenticated_ashare_full(u64 size);
        shark::span<FKOS> recv_authenticated_bshare(u64 size);
        shark::span<crypto::DCFBitKey> recv_dcfbit(u64 size, int bin);
        shark::span<crypto::DCFRingKey> recv_dcfring(u64 size, int bin);
        shark::span<crypto::DPFRingKey> recv_dpfring(u64 size, int bin);

        void authenticated_reconstruct(shark::span<u64> &share, const shark::span<u128> &share_tag, shark::span<u64> &res);
        shark::span<u64> authenticated_reconstruct(shark::span<u64> &share, const shark::span<u128> &share_tag);
        void authenticated_reconstruct(shark::span<u128> &share, const shark::span<u128> &share_tag, shark::span<u64> &res);
        shark::span<u64> authenticated_reconstruct(shark::span<u128> &share, const shark::span<u128> &share_tag);
        void authenticated_reconstruct_full(shark::span<u128> &share, const shark::span<u128> &share_tag, shark::span<u128> &res);
        shark::span<u128> authenticated_reconstruct_full(shark::span<u128> &share, const shark::span<u128> &share_tag);
        shark::span<u8> authenticated_reconstruct(shark::span<FKOS> &share);

        void batch_check();
        void debug_batch_check(const char *label);
        // Stats output gated by SHARK_DEBUG_BATCHCHECK.
        void debug_batch_check_force(const char *label);

        FKOS xor_fkos(FKOS x, FKOS y);
        FKOS not_fkos(FKOS x);
    }
}
