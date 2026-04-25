#include <shark/protocols/common.hpp>
#include <shark/protocols/init.hpp>
#include <shark/utils/assert.hpp>
#include <fstream>
#include <cryptoTools/Common/Defines.h>
#include <chrono>
#include <cstdlib>
#include <Eigen/Core>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace shark {
    namespace protocols {

        namespace {
            void configure_parallel_backends()
            {
#ifdef _OPENMP
                Eigen::initParallel();
                Eigen::setNbThreads(omp_get_max_threads());
#else
                Eigen::setNbThreads(1);
#endif
            }
        }

        namespace init {
            void gen(uint64_t key)
            {
                party = DEALER;
                configure_parallel_backends();
                u64 seed64 = static_cast<u64>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
                prngGlobal.SetSeed(osuCrypto::toBlock(seed64));

                server = new Peer(filename[0]);
                client = new Peer(filename[1]);

                ring_key = u128(rand<u64>());
                u128 ring_key_0 = rand<u128>();
                u128 ring_key_1 = ring_key - ring_key_0;
                server->send(ring_key_0);
                client->send(ring_key_1);

                u64 bit_key_0 = rand<u64>();
                u64 bit_key_1 = rand<u64>();
                bit_key = bit_key_0 ^ bit_key_1;

                server->send(bit_key_0);
                client->send(bit_key_1);

                osuCrypto::block prng_seed = rand<block>();
                server->send(prng_seed);
                client->send(prng_seed);

                prngGlobal.SetSeed(prng_seed);
            }

            void eval(int _party, std::string ip, int port, bool oneShot)
            {
                always_assert(_party == SERVER || _party == CLIENT);
                party = _party;
                configure_parallel_backends();
                dealer = new Dealer(filename[party], oneShot);

                ring_key = dealer->recv<u128>();
                bit_key = dealer->recv<u64>();
                osuCrypto::block prng_seed = dealer->recv<osuCrypto::block>();

                // setup communication between evaluating parties
                if (party == SERVER)
                {
                    peer = waitForPeer(port);
                }
                else
                {
                    peer = new Peer(ip, port);
                }

                prngGlobal.SetSeed(prng_seed);

                if (public_alpha_enabled())
                {
                    // Reconstruct full arithmetic MAC key (ring_key) for public-alpha mode.
                    u128 my_ring = ring_key;
                    u128 peer_ring = 0;
                    if (party == SERVER)
                    {
                        peer->send(my_ring);
                        peer_ring = peer->recv<u128>();
                    }
                    else
                    {
                        peer_ring = peer->recv<u128>();
                        peer->send(my_ring);
                    }
                    ring_key = my_ring + peer_ring;
                }
            }

            void from_args(int argc, char ** argv)
            {
                always_assert(argc > 1);
                int _party = atoi(argv[1]);
                always_assert(_party == DEALER || _party == SERVER || _party == CLIENT || _party == EMUL);

                if (_party == EMUL)
                {
                    party = EMUL;
                    return;
                }
                
                std::string ip = (argc > 2) ? argv[2] : "127.0.0.1";
                int port = (argc > 3) ? atoi(argv[3]) : 42069;
                if (_party == DEALER)
                    init::gen(0xdeadbeef);
                else
                    eval(_party, ip, port, false);
            }
        }
    }
}
