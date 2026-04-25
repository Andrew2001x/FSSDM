#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <shark/types/u128.hpp>
#include <shark/types/span.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>

namespace shark
{
    typedef enum BufType
    {
        BUF_FILE,
        BUF_SOCKET,
        BUF_MEM
    } BufType;

    class KeyBuf
    {
    public:
        uint64_t bytesSent = 0;
        uint64_t bytesReceived = 0;
        uint64_t roundsSent = 0;
        uint64_t roundsReceived = 0;
        BufType t;
        virtual void sync() {}
        virtual void read(char *buf, u64 bytes) = 0;
        virtual char *read(u64 bytes) = 0;
        virtual void write(char *buf, u64 bytes) = 0;
        virtual void close() = 0;
        bool isMem() { return t == BUF_MEM; }
    };

    typedef enum FileMode
    {
        F_RD_ONLY,
        F_WR_ONLY
    } FileMode;

    namespace wire
    {
        inline void write_le_u32(char *dst, std::uint32_t value)
        {
            for (size_t i = 0; i < sizeof(std::uint32_t); ++i)
            {
                dst[i] = static_cast<char>((value >> (8 * i)) & 0xffu);
            }
        }

        inline std::uint32_t read_le_u32(const char *src)
        {
            std::uint32_t value = 0;
            for (size_t i = 0; i < sizeof(std::uint32_t); ++i)
            {
                value |= (std::uint32_t(static_cast<unsigned char>(src[i])) << (8 * i));
            }
            return value;
        }

        inline void write_le_u64(char *dst, u64 value)
        {
            for (size_t i = 0; i < sizeof(u64); ++i)
            {
                dst[i] = static_cast<char>((value >> (8 * i)) & 0xffu);
            }
        }

        inline u64 read_le_u64(const char *src)
        {
            u64 value = 0;
            for (size_t i = 0; i < sizeof(u64); ++i)
            {
                value |= (u64(static_cast<unsigned char>(src[i])) << (8 * i));
            }
            return value;
        }

        template <typename T>
        struct Serializer;

        template <>
        struct Serializer<u8>
        {
            static constexpr size_t size = 1;

            static std::array<char, size> serialize(u8 value)
            {
                return {static_cast<char>(value)};
            }

            static u8 deserialize(const char *src)
            {
                return static_cast<u8>(static_cast<unsigned char>(src[0]));
            }
        };

        template <>
        struct Serializer<int>
        {
            static constexpr size_t size = sizeof(std::uint32_t);

            static std::array<char, size> serialize(int value)
            {
                static_assert(sizeof(int) == sizeof(std::uint32_t));
                std::array<char, size> out{};
                std::int32_t signed_value = static_cast<std::int32_t>(value);
                std::uint32_t raw = 0;
                std::memcpy(&raw, &signed_value, sizeof(raw));
                write_le_u32(out.data(), raw);
                return out;
            }

            static int deserialize(const char *src)
            {
                static_assert(sizeof(int) == sizeof(std::uint32_t));
                std::uint32_t raw = read_le_u32(src);
                std::int32_t signed_value = 0;
                std::memcpy(&signed_value, &raw, sizeof(raw));
                return static_cast<int>(signed_value);
            }
        };

        template <>
        struct Serializer<u64>
        {
            static constexpr size_t size = sizeof(u64);

            static std::array<char, size> serialize(u64 value)
            {
                std::array<char, size> out{};
                write_le_u64(out.data(), value);
                return out;
            }

            static u64 deserialize(const char *src)
            {
                return read_le_u64(src);
            }
        };

        template <>
        struct Serializer<u128>
        {
            static constexpr size_t size = 2 * sizeof(u64);

            static std::array<char, size> serialize(u128 value)
            {
                std::array<char, size> out{};
                write_le_u64(out.data(), getLow(value));
                write_le_u64(out.data() + sizeof(u64), getHigh(value));
                return out;
            }

            static u128 deserialize(const char *src)
            {
                u128 value = 0;
                setLow(value, read_le_u64(src));
                setHigh(value, read_le_u64(src + sizeof(u64)));
                return value;
            }
        };

        template <>
        struct Serializer<block>
        {
            static constexpr size_t size = 16;

            static std::array<char, size> serialize(const block &value)
            {
                static_assert(sizeof(block) == size);
                std::array<char, size> out{};
                std::memcpy(out.data(), &value, size);
                return out;
            }

            static block deserialize(const char *src)
            {
                static_assert(sizeof(block) == size);
                block value{};
                std::memcpy(&value, src, size);
                return value;
            }
        };

        template <typename A, typename B>
        struct Serializer<std::pair<A, B>>
        {
            static constexpr size_t first_size = Serializer<A>::size;
            static constexpr size_t second_size = Serializer<B>::size;
            static constexpr size_t size = first_size + second_size;

            static std::array<char, size> serialize(const std::pair<A, B> &value)
            {
                std::array<char, size> out{};
                auto first = Serializer<A>::serialize(value.first);
                auto second = Serializer<B>::serialize(value.second);
                std::copy(first.begin(), first.end(), out.begin());
                std::copy(second.begin(), second.end(), out.begin() + first_size);
                return out;
            }

            static std::pair<A, B> deserialize(const char *src)
            {
                return {
                    Serializer<A>::deserialize(src),
                    Serializer<B>::deserialize(src + first_size),
                };
            }
        };

        template <typename T>
        auto serialize_value(const T &value)
        {
            return Serializer<T>::serialize(value);
        }

        template <typename T>
        T deserialize_value(const char *src)
        {
            return Serializer<T>::deserialize(src);
        }
    }

    class FileBuf : public KeyBuf
    {
    public:
        std::fstream file;
        std::vector<char> ioBuffer;

        FileBuf(std::string filename, FileMode f)
        {
            printf("Opening file=%s, mode=%d\n", filename.data(), f);
            this->t = BUF_FILE;
            ioBuffer.resize(8 * 1024 * 1024);
            file.rdbuf()->pubsetbuf(ioBuffer.data(), (std::streamsize)ioBuffer.size());
            if (f == F_WR_ONLY)
                this->file.open(filename, std::ios::out | std::ios::binary);
            else
                this->file.open(filename, std::ios::in | std::ios::binary);
        }

        void read(char *buf, u64 bytes)
        {
            this->file.read(buf, bytes);
            bytesReceived += bytes;
            roundsReceived += 1;
        }

        char *read(u64 bytes)
        {
            char *newBuf = new char[bytes];
            this->read(newBuf, bytes);
            return newBuf;
        }

        void write(char *buf, u64 bytes)
        {
            this->file.write(buf, bytes);
            bytesSent += bytes;
            roundsSent += 1;
        }

        void close()
        {
            this->file.close();
        }
    };

    class SocketBuf : public KeyBuf
    {
    public:
        int sendsocket, recvsocket;

        SocketBuf(std::string ip, int port, bool onlyRecv);
        SocketBuf(int sendsocket, int recvsocket) : sendsocket(sendsocket), recvsocket(recvsocket)
        {
            this->t = BUF_SOCKET;
        }
        void sync();
        void read(char *buf, u64 bytes);
        char *read(u64 bytes);
        void write(char *buf, u64 bytes);
        void close();
    };

    class MemBuf : public KeyBuf
    {
    public:
        char **memBufPtr;
        char *startPtr;

        MemBuf(char **mBufPtr)
        {
            this->t = BUF_MEM;
            memBufPtr = mBufPtr;
            startPtr = *mBufPtr;
        }

        void read(char *buf, u64 bytes)
        {
            memcpy(buf, *memBufPtr, bytes);
            *memBufPtr += bytes;
            bytesReceived += bytes;
            roundsReceived += 1;
        }

        char *read(u64 bytes)
        {
            char *newBuf = *memBufPtr;
            *memBufPtr += bytes;
            bytesReceived += bytes;
            roundsReceived += 1;
            return newBuf;
        }

        void write(char *buf, u64 bytes)
        {
            memcpy(*memBufPtr, buf, bytes);
            *memBufPtr += bytes;
            bytesSent += bytes;
            roundsSent += 1;
        }

        void close()
        {
            // do nothing yet
        }
    };

    class Peer
    {
    public:
        KeyBuf *keyBuf;

        Peer(std::string ip, int port)
        {
            keyBuf = new SocketBuf(ip, port, false);
        }

        Peer(int sendsocket, int recvsocket)
        {
            keyBuf = new SocketBuf(sendsocket, recvsocket);
        }

        Peer(std::string filename)
        {
            keyBuf = new FileBuf(filename, F_WR_ONLY);
        }

        Peer(char **mBufPtr)
        {
            keyBuf = new MemBuf(mBufPtr);
        }

        inline uint64_t bytesSent()
        {
            return keyBuf->bytesSent;
        }

        inline uint64_t bytesReceived()
        {
            return keyBuf->bytesReceived;
        }

        inline uint64_t roundsSent()
        {
            return keyBuf->roundsSent;
        }

        inline uint64_t roundsReceived()
        {
            return keyBuf->roundsReceived;
        }

        void inline zeroBytesSent()
        {
            keyBuf->bytesSent = 0;
        }

        void inline zeroBytesReceived()
        {
            keyBuf->bytesReceived = 0;
        }

        void close();

        void send_seed(block seed)
        {
            send(seed);
        }

        template <typename T>
        void send(const T &val)
        {
            auto buf = wire::serialize_value(val);
            keyBuf->write(buf.data(), (u64)buf.size());
        }

        template <typename T>
        T recv()
        {
            std::array<char, wire::Serializer<T>::size> buf{};
            keyBuf->read(buf.data(), (u64)buf.size());
            return wire::deserialize_value<T>(buf.data());
        }

        template <typename T>
        void send_array(const shark::span<T> &arr)
        {
            keyBuf->write((char *)arr.data(), arr.size() * sizeof(T));
        }

        template <typename T>
        void recv_array(shark::span<T> &arr)
        {
            keyBuf->read((char *)arr.data(), arr.size() * sizeof(T));
        }

        template <typename T>
        shark::span<T> recv_array(u64 size)
        {
            if (keyBuf->isMem())
            {
                // Zero-copy path when the backing store is an in-memory key buffer.
                T *buf = reinterpret_cast<T *>(keyBuf->read(size * sizeof(T)));
                shark::span<T> arr(buf, size);
                return arr;
            }
            // File/socket path: allocate managed storage so memory is reclaimed with span lifetime.
            shark::span<T> arr(size);
            keyBuf->read(reinterpret_cast<char *>(arr.data()), size * sizeof(T));
            return arr;
        }

        void send_u128(u128 val)
        {
            send(val);
        }

        u128 recv_u128()
        {
            return recv<u128>();
        }

        void sync()
        {
            keyBuf->write((char *)"a", 1);
            char ack = 0;
            keyBuf->read(&ack, 1);
        }

    };

    Peer *waitForPeer(int port);

    class Dealer
    {
    public:
        KeyBuf *keyBuf;
        char *_kbuf;

        Dealer(std::string ip, int port)
        {
            keyBuf = new SocketBuf(ip, port, true);
        }

        Dealer(std::string filename, bool oneShot = true)
        {
            if (oneShot)
            {
                // get file size
                std::ifstream file(filename, std::ios::binary | std::ios::ate);
                u64 size = file.tellg();
                file.close();

                _kbuf = new char[size];
                std::ifstream file2(filename, std::ios::binary);
                always_assert(file2.is_open());
                if (!file2.read(_kbuf, size))
                {
                    // Print system error message
                    std::perror("Key File Read Error: ");
                    exit(1);
                }
                always_assert(file2.gcount() == size);
                file2.close();
                keyBuf = new MemBuf(&_kbuf);
            }
            else
            {
                keyBuf = new FileBuf(filename, F_RD_ONLY);
            }
        }

        Dealer(char **mBufPtr)
        {
            keyBuf = new MemBuf(mBufPtr);
        }

        inline uint64_t bytesReceived()
        {
            return keyBuf->bytesReceived;
        }

        void close()
        {
            keyBuf->close();
        }

        block recv_seed()
        {
            return recv<block>();
        }

        template <typename T>
        T recv()
        {
            std::array<char, wire::Serializer<T>::size> buf{};
            shark::utils::start_timer("dealer_read_local");
            keyBuf->read(buf.data(), (u64)buf.size());
            shark::utils::stop_timer("dealer_read_local");
            return wire::deserialize_value<T>(buf.data());
        }

        template <typename T>
        void recv_array(shark::span<T> &arr)
        {
            shark::utils::start_timer("dealer_read_local");
            keyBuf->read((char *)arr.data(), arr.size() * sizeof(T));
            shark::utils::stop_timer("dealer_read_local");
        }

        template <typename T>
        shark::span<T> recv_array(u64 size)
        {
            shark::utils::start_timer("dealer_read_local");
            if (keyBuf->isMem())
            {
                // Zero-copy path when the backing store is an in-memory key buffer.
                T *buf = reinterpret_cast<T *>(keyBuf->read(size * sizeof(T)));
                shark::utils::stop_timer("dealer_read_local");
                shark::span<T> arr(buf, size);
                return arr;
            }
            // File/socket path: allocate managed storage so memory is reclaimed with span lifetime.
            shark::span<T> arr(size);
            keyBuf->read(reinterpret_cast<char *>(arr.data()), size * sizeof(T));
            shark::utils::stop_timer("dealer_read_local");
            return arr;
        }

        u128 recv_u128()
        {
            return recv<u128>();
        }
    };
}
