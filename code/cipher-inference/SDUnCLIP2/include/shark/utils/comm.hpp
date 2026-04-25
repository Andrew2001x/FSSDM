#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <shark/types/u128.hpp>
#include <shark/types/span.hpp>
#include <shark/utils/assert.hpp>

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

    class FileBuf : public KeyBuf
    {
    public:
        std::string filename;
        FileMode mode;
        std::fstream file;
        std::vector<char> ioBuffer;

        static size_t env_mb_or(const char *name, size_t default_mb)
        {
            const char *env = std::getenv(name);
            if (env == nullptr || *env == '\0')
            {
                return default_mb;
            }
            char *end = nullptr;
            unsigned long long parsed = std::strtoull(env, &end, 10);
            if (end == env || parsed == 0)
            {
                return default_mb;
            }
            return (size_t)parsed;
        }

        FileBuf(std::string filename, FileMode f)
        {
            printf("Opening file=%s, mode=%d\n", filename.data(), f);
            this->t = BUF_FILE;
            this->filename = std::move(filename);
            this->mode = f;
            const size_t io_buffer_mb =
                env_mb_or("SHARK_KEYBUF_IO_MB", (f == F_WR_ONLY) ? 64 : 8);
            ioBuffer.resize(io_buffer_mb * 1024ull * 1024ull);

            if (f == F_WR_ONLY)
            {
                file.rdbuf()->pubsetbuf(ioBuffer.data(), (std::streamsize)ioBuffer.size());
                this->file.open(this->filename, std::ios::out | std::ios::binary | std::ios::trunc);
            }
            else
            {
                file.rdbuf()->pubsetbuf(ioBuffer.data(), (std::streamsize)ioBuffer.size());
                this->file.open(this->filename, std::ios::in | std::ios::binary);
            }
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
            keyBuf->write((char *)&seed, sizeof(block));
        }

        template <typename T>
        void send(T val)
        {
            keyBuf->write((char *)&val, sizeof(T));
        }

        template <typename T>
        T recv()
        {
            T val;
            keyBuf->read((char *)&val, sizeof(T));
            return val;
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
            keyBuf->write((char *)&val, sizeof(u128));
        }

        u128 recv_u128()
        {
            u128 val;
            keyBuf->read((char *)&val, sizeof(u128));
            return val;
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
            block seed;
            keyBuf->read((char *)&seed, sizeof(block));
            return seed;
        }

        template <typename T>
        T recv()
        {
            T val;
            keyBuf->read((char *)&val, sizeof(T));
            return val;
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

        u128 recv_u128()
        {
            u128 val;
            keyBuf->read((char *)&val, sizeof(u128));
            return val;
        }
    };
}
