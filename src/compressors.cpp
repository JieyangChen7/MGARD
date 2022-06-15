#include "compressors.hpp"

#include <cassert>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <zlib.h>

#include "format.hpp"
#include "huffman.hpp"
#include "utilities.hpp"

#ifdef MGARD_ZSTD
#include <zstd.h>
#endif

namespace mgard {

void decompress_memory_huffman(unsigned char const *const src,
                               const std::size_t srcLen, long int *const dst,
                               const std::size_t dstLen) {
  // Dummy header until we change the signature of `decompress_memory_huffman`.
  pb::Header header;
  header.mutable_encoding()->set_compressor(
#ifdef MGARD_ZSTD
      pb::Encoding::CPU_HUFFMAN_ZSTD
#else
      pb::Encoding::CPU_HUFFMAN_ZLIB
#endif
  );
  const HuffmanEncodedStream encoded =
      decompress_deserialize(header, src, srcLen);
  const MemoryBuffer<long int> decoded = huffman_decoding(encoded);
  {
    long int const *const p = decoded.data.get();
    if (decoded.size * sizeof(*p) != dstLen) {
      throw std::runtime_error(
          "mismatch between expected and obtained decompressed buffer sizes");
    }
    std::copy(p, p + decoded.size, dst);
  }
}

MemoryBuffer<unsigned char> compress_memory_huffman(long int const *const src,
                                                    const std::size_t srcLen) {
  const HuffmanEncodedStream encoded = huffman_encoding(src, srcLen);
  // Dummy header until we change the signature of `compress_memory_huffman`.
  pb::Header header;
  header.mutable_encoding()->set_compressor(
#ifdef MGARD_ZSTD
      pb::Encoding::CPU_HUFFMAN_ZSTD
#else
      pb::Encoding::CPU_HUFFMAN_ZLIB
#endif
  );
  return serialize_compress(header, encoded);
}

#ifdef MGARD_ZSTD
/*! CHECK
 * Check that the condition holds. If it doesn't print a message and die.
 */
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d CHECK(%s) failed: ", __FILE__, __LINE__, #cond);  \
      fprintf(stderr, "" __VA_ARGS__);                                         \
      fprintf(stderr, "\n");                                                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/*! CHECK_ZSTD
 * Check the zstd error code and die if an error occurred after printing a
 * message.
 */
/*! CHECK_ZSTD
 * Check the zstd error code and die if an error occurred after printing a
 * message.
 */
#define CHECK_ZSTD(fn, ...)                                                    \
  do {                                                                         \
    size_t const err = (fn);                                                   \
    CHECK(!ZSTD_isError(err), "%s", ZSTD_getErrorName(err));                   \
  } while (0)

MemoryBuffer<unsigned char> compress_memory_zstd(void const *const src,
                                                 const std::size_t srcLen) {
  const size_t cBuffSize = ZSTD_compressBound(srcLen);
  unsigned char *const buffer = new unsigned char[cBuffSize];
  const std::size_t cSize = ZSTD_compress(buffer, cBuffSize, src, srcLen, 1);
  CHECK_ZSTD(cSize);
  return MemoryBuffer<unsigned char>(buffer, cSize);
}
#endif

MemoryBuffer<unsigned char> compress_memory_z(void z_const *const src,
                                              const std::size_t srcLen) {
  const std::size_t BUFSIZE = 2048 * 1024;
  std::vector<Bytef *> buffers;
  std::vector<std::size_t> bufferLengths;

  z_stream strm;
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.next_in = static_cast<Bytef z_const *>(src);
  strm.avail_in = srcLen;
  buffers.push_back(strm.next_out = new Bytef[BUFSIZE]);
  bufferLengths.push_back(strm.avail_out = BUFSIZE);

  deflateInit(&strm, Z_BEST_COMPRESSION);

  while (strm.avail_in != 0) {
    [[maybe_unused]] const int res = deflate(&strm, Z_NO_FLUSH);
    assert(res == Z_OK);
    if (strm.avail_out == 0) {
      buffers.push_back(strm.next_out = new Bytef[BUFSIZE]);
      bufferLengths.push_back(strm.avail_out = BUFSIZE);
    }
  }

  int res = Z_OK;
  while (res == Z_OK) {
    if (strm.avail_out == 0) {
      buffers.push_back(strm.next_out = new Bytef[BUFSIZE]);
      bufferLengths.push_back(strm.avail_out = BUFSIZE);
    }
    res = deflate(&strm, Z_FINISH);
  }

  assert(res == Z_STREAM_END);
  bufferLengths.back() -= strm.avail_out;
  // Could just do `nbuffers * BUFSIZE - strm.avail_out`.
  const std::size_t bufferLen =
      std::accumulate(bufferLengths.begin(), bufferLengths.end(), 0);
  unsigned char *const buffer = new unsigned char[bufferLen];
  {
    const std::size_t nbuffers = buffers.size();
    unsigned char *p = buffer;
    for (std::size_t i = 0; i < nbuffers; ++i) {
      unsigned char const *const buffer = buffers.at(i);
      const std::size_t bufferLength = bufferLengths.at(i);
      std::copy(buffer, buffer + bufferLength, p);
      p += bufferLength;
      delete[] buffer;
    }
  }
  deflateEnd(&strm);

  return MemoryBuffer<unsigned char>(buffer, bufferLen);
}

void decompress_memory_z(void z_const *const src, const std::size_t srcLen,
                         unsigned char *const dst, const std::size_t dstLen) {
  z_stream strm = {};
  strm.total_in = strm.avail_in = srcLen;
  strm.total_out = strm.avail_out = dstLen;
  strm.next_in = static_cast<Bytef z_const *>(src);
  strm.next_out = reinterpret_cast<Bytef *>(dst);

  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;

  [[maybe_unused]] int res;
  res = inflateInit2(&strm, (15 + 32)); // 15 window bits, and the +32 tells
                                        // zlib to to detect if using gzip or
                                        // zlib
  assert(res == Z_OK);
  res = inflate(&strm, Z_FINISH);
  assert(res == Z_STREAM_END);
  res = inflateEnd(&strm);
  assert(res == Z_OK);
}

#ifdef MGARD_ZSTD
void decompress_memory_zstd(void const *const src, const std::size_t srcLen,
                            unsigned char *const dst,
                            const std::size_t dstLen) {
  size_t const dSize = ZSTD_decompress(dst, dstLen, src, srcLen);
  CHECK_ZSTD(dSize);

  /* When zstd knows the content size, it will error if it doesn't match. */
  CHECK(dstLen == dSize, "Impossible because zstd will check this condition!");
}
#endif

MemoryBuffer<unsigned char> compress(const pb::Header &header, void *const src,
                                     const std::size_t srcLen) {
  switch (header.encoding().compressor()) {
  case pb::Encoding::CPU_HUFFMAN_ZSTD:
#ifdef MGARD_ZSTD
  {
    if (header.quantization().type() != mgard::pb::Quantization::INT64_T) {
      throw std::runtime_error("Huffman tree not implemented for quantization "
                               "types other than `std::int64_t`");
    }
    // Quantization type size.
    const std::size_t qts = quantization_buffer(header, 1).size;
    if (srcLen % qts) {
      throw std::runtime_error("incorrect quantization buffer size");
    }
    return compress_memory_huffman(reinterpret_cast<long int *>(src),
                                   srcLen / qts);
  }
#else
    throw std::runtime_error("MGARD compiled without ZSTD support");
#endif
  case pb::Encoding::CPU_HUFFMAN_ZLIB:
    return compress_memory_z(src, srcLen);
  default:
    throw std::runtime_error("unrecognized lossless compressor");
  }
}

void decompress(const pb::Header &header, void *const src,
                const std::size_t srcLen, void *const dst,
                const std::size_t dstLen) {
  switch (read_encoding_compressor(header)) {
  case pb::Encoding::NOOP_COMPRESSOR:
    if (srcLen != dstLen) {
      throw std::invalid_argument(
          "source and destination lengths must be equal");
    }
    {
      unsigned char const *const p = static_cast<unsigned char const *>(src);
      unsigned char *const q = static_cast<unsigned char *>(dst);
      std::copy(p, p + srcLen, q);
    }
    break;
  case pb::Encoding::CPU_HUFFMAN_ZLIB:
    decompress_memory_z(const_cast<void z_const *>(src), srcLen,
                        static_cast<unsigned char *>(dst), dstLen);
    break;
  case pb::Encoding::CPU_HUFFMAN_ZSTD:
#ifdef MGARD_ZSTD
    decompress_memory_huffman(static_cast<unsigned char *>(src), srcLen,
                              static_cast<long int *>(dst), dstLen);
    break;
#else
    throw std::runtime_error("MGARD compiled without ZSTD support");
#endif
  default:
    throw std::runtime_error("unsupported lossless encoder");
  }
}

} // namespace mgard
