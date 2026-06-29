/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "BlockDelta/BlockDelta.hpp"
#include "CPU.hpp"
#include "LZ4/LZ4.hpp"
#include "LosslessCompressorInterface.hpp"
#include "ParallelHuffman/Huffman.hpp"
#include "Zstd.hpp"

#ifndef MGARD_X_LOSSLESS_TEMPLATE_HPP
#define MGARD_X_LOSSLESS_TEMPLATE_HPP

namespace mgard_x {

template <typename T, typename H, typename DeviceType>
class ComposedLosslessCompressor
    : public LosslessCompressorInterface<T, DeviceType> {
public:
  using S = typename std::make_signed<T>::type;
  using Q = typename std::make_unsigned<T>::type;

  ComposedLosslessCompressor() : initialized(false) {}

  // Whether the configured lossless path actually uses the (workspace-heavy)
  // Huffman backend. BlockDelta and standalone LZ4 have their own
  // (de)compressors and never touch the Huffman workspace; every other type is
  // Huffman / Huffman+LZ4 / Huffman+Zstd (or CPU_Lossless falling through to
  // Huffman in Compress()).
  static bool uses_huffman(enum lossless_type lossless) {
    return lossless != lossless_type::BlockDelta &&
           lossless != lossless_type::LZ4;
  }

  ComposedLosslessCompressor(SIZE n, Config config)
      : initialized(true), n(n), config(config) {
    static_assert(!std::is_floating_point<T>::value,
                  "ComposedLosslessCompressor: Type of T must be integer.");
    if (uses_huffman(config.lossless)) {
      huffman.Resize(n, config.huff_dict_size, config.huff_block_size,
                     config.estimate_outlier_ratio, 0);
    }
    if (config.lossless == lossless_type::Huffman_LZ4) {
      lz4.Resize(n * sizeof(H), config.lz4_block_size, 0);
    }
    if (config.lossless == lossless_type::LZ4) {
      lz4.Resize(n * sizeof(T), config.lz4_block_size, 0);
    }
    if (config.lossless == lossless_type::Huffman_Zstd) {
      zstd.Resize(n * sizeof(H), config.zstd_compress_level, 0);
    }
    if (config.lossless == lossless_type::BlockDelta) {
      blockdelta.Resize(n, config.block_delta_block_size,
                        config.block_delta_mode, 0);
    }
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  void Adapt(SIZE n, Config config, int queue_idx) {
    this->initialized = true;
    this->n = n;
    this->config = config;
    if (uses_huffman(config.lossless)) {
      huffman.Resize(n, config.huff_dict_size, config.huff_block_size,
                     config.estimate_outlier_ratio, queue_idx);
    }
    if (config.lossless == lossless_type::Huffman_LZ4) {
      lz4.Resize(n * sizeof(H), config.lz4_block_size, queue_idx);
    }
    if (config.lossless == lossless_type::LZ4) {
      lz4.Resize(n * sizeof(T), config.lz4_block_size, queue_idx);
    }
    if (config.lossless == lossless_type::Huffman_Zstd) {
      zstd.Resize(n * sizeof(H), config.zstd_compress_level, queue_idx);
    }
    if (config.lossless == lossless_type::BlockDelta) {
      blockdelta.Resize(n, config.block_delta_block_size,
                        config.block_delta_mode, queue_idx);
    }
  }

  static size_t EstimateMemoryFootprint(SIZE primary_count, Config config) {
    size_t size = 0;
    if (uses_huffman(config.lossless)) {
      size += Huffman<Q, S, H, DeviceType>::EstimateMemoryFootprint(
          primary_count, config.huff_dict_size, config.huff_block_size,
          config.estimate_outlier_ratio);
    }
    if (config.lossless == lossless_type::Huffman_LZ4) {
      size += LZ4<DeviceType>::EstimateMemoryFootprint(
          primary_count * sizeof(H), config.lz4_block_size);
    }
    if (config.lossless == lossless_type::LZ4) {
      size += LZ4<DeviceType>::EstimateMemoryFootprint(
          primary_count * sizeof(T), config.lz4_block_size);
    }
    if (config.lossless == lossless_type::Huffman_Zstd) {
      size +=
          Zstd<DeviceType>::EstimateMemoryFootprint(primary_count * sizeof(H));
    }
    if (config.lossless == lossless_type::BlockDelta) {
      size += BlockDeltaLossless<T, DeviceType>::EstimateMemoryFootprint(
          primary_count, config.block_delta_block_size);
    }
    return size;
  }

  void Compress(Array<1, T, DeviceType> &original_data,
                Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {

    if (config.lossless == lossless_type::BlockDelta) {
      blockdelta.Compress(original_data, compressed_data, queue_idx);
      return;
    }

    if (config.lossless == lossless_type::LZ4) {
      // LZ4 directly on the raw quantized stream (no Huffman). View the T array
      // as bytes into compressed_data, then compress it in place.
      SIZE nbytes = original_data.shape(0) * sizeof(T);
      compressed_data.resize({nbytes}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(compressed_data.data(),
                                        (Byte *)original_data.data(), nbytes,
                                        queue_idx);
      lz4.Compress(compressed_data, queue_idx);
      return;
    }

    huffman.Compress(original_data, compressed_data, 0.0, queue_idx);

    if (config.lossless == lossless_type::Huffman_LZ4) {
      huffman.Serialize(compressed_data, queue_idx);
      lz4.Compress(compressed_data, queue_idx);
    }

    if (config.lossless == lossless_type::Huffman_Zstd) {
      huffman.Serialize(compressed_data, queue_idx);
      zstd.Compress(compressed_data, queue_idx);
    }
  }

  void Serialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    if (config.lossless == lossless_type::Huffman) {
      huffman.Serialize(compressed_data, queue_idx);
    }
    if (config.lossless == lossless_type::BlockDelta) {
      blockdelta.Serialize(compressed_data, queue_idx);
    }
  }

  void Deserialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    if (config.lossless == lossless_type::Huffman) {
      huffman.Deserialize(compressed_data, queue_idx);
    }
    if (config.lossless == lossless_type::BlockDelta) {
      blockdelta.Deserialize(compressed_data, queue_idx);
    }
  }

  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  Array<1, T, DeviceType> &decompressed_data, int queue_idx) {

    if (config.lossless == lossless_type::BlockDelta) {
      // Deserialize (the memory-movement stage) was already run separately by
      // the pipeline; Decompress is computation only.
      blockdelta.Decompress(compressed_data, decompressed_data, queue_idx);
      return;
    }

    if (config.lossless == lossless_type::LZ4) {
      // Inverse of the standalone LZ4 path: decompress to the raw byte stream,
      // then reinterpret it back into the quantized T array.
      lz4.Decompress(compressed_data, queue_idx);
      SIZE nbytes = compressed_data.shape(0);
      decompressed_data.resize({(SIZE)(nbytes / sizeof(T))}, queue_idx);
      MemoryManager<DeviceType>::Copy1D((Byte *)decompressed_data.data(),
                                        compressed_data.data(), nbytes,
                                        queue_idx);
      return;
    }

    if (config.lossless == lossless_type::Huffman_LZ4) {
      lz4.Decompress(compressed_data, queue_idx);
      huffman.Deserialize(compressed_data, queue_idx);
    }

    if (config.lossless == lossless_type::Huffman_Zstd) {
      zstd.Decompress(compressed_data, queue_idx);
      huffman.Deserialize(compressed_data, queue_idx);
    }

    huffman.Decompress(compressed_data, decompressed_data, queue_idx);
  }

  bool initialized;
  SIZE n;
  Config config;
  Huffman<Q, S, H, DeviceType> huffman;
  LZ4<DeviceType> lz4;
  Zstd<DeviceType> zstd;
  BlockDeltaLossless<T, DeviceType> blockdelta;
};

} // namespace mgard_x

#endif
