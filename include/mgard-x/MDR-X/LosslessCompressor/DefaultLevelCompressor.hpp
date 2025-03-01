#ifndef _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP
#define _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP

#include "../../Lossless/ParallelHuffman/Huffman.hpp"
#include "../../Lossless/Zstd.hpp"
// #include "../RefactorUtils.hpp"
#include "LevelCompressorInterface.hpp"
#include "LosslessCompressor.hpp"

namespace mgard_x {
namespace MDR {

// interface for lossless compressor
template <typename T_bitplane, typename DeviceType>
class DefaultLevelCompressor
    : public concepts::LevelCompressorInterface<T_bitplane, DeviceType> {
public:
  DefaultLevelCompressor() : initialized(false) {}
  DefaultLevelCompressor(SIZE max_n, Config config) {
    Adapt(max_n, config, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }
  ~DefaultLevelCompressor(){};

  void Adapt(SIZE max_n, Config config, int queue_idx) {
    this->initialized = true;
    this->config = config;
    huffman.Resize(max_n, config.huff_dict_size, config.huff_block_size,
                   config.estimate_outlier_ratio, queue_idx);
    zstd.Resize(max_n * sizeof(T_bitplane), config.zstd_compress_level,
                queue_idx);
  }
  static size_t EstimateMemoryFootprint(SIZE max_n, Config config) {
    size_t size = 0;
    size += Huffman<T_bitplane, T_bitplane, HUFFMAN_CODE, DeviceType>::
        EstimateMemoryFootprint(max_n, config.huff_dict_size,
                                config.huff_block_size,
                                config.estimate_outlier_ratio);
    size +=
        Zstd<DeviceType>::EstimateMemoryFootprint(max_n * sizeof(T_bitplane));
    return size;
  }
  // compress level, overwrite and free original streams; rewrite streams sizes
  void
  compress_level(std::vector<SIZE> &bitplane_sizes,
                 Array<2, T_bitplane, DeviceType> &encoded_bitplanes,
                 std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
                 int queue_idx) {

    SubArray<2, T_bitplane, DeviceType> encoded_bitplanes_subarray(
        encoded_bitplanes);

    for (SIZE bitplane_idx = 0;
         bitplane_idx < encoded_bitplanes_subarray.shape(0); bitplane_idx++) {
      T_bitplane *bitplane = encoded_bitplanes_subarray(bitplane_idx, 0);
      // Huffman
      Adapt(encoded_bitplanes_subarray.shape(1), config, queue_idx);
      Array<1, T_bitplane, DeviceType> encoded_bitplane(
          {encoded_bitplanes_subarray.shape(1)}, bitplane);
      huffman.Compress(encoded_bitplane, compressed_bitplanes[bitplane_idx],
                       queue_idx);
      huffman.Serialize(compressed_bitplanes[bitplane_idx], queue_idx);
      bitplane_sizes[bitplane_idx] =
          compressed_bitplanes[bitplane_idx].shape(0);

      // Array<1, Byte, DeviceType> compressed_bitplane(
      //     {bitplane_sizes[bitplane_idx]});
      // MemoryManager<DeviceType>::Copy1D(
      //     compressed_bitplane.data(), (uint8_t *)bitplane,
      //     bitplane_sizes[bitplane_idx], queue_idx);
      // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      // int old_log_level = log::level;
      // log::level = log::ERR;
      // zstd.Compress(compressed_bitplane, queue_idx);
      // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      // log::level = old_log_level;
      // compressed_bitplanes[bitplane_idx] = compressed_bitplane;
      // bitplane_sizes[bitplane_idx] = compressed_bitplane.shape(0);
    }
  }

  // decompress level, create new buffer and overwrite original streams; will
  // not change stream sizes
  void decompress_level(
      std::vector<SIZE> &bitplane_sizes,
      std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
      Array<2, T_bitplane, DeviceType> &encoded_bitplanes,
      uint8_t starting_bitplane, uint8_t num_bitplanes, int queue_idx) {

    SubArray<2, T_bitplane, DeviceType> encoded_bitplanes_subarray(
        encoded_bitplanes);

    for (SIZE bitplane_idx = starting_bitplane; bitplane_idx < num_bitplanes;
         bitplane_idx++) {
      T_bitplane *bitplane = encoded_bitplanes_subarray(bitplane_idx, 0);

      // Huffman
      Adapt(encoded_bitplanes_subarray.shape(1), config, queue_idx);
      Array<1, T_bitplane, DeviceType> encoded_bitplane(
          {encoded_bitplanes_subarray.shape(1)}, bitplane);
      huffman.Deserialize(compressed_bitplanes[bitplane_idx], queue_idx);
      huffman.Decompress(compressed_bitplanes[bitplane_idx], encoded_bitplane,
                         queue_idx);

      // std::cout << "decompress level: " << bitplane_idx << "\n";
      // int old_log_level = log::level;
      // log::level = log::ERR;
      // zstd.Decompress(compressed_bitplanes[bitplane_idx], queue_idx);
      // log::level = old_log_level;
      // MemoryManager<DeviceType>::Copy1D(
      //     (uint8_t *)bitplane, compressed_bitplanes[bitplane_idx].data(),
      //     compressed_bitplanes[bitplane_idx].shape(0), queue_idx);
      // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    }
  }

  // release the buffer created
  void decompress_release() {}

  void print() const {}
  bool initialized;
  Huffman<T_bitplane, T_bitplane, HUFFMAN_CODE, DeviceType> huffman;
  Zstd<DeviceType> zstd;
  Config config;
};

} // namespace MDR
} // namespace mgard_x
#endif
