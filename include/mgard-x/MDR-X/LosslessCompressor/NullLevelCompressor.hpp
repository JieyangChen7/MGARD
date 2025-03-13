#ifndef _MDR_NULL_LEVEL_COMPRESSOR_HPP
#define _MDR_NULL_LEVEL_COMPRESSOR_HPP

#include "LevelCompressorInterface.hpp"
namespace MDR {
// Null lossless compressor
class NullLevelCompressor : public concepts::LevelCompressorInterface {
public:
  NullLevelCompressor() {}
  uint8_t compress_level(std::vector<uint8_t *> &streams,
                         std::vector<uint32_t> &stream_sizes) const {
    return 0;
  }
  void decompress_level(std::vector<const uint8_t *> &streams,
                        const std::vector<uint32_t> &stream_sizes,
                        uint8_t starting_bitplane, uint8_t num_bitplanes,
                        uint8_t stopping_index) {}
  void decompress_release() {}
  void print() const { std::cout << "Null level compressor" << std::endl; }
};
} // namespace MDR

namespace mgard_x {
namespace MDR {

// interface for lossless compressor
template <typename T_bitplane, typename DeviceType>
class NullLevelCompressor
    : public concepts::LevelCompressorInterface<T_bitplane, DeviceType> {
public:
  NullLevelCompressor() : initialized(false) {}
  NullLevelCompressor(SIZE max_n, Config config) {
    Adapt(max_n, config, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }
  ~NullLevelCompressor(){};

  void Adapt(SIZE max_n, Config config, int queue_idx) {
    this->initialized = true;
    this->config = config;
  }

  static size_t EstimateMemoryFootprint(SIZE max_n, Config config) {
    size_t size = 0;
    size += Huffman<T_bitplane, T_bitplane, HUFFMAN_CODE, DeviceType>::
        EstimateMemoryFootprint(max_n, config.huff_dict_size,
                                config.huff_block_size,
                                config.estimate_outlier_ratio);
    return size;
  }
  // compress level, overwrite and free original streams; rewrite streams sizes
  void
  compress_level(SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
                 std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
                 int queue_idx) {

    for (SIZE bitplane_idx = 0; bitplane_idx < encoded_bitplanes.shape(0);
         bitplane_idx++) {
      T_bitplane *bitplane = encoded_bitplanes(bitplane_idx, 0);

      compressed_bitplanes[bitplane_idx].resize(
          {encoded_bitplanes.shape(1) * sizeof(T_bitplane)});
      MemoryManager<DeviceType>::Copy1D(
          compressed_bitplanes[bitplane_idx].data(), (Byte *)bitplane,
          encoded_bitplanes.shape(1) * sizeof(T_bitplane), queue_idx);
    }
  }

  // decompress level, create new buffer and overwrite original streams; will
  // not change stream sizes
  void decompress_level(
      std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
      SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
      uint8_t starting_bitplane, uint8_t num_bitplanes, int queue_idx) {

    for (SIZE bitplane_idx = starting_bitplane;
         bitplane_idx < starting_bitplane + num_bitplanes; bitplane_idx++) {
      T_bitplane *bitplane = encoded_bitplanes(bitplane_idx, 0);
      MemoryManager<DeviceType>::Copy1D(
          (uint8_t *)bitplane, compressed_bitplanes[bitplane_idx].data(),
          compressed_bitplanes[bitplane_idx].shape(0), queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    }
  }

  // release the buffer created
  void decompress_release() {}

  void print() const {}

  bool initialized;
  Config config;
};

} // namespace MDR
} // namespace mgard_x
#endif