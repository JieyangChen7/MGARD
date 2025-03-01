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
  compress_level(std::vector<SIZE> &bitplane_sizes,
                 Array<2, T_bitplane, DeviceType> &encoded_bitplanes,
                 std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
                 int queue_idx) {

    SubArray<2, T_bitplane, DeviceType> encoded_bitplanes_subarray(
        encoded_bitplanes);
    for (SIZE bitplane_idx = 0;
         bitplane_idx < encoded_bitplanes_subarray.shape(0); bitplane_idx++) {
      T_bitplane *bitplane = encoded_bitplanes_subarray(bitplane_idx, 0);

      Array<1, Byte, DeviceType> compressed_bitplane(
          {bitplane_sizes[bitplane_idx]});
      MemoryManager<DeviceType>::Copy1D(
          compressed_bitplane.data(), (Byte *)bitplane,
          bitplane_sizes[bitplane_idx], queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      compressed_bitplanes[bitplane_idx] = compressed_bitplane;
      bitplane_sizes[bitplane_idx] = bitplane_sizes[bitplane_idx];
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

    for (SIZE bitplane_idx = starting_bitplane;
         bitplane_idx < starting_bitplane + num_bitplanes; bitplane_idx++) {
      // std::cout << "decompress level: " << bitplane_idx << " " <<
      // (int)num_bitplanes << "\n";
      T_bitplane *bitplane = encoded_bitplanes_subarray(bitplane_idx, 0);
      // MDR::Zstd
      // SIZE compressed_size = bitplane_sizes[starting_bitplane +
      // bitplane_idx]; Byte *compressed_host = new Byte[compressed_size];
      // MemoryManager<DeviceType>::Copy1D(
      //     compressed_host,
      //     compressed_bitplanes[starting_bitplane + bitplane_idx].data(),
      //     compressed_size, 0);
      // DeviceRuntime<DeviceType>::SyncQueue(0);

      // Byte *bitplane_host = NULL;
      // SIZE decompressed_size = ::MDR::ZSTD::decompress(
      //     compressed_host, compressed_size, &bitplane_host);

      // MemoryManager<DeviceType>::Copy1D(bitplane, (T_bitplane
      // *)bitplane_host,
      //                                   decompressed_size /
      //                                   sizeof(T_bitplane), 0);
      // DeviceRuntime<DeviceType>::SyncQueue(0);

      // Huffman
      // Array<1, T_bitplane, DeviceType>
      // encoded_bitplane({encoded_bitplanes_subarray.shape(1)}, bitplane);
      // huffman.Decompress(compressed_bitplanes[bitplane_idx],
      // encoded_bitplane, queue_idx);
      // int old_log_level = log::level;
      // log::level = log::ERR;
      // ZstdDecompress(compressed_bitplanes[bitplane_idx]);
      // log::level = old_log_level;
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