#ifndef _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP
#define _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP

#include "../../Lossless/ParallelHuffman/Huffman.hpp"
#include "../../Lossless/ParallelRLE/RunLengthEncoding.hpp"
#include "../../Lossless/Zstd.hpp"
// #include "../RefactorUtils.hpp"
#include "LevelCompressorInterface.hpp"
#include "LosslessCompressor.hpp"

namespace mgard_x {
namespace MDR {

struct HUFFMAN {};
struct RLE {};

// interface for lossless compressor
template <typename T_bitplane, typename CompressorType, typename DeviceType>
class DefaultLevelCompressor
    : public concepts::LevelCompressorInterface<T_bitplane, DeviceType> {
public:
  using T_compress = u_int8_t;
  // using T_compress = u_int16_t;

  static constexpr int byte_ratio = sizeof(T_bitplane) / sizeof(T_compress);
  static constexpr int _huff_dict_size = 256;
  static constexpr int _huff_block_size = 1024;
  static constexpr int num_merged_bitplanes = 4;

  DefaultLevelCompressor() : initialized(false) {}
  DefaultLevelCompressor(SIZE max_n, Config config) {
    this->initialized = true;
    Adapt(max_n * byte_ratio, config, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }
  ~DefaultLevelCompressor(){};

  void Adapt(SIZE max_n, Config config, int queue_idx) {
    this->initialized = true;
    this->config = config;
    if constexpr (std::is_same<CompressorType, HUFFMAN>::value) {
      huffman.Resize(max_n * byte_ratio * num_merged_bitplanes, _huff_dict_size,
                     _huff_block_size, config.estimate_outlier_ratio,
                     queue_idx);
    }
    if constexpr (std::is_same<CompressorType, RLE>::value) {
      rle.Resize(max_n * byte_ratio * num_merged_bitplanes, queue_idx);
    }
  }
  static size_t EstimateMemoryFootprint(SIZE max_n, Config config) {
    size_t size = 0;
    if constexpr (std::is_same<CompressorType, HUFFMAN>::value) {
      size += Huffman<T_bitplane, T_bitplane, HUFFMAN_CODE, DeviceType>::
          EstimateMemoryFootprint(max_n * byte_ratio * num_merged_bitplanes,
                                  _huff_dict_size, _huff_block_size,
                                  config.estimate_outlier_ratio);
    }
    if constexpr (std::is_same<CompressorType, RLE>::value) {
      size += parallel_rle::RunLengthEncoding<
          T_compress, u_int32_t, u_int32_t,
          DeviceType>::EstimateMemoryFootprint(max_n * byte_ratio *
                                               num_merged_bitplanes);
    }
    return size;
  }
  // compress level, overwrite and free original streams; rewrite streams sizes
  void
  compress_level(SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
                 std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
                 int queue_idx) {

    if (0) {

      parallel_rle::RunLengthEncoding<T_compress, u_int32_t, u_int32_t,
                                      DeviceType>
          rle;

      std::vector<T_compress> data = {1,  2,  3,  6,  6,  6, 5, 5, 10, 10,
                                      10, 23, 23, 23, 23, 1, 2, 3, 4,  6};
      rle.Resize(data.size(), queue_idx);
      Array<1, T_compress, DeviceType> original_data({data.size()});
      Array<1, T_compress, DeviceType> decompressed_data({data.size()});
      MemoryManager<DeviceType>::Copy1D(original_data.data(), data.data(),
                                        data.size(), queue_idx);
      Array<1, Byte, DeviceType> compressed_data(
          {data.size() * sizeof(T_compress)});
      rle.Compress(original_data, compressed_data, queue_idx);

      rle.Deserialize(compressed_data, queue_idx);
      rle.Decompress(compressed_data, decompressed_data, queue_idx);

      exit(0);
    }
    std::vector<float> cr;
    for (SIZE bitplane_idx = 0; bitplane_idx < encoded_bitplanes.shape(0);
         bitplane_idx++) {
      if (bitplane_idx % num_merged_bitplanes == 0) {
        SIZE merged_bitplane_size =
            encoded_bitplanes.shape(1) * byte_ratio * num_merged_bitplanes;
        // Timer timer;
        // timer.start();
        T_compress *bitplane = (T_compress *)encoded_bitplanes(bitplane_idx, 0);

        Array<1, T_compress, DeviceType> encoded_bitplane(
            {merged_bitplane_size}, bitplane);

        if constexpr (std::is_same<CompressorType, HUFFMAN>::value) {
          ATOMIC_IDX zero = 0;
          MemoryManager<DeviceType>::Copy1D(
              huffman.workspace.outlier_count_subarray.data(), &zero, 1,
              queue_idx);
          MemoryManager<DeviceType>::Copy1D(
              &huffman.outlier_count,
              huffman.workspace.outlier_count_subarray.data(), 1, queue_idx);
          int old_log_level = log::level;
          log::level = 0;

          huffman.CompressPrimary(
              encoded_bitplane, compressed_bitplanes[bitplane_idx], queue_idx);
          huffman.Serialize(compressed_bitplanes[bitplane_idx], queue_idx);
          log::level = old_log_level;
        }

        if constexpr (std::is_same<CompressorType, RLE>::value) {
          rle.Compress(encoded_bitplane, compressed_bitplanes[bitplane_idx],
                       queue_idx);
          rle.Serialize(compressed_bitplanes[bitplane_idx], queue_idx);
        }

        // cr.push_back((float)merged_bitplane_size /
        //               compressed_bitplanes[bitplane_idx].shape(0));

        // timer.end();
        // timer.print("Compressing bitplane", merged_bitplane_size);
        // timer.clear();
      } else {
        compressed_bitplanes[bitplane_idx].resize({1}, queue_idx);
      }
    }
    // std::string cr_string = "";
    // for (auto x : cr) {
    //   cr_string += std::to_string(x) + " ";
    // }
    // log::info("CR: " + cr_string);
  }

  // decompress level, create new buffer and overwrite original streams; will
  // not change stream sizes
  void decompress_level(
      std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
      SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
      uint8_t starting_bitplane, uint8_t num_bitplanes, int queue_idx) {

    for (SIZE bitplane_idx = starting_bitplane;
         bitplane_idx < starting_bitplane + num_bitplanes; bitplane_idx++) {

      if (bitplane_idx % num_merged_bitplanes == 0) {
        T_compress *bitplane = (T_compress *)encoded_bitplanes(bitplane_idx, 0);
        SIZE merged_bitplane_size =
            encoded_bitplanes.shape(1) * byte_ratio * num_merged_bitplanes;

        Array<1, T_compress, DeviceType> encoded_bitplane(
            {merged_bitplane_size}, bitplane);

        if constexpr (std::is_same<CompressorType, HUFFMAN>::value) {
          int old_log_level = log::level;
          log::level = 0;
          huffman.Deserialize(compressed_bitplanes[bitplane_idx], queue_idx);
          huffman.DecompressPrimary(compressed_bitplanes[bitplane_idx],
                                    encoded_bitplane, queue_idx);
          log::level = old_log_level;
        }

        if constexpr (std::is_same<CompressorType, RLE>::value) {
          rle.Deserialize(compressed_bitplanes[bitplane_idx], queue_idx);
          rle.Decompress(compressed_bitplanes[bitplane_idx], encoded_bitplane,
                         queue_idx);
        }
      }
    }
  }

  // release the buffer created
  void decompress_release() {}

  void print() const {}
  bool initialized;
  Huffman<T_compress, T_compress, uint64_t, DeviceType> huffman;
  parallel_rle::RunLengthEncoding<T_compress, u_int32_t, u_int32_t, DeviceType>
      rle;
  Config config;
};

} // namespace MDR
} // namespace mgard_x
#endif
