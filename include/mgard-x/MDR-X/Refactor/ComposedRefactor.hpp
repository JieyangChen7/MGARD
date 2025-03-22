#ifndef _MDR_COMPOSED_REFACTOR_HPP
#define _MDR_COMPOSED_REFACTOR_HPP

#include "../BitplaneEncoder/BitplaneEncoder.hpp"
#include "../Decomposer/Decomposer.hpp"
#include "../ErrorCollector/ErrorCollector.hpp"
#include "../Interleaver/Interleaver.hpp"
#include "../LosslessCompressor/LevelCompressor.hpp"
// #include "../RefactorUtils.hpp"
#include "../Writer/Writer.hpp"
#include "RefactorInterface.hpp"
// #include "../DataStructures/MDRData.hpp"
#include <algorithm>
#include <iostream>
namespace mgard_x {
namespace MDR {
// a decomposition-based scientific data refactor: compose a refactor using
// decomposer, interleaver, encoder, and error collector
template <DIM D, typename T_data, typename DeviceType>
class ComposedRefactor
    : public concepts::RefactorInterface<D, T_data, DeviceType> {
public:
  constexpr static bool CONTROL_L2 = false;
  constexpr static bool NegaBinary = false;
  using HierarchyType = Hierarchy<D, T_data, DeviceType>;
  using T_bitplane = uint32_t;
  using T_error = double;
  using Basis = Orthogonal;
  // using Basis = Hierarchical;
  using Decomposer = MGARDDecomposer<D, T_data, Basis, DeviceType>;
  using Interleaver = DirectInterleaver<D, T_data, DeviceType>;
  // using Encoder = GroupedBPEncoder<D, T_data, T_bitplane, T_error,
  // CONTROL_L2, DeviceType>;
  using Encoder = BPEncoderOptV1<D, T_data, T_bitplane, T_error, NegaBinary,
                                 CONTROL_L2, DeviceType>;
  // using Compressor = DefaultLevelCompressor<T_bitplane, HUFFMAN, DeviceType>;
  // using Compressor = DefaultLevelCompressor<T_bitplane, RLE, DeviceType>;
  using Compressor = HybridLevelCompressor<T_bitplane, DeviceType>;
  // using Compressor = NullLevelCompressor<T_bitplane, DeviceType>;

  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;

  ComposedRefactor() : initialized(false) {}

  ComposedRefactor(Hierarchy<D, T_data, DeviceType> &hierarchy, Config config) {
    Adapt(hierarchy, config, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  static SIZE MaxOutputDataSize(std::vector<SIZE> shape, Config config) {
    Hierarchy<D, T_data, DeviceType> hierarchy;
    hierarchy.EstimateMemoryFootprint(shape);
    SIZE size = 0;
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += Encoder::MAX_BITPLANES *
              Encoder::bitplane_length(hierarchy.level_num_elems(level_idx)) *
              sizeof(T_bitplane);
    }
    return size;
  }

  ~ComposedRefactor() {}

  void Adapt(Hierarchy<D, T_data, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    decomposer.Adapt(hierarchy, config, queue_idx);
    interleaver.Adapt(hierarchy, queue_idx);
    encoder.Adapt(hierarchy, queue_idx);
    // batched_encoder.Adapt(hierarchy, queue_idx);
    compressor.Adapt(encoder.bitplane_length(
                         hierarchy.level_num_elems(hierarchy.l_target())),
                     hierarchy.l_target() + 1, Encoder::MAX_BITPLANES, config,
                     queue_idx);

    level_data_array.resize(hierarchy.l_target() + 1);
    level_data_subarray.resize(hierarchy.l_target() + 1);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      level_data_array[level_idx].resize({hierarchy.level_num_elems(level_idx)},
                                         queue_idx);
      level_data_subarray[level_idx] =
          SubArray<1, T_data, DeviceType>(level_data_array[level_idx]);
    }
    abs_max_result_array.resize({1}, queue_idx);
    DeviceCollective<DeviceType>::AbsMax(
        hierarchy.level_num_elems(hierarchy.l_target()),
        SubArray<1, T_data, DeviceType>(), SubArray<1, T_data, DeviceType>(),
        abs_max_workspace, false, 0);
    encoded_bitplanes_array.resize(hierarchy.l_target() + 1);
    encoded_bitplanes_subarray.resize(hierarchy.l_target() + 1);
    level_num_elems.resize(hierarchy.l_target() + 1);
    level_errors_array.resize(hierarchy.l_target() + 1);
    level_errors_subarray.resize(hierarchy.l_target() + 1);
    exp.resize(hierarchy.l_target() + 1);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      encoded_bitplanes_array[level_idx].resize(
          {(SIZE)Encoder::MAX_BITPLANES,
           encoder.bitplane_length(hierarchy.level_num_elems(level_idx))},
          queue_idx);
      encoded_bitplanes_subarray[level_idx] =
          SubArray<2, T_bitplane, DeviceType>(
              encoded_bitplanes_array[level_idx]);
      level_num_elems[level_idx] = hierarchy.level_num_elems(level_idx);
      level_errors_array[level_idx].resize({(SIZE)Encoder::MAX_BITPLANES + 1},
                                           queue_idx);
      level_errors_subarray[level_idx] =
          SubArray<1, T_error, DeviceType>(level_errors_array[level_idx]);
    }
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape,
                                        Config config) {
    Hierarchy<D, T_data, DeviceType> hierarchy;
    size_t size = 0;
    size += hierarchy.EstimateMemoryFootprint(shape);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += hierarchy.level_num_elems(level_idx) * sizeof(T_data);
    }
    size += sizeof(T_data);
    Array<1, Byte, DeviceType> tmp;
    DeviceCollective<DeviceType>::AbsMax(
        hierarchy.level_num_elems(hierarchy.l_target()),
        SubArray<1, T_data, DeviceType>(), SubArray<1, T_data, DeviceType>(),
        tmp, false, 0);
    size += tmp.shape(0);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += Encoder::MAX_BITPLANES *
              Encoder::bitplane_length(hierarchy.level_num_elems(level_idx)) *
              sizeof(T_bitplane);
      size += sizeof(T_error) * (Encoder::MAX_BITPLANES + 1);
    }

    SIZE max_n = Encoder::bitplane_length(
        hierarchy.level_num_elems(hierarchy.l_target()));

    size += (Encoder::MAX_BITPLANES + 1) * sizeof(T_error);
    size += Decomposer::EstimateMemoryFootprint(shape);
    size += Interleaver::EstimateMemoryFootprint(shape);
    size += Encoder::EstimateMemoryFootprint(shape);
    size += Compressor::EstimateMemoryFootprint(max_n, config);
    return size;
  }

  void Refactor(Array<D, T_data, DeviceType> &data_array,
                MDRMetadata &mdr_metadata, MDRData<DeviceType> &mdr_data,
                int queue_idx) {
    SIZE target_level = hierarchy->l_target();
    mdr_metadata.Initialize(hierarchy->l_target() + 1, Encoder::MAX_BITPLANES);
    mdr_data.Resize(hierarchy->l_target() + 1, Encoder::MAX_BITPLANES);

    SubArray<D, T_data, DeviceType> data(data_array);

    Timer timer, timer_all;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer_all.start();
    }
    decomposer.decompose(data_array, hierarchy->l_target(), 0, queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    interleaver.interleave(data, level_data_subarray, hierarchy->l_target(),
                           queue_idx);
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Interleave", hierarchy->total_num_elems() * sizeof(T_data));
      timer.clear();
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    for (int level_idx = 0; level_idx < hierarchy->l_target() + 1;
         level_idx++) {

      SubArray<1, T_data, DeviceType> result(abs_max_result_array);
      DeviceCollective<DeviceType>::AbsMax(
          level_data_subarray[level_idx].shape(0),
          level_data_subarray[level_idx], result, abs_max_workspace, true,
          queue_idx);
      T_data level_max_error;
      MemoryManager<DeviceType>::Copy1D(&level_max_error, result.data(), 1,
                                        queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

      int level_exp = 0;
      frexp(level_max_error, &level_exp);
      exp[level_idx] = level_exp;
      // printf("level: %d, level_max_error: %.10f, level_exp: %d\n", level_idx,
      // level_max_error, level_exp);
      mdr_metadata.level_error_bounds[level_idx] = level_max_error;
      mdr_metadata.level_num_elems[level_idx] =
          hierarchy->level_num_elems(level_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Max Error", hierarchy->total_num_elems() * sizeof(T_data));
      timer.clear();
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    for (int level_idx = 0; level_idx < hierarchy->l_target() + 1;
         level_idx++) {
      encoder.encode(hierarchy->level_num_elems(level_idx),
                     Encoder::MAX_BITPLANES, exp[level_idx],
                     level_data_subarray[level_idx],
                     encoded_bitplanes_subarray[level_idx],
                     level_errors_subarray[level_idx], queue_idx);
    }

    for (int level_idx = 0; level_idx < hierarchy->l_target() + 1;
         level_idx++) {
      std::vector<T_error> squared_error(Encoder::MAX_BITPLANES + 1);
      MemoryManager<DeviceType>::Copy1D(squared_error.data(),
                                        level_errors_array[level_idx].data(),
                                        Encoder::MAX_BITPLANES + 1, queue_idx);
      mdr_metadata.level_squared_errors[level_idx] = squared_error;
      // PrintSubarray("level_errors", level_errors_subarray[level_idx]);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Encoding", hierarchy->total_num_elems() * sizeof(T_data));
      timer.clear();
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    for (int level_idx = 0; level_idx < hierarchy->l_target() + 1;
         level_idx++) {
      compressor.compress_level(encoded_bitplanes_subarray[level_idx],
                                mdr_data.compressed_bitplanes[level_idx],
                                level_idx, queue_idx);
      for (int bitplane_idx = 0; bitplane_idx < Encoder::MAX_BITPLANES;
           bitplane_idx++) {
        mdr_metadata.level_sizes[level_idx][bitplane_idx] +=
            mdr_data.compressed_bitplanes[level_idx][bitplane_idx].shape(0);
      }
    }
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Lossless", hierarchy->total_num_elems() * sizeof(T_data));
      timer.clear();
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer_all.end();
      timer_all.print("Low-level refactoring",
                      hierarchy->total_num_elems() * sizeof(T_data));
      timer_all.clear();
    }
  }

  void print() const {
    std::cout << "Composed refactor with the following components."
              << std::endl;
    std::cout << "Decomposer: ";
    decomposer.print();
    std::cout << "Interleaver: ";
    interleaver.print();
    std::cout << "Encoder: ";
    encoder.print();
  }

  bool initialized = false;

private:
  Hierarchy<D, T_data, DeviceType> *hierarchy;
  Decomposer decomposer;
  Interleaver interleaver;
  Encoder encoder;
  // BatchedEncoder batched_encoder;
  Compressor compressor;

  std::vector<Array<1, T_data, DeviceType>> level_data_array;
  std::vector<SubArray<1, T_data, DeviceType>> level_data_subarray;

  Array<1, T_data, DeviceType> abs_max_result_array;
  Array<1, Byte, DeviceType> abs_max_workspace;

  std::vector<Array<2, T_bitplane, DeviceType>> encoded_bitplanes_array;
  std::vector<SubArray<2, T_bitplane, DeviceType>> encoded_bitplanes_subarray;

  std::vector<Array<1, T_error, DeviceType>> level_errors_array;
  std::vector<SubArray<1, T_error, DeviceType>> level_errors_subarray;

  std::vector<SIZE> level_num_elems;
  std::vector<int32_t> exp;
};
} // namespace MDR
} // namespace mgard_x
#endif
