#ifndef _MDR_BP_ENCODER_OPT_HPP
#define _MDR_BP_ENCODER_OPT_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "BitplaneEncoderInterface.hpp"
#include <string.h>

namespace mgard_x {
namespace MDR {

template <typename T, typename T_fp, typename T_sfp, typename T_bitplane,
          typename T_error, typename DeviceType>
class BPEncoderOptFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPEncoderOptFunctor() {}
  MGARDX_CONT
  BPEncoderOptFunctor(SIZE n, SIZE num_bitplanes,
                        SIZE exp, SubArray<1, T, DeviceType> v,
                        SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                        SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), num_bitplanes(num_bitplanes), exp(exp),
        encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {
    Functor<DeviceType>();
  }
  // exponent align
  // calculate error
  // store signs
  // find the most significant bit
  MGARDX_EXEC void Operation1() {

    
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  // parameters
  SIZE n;
  SIZE num_bitplanes;
  SIZE exp;
  SubArray<1, T, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
};

template <typename T, typename T_bitplane, typename T_error, typename DeviceType>
class BPEncoderOptKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "bp encoder opt";
  MGARDX_CONT
  BPEncoderOptKernel(SIZE n, SIZE num_bitplanes,
                       SIZE exp, SubArray<1, T, DeviceType> v,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), num_bitplanes(num_bitplanes), exp(exp),
        encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {}

  using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                         uint64_t, uint32_t>::type;
  using FunctorType =
      BPEncoderOptFunctor<T, T_fp, T_sfp, T_bitplane, T_error, DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {
    FunctorType functor(n, num_bitplanes, exp, v, encoded_bitplanes, level_errors_workspace);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 32;
    gridz = 1;
    gridy = 1;
    gridx = (n - 1) / 32 + 1; // modify this to the correct value
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SIZE n;
  SIZE num_bitplanes;
  SIZE exp;
  SubArray<1, T, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
};

template <typename T, typename T_fp, typename T_sfp, typename T_bitplane, typename DeviceType>
class BPDecoderOptFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPDecoderOptFunctor() {}
  MGARDX_CONT
  BPDecoderOptFunctor(SIZE n, SIZE starting_bitplane,
                        SIZE num_bitplanes, SIZE exp,
                        SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                        SubArray<1, bool, DeviceType> signs,
                        SubArray<1, T, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane), num_bitplanes(num_bitplanes),
        exp(exp), encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  // parameters
  SIZE n;
  SIZE starting_bitplane;
  SIZE num_bitplanes;
  SIZE exp;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T, DeviceType> v;
};

template <typename T, typename T_bitplane, typename DeviceType>
class BPDecoderOptKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "bp decoder opt";
  MGARDX_CONT
  BPDecoderOptKernel(SIZE n, SIZE starting_bitplane,
                       SIZE num_bitplanes, SIZE exp,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<1, bool, DeviceType> signs,
                       SubArray<1, T, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane), num_bitplanes(num_bitplanes),
        exp(exp), encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {}

  using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                         uint64_t, uint32_t>::type;
  using FunctorType =
      BPDecoderOptFunctor<T, T_fp, T_sfp, T_bitplane, DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {

    FunctorType functor(n, starting_bitplane, num_bitplanes,
                        exp, encoded_bitplanes, signs, v);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 32;
    gridz = 1;
    gridy = 1;
    gridx = (n - 1) / 32 + 1; // modify this to the correct value
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SIZE n;
  SIZE starting_bitplane;
  SIZE num_bitplanes;
  SIZE exp;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T, DeviceType> v;
};

// general bitplane encoder that encodes data by block using T_stream type
// buffer
template <DIM D, typename T_data, typename T_bitplane, typename T_error,
          typename DeviceType>
class GroupedBPEncoderOpt
    : public concepts::BitplaneEncoderInterface<D, T_data, T_bitplane, T_error,
                                                DeviceType> {
public:
  GroupedBPEncoderOpt() : initialized(false) {
    static_assert(std::is_floating_point<T_data>::value,
                  "GeneralBPEncoder: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value,
                  "GeneralBPEncoder: long double is not supported.");
    static_assert(std::is_unsigned<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
    static_assert(std::is_integral<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
  }
  GroupedBPEncoderOpt(Hierarchy<D, T_data, DeviceType> &hierarchy) {
    static_assert(std::is_floating_point<T_data>::value,
                  "GeneralBPEncoder: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value,
                  "GeneralBPEncoder: long double is not supported.");
    static_assert(std::is_unsigned<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
    static_assert(std::is_integral<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
    Adapt(hierarchy, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  void Adapt(Hierarchy<D, T_data, DeviceType> &hierarchy, int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    SIZE max_level_num_elems = hierarchy.level_num_elems(hierarchy.l_target());

    SIZE max_bitplane = 64;
    level_errors_work_array.resize(
        {max_bitplane + 1, num_blocks(max_level_num_elems)}, queue_idx);
    DeviceCollective<DeviceType>::Sum(
        num_blocks(max_level_num_elems), SubArray<1, T_error, DeviceType>(),
        SubArray<1, T_error, DeviceType>(), level_error_sum_work_array, false,
        queue_idx);
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    Hierarchy<D, T_data, DeviceType> hierarchy(shape, Config());
    SIZE max_level_num_elems = hierarchy.level_num_elems(hierarchy.l_target());
    SIZE max_bitplane = 64;
    size_t size = 0;
    size += hierarchy.EstimateMemoryFootprint(shape);
    size +=
        (max_bitplane + 1) * num_blocks(max_level_num_elems) * sizeof(T_error);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += hierarchy.level_num_elems(level_idx) * sizeof(bool);
    }
    return size;
  }

  void encode(SIZE n, SIZE num_bitplanes, int32_t exp,
              SubArray<1, T_data, DeviceType> v,
              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
              SubArray<1, T_error, DeviceType> level_errors,
              std::vector<SIZE> &streams_sizes, int queue_idx) {

    SubArray<2, T_error, DeviceType> level_errors_work(level_errors_work_array);

    DeviceLauncher<DeviceType>::Execute(
        BPEncoderOptKernel<T_data, T_bitplane, T_error, DeviceType>(
            n, num_bitplanes, exp, v, encoded_bitplanes,
            level_errors_work),
        queue_idx);

    SIZE reduce_size = num_blocks(n);
    for (int i = 0; i < num_bitplanes + 1; i++) {
      SubArray<1, T_error, DeviceType> curr_errors({reduce_size},
                                                   level_errors_work(i, 0));
      SubArray<1, T_error, DeviceType> sum_error({1}, level_errors(i));
      DeviceCollective<DeviceType>::Sum(reduce_size, curr_errors, sum_error,
                                        level_error_sum_work_array, true,
                                        queue_idx);
    }
    for (int i = 0; i < num_bitplanes; i++) {
      streams_sizes[i] = buffer_size(n) * sizeof(T_bitplane);
    }
  }

  void decode(SIZE n, SIZE num_bitplanes, int32_t exp,
              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes, int level,
              SubArray<1, T_data, DeviceType> v, int queue_idx) {
                // do not implement
              }

  // decode the data and record necessary information for progressiveness
  void progressive_decode(SIZE n, SIZE starting_bitplane, SIZE num_bitplanes,
                          int32_t exp,
                          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                          SubArray<1, bool, DeviceType> level_signs, int level,
                          SubArray<1, T_data, DeviceType> v, int queue_idx) {
    if (num_bitplanes > 0) {
      DeviceLauncher<DeviceType>::Execute(
          BPDecoderOptKernel<T_data, T_bitplane, DeviceType>(
              n, starting_bitplane, num_bitplanes, exp,
              encoded_bitplanes, level_signs, v),
          queue_idx);
    }
  }

  static SIZE buffer_size(SIZE n) {
    const SIZE num_elems_per_batch = sizeof(T_bitplane) * 8;
    SIZE bitplane_max_length_total = (n - 1) / num_elems_per_batch + 1;
    bitplane_max_length_total *= 2;
    return bitplane_max_length_total;
  }

  static SIZE num_blocks(SIZE n) {
    SIZE num_blocks = (n - 1) / 32 + 1;
    // modify this to the correct calculation
    return num_blocks;
  }

  void print() const { std::cout << "Grouped bitplane encoder" << std::endl; }

private:
  bool initialized;
  Hierarchy<D, T_data, DeviceType> *hierarchy;
  Array<2, T_error, DeviceType> level_errors_work_array;
  Array<1, Byte, DeviceType> level_error_sum_work_array;
  std::vector<std::vector<uint8_t>> level_recording_bitplanes;
};
} // namespace MDR
} // namespace mgard_x
#endif
