#ifndef _MDR_BP_ENCODER_OPT_V1_HPP
#define _MDR_BP_ENCODER_OPT_V1_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "BitplaneEncoderInterface.hpp"
#include <string.h>

#define BINARY_TYPE BINARY
// #define BINARY_TYPE NEGABINARY

namespace mgard_x {
namespace MDR {

template <typename T, typename T_fp, typename T_sfp, typename T_bitplane,
          typename T_error, OPTION BinaryType, typename DeviceType>
class BPEncoderOptV1Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPEncoderOptV1Functor() {}
  MGARDX_CONT
  BPEncoderOptV1Functor(SIZE n, SIZE num_bitplanes, SIZE exp,
                        SubArray<1, T, DeviceType> v,
                        SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                        SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), num_bitplanes(num_bitplanes), exp(exp),
        encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void encode_batch(T_fp *v, T_bitplane *encoded, int batch_size,
                                int num_bitplanes) {
    for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
      T_bitplane buffer = 0;
      for (int data_idx = 0; data_idx < batch_size; data_idx++) {
        T_bitplane bit = (v[data_idx] >> (sizeof(T_fp) * 8 - 1 - bp_idx)) & 1u;
        buffer += bit << sizeof(T_bitplane) * 8 - 1 - data_idx;
      }
      encoded[bp_idx] = buffer;
    }
  }

  MGARDX_EXEC void error_collect(T *v, T_error *errors, int batch_size,
                                 SIZE num_bitplanes, SIZE exp) {
    for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
      for (int data_idx = 0; data_idx < batch_size; data_idx++) {

        T data = v[data_idx];

        T_fp fp_data = (T_fp)fabs(v[data_idx]);
        T_sfp fps_data = (T_sfp)data;
        T_error mantissa = fabs(data) - fp_data;
        T_fp mask = ((T_fp)1 << bp_idx) - 1;
        T_error diff = 0;
        diff = (T_error)(fp_data & mask) + mantissa;
        errors[num_bitplanes - bp_idx] += diff * diff;
      }
    }
    for (int data_idx = 0; data_idx < batch_size; data_idx++) {
      T data = v[data_idx];
      errors[0] += data * data;
    }

    for (int bp_idx = 0; bp_idx < num_bitplanes + 1; bp_idx++) {
      errors[bp_idx] = ldexp(errors[bp_idx], 2 * (-(int)num_bitplanes + exp));
    }
  }

  MGARDX_EXEC void Operation1() {
    int batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                        FunctorBase<DeviceType>::GetBlockDimX() +
                    FunctorBase<DeviceType>::GetThreadIdX();

    int num_batches = (n - 1) / BATCH_SIZE + 1;
    T shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_fp signs[BATCH_SIZE];
    T_bitplane encoded_data[MAX_BITPLANES];
    T_bitplane encoded_sign[MAX_BITPLANES];
    T_error errors[MAX_BITPLANES + 1];

    if (batch_idx < num_batches) {
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T data = 0;
        if (batch_idx * BATCH_SIZE + data_idx < n) {
          data = *v(batch_idx * BATCH_SIZE + data_idx);
        }
        shifted_data[data_idx] = ldexp(data, num_bitplanes - exp);
        fp_data[data_idx] = (T_fp)fabs(shifted_data[data_idx]);
        signs[data_idx] = ((T_sfp)signbit(data)) << (sizeof(T_fp) * 8 - 1);
        // printf("%f: ", data); print_bits(fp_data[data_idx], b);
      }
      // encode data
      encode_batch(fp_data, encoded_data, BATCH_SIZE, num_bitplanes);
      // encode sign
      encode_batch(signs, encoded_sign, BATCH_SIZE, 1);

      error_collect(shifted_data, errors, BATCH_SIZE, num_bitplanes, exp);

      for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
        *encoded_bitplanes(bp_idx, batch_idx * 2) = encoded_data[bp_idx];
        // print_bits(encoded_bitplanes[bp_idx * b + batch_idx * 2],
        // batch_size);
      }
      *encoded_bitplanes(0, batch_idx * 2 + 1) = encoded_sign[0];
      // print_bits(encoded_bitplanes[0 * b + batch_idx * 2 + 1], batch_size);
      for (int bp_idx = 0; bp_idx < num_bitplanes + 1; bp_idx++) {
        *level_errors_workspace(bp_idx, batch_idx) = errors[bp_idx];
      }
    }
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
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T) * 8;
};

template <typename T, typename T_bitplane, typename T_error, OPTION BinaryType,
          typename DeviceType>
class BPEncoderOptV1Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "grouped bp encoder";
  MGARDX_CONT
  BPEncoderOptV1Kernel(SIZE n, SIZE num_bitplanes, SIZE exp,
                       SubArray<1, T, DeviceType> v,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), num_bitplanes(num_bitplanes), exp(exp),
        encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {}

  using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                         uint64_t, uint32_t>::type;
  using FunctorType = BPEncoderOptV1Functor<T, T_fp, T_sfp, T_bitplane, T_error,
                                            BinaryType, DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {
    FunctorType functor(n, num_bitplanes, exp, v, encoded_bitplanes,
                        level_errors_workspace);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (n - 1) / tbx + 1;
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

template <typename T, typename T_fp, typename T_sfp, typename T_bitplane,
          OPTION BinaryType, typename DeviceType>
class BPDecoderOptV1Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPDecoderOptV1Functor() {}
  MGARDX_CONT
  BPDecoderOptV1Functor(SIZE n, SIZE starting_bitplane, SIZE num_bitplanes,
                        SIZE exp,
                        SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                        SubArray<1, bool, DeviceType> signs,
                        SubArray<1, T, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane),
        num_bitplanes(num_bitplanes), exp(exp),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void decode_batch(T_fp *v, T_bitplane *encoded, int batch_size,
                                int num_bitplanes) {
    for (int data_idx = 0; data_idx < batch_size; data_idx++) {
      T_fp buffer = 0;
      for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
        T_fp bit =
            (encoded[bp_idx] >> (sizeof(T_bitplane) * 8 - 1 - data_idx)) & 1u;
        buffer += bit << (num_bitplanes - 1 - bp_idx);
      }
      v[data_idx] = buffer;
    }
  }

  MGARDX_EXEC void Operation1() {
    int batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                        FunctorBase<DeviceType>::GetBlockDimX() +
                    FunctorBase<DeviceType>::GetThreadIdX();

    int num_batches = (n - 1) / BATCH_SIZE + 1;

    T shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_fp signs[BATCH_SIZE];
    T_bitplane encoded_data[MAX_BITPLANES];
    T_bitplane encoded_sign[MAX_BITPLANES];

    int ending_bitplane = starting_bitplane + num_bitplanes;

    // for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    if (batch_idx < num_batches) {

      for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
        encoded_data[bp_idx] = *encoded_bitplanes(bp_idx, batch_idx * 2);
        // print_bits(encoded_data[bp_idx], batch_size);
      }
      encoded_sign[0] = *encoded_bitplanes(0, batch_idx * 2 + 1);
      // print_bits(encoded_sign[0], batch_size);

      // encode data
      decode_batch(fp_data, encoded_data, BATCH_SIZE, num_bitplanes);
      // encode sign
      decode_batch(signs, encoded_sign, BATCH_SIZE, 1);
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {

        T data = ldexp((T)fp_data[data_idx], -ending_bitplane + exp);
        *v(batch_idx * BATCH_SIZE + data_idx) = signs[data_idx] ? -data : data;
        // printf("%f: ", data); print_bits(fp_data[data_idx], b);
      }
    }
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
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T) * 8;
};

template <typename T, typename T_bitplane, OPTION BinaryType,
          typename DeviceType>
class BPDecoderOptV1Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "grouped bp decoder";
  MGARDX_CONT
  BPDecoderOptV1Kernel(SIZE n, SIZE starting_bitplane, SIZE num_bitplanes,
                       SIZE exp,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<1, bool, DeviceType> signs,
                       SubArray<1, T, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane),
        num_bitplanes(num_bitplanes), exp(exp),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {}

  using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                         uint64_t, uint32_t>::type;
  using FunctorType =
      BPDecoderOptV1Functor<T, T_fp, T_sfp, T_bitplane, BinaryType, DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {

    FunctorType functor(n, starting_bitplane, num_bitplanes, exp,
                        encoded_bitplanes, signs, v);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (n - 1) / tbx + 1;
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
class BPEncoderOptV1
    : public concepts::BitplaneEncoderInterface<D, T_data, T_bitplane, T_error,
                                                DeviceType> {
public:
  BPEncoderOptV1() : initialized(false) {
    static_assert(std::is_floating_point<T_data>::value,
                  "GeneralBPEncoder: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value,
                  "GeneralBPEncoder: long double is not supported.");
    static_assert(std::is_unsigned<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
    static_assert(std::is_integral<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
  }
  BPEncoderOptV1(Hierarchy<D, T_data, DeviceType> &hierarchy) {
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

  static SIZE buffer_size(SIZE n) {
    return num_blocks(n) * sizeof(T_bitplane) * 2;
  }

  static SIZE num_blocks(SIZE n) {
    const SIZE batch_size = sizeof(T_bitplane) * 8;
    SIZE num_blocks = (n - 1) / batch_size + 1;
    return num_blocks;
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
        BPEncoderOptV1Kernel<T_data, T_bitplane, T_error, BINARY_TYPE,
                             DeviceType>(n, num_bitplanes, exp, v,
                                         encoded_bitplanes, level_errors_work),
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
              SubArray<1, T_data, DeviceType> v, int queue_idx) {}

  // decode the data and record necessary information for progressiveness
  void progressive_decode(SIZE n, SIZE starting_bitplanes, SIZE num_bitplanes,
                          int32_t exp,
                          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                          SubArray<1, bool, DeviceType> level_signs, int level,
                          SubArray<1, T_data, DeviceType> v, int queue_idx) {

    if (num_bitplanes > 0) {
      DeviceLauncher<DeviceType>::Execute(
          BPDecoderOptV1Kernel<T_data, T_bitplane, BINARY_TYPE, DeviceType>(
              n, starting_bitplanes, num_bitplanes, exp, encoded_bitplanes,
              level_signs, v),
          queue_idx);
    }
  }

  void print() const { std::cout << "Grouped bitplane encoder" << std::endl; }

private:
  bool initialized;
  Hierarchy<D, T_data, DeviceType> *hierarchy;
  Array<2, T_error, DeviceType> level_errors_work_array;
  Array<1, Byte, DeviceType> level_error_sum_work_array;
};
} // namespace MDR
} // namespace mgard_x
#endif
