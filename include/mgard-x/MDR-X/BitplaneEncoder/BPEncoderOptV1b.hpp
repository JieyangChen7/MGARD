#ifndef _MDR_BP_ENCODER_OPT_V1b_HPP
#define _MDR_BP_ENCODER_OPT_V1b_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "BitplaneEncoderInterface.hpp"
#include <string.h>

namespace mgard_x {
namespace MDR {

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          typename T_error, int NUM_BITPLANES, bool NegaBinary, bool CollectError,
          typename DeviceType>
class BPEncoderOptV1bFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPEncoderOptV1bFunctor() {}
  MGARDX_CONT
  BPEncoderOptV1bFunctor(SIZE n, SubArray<1, T_data, DeviceType> abs_max,
                        SubArray<1, T_data, DeviceType> v,
                        SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                        SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void encode_batch(T_fp *v, T_bitplane *encoded) {

    #pragma unroll                            
    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      T_bitplane buffer = 0;
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_bitplane bit = (v[data_idx] >> (NUM_BITPLANES - 1 - bp_idx)) & (T_bitplane)1;
        buffer |= bit << BATCH_SIZE - 1 - data_idx;
      }
      encoded[bp_idx] = buffer;
    }
  }

  MGARDX_EXEC void error_collect_binary(T_data *shifted_data, T_error *errors, int exp) {

    int batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                        FunctorBase<DeviceType>::GetBlockDimX() +
                    FunctorBase<DeviceType>::GetThreadIdX();

    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = shifted_data[data_idx];
        T_fp fp_data = (T_fp)fabs(data);
        T_error mantissa = fabs(data) - fp_data;
        T_fp mask = ((T_fp)1 << bp_idx) - 1;
        T_error diff = (T_error)(fp_data & mask) + mantissa;
        // if (bp_idx == 31 && batch_idx == 0) {
        //   printf(
        //       "data: %f  fp_data: %llu  fps_data: %lld  mask: %llu  diff:
        //       %f\n", data, fp_data, sfp_data, mask, diff);
        // }
        errors[NUM_BITPLANES - bp_idx] += diff * diff;
      }
    }
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      T_data data = shifted_data[data_idx];
      errors[0] += data * data;
    }

    for (int bp_idx = 0; bp_idx < NUM_BITPLANES + 1; bp_idx++) {
      errors[bp_idx] = ldexp(errors[bp_idx], 2 * (-NUM_BITPLANES + exp));
    }
  }

  MGARDX_EXEC void error_collect_negabinary(T_data *shifted_data,
                                            T_error *errors,
                                            int exp) {

    int batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                        FunctorBase<DeviceType>::GetBlockDimX() +
                    FunctorBase<DeviceType>::GetThreadIdX();

    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = shifted_data[data_idx];
        T_fp fp_data = (T_fp)fabs(data);
        T_error mantissa = fabs(data) - fp_data;
        T_fp mask = ((T_fp)1 << bp_idx) - 1;
        T_fp ngb_data = Math<DeviceType>::binary2negabinary((T_sfp)data);
        T_error diff =
            (T_error)Math<DeviceType>::negabinary2binary(ngb_data & mask) +
            mantissa;
        // if (bp_idx == 31 && batch_idx == 0) {
        //   printf(
        //       "data: %f  fp_data: %llu  fps_data: %lld  mask: %llu  diff:
        //       %f\n", data, fp_data, sfp_data, mask, diff);
        // }
        errors[NUM_BITPLANES - bp_idx] += diff * diff;
      }
    }
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      T_data data = shifted_data[data_idx];
      errors[0] += data * data;
    }

    for (int bp_idx = 0; bp_idx < NUM_BITPLANES + 1; bp_idx++) {
      errors[bp_idx] = ldexp(errors[bp_idx], 2 * (-NUM_BITPLANES + exp));
    }
  }

  MGARDX_EXEC void EncodeBinary() {
    SIZE batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                   FunctorBase<DeviceType>::GetBlockDimX() +
               FunctorBase<DeviceType>::GetThreadIdX();

    SIZE num_full_batches = n / BATCH_SIZE;

    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_bitplane encoded_data[NUM_BITPLANES];
    T_bitplane encoded_sign = 0;
    T_error errors[NUM_BITPLANES + 1];

    int exp;
    frexp(*abs_max((IDX)0), &exp);  

    if (batch_idx >= num_full_batches) {
      return;
    }

    if (exp > 0) {
      #pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = *v(data_idx * num_full_batches + batch_idx);
        // this can cause overflow
        shifted_data[data_idx] = data * ((T_fp)1 << NUM_BITPLANES - exp);
        // ldexp without constant argument is slow
        // shifted_data[data_idx] = ldexp(data, NUM_BITPLANES - exp);
        fp_data[data_idx] = (T_fp)fabs(shifted_data[data_idx]);
        
        // if (num_full_batches == 1) printf("data: %f * %d %d, shifted_data: %f fp_data: %llu \n", data, NUM_BITPLANES, exp, shifted_data[data_idx], fp_data[data_idx]);
      }
    } else {
      #pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = *v(data_idx * num_full_batches + batch_idx);
        shifted_data[data_idx] = data * pow(2, NUM_BITPLANES - exp);
        fp_data[data_idx] = (T_fp)fabs(shifted_data[data_idx]);
      }
    } 

    // encode sign
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      encoded_sign += (T_fp)(signbit(shifted_data[data_idx]) == 0 ? 0 : 1) << (BATCH_SIZE - 1 - data_idx);
    }
    // encode data
    encode_batch(fp_data, encoded_data);
    // store data
    #pragma unroll
    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      // if (num_full_batches == 1) printf("encoded_data: %u\n", encoded_data[bp_idx]);
      *encoded_bitplanes(bp_idx, batch_idx) = encoded_data[bp_idx];
    }
    // store sign
    *encoded_bitplanes(0, num_full_batches + batch_idx) = encoded_sign;
    // set rest of the bitplanes to 0
    #pragma unroll
    for (int bp_idx = 1; bp_idx < NUM_BITPLANES; bp_idx++) {
      *encoded_bitplanes(bp_idx, num_full_batches + batch_idx) = (T_bitplane)0;
    }
    if constexpr (CollectError) {
      error_collect_binary(shifted_data, errors, exp);
      for (int bp_idx = 0; bp_idx < NUM_BITPLANES + 1; bp_idx++) {
        *level_errors_workspace(bp_idx, batch_idx) = errors[bp_idx];
      }
    }
  }

  MGARDX_EXEC void EncodeNegaBinary() {
    SIZE batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                   FunctorBase<DeviceType>::GetBlockDimX() +
               FunctorBase<DeviceType>::GetThreadIdX();

    SIZE num_full_batches = n / BATCH_SIZE;

    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_bitplane encoded_data[NUM_BITPLANES];
    T_error errors[NUM_BITPLANES + 1];

    int exp;
    frexp(*abs_max((IDX)0), &exp);  
    exp += 2;

    if (batch_idx >= num_full_batches) {
      return;
    }
    
    if (exp > 0) {
      #pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = 0;
          data = *v(data_idx * num_full_batches + batch_idx);
          // This can cause overflow
          shifted_data[data_idx] = data * ((T_fp)1 << NUM_BITPLANES - exp);
          // ldexp without constant argument is slow
          // shifted_data[data_idx] = ldexp(data, NUM_BITPLANES - exp);
          fp_data[data_idx] =
              Math<DeviceType>::binary2negabinary((T_sfp)shifted_data[data_idx]);
      }
    } else {
      #pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = 0;
          data = *v(data_idx * num_full_batches + batch_idx);
          shifted_data[data_idx] = data * pow(2, NUM_BITPLANES - exp);   
          // ldexp without constant argument is slow
          // shifted_data[data_idx] = ldexp(data, NUM_BITPLANES - exp);
          fp_data[data_idx] =
              Math<DeviceType>::binary2negabinary((T_sfp)shifted_data[data_idx]);
      }
    }
    
    // encode data
    encode_batch(fp_data, encoded_data);
    // store data
    #pragma unroll
    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      *encoded_bitplanes(bp_idx, batch_idx) = encoded_data[bp_idx];
    }

    if constexpr (CollectError) {
      error_collect_negabinary(shifted_data, errors, exp);
      #pragma unroll
      for (int bp_idx = 0; bp_idx < NUM_BITPLANES + 1; bp_idx++) {
        *level_errors_workspace(bp_idx, batch_idx) = errors[bp_idx];
      }
    }
  }

  MGARDX_EXEC void Operation1() {
    if constexpr (NegaBinary) {
      EncodeNegaBinary();
    } else {
      EncodeBinary();
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  // parameters
  SIZE n;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
};

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          typename T_error, int NUM_BITPLANES, bool NegaBinary, bool CollectError,
          typename DeviceType>
class BPEncoderOptV1bKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "grouped bp encoder";
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  MGARDX_CONT
  BPEncoderOptV1bKernel(SIZE n, SubArray<1, T_data, DeviceType> abs_max,
                       SubArray<1, T_data, DeviceType> v,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {}

  using FunctorType =
      BPEncoderOptV1bFunctor<T_data, T_fp, T_sfp, T_bitplane, T_error, NUM_BITPLANES,
                            NegaBinary, CollectError, DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {
    FunctorType functor(n, abs_max, v, encoded_bitplanes,
                        level_errors_workspace);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE total_thread = std::max((SIZE)1, n / BATCH_SIZE);
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (total_thread - 1) / tbx + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SIZE n;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
};

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          int NUM_BITPLANES, bool NegaBinary, typename DeviceType>
class BPDecoderOptV1bFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPDecoderOptV1bFunctor() {}
  MGARDX_CONT
  BPDecoderOptV1bFunctor(SIZE n, int starting_bitplane,
                        SubArray<1, T_data, DeviceType> abs_max,
                        SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                        SubArray<1, bool, DeviceType> signs,
                        SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane),
         abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void decode_batch(T_fp *v, T_bitplane *encoded) {
    #pragma unroll
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      T_fp buffer = 0;
      for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
        T_fp bit = (encoded[bp_idx] >> (BATCH_SIZE - 1 - data_idx)) & (T_fp)1;
        buffer += bit << (NUM_BITPLANES - 1 - bp_idx);
        // printf("bit: %llu, buffer: %llu\n", bit, buffer);
      }
      v[data_idx] = buffer;
    }
  }

  MGARDX_EXEC void DecodeBinary() {
    SIZE batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                   FunctorBase<DeviceType>::GetBlockDimX() +
               FunctorBase<DeviceType>::GetThreadIdX();
    
    SIZE num_full_batches = n / BATCH_SIZE;

    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_fp fp_sign[BATCH_SIZE];
    T_bitplane encoded_data[NUM_BITPLANES];
    T_bitplane encoded_sign;

    int exp;
    frexp(*abs_max((IDX)0), &exp);

    if (batch_idx >= num_full_batches) {
      return;
    }

    int ending_bitplane = starting_bitplane + NUM_BITPLANES;

    #pragma unroll
    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      encoded_data[bp_idx] =
          *encoded_bitplanes(starting_bitplane + bp_idx, batch_idx);
      // if (num_full_batches == 1) printf("encoded_data: %u\n", encoded_data[bp_idx]);
    }
    // decode data
    decode_batch(fp_data, encoded_data);

    if (starting_bitplane == 0) {
      // decode sign
      encoded_sign = *encoded_bitplanes(0, num_full_batches + batch_idx);
      #pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        fp_sign[data_idx] = (encoded_sign >> (BATCH_SIZE - 1 - data_idx)) & (T_fp)1;
        *signs(data_idx * num_full_batches + batch_idx) = fp_sign[data_idx];
      }
    } else {
      #pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        fp_sign[data_idx] = *signs(data_idx * num_full_batches + batch_idx);
      }
    }
    #pragma unroll
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      shifted_data[data_idx] = (T_data)fp_data[data_idx];
      // It is beneficial to use pow instead of ldexp
      T_data data = shifted_data[data_idx] * pow(2, -ending_bitplane + exp);
      // T_data data = ldexp(shifted_data[data_idx], -ending_bitplane + exp);
      data = fp_sign[data_idx] ? -data : data;
      *v(data_idx * num_full_batches + batch_idx) = data;

      // if (num_full_batches == 1) printf("%llu %f %f\n", fp_data[data_idx], shifted_data[data_idx], data);
    }
  }

  MGARDX_EXEC void DecodeNegaBinary() {
    SIZE batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                   FunctorBase<DeviceType>::GetBlockDimX() +
               FunctorBase<DeviceType>::GetThreadIdX();

    SIZE num_full_batches = n / BATCH_SIZE;

    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_bitplane encoded_data[NUM_BITPLANES];

    int exp;
    frexp(*abs_max((IDX)0), &exp);  
    exp += 2;

    if (batch_idx >= num_full_batches) {
      return;
    }

    int ending_bitplane = starting_bitplane + NUM_BITPLANES;

    // load bitplanes
    #pragma unroll
    for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
      encoded_data[bp_idx] =
          *encoded_bitplanes(starting_bitplane + bp_idx, batch_idx);
      // print_bits(encoded_data[bp_idx], batch_size);
    }
    // decode data
    decode_batch(fp_data, encoded_data);

    // store data
    #pragma unroll
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      shifted_data[data_idx] = Math<DeviceType>::negabinary2binary(fp_data[data_idx]);
      // No noticing difference between the two
      T_data data = shifted_data[data_idx] * pow(2, -ending_bitplane + exp);
      // T_data data = ldexp(shifted_data[data_idx], -ending_bitplane + exp);
      data = ending_bitplane % 2 != 0 ? -data : data;
      *v(data_idx * num_full_batches + batch_idx) = data;
      // printf("%f: ", data); print_bits(fp_data[data_idx], b);
    }
  }

  MGARDX_EXEC void Operation1() {
    if constexpr (NegaBinary) {
      DecodeNegaBinary();
    } else {
      DecodeBinary();
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  // parameters
  SIZE n;
  int starting_bitplane;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T_data, DeviceType> v;
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
};

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          int NUM_BITPLANES, bool NegaBinary, typename DeviceType>
class BPDecoderOptV1bKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "grouped bp decoder";
  static constexpr SIZE BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
  MGARDX_CONT
  BPDecoderOptV1bKernel(SIZE n, int starting_bitplane,
                       SubArray<1, T_data, DeviceType> abs_max,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<1, bool, DeviceType> signs,
                       SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane),
        abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {}

  using FunctorType = BPDecoderOptV1bFunctor<T_data, T_fp, T_sfp, T_bitplane,
                                            NUM_BITPLANES, NegaBinary, DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {

    FunctorType functor(n, starting_bitplane, abs_max,
                        encoded_bitplanes, signs, v);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE total_thread = std::max((SIZE)1, n / BATCH_SIZE);
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (total_thread - 1) / tbx + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SIZE n;
  int starting_bitplane;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T_data, DeviceType> v;
};

// general bitplane encoder that encodes data by block using T_stream type
// buffer
template <DIM D, typename T_data, typename T_bitplane, typename T_error,
          bool NegaBinary, bool CollectError, typename DeviceType>
class BPEncoderOptV1b
    : public concepts::BitplaneEncoderInterface<D, T_data, T_bitplane, T_error,
                                                CollectError, DeviceType> {
public:
  static constexpr SIZE BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
  using T_sfp = typename std::conditional<std::is_same<T_data, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T_data, double>::value,
                                         uint64_t, uint32_t>::type;

  BPEncoderOptV1b() : initialized(false) {
    static_assert(std::is_floating_point<T_data>::value,
                  "GeneralBPEncoder: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value,
                  "GeneralBPEncoder: long double is not supported.");
    static_assert(std::is_unsigned<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
    static_assert(std::is_integral<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
  }
  BPEncoderOptV1b(Hierarchy<D, T_data, DeviceType> &hierarchy) {
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

  static SIZE bitplane_length(SIZE n) {
    if constexpr (!NegaBinary) {
      return num_blocks(n) * 2;
    } else {
      return num_blocks(n);
    }
  }

  static SIZE num_blocks(SIZE n) {
    const SIZE batch_size = sizeof(T_bitplane) * 8;
    SIZE num_blocks = (n - 1) / batch_size + 1;
    return num_blocks;
  }

  void Adapt(Hierarchy<D, T_data, DeviceType> &hierarchy, int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    SIZE max_level_num_elems = round_up(hierarchy.level_num_elems(hierarchy.l_target()), BATCH_SIZE);

    level_errors_work_array.resize(
        {MAX_BITPLANES + 1, num_blocks(max_level_num_elems)}, queue_idx);
    DeviceCollective<DeviceType>::Sum(
        num_blocks(max_level_num_elems), SubArray<1, T_error, DeviceType>(),
        SubArray<1, T_error, DeviceType>(), level_error_sum_work_array, false,
        queue_idx);
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    Hierarchy<D, T_data, DeviceType> hierarchy(shape, Config());
    SIZE max_level_num_elems = hierarchy.level_num_elems(hierarchy.l_target());
    size_t size = 0;
    size += hierarchy.EstimateMemoryFootprint(shape);
    size +=
        (MAX_BITPLANES + 1) * num_blocks(max_level_num_elems) * sizeof(T_error);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += hierarchy.level_num_elems(level_idx) * sizeof(bool);
    }
    return size;
  }

  // TODO: remove num_bitplanes in the future
  void encode(SIZE n, int num_bitplanes, SubArray<1, T_data, DeviceType> abs_max,
              SubArray<1, T_data, DeviceType> v,
              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
              SubArray<1, T_error, DeviceType> level_errors, int queue_idx) {

    if (n % BATCH_SIZE != 0) {
      log::err("BPEncoderV1b: n is not a multiple of BATCH_SIZE");
      exit(-1);
    }
    SubArray<2, T_error, DeviceType> level_errors_work(level_errors_work_array);

    DeviceLauncher<DeviceType>::Execute(
        BPEncoderOptV1bKernel<T_data, T_fp, T_sfp, T_bitplane, T_error, MAX_BITPLANES,
                             NegaBinary, CollectError, DeviceType>(
            n, abs_max, v, encoded_bitplanes, level_errors_work),
        queue_idx);

    if constexpr (CollectError) {
      SIZE reduce_size = num_blocks(n);
      for (int i = 0; i < MAX_BITPLANES + 1; i++) {
        SubArray<1, T_error, DeviceType> curr_errors({reduce_size},
                                                     level_errors_work(i, 0));
        SubArray<1, T_error, DeviceType> sum_error({1}, level_errors(i));
        DeviceCollective<DeviceType>::Sum(reduce_size, curr_errors, sum_error,
                                          level_error_sum_work_array, true,
                                          queue_idx);
      }
    }
  }

  void decode(SIZE n, int num_bitplanes, SubArray<1, T_data, DeviceType> abs_max,
              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes, int level,
              SubArray<1, T_data, DeviceType> v, int queue_idx) {}

  // decode the data and record necessary information for progressiveness
  void progressive_decode(SIZE n, int starting_bitplane, int num_bitplanes,
                          SubArray<1, T_data, DeviceType> abs_max,
                          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                          SubArray<1, bool, DeviceType> level_signs, int level,
                          SubArray<1, T_data, DeviceType> v, int queue_idx) {

    // if (num_bitplanes > 0) {
    //   DeviceLauncher<DeviceType>::Execute(
    //       BPDecoderOptV1bKernel<T_data, T_fp, T_sfp, T_bitplane, NegaBinary,
    //                            DeviceType>(n, starting_bitplane, num_bitplanes,
    //                                        abs_max, encoded_bitplanes, level_signs,
    //                                        v),
    //       queue_idx);
    // }

    #define V1B_DECODE(NUM_BITPLANES) \
      if (num_bitplanes == NUM_BITPLANES) { \
        DeviceLauncher<DeviceType>::Execute( \
            BPDecoderOptV1bKernel<T_data, T_fp, T_sfp, T_bitplane, NUM_BITPLANES, NegaBinary, \
                                 DeviceType>(n, starting_bitplane, \
                                             abs_max, encoded_bitplanes, \
                                             level_signs, v), \
            queue_idx); \
      }
    V1B_DECODE(1); V1B_DECODE(2); V1B_DECODE(3);  V1B_DECODE(4); 
    V1B_DECODE(5);  V1B_DECODE(6); V1B_DECODE(7); V1B_DECODE(8); 
    V1B_DECODE(9); V1B_DECODE(10); V1B_DECODE(11); V1B_DECODE(12); 
    V1B_DECODE(13); V1B_DECODE(14); V1B_DECODE(15); V1B_DECODE(16);
    V1B_DECODE(17); V1B_DECODE(18); V1B_DECODE(19); V1B_DECODE(20);
    V1B_DECODE(21); V1B_DECODE(22); V1B_DECODE(23); V1B_DECODE(24);
    V1B_DECODE(25); V1B_DECODE(26); V1B_DECODE(27); V1B_DECODE(28);
    V1B_DECODE(29); V1B_DECODE(30); V1B_DECODE(31); V1B_DECODE(32);
    V1B_DECODE(33); V1B_DECODE(34); V1B_DECODE(35); V1B_DECODE(36);
    V1B_DECODE(37); V1B_DECODE(38); V1B_DECODE(39); V1B_DECODE(40);
    V1B_DECODE(41); V1B_DECODE(42); V1B_DECODE(43); V1B_DECODE(44);
    V1B_DECODE(45); V1B_DECODE(46); V1B_DECODE(47); V1B_DECODE(48);
    V1B_DECODE(49); V1B_DECODE(50); V1B_DECODE(51); V1B_DECODE(52);
    V1B_DECODE(53); V1B_DECODE(54); V1B_DECODE(55); V1B_DECODE(56); 
    V1B_DECODE(57); V1B_DECODE(58); V1B_DECODE(59); V1B_DECODE(60); 
    V1B_DECODE(61); V1B_DECODE(62); V1B_DECODE(63); V1B_DECODE(64); 

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
