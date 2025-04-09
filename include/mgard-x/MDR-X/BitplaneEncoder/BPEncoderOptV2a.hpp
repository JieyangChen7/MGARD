#ifndef _MDR_BP_ENCODER_OPT_V2a_HPP
#define _MDR_BP_ENCODER_OPT_V2a_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "BitplaneEncoderInterface.hpp"
#include <string.h>

namespace mgard_x {
namespace MDR {

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          typename T_error, SIZE NUM_BITPLANES, SIZE M, bool NegaBinary, bool CollectError,
          typename DeviceType>
class BPEncoderOptV2aFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPEncoderOptV2aFunctor() {}
  MGARDX_CONT
  BPEncoderOptV2aFunctor(SIZE n, int num_bitplanes, SubArray<1, T_data, DeviceType> abs_max,
                        SubArray<1, T_data, DeviceType> v,
                        SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                        SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), num_bitplanes(num_bitplanes), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void encode_batch(T_fp *v, T_bitplane *encoded,
                                int num_bitplanes) {
    for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
      T_bitplane buffer = 0;
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_bitplane bit = (v[data_idx] >> (num_bitplanes - 1 - bp_idx)) & 1u;
        buffer += bit << BATCH_SIZE - 1 - data_idx;
      }
      encoded[bp_idx] = buffer;
    }
  }

  MGARDX_EXEC void error_collect_binary(T_data *shifted_data, T_error *errors,
                                        int num_bitplanes, int exp) {

    int batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                        FunctorBase<DeviceType>::GetBlockDimX() +
                    FunctorBase<DeviceType>::GetThreadIdX();

    for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
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
        errors[num_bitplanes - bp_idx] += diff * diff;
      }
    }
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      T_data data = shifted_data[data_idx];
      errors[0] += data * data;
    }

    for (int bp_idx = 0; bp_idx < num_bitplanes + 1; bp_idx++) {
      errors[bp_idx] = ldexp(errors[bp_idx], 2 * (-(int)num_bitplanes + exp));
    }
  }

  MGARDX_EXEC void error_collect_negabinary(T_data *shifted_data,
                                            T_error *errors, int num_bitplanes,
                                            int exp) {

    int batch_idx = FunctorBase<DeviceType>::GetBlockIdX() *
                        FunctorBase<DeviceType>::GetBlockDimX() +
                    FunctorBase<DeviceType>::GetThreadIdX();

    for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
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
        errors[num_bitplanes - bp_idx] += diff * diff;
      }
    }
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      T_data data = shifted_data[data_idx];
      errors[0] += data * data;
    }

    for (int bp_idx = 0; bp_idx < num_bitplanes + 1; bp_idx++) {
      errors[bp_idx] = ldexp(errors[bp_idx], 2 * (-(int)num_bitplanes + exp));
    }
  }

  MGARDX_EXEC void EncodeBinary() {
    SIZE gid = FunctorBase<DeviceType>::GetBlockIdX() *
                       FunctorBase<DeviceType>::GetBlockDimX() +
                   FunctorBase<DeviceType>::GetThreadIdX();

    SIZE tid = FunctorBase<DeviceType>::GetThreadIdX();

    SIZE lane_id = tid % BATCH_SIZE;
    SIZE warp_id = gid / BATCH_SIZE;

    SIZE num_batches = n / BATCH_SIZE;

    T_bitplane buffer;
    T_bitplane encoded_data[NUM_BITPLANES];
    T_bitplane encoded_sign;
    T_error errors;

    T_fp *sm_p = (T_fp *)FunctorBase<DeviceType>::GetSharedMemory();
    T_fp *fp_data = sm_p + BATCH_SIZE  * (tid/BATCH_SIZE);

    int exp;
    frexp(*abs_max((IDX)0), &exp);

    
    SIZE actual_batch_per_warp = min(M, num_batches - warp_id * M);

    #pragma unroll
    for (int i = 0; i < M; i++) {
      SIZE batch_idx = warp_id * M + i;
      // if (batch_idx < num_batches) {
        // actual_batch_per_warp++;
        T_data data = *v(batch_idx * BATCH_SIZE + lane_id);
        T_data shifted_data = ldexp(data, NUM_BITPLANES - exp);
        T_fp fp_data = (T_fp)fabs(shifted_data);
        T_fp fp_sign = (T_fp)(signbit(data) == 0 ? 0 : 1);
        #define FULL_MASK 0xffffffff

        #pragma unroll
        for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
          T_bitplane bit = (fp_data >> (NUM_BITPLANES - 1 - bp_idx)) & 1u;

          // option 1
          // T_bitplane shifted_bit = bit << BATCH_SIZE - 1 - data_idx;
          // for (int offset = 16; offset > 0; offset /= 2) {
          //   buffer |= __shfl_down_sync(FULL_MASK, shifted_bit, offset);
          // }
          // buffer = __shfl_sync(FULL_MASK, buffer, 0);

          // option 2
          // T_bitplane shifted_bit = bit << BATCH_SIZE - 1 - data_idx;
          // buffer = __reduce_add_sync(FULL_MASK, shifted_bit);
          // buffer = __shfl_sync(FULL_MASK, buffer, 0);

          // option 3
          // buffer = __match_any_sync(FULL_MASK, bit);
          // if (!bit) buffer ^= FULL_MASK;
          // buffer = __shfl_sync(FULL_MASK, buffer, 0);

          // option 4
          buffer = __ballot_sync(FULL_MASK, bit);

          // Save to mine registers
          if (lane_id == i) {
            encoded_data[bp_idx] = buffer;
          }
        }

        // option 5
        // fp_data[lane_id] = fp_data;
        


      
        // option 1
        // encoded_sign = fp_sign << BATCH_SIZE - 1 - data_idx;
        // for (int offset = 16; offset > 0; offset /= 2) {
        //   encoded_sign |= __shfl_down_sync(FULL_MASK, encoded_sign, offset);
        // }
        // option 2
        // encoded_sign = fp_sign << BATCH_SIZE - 1 - data_idx;
        // encoded_sign = __reduce_add_sync(FULL_MASK, encoded_sign);

        // option 3
        buffer = __ballot_sync(FULL_MASK, fp_sign);

        if (lane_id == i) {
          encoded_sign = buffer;
        }
      // }
    }

    // if (lane_id < actual_batch_per_warp) {
      #pragma unroll
      for (int bp_idx = 0; bp_idx < NUM_BITPLANES; bp_idx++) {
        *encoded_bitplanes(bp_idx, warp_id * M + lane_id) = encoded_data[bp_idx];
      }
      *encoded_bitplanes(0, num_batches + warp_id * M + lane_id) = encoded_sign;
      #pragma unroll
      for (int bp_idx = 1; bp_idx < NUM_BITPLANES; bp_idx++) {
        *encoded_bitplanes(bp_idx, num_batches + warp_id * M + lane_id) = (T_bitplane)0;
      }
    // }
  }

  MGARDX_EXEC void EncodeNegaBinary() {
    SIZE max_batches_per_warp = 32;
    SIZE global_batch_start = FunctorBase<DeviceType>::GetBlockIdX() * max_batches_per_warp;
    SIZE tid = FunctorBase<DeviceType>::GetThreadIdX();

    SIZE num_batches = (n - 1) / BATCH_SIZE + 1;
    SIZE num_batches_this_warp = min(num_batches - global_batch_start, max_batches_per_warp);
    T_data data;
    T_data shifted_data;
    T_fp fp_data;
    T_fp fp_sign;
    T_bitplane buffer;
    T_bitplane encoded_data[MAX_BITPLANES];
    T_bitplane encoded_sign;
    T_error errors;


    int exp;
    frexp(*abs_max((IDX)0), &exp); 
    exp += 2;

    SIZE data_idx = tid;
    SIZE my_batch_idx = tid;

    for (SIZE local_batch_idx = 0; local_batch_idx < num_batches_this_warp; local_batch_idx++) {
      SIZE global_batch_idx = global_batch_start + local_batch_idx;
      data = 0;
      if (global_batch_idx * BATCH_SIZE + data_idx < n) {
        data = *v(global_batch_idx * BATCH_SIZE + data_idx);
      }
      shifted_data = ldexp(data, num_bitplanes - exp);
      fp_data =
            Math<DeviceType>::binary2negabinary((T_sfp)shifted_data);
      #define FULL_MASK 0xffffffff
      for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
        // T_bitplane bit = (fp_data >> (num_bitplanes - 1 - bp_idx)) & 1u;
        T_bitplane bit = 1u;
        // T_bitplane shifted_bit = bit << BATCH_SIZE - 1 - data_idx;
        // option 1
        // for (int offset = 16; offset > 0; offset /= 2) {
        //   buffer |= __shfl_down_sync(FULL_MASK, shifted_bit, offset);
        // }
        // option 2
        // buffer = __reduce_add_sync(FULL_MASK, shifted_bit);

        // option 3
        // buffer = __match_any_sync(FULL_MASK, bit);
        // if (!bit) buffer ^= FULL_MASK;

        // option 4
        buffer = __ballot_sync(FULL_MASK, bit);

        // buffer = __shfl_sync(FULL_MASK, buffer, 0);
        if (my_batch_idx == local_batch_idx) {
          encoded_data[bp_idx] = buffer;
        }
      }


      for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
        // printf("thread %llu, encoded_data %u, \n", tid, encoded_data[bp_idx]);
        // print_bits(encoded_data[bp_idx], b);
        *encoded_bitplanes(bp_idx, global_batch_start + my_batch_idx) = encoded_data[bp_idx];
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
    size += sizeof(T_fp) * BATCH_SIZE * (256/32);
    return size;
  }

private:
  // parameters
  SIZE n;
  int num_bitplanes;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
};

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          typename T_error, SIZE NUM_BITPLANES, SIZE M, bool NegaBinary, bool CollectError,
          typename DeviceType>
class BPEncoderOptV2aKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "grouped bp encoder";
  static constexpr SIZE BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
  MGARDX_CONT
  BPEncoderOptV2aKernel(SIZE n, int num_bitplanes, SubArray<1, T_data, DeviceType> abs_max,
                       SubArray<1, T_data, DeviceType> v,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), num_bitplanes(num_bitplanes), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {}

  using FunctorType =
      BPEncoderOptV2aFunctor<T_data, T_fp, T_sfp, T_bitplane, T_error,
                            NUM_BITPLANES, M, NegaBinary, CollectError, DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {
    FunctorType functor(n, num_bitplanes, abs_max, v, encoded_bitplanes,
                        level_errors_workspace);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE num_batches = n / BATCH_SIZE;
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    // gridx = num_batches  / ((tbx/32)*32);
    gridx = (num_batches - 1) / ((tbx/32)*M) + 1;

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SIZE n;
  int num_bitplanes;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
};

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          SIZE M, bool NegaBinary, typename DeviceType>
class BPDecoderOptV2aFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPDecoderOptV2aFunctor() {}
  MGARDX_CONT
  BPDecoderOptV2aFunctor(SIZE n, int starting_bitplane, int num_bitplanes,
                        SubArray<1, T_data, DeviceType> abs_max,
                        SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                        SubArray<1, bool, DeviceType> signs,
                        SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane),
        num_bitplanes(num_bitplanes), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void decode_batch(T_fp *v, T_bitplane *encoded) {
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      T_fp buffer = 0;
      for (int bp_idx = 0; bp_idx < MAX_BITPLANES; bp_idx++) {
        T_fp bit = (encoded[bp_idx] >> (BATCH_SIZE - 1 - data_idx)) & 1u;
        buffer += bit << (MAX_BITPLANES - 1 - bp_idx);
      }
      v[data_idx] = buffer;
    }
  }

  MGARDX_EXEC void DecodeBinary() {
    SIZE gid = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();

    SIZE tid = FunctorBase<DeviceType>::GetThreadIdX();

    uint32_t lane_id = tid % BATCH_SIZE;
    SIZE warp_id = gid / BATCH_SIZE;

    SIZE num_batches = n / BATCH_SIZE;

    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_fp fp_sign[BATCH_SIZE];
    T_bitplane encoded_data[MAX_BITPLANES];
    T_bitplane encoded_sign;

    int exp;
    frexp(*abs_max((IDX)0), &exp);

    int ending_bitplane = starting_bitplane + num_bitplanes;

    SIZE actual_batch_per_warp = min(M, num_batches - warp_id * M);

    if (lane_id < actual_batch_per_warp) {
      #pragma unroll
      for (int bp_idx = 0; bp_idx < MAX_BITPLANES; bp_idx++) {
        encoded_data[bp_idx] = *encoded_bitplanes(bp_idx, warp_id * M + lane_id);
      }
      encoded_sign = *encoded_bitplanes(0, num_batches + warp_id * M + lane_id);

    }


    if (lane_id < actual_batch_per_warp) {
      // decode data
      decode_batch(fp_data, encoded_data);
      #pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        fp_sign[data_idx] = (encoded_sign >> (BATCH_SIZE - 1 - data_idx)) & 1u;
      }
      #pragma unroll
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        shifted_data[data_idx] = (T_data)fp_data[data_idx];
        // It is beneficial to use pow instead of ldexp
        T_data data = shifted_data[data_idx] * pow(2, -ending_bitplane + exp);
        // T_data data = ldexp(shifted_data[data_idx], -ending_bitplane + exp);
        data = fp_sign[data_idx] ? -data : data;
        shifted_data[data_idx] = data;
      }
    }

    for (u_int32_t mask = 0; mask < BATCH_SIZE; mask++) {
      // printf("lane_id: %d, mask: %u, lane_id^mask: %d\n", lane_id, mask, lane_id^mask);
      #define FULL_MASK 0xffffffff
      // T_data buffer = __shfl_xor_sync(FULL_MASK, shifted_data[lane_id^mask], mask);
      // shifted_data[lane_id^mask] = buffer;
    }

    for (int i = 0; i < actual_batch_per_warp; i++) {
      SIZE batch_idx = warp_id * M + i;
      *v(batch_idx * BATCH_SIZE + lane_id) = shifted_data[i];
    }

  }

  MGARDX_EXEC void DecodeNegaBinary() {
    SIZE gid = FunctorBase<DeviceType>::GetBlockIdX() *
                   FunctorBase<DeviceType>::GetBlockDimX() +
               FunctorBase<DeviceType>::GetThreadIdX();
    SIZE grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                     FunctorBase<DeviceType>::GetBlockDimX();
    SIZE num_batches = (n - 1) / BATCH_SIZE + 1;

    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_bitplane encoded_data[MAX_BITPLANES];

    int exp;
    frexp(*abs_max((IDX)0), &exp); 
    exp += 2;

    int ending_bitplane = starting_bitplane + num_bitplanes;

    for (SIZE batch_idx = gid; batch_idx < num_batches;
         batch_idx += grid_size) {

      for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
        encoded_data[bp_idx] =
            *encoded_bitplanes(starting_bitplane + bp_idx, batch_idx);
        // print_bits(encoded_data[bp_idx], batch_size);
      }
      // encode data
      decode_batch(fp_data, encoded_data, num_bitplanes);

      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = ldexp(
            (T_data)Math<DeviceType>::negabinary2binary(fp_data[data_idx]),
            -ending_bitplane + exp);
        if (batch_idx * BATCH_SIZE + data_idx < n) {
          *v(batch_idx * BATCH_SIZE + data_idx) =
              ending_bitplane % 2 != 0 ? -data : data;
        }
        // printf("%f: ", data); print_bits(fp_data[data_idx], b);
      }
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
  int num_bitplanes;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T_data, DeviceType> v;
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
};

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          SIZE M, bool NegaBinary, typename DeviceType>
class BPDecoderOptV2aKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "grouped bp decoder";
  static constexpr SIZE BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
  MGARDX_CONT
  BPDecoderOptV2aKernel(SIZE n, int starting_bitplane, int num_bitplanes,
                       SubArray<1, T_data, DeviceType> abs_max,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<1, bool, DeviceType> signs,
                       SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane),
        num_bitplanes(num_bitplanes), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {}

  using FunctorType = BPDecoderOptV2aFunctor<T_data, T_fp, T_sfp, T_bitplane,
                                            M, NegaBinary, DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {

    FunctorType functor(n, starting_bitplane, num_bitplanes, abs_max,
                        encoded_bitplanes, signs, v);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE num_batches = n / BATCH_SIZE;
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    // gridx = num_batches  / ((tbx/32)*32);
    gridx = (num_batches - 1) / ((tbx/32)*M) + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SIZE n;
  int starting_bitplane;
  int num_bitplanes;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T_data, DeviceType> v;
};

// general bitplane encoder that encodes data by block using T_stream type
// buffer
template <DIM D, typename T_data, typename T_bitplane, typename T_error,
          bool NegaBinary, bool CollectError, typename DeviceType>
class BPEncoderOptV2a
    : public concepts::BitplaneEncoderInterface<D, T_data, T_bitplane, T_error,
                                                CollectError, DeviceType> {
public:
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
  using T_sfp = typename std::conditional<std::is_same<T_data, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T_data, double>::value,
                                         uint64_t, uint32_t>::type;

  BPEncoderOptV2a() : initialized(false) {
    static_assert(std::is_floating_point<T_data>::value,
                  "GeneralBPEncoder: input data must be floating points.");
    static_assert(!std::is_same<T_data, long double>::value,
                  "GeneralBPEncoder: long double is not supported.");
    static_assert(std::is_unsigned<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
    static_assert(std::is_integral<T_bitplane>::value,
                  "GroupedBPBlockEncoder: streams must be unsigned integers.");
  }
  BPEncoderOptV2a(Hierarchy<D, T_data, DeviceType> &hierarchy) {
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
    SIZE max_level_num_elems = hierarchy.level_num_elems(hierarchy.l_target());

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

  void encode(SIZE n, int num_bitplanes, SubArray<1, T_data, DeviceType> abs_max,
              SubArray<1, T_data, DeviceType> v,
              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
              SubArray<1, T_error, DeviceType> level_errors, int queue_idx) {

    SubArray<2, T_error, DeviceType> level_errors_work(level_errors_work_array);

    constexpr SIZE max_batch_per_warp = 8;
    DeviceLauncher<DeviceType>::Execute(
        BPEncoderOptV2aKernel<T_data, T_fp, T_sfp, T_bitplane, T_error, MAX_BITPLANES, max_batch_per_warp,
                             NegaBinary, CollectError, DeviceType>(
            n, num_bitplanes, abs_max, v, encoded_bitplanes, level_errors_work),
        queue_idx);

    if constexpr (CollectError) {
      SIZE reduce_size = num_blocks(n);
      for (int i = 0; i < num_bitplanes + 1; i++) {
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
  void progressive_decode(SIZE n, int starting_bitplanes, int num_bitplanes,
                          SubArray<1, T_data, DeviceType> abs_max,
                          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                          SubArray<1, bool, DeviceType> level_signs, int level,
                          SubArray<1, T_data, DeviceType> v, int queue_idx) {

    constexpr SIZE max_batch_per_warp = 8;
    if (num_bitplanes > 0) {
      DeviceLauncher<DeviceType>::Execute(
          BPDecoderOptV2aKernel<T_data, T_fp, T_sfp, T_bitplane, max_batch_per_warp, NegaBinary,
                               DeviceType>(n, starting_bitplanes, num_bitplanes,
                                           abs_max, encoded_bitplanes, level_signs,
                                           v),
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
