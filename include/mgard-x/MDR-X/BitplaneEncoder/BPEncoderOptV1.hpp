#ifndef _MDR_BP_ENCODER_OPT_V1_HPP
#define _MDR_BP_ENCODER_OPT_V1_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "BitplaneEncoderInterface.hpp"
#include <string.h>

namespace mgard_x {
namespace MDR {

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          typename T_error, bool NegaBinary, bool CollectError,
          typename DeviceType>
class BPEncoderOptV1Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPEncoderOptV1Functor() {}
  MGARDX_CONT
  BPEncoderOptV1Functor(SIZE n, int num_bitplanes,
                        SubArray<1, T_data, DeviceType> abs_max,
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
        T_bitplane bit =
            (v[data_idx] >> (num_bitplanes - 1 - bp_idx)) & (T_bitplane)1;
        buffer += bit << BATCH_SIZE - 1 - data_idx;
      }
      encoded[bp_idx] = buffer;
    }
  }

  MGARDX_EXEC void encode_batch_with_prediction(T_fp *v, T_bitplane *encoded,
                                                int num_bitplanes) {
    T_fp data_values[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) {
      data_values[i] = 0;
    }
    for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
      T_bitplane buffer = 0;
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_bitplane cur_bit =
            (v[data_idx] >> (num_bitplanes - 1 - bp_idx)) & (T_bitplane)1;
        if (bp_idx == 0) {
          buffer += cur_bit << BATCH_SIZE - 1 - data_idx;
        } else {
          if constexpr (NegaBinary) {
            data_values[data_idx] = data_values[data_idx] * (-2);
          } else {
            data_values[data_idx] = data_values[data_idx] * 2;
          }
          if (data_idx) {
            T_bitplane pred_bit = 0;
            T_fp d0 = abs((T_sfp)data_values[data_idx - 1] -
                          (T_sfp)data_values[data_idx]);
            T_fp d1 = abs((T_sfp)data_values[data_idx - 1] -
                          (T_sfp)data_values[data_idx] - 1);
            if (d0 > d1) {
              pred_bit = 1;
            }
            if (pred_bit != cur_bit) {
              buffer += (T_bitplane)1 << BATCH_SIZE - 1 - data_idx;
            }
          } else {
            buffer += cur_bit << BATCH_SIZE - 1 - data_idx;
          }
          data_values[data_idx] += cur_bit;
        }
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

    SIZE grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                     FunctorBase<DeviceType>::GetBlockDimX();

    SIZE num_batches = (n - 1) / BATCH_SIZE + 1;
    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_fp fp_sign[BATCH_SIZE];
    T_bitplane encoded_data[MAX_BITPLANES];
    T_bitplane encoded_sign[1];
    T_error errors[MAX_BITPLANES + 1];

    int exp;
    frexp(*abs_max((IDX)0), &exp);

    for (SIZE batch_idx = gid; batch_idx < num_batches;
         batch_idx += grid_size) {
      // SIZE batch_idx = gid;
      // if (batch_idx < num_batches) {
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = 0;
        if (batch_idx * BATCH_SIZE + data_idx < n) {
          data = *v(batch_idx * BATCH_SIZE + data_idx);
        }
        shifted_data[data_idx] = ldexp(data, num_bitplanes - exp);
        fp_data[data_idx] = (T_fp)fabs(shifted_data[data_idx]);
        fp_sign[data_idx] = (T_fp)(signbit(data) == 0 ? 0 : 1);
        // fp_data[data_idx] = (T_fp)fabs(shifted_data[data_idx]);
        // if (batch_idx == 0) {
        //   printf("fp_data[data_idx]: %llu\n", fp_data[data_idx]);
        // }
        // printf("%f: ", data); print_bits(fp_data[data_idx], b);
        // printf("data: %f, fp_data[data_idx]: %llu, signbit(data): %lld,
        // fp_sign[data_idx]: %llu \n", data, fp_data[data_idx], signbit(data),
        // fp_sign[data_idx]);
      }
      // encode data
      encode_batch(fp_data, encoded_data, num_bitplanes);
      for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
        *encoded_bitplanes(bp_idx, batch_idx) = encoded_data[bp_idx];
        // if (batch_idx == 0) {
        //   printf("encoded_data: %llu\n", encoded_data[bp_idx]);
        // }
        // print_bits(encoded_bitplanes[bp_idx * b + batch_idx * 2],
        // batch_size);
      }
      // encode sign
      encode_batch(fp_sign, encoded_sign, 1);

      // if (batch_idx == 0) {
      //   printf("encoded_sign: %u\n", encoded_sign[0]);
      // }

      *encoded_bitplanes(0, num_batches + batch_idx) = encoded_sign[0];
      // set rest of the bitplanes to 0
      for (int bp_idx = 1; bp_idx < num_bitplanes; bp_idx++) {
        *encoded_bitplanes(bp_idx, num_batches + batch_idx) = (T_bitplane)0;
      }
      // // encode sign
      // encode_batch(signs, encoded_sign, BATCH_SIZE, 1);
      // print_bits(encoded_bitplanes[0 * b + batch_idx * 2 + 1], batch_size);

      if constexpr (CollectError) {
        error_collect_binary(shifted_data, errors, num_bitplanes, exp);
        for (int bp_idx = 0; bp_idx < num_bitplanes + 1; bp_idx++) {
          *level_errors_workspace(bp_idx, batch_idx) = errors[bp_idx];
        }
      }
    }
  }

  MGARDX_EXEC void EncodeNegaBinary() {
    SIZE gid = FunctorBase<DeviceType>::GetBlockIdX() *
                   FunctorBase<DeviceType>::GetBlockDimX() +
               FunctorBase<DeviceType>::GetThreadIdX();

    SIZE grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                     FunctorBase<DeviceType>::GetBlockDimX();

    SIZE num_batches = (n - 1) / BATCH_SIZE + 1;
    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_bitplane encoded_data[MAX_BITPLANES];
    T_error errors[MAX_BITPLANES + 1];

    int exp;
    frexp(*abs_max((IDX)0), &exp);

    exp += 2;

    for (SIZE batch_idx = gid; batch_idx < num_batches;
         batch_idx += grid_size) {
      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = 0;
        if (batch_idx * BATCH_SIZE + data_idx < n) {
          data = *v(batch_idx * BATCH_SIZE + data_idx);
        }
        shifted_data[data_idx] = ldexp(data, num_bitplanes - exp);
        fp_data[data_idx] =
            Math<DeviceType>::binary2negabinary((T_sfp)shifted_data[data_idx]);
        // fp_data[data_idx] = (T_fp)fabs(shifted_data[data_idx]);

        // printf("%f: ", data); print_bits(fp_data[data_idx], b);
      }
      // encode data
      encode_batch(fp_data, encoded_data, num_bitplanes);
      for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
        *encoded_bitplanes(bp_idx, batch_idx) = encoded_data[bp_idx];
        // print_bits(encoded_bitplanes[bp_idx * b + batch_idx * 2],
        // batch_size);
      }

      if constexpr (CollectError) {
        error_collect_negabinary(shifted_data, errors, num_bitplanes, exp);
        for (int bp_idx = 0; bp_idx < num_bitplanes + 1; bp_idx++) {
          *level_errors_workspace(bp_idx, batch_idx) = errors[bp_idx];
        }
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
  int num_bitplanes;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<1, T_data, DeviceType> v;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<2, T_error, DeviceType> level_errors_workspace;
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
};

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          typename T_error, bool NegaBinary, bool CollectError,
          typename DeviceType>
class BPEncoderOptV1Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "grouped bp encoder";
  MGARDX_CONT
  BPEncoderOptV1Kernel(SIZE n, int num_bitplanes,
                       SubArray<1, T_data, DeviceType> abs_max,
                       SubArray<1, T_data, DeviceType> v,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<2, T_error, DeviceType> level_errors_workspace)
      : n(n), num_bitplanes(num_bitplanes), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), v(v),
        level_errors_workspace(level_errors_workspace) {}

  using FunctorType =
      BPEncoderOptV1Functor<T_data, T_fp, T_sfp, T_bitplane, T_error,
                            NegaBinary, CollectError, DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {
    FunctorType functor(n, num_bitplanes, abs_max, v, encoded_bitplanes,
                        level_errors_workspace);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE repeat_factor = 16;
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (n - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(),
                     gridx / repeat_factor);
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
          bool NegaBinary, typename DeviceType>
class BPDecoderOptV1Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  BPDecoderOptV1Functor() {}
  MGARDX_CONT
  BPDecoderOptV1Functor(SIZE n, SIZE starting_bitplane, int num_bitplanes,
                        SubArray<1, T_data, DeviceType> abs_max,
                        SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                        SubArray<1, bool, DeviceType> signs,
                        SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane),
        num_bitplanes(num_bitplanes), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void decode_batch(T_fp *v, T_bitplane *encoded,
                                int num_bitplanes) {
    for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
      T_fp buffer = 0;
      for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++) {
        T_fp bit = (encoded[bp_idx] >> (BATCH_SIZE - 1 - data_idx)) & (T_fp)1;
        buffer += bit << (num_bitplanes - 1 - bp_idx);
      }
      v[data_idx] = buffer;
    }
  }

  MGARDX_EXEC void DecodeBinary() {
    SIZE gid = FunctorBase<DeviceType>::GetBlockIdX() *
                   FunctorBase<DeviceType>::GetBlockDimX() +
               FunctorBase<DeviceType>::GetThreadIdX();
    SIZE grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                     FunctorBase<DeviceType>::GetBlockDimX();
    SIZE num_batches = (n - 1) / BATCH_SIZE + 1;

    T_data shifted_data[BATCH_SIZE];
    T_fp fp_data[BATCH_SIZE];
    T_fp fp_sign[BATCH_SIZE];
    T_bitplane encoded_data[MAX_BITPLANES];
    T_bitplane encoded_sign[MAX_BITPLANES];

    int exp;
    frexp(*abs_max((IDX)0), &exp);

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

      if (starting_bitplane == 0) {
        // decode sign
        encoded_sign[0] = *encoded_bitplanes(0, num_batches + batch_idx);
        decode_batch(fp_sign, encoded_sign, 1);
        for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
          *signs(batch_idx * BATCH_SIZE + data_idx) = fp_sign[data_idx];
        }
      } else {
        for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
          fp_sign[data_idx] = *signs(batch_idx * BATCH_SIZE + data_idx);
        }
      }

      for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
        T_data data = ldexp((T_data)fp_data[data_idx], -ending_bitplane + exp);
        if (batch_idx * BATCH_SIZE + data_idx < n) {
          *v(batch_idx * BATCH_SIZE + data_idx) =
              fp_sign[data_idx] ? -data : data;
        }
        // printf("data: %f, fp_data[data_idx]: %llu\n", *v(batch_idx *
        // BATCH_SIZE + data_idx), fp_data[data_idx]); printf("%f: ", data);
        // print_bits(fp_data[data_idx], b);
      }
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
  SIZE starting_bitplane;
  int num_bitplanes;
  SubArray<1, T_data, DeviceType> abs_max;
  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
  SubArray<1, bool, DeviceType> signs;
  SubArray<1, T_data, DeviceType> v;
  static constexpr int BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
};

template <typename T_data, typename T_fp, typename T_sfp, typename T_bitplane,
          bool NegaBinary, typename DeviceType>
class BPDecoderOptV1Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "grouped bp decoder";
  MGARDX_CONT
  BPDecoderOptV1Kernel(SIZE n, SIZE starting_bitplane, int num_bitplanes,
                       SubArray<1, T_data, DeviceType> abs_max,
                       SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                       SubArray<1, bool, DeviceType> signs,
                       SubArray<1, T_data, DeviceType> v)
      : n(n), starting_bitplane(starting_bitplane),
        num_bitplanes(num_bitplanes), abs_max(abs_max),
        encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {}

  using FunctorType = BPDecoderOptV1Functor<T_data, T_fp, T_sfp, T_bitplane,
                                            NegaBinary, DeviceType>;
  using TaskType = Task<FunctorType>;

  MGARDX_CONT TaskType GenTask(int queue_idx) {

    FunctorType functor(n, starting_bitplane, num_bitplanes, abs_max,
                        encoded_bitplanes, signs, v);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE repeat_factor = 8;
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (n - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(),
                     gridx / repeat_factor);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SIZE n;
  SIZE starting_bitplane;
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
class BPEncoderOptV1
    : public concepts::BitplaneEncoderInterface<D, T_data, T_bitplane, T_error,
                                                CollectError, DeviceType> {
public:
  static constexpr SIZE BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr int MAX_BITPLANES = sizeof(T_data) * 8;
  using T_sfp = typename std::conditional<std::is_same<T_data, double>::value,
                                          int64_t, int32_t>::type;
  using T_fp = typename std::conditional<std::is_same<T_data, double>::value,
                                         uint64_t, uint32_t>::type;

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
    SIZE max_level_num_elems =
        round_up(hierarchy.level_num_elems(hierarchy.l_target()), BATCH_SIZE);

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

  void encode(SIZE n, int num_bitplanes,
              SubArray<1, T_data, DeviceType> abs_max,
              SubArray<1, T_data, DeviceType> v,
              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
              SubArray<1, T_error, DeviceType> level_errors, int queue_idx) {

    SubArray<2, T_error, DeviceType> level_errors_work(level_errors_work_array);

    DeviceLauncher<DeviceType>::Execute(
        BPEncoderOptV1Kernel<T_data, T_fp, T_sfp, T_bitplane, T_error,
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

  void decode(SIZE n, int num_bitplanes,
              SubArray<1, T_data, DeviceType> abs_max,
              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes, int level,
              SubArray<1, T_data, DeviceType> v, int queue_idx) {}

  // decode the data and record necessary information for progressiveness
  void progressive_decode(SIZE n, int starting_bitplane, int num_bitplanes,
                          SubArray<1, T_data, DeviceType> abs_max,
                          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                          SubArray<1, bool, DeviceType> level_signs, int level,
                          SubArray<1, T_data, DeviceType> v, int queue_idx) {

    if (num_bitplanes > 0) {
      DeviceLauncher<DeviceType>::Execute(
          BPDecoderOptV1Kernel<T_data, T_fp, T_sfp, T_bitplane, NegaBinary,
                               DeviceType>(n, starting_bitplane, num_bitplanes,
                                           abs_max, encoded_bitplanes,
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
