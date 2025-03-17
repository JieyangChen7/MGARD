#ifndef _MDR_BP_ENCODER_OPT_HPP
#define _MDR_BP_ENCODER_OPT_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "BitplaneEncoderInterface.hpp"
#include <string.h>

namespace mgard_x
{
  namespace MDR
  {
    template <typename T, typename T_fp, typename T_sfp, typename T_bitplane,
              typename T_error, typename DeviceType, int NumBitplanes>
    class BPEncoderOptFunctor : public Functor<DeviceType>
    {
    public:
      MGARDX_CONT
      BPEncoderOptFunctor() {}
      MGARDX_CONT
      BPEncoderOptFunctor(SIZE n, SIZE batches_per_thread, SIZE exp,
                          SubArray<1, T, DeviceType> v,
                          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                          SubArray<2, T_error, DeviceType> level_errors_workspace)
          : n(n), batches_per_thread(batches_per_thread), exp(exp), encoded_bitplanes(encoded_bitplanes),
            v(v), level_errors_workspace(level_errors_workspace)
      {
        Functor<DeviceType>();
      }

      template <typename T_org, typename T_trans>
      MGARDX_EXEC void encode_batch(T_org *v, T_trans *encoded, SIZE batch_size,
                                    SIZE b)
      {
        for (SIZE bp_idx = 0; bp_idx < b; bp_idx++)
        {
          T_trans buffer = 0;
          for (SIZE data_idx = 0; data_idx < batch_size; data_idx++)
          {
            T_trans bit = (v[data_idx] >> (sizeof(T_org) * 8 - 1 - bp_idx)) & 1u;
            buffer += bit << (sizeof(T_trans) * 8 - 1 - data_idx);
          }
          encoded[bp_idx] = buffer;
        }
      }

      MGARDX_EXEC void ErrorCollect(T *v, T_error *errors, SIZE num_elems, SIZE num_bitplanes)
      {
        for (int bp_idx = 0; bp_idx < num_bitplanes; bp_idx++)
        {
          T_fp mask = ((T_fp)1 << bp_idx) - 1;
          for (int data_idx = 0; data_idx < num_elems; data_idx++)
          {
            T data = v[data_idx];
            T_fp fp_data = (T_fp)fabs(v[data_idx]);
            T_error mantissa = fabs(data) - fp_data;
            T_error diff = (T_error)(fp_data & mask) + mantissa;
            errors[num_bitplanes - bp_idx] += diff * diff;
          }
        }
        for (int data_idx = 0; data_idx < num_elems; data_idx++)
        {
          T data = v[data_idx];
          errors[0] += (data * data);
        }
      }

      MGARDX_EXEC void Operation1()
      {
        constexpr SIZE batch_size = sizeof(T_bitplane) * 8;
        SIZE total_batches = (n + batch_size - 1) / batch_size;

        SIZE total_batches_per_TB = FunctorBase<DeviceType>::GetBlockDimX() * batches_per_thread;

        // block_offset is used for storing data and signs(for encoded_bitplanes)
        SIZE block_offset = FunctorBase<DeviceType>::GetBlockIdX() * (2 * total_batches_per_TB);
        SIZE local_thread_idx = FunctorBase<DeviceType>::GetThreadIdX();

        T_error error_vals[NumBitplanes + 1];
        for (int i = 0; i < NumBitplanes + 1; i++)
        {
          error_vals[i] = 0;
        }
        // Every thread just store 1 error value
        SIZE global_error_index = FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetBlockIdX() + local_thread_idx;

        for (int b = 0; b < batches_per_thread; b++)
        {
          // Variables for input
          SIZE local_batch_idx = local_thread_idx * batches_per_thread + b;
          // points to current batch index
          SIZE global_batch_index = FunctorBase<DeviceType>::GetBlockIdX() * total_batches_per_TB + local_batch_idx;

          // Variables for output
          // Address to store sign
          SIZE global_batch_sign = block_offset + local_thread_idx * batches_per_thread * 2 + batches_per_thread + b;
          // Address to store data
          SIZE global_batch_data = block_offset + local_thread_idx * batches_per_thread * 2 + b;

          if (global_batch_index >= total_batches)
            continue;

          T shifted_data[batch_size];
          T_fp fp_data[batch_size];
          T_fp signs[batch_size];
          T_bitplane encoded_data[NumBitplanes];
          T_bitplane encoded_sign[1];

          for (SIZE local_data_idx = 0; local_data_idx < batch_size; local_data_idx++)
          {
            SIZE global_index = global_batch_index * batch_size + local_data_idx;

            T d = (global_index < n) ? *v(global_index) : 0;
            T shifted = ldexp(d, (int)NumBitplanes - (int)exp);

            // By default we are using BINARY data types
            T_fp fix_point = (T_fp)fabs(shifted);

            shifted_data[local_data_idx] = shifted;
            fp_data[local_data_idx] = fix_point;
            signs[local_data_idx] = ((T_sfp)signbit(d)) << (sizeof(T_fp) * 8 - 1);
          }

          // Use encode_batch to encode both data and sign
          encode_batch<T_fp, T_bitplane>(fp_data, encoded_data, batch_size, NumBitplanes);
          encode_batch<T_fp, T_bitplane>(signs, encoded_sign, batch_size, 1);

          // Write data
          for (SIZE bp_idx = 0; bp_idx < NumBitplanes; bp_idx++)
          {
            *encoded_bitplanes(bp_idx, global_batch_data) = encoded_data[bp_idx];
          }
          // Write sign
          *encoded_bitplanes(0, global_batch_sign) = encoded_sign[0];

          // Collect errors
          ErrorCollect(shifted_data, error_vals, batch_size, NumBitplanes);
        }
        // Boundary check
        if (global_error_index < (total_batches + batches_per_thread - 1) / batches_per_thread)
        {
          // Scale and store errors
          for (int bp_idx = 0; bp_idx < NumBitplanes + 1; bp_idx++)
          {
            error_vals[bp_idx] = ldexp(error_vals[bp_idx], 2 * (exp - (int)NumBitplanes));
          }
          // Following the original storing format
          for (int bp_idx = 0; bp_idx < NumBitplanes + 1; bp_idx++)
          {
            // In block index, for here we can use threadIdx
            *level_errors_workspace(bp_idx, global_error_index) = error_vals[bp_idx];
          }
        }
      }

      // Save back to global memory
      MGARDX_EXEC void Operation3() {}

      MGARDX_CONT size_t shared_memory_size()
      {
        size_t size = 0;

        return size;
      }

    private:
      // parameters
      SIZE n;
      SIZE exp;
      SIZE batches_per_thread;
      SubArray<1, T, DeviceType> v;
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
      SubArray<2, T_error, DeviceType> level_errors_workspace;

      IDX batch_idx;
    };

    template <typename T, typename T_bitplane, typename T_error,
              typename DeviceType, int NumBitplanes>
    class BPEncoderOptKernel : public Kernel
    {
    public:
      constexpr static bool EnableAutoTuning() { return false; }
      constexpr static std::string_view Name = "bp encoder opt";
      MGARDX_CONT
      BPEncoderOptKernel(SIZE n, SIZE batches_per_thread, SIZE exp, SubArray<1, T, DeviceType> v,
                         SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                         SubArray<2, T_error, DeviceType> level_errors_workspace)
          : n(n), batches_per_thread(batches_per_thread), exp(exp), encoded_bitplanes(encoded_bitplanes), v(v),
            level_errors_workspace(level_errors_workspace) {}

      using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                              int64_t, int32_t>::type;
      using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                             uint64_t, uint32_t>::type;
      using FunctorType = BPEncoderOptFunctor<T, T_fp, T_sfp, T_bitplane, T_error,
                                              DeviceType, NumBitplanes>;
      using TaskType = Task<FunctorType>;

      MGARDX_CONT TaskType GenTask(int queue_idx)
      {
        FunctorType functor(n, batches_per_thread, exp, v, encoded_bitplanes, level_errors_workspace);
        SIZE tbx, tby, tbz, gridx, gridy, gridz;
        size_t sm_size = functor.shared_memory_size();
        tbz = 1;
        tby = 1;
        tbx = 32;
        constexpr SIZE batch_size = sizeof(T_bitplane) * 8;
        SIZE num_batches = (n - 1) / batch_size + 1;
        gridz = 1;
        gridy = 1;
        gridx = (num_batches - 1) / (tbx * batches_per_thread) + 1;
        return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                    std::string(Name));
      }

    private:
      SIZE n;
      SIZE batches_per_thread;
      SIZE exp;
      SubArray<1, T, DeviceType> v;
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
      SubArray<2, T_error, DeviceType> level_errors_workspace;
    };

    template <typename T, typename T_fp, typename T_sfp, typename T_bitplane,
              typename DeviceType, int NumBitplanes>
    class BPDecoderOptFunctor : public Functor<DeviceType>
    {
    public:
      MGARDX_CONT
      BPDecoderOptFunctor() {}
      MGARDX_CONT
      BPDecoderOptFunctor(SIZE n, SIZE num_batches_per_TB, SIZE starting_bitplane,
                          SIZE num_bitplanes, SIZE exp,
                          SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                          SubArray<1, bool, DeviceType> signs,
                          SubArray<1, T, DeviceType> v)
          : n(n), num_batches_per_TB(num_batches_per_TB), starting_bitplane(starting_bitplane),
            num_bitplanes(num_bitplanes), exp(exp), encoded_bitplanes(encoded_bitplanes),
            signs(signs), v(v)
      {
        Functor<DeviceType>();
      }

      MGARDX_EXEC void Operation1()
      {
        constexpr SIZE batch_size = sizeof(T_bitplane) * 8;
        SIZE num_batches = (n + batch_size - 1) / batch_size;

        SIZE block_offset = FunctorBase<DeviceType>::GetBlockIdX() * (2 * num_batches_per_TB);

        SIZE tid = FunctorBase<DeviceType>::GetThreadIdX();

        SIZE ending_bitplane = starting_bitplane + num_bitplanes;

        for (SIZE i = 0; i < num_batches_per_TB; i++)
        {
          // Re-calculate the index for data and sign
          SIZE local_batch_idx = i;
          SIZE global_batch_idx = block_offset / 2 + local_batch_idx;
          SIZE global_batch_data = block_offset + local_batch_idx;
          SIZE global_batch_sign = block_offset + num_batches_per_TB + local_batch_idx;

          if (global_batch_idx >= num_batches)
          {
            break;
          }

          SIZE data_idx = tid % 32;
          // Index for original data
          SIZE global_data_idx = global_batch_idx * batch_size + data_idx;

          T_bitplane sign_mask = *encoded_bitplanes(0, global_batch_sign);
          bool sign = ((sign_mask >> (sizeof(T_bitplane) * 8 - 1 - data_idx)) & 1u) != 0;

          // by default we use ALIGN_LEFT
          T_fp fp_val = 0;

          for (SIZE bp = starting_bitplane; bp < ending_bitplane; bp++)
          {
            T_bitplane data_mask = *encoded_bitplanes(bp, global_batch_data);
            T_fp bit = (data_mask >> (sizeof(T_bitplane) * 8 - 1 - data_idx)) & 1u;
            // Pay attention to possible changes here
            fp_val += bit << (ending_bitplane - 1 - bp);
          }
          // printf("Original Value: %u\n", fp_val);
          if (global_data_idx < n)
          {

            T shifted = ldexp((T)(fp_val), -ending_bitplane + exp);
            *v(global_data_idx) = sign ? -shifted : shifted;
            *signs(global_data_idx) = sign;
          }
        }
      }

      MGARDX_CONT size_t shared_memory_size()
      {
        return 0;
      }

    private:
      SIZE n;
      SIZE starting_bitplane;
      SIZE num_batches_per_TB;
      SIZE num_bitplanes;
      SIZE exp;
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes;
      SubArray<1, bool, DeviceType> signs;
      SubArray<1, T, DeviceType> v;
    };

    template <typename T, typename T_bitplane, typename DeviceType, int NumBitplanes>
    class BPDecoderOptKernel : public Kernel
    {
    public:
      constexpr static bool EnableAutoTuning() { return false; }
      constexpr static std::string_view Name = "bp decoder opt";
      MGARDX_CONT
      BPDecoderOptKernel(SIZE n, SIZE num_batches_per_TB, SIZE starting_bitplane,
                         SIZE num_bitplanes, SIZE exp,
                         SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                         SubArray<1, bool, DeviceType> signs,
                         SubArray<1, T, DeviceType> v)
          : n(n), num_batches_per_TB(num_batches_per_TB), starting_bitplane(starting_bitplane),
            num_bitplanes(num_bitplanes), exp(exp),
            encoded_bitplanes(encoded_bitplanes), signs(signs), v(v) {}

      using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                              int64_t, int32_t>::type;
      using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                             uint64_t, uint32_t>::type;
      using FunctorType =
          BPDecoderOptFunctor<T, T_fp, T_sfp, T_bitplane,
                              DeviceType, NumBitplanes>;
      using TaskType = Task<FunctorType>;

      MGARDX_CONT TaskType GenTask(int queue_idx)
      {
        FunctorType functor(n, num_batches_per_TB, starting_bitplane, num_bitplanes, exp, encoded_bitplanes, signs, v);
        SIZE tbx, tby, tbz, gridx, gridy, gridz;
        size_t sm_size = functor.shared_memory_size();
        tbz = 1;
        tby = 1;
        tbx = 32;
        gridz = 1;
        gridy = 1;
        constexpr SIZE batch_size = sizeof(T_bitplane) * 8;
        SIZE num_batches = (n + batch_size - 1) / batch_size;
        gridx = (num_batches + num_batches_per_TB - 1) / num_batches_per_TB;
        return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx, std::string(Name));
      }

    private:
      SIZE n;
      SIZE num_batches_per_TB;
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
              typename DeviceType, int NumBitplanes>
    class GroupedBPEncoderOpt
        : public concepts::BitplaneEncoderInterface<D, T_data, T_bitplane, T_error,
                                                    DeviceType>
    {
    public:
      GroupedBPEncoderOpt() : initialized(false)
      {
        static_assert(std::is_floating_point<T_data>::value,
                      "GeneralBPEncoder: input data must be floating points.");
        static_assert(!std::is_same<T_data, long double>::value,
                      "GeneralBPEncoder: long double is not supported.");
        static_assert(std::is_unsigned<T_bitplane>::value,
                      "GroupedBPBlockEncoder: streams must be unsigned integers.");
        static_assert(std::is_integral<T_bitplane>::value,
                      "GroupedBPBlockEncoder: streams must be unsigned integers.");
      }
      GroupedBPEncoderOpt(Hierarchy<D, T_data, DeviceType> &hierarchy)
      {
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

      void Adapt(Hierarchy<D, T_data, DeviceType> &hierarchy, int queue_idx)
      {
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

      static size_t EstimateMemoryFootprint(std::vector<SIZE> shape)
      {
        Hierarchy<D, T_data, DeviceType> hierarchy(shape, Config());
        SIZE max_level_num_elems = hierarchy.level_num_elems(hierarchy.l_target());
        SIZE max_bitplane = 64;
        size_t size = 0;
        size += hierarchy.EstimateMemoryFootprint(shape);
        size +=
            (max_bitplane + 1) * num_blocks(max_level_num_elems) * sizeof(T_error);
        for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++)
        {
          size += hierarchy.level_num_elems(level_idx) * sizeof(bool);
        }
        return size;
      }

      void encode(SIZE n, int32_t exp, SubArray<1, T_data, DeviceType> v,
                  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                  SubArray<1, T_error, DeviceType> level_errors,
                  std::vector<SIZE> &streams_sizes, int queue_idx)
      {

        SubArray<2, T_error, DeviceType> level_errors_work(level_errors_work_array);

        DeviceLauncher<DeviceType>::Execute(
            BPEncoderOptKernel<T_data, T_bitplane, T_error, DeviceType,
                               NumBitplanes>(n, num_batches_per_TB, exp, v, encoded_bitplanes,
                                             level_errors_work),
            queue_idx);

        SIZE reduce_size = num_blocks(n);
        for (int i = 0; i < NumBitplanes + 1; i++)
        {
          SubArray<1, T_error, DeviceType> curr_errors({reduce_size},
                                                       level_errors_work(i, 0));
          SubArray<1, T_error, DeviceType> sum_error({1}, level_errors(i));
          DeviceCollective<DeviceType>::Sum(reduce_size, curr_errors, sum_error,
                                            level_error_sum_work_array, true,
                                            queue_idx);
        }
        for (int i = 0; i < NumBitplanes; i++)
        {
          streams_sizes[i] = buffer_size(n) * sizeof(T_bitplane);
        }
      }

      // Do not implement
      void encode(SIZE n, SIZE num_bitplanes, int32_t exp,
                  SubArray<1, T_data, DeviceType> v,
                  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                  SubArray<1, T_error, DeviceType> level_errors,
                  std::vector<SIZE> &streams_sizes, int queue_idx) {}

      // Do not implement
      void decode(SIZE n, SIZE num_bitplanes, int32_t exp,
                  SubArray<2, T_bitplane, DeviceType> encoded_bitplanes, int level,
                  SubArray<1, T_data, DeviceType> v, int queue_idx) {}

      // decode the data and record necessary information for progressiveness
      void progressive_decode(SIZE n, SIZE starting_bitplane, SIZE num_bitplanes, int32_t exp,
                              SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                              SubArray<1, bool, DeviceType> level_signs, int level,
                              SubArray<1, T_data, DeviceType> v, int queue_idx)
      {
        if (num_bitplanes > 0)
        {
          Timer timer;
          DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
          timer.start();
          DeviceLauncher<DeviceType>::Execute(
              BPDecoderOptKernel<T_data, T_bitplane, DeviceType, NumBitplanes>(
                  n, num_batches_per_TB, starting_bitplane, num_bitplanes, exp,
                  encoded_bitplanes, level_signs, v),
              queue_idx);
          DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
          timer.end();
          timer.print("progressive_decode");
        }
      }

      static SIZE buffer_size(SIZE n)
      {
        constexpr SIZE batch_size = sizeof(T_bitplane) * 8;
        SIZE num_elems_per_TB = batch_size * num_batches_per_TB;
        SIZE num_blocks = (n - 1) / num_elems_per_TB + 1;

        constexpr SIZE bitplane_max_length_per_TB = num_batches_per_TB * 2;
        SIZE bitplane_max_length_total = bitplane_max_length_per_TB * num_blocks;
        return bitplane_max_length_total;
      }

      static SIZE num_blocks(SIZE n)
      {
        constexpr SIZE batch_size = sizeof(T_bitplane) * 8;
        return (n - 1) / batch_size + 1;
      }

      void print() const { std::cout << "Grouped bitplane encoder" << std::endl; }

    private:
      bool initialized;
      Hierarchy<D, T_data, DeviceType> *hierarchy;
      static constexpr SIZE num_batches_per_TB = 2;
      Array<2, T_error, DeviceType> level_errors_work_array;
      Array<1, Byte, DeviceType> level_error_sum_work_array;
      std::vector<std::vector<uint8_t>> level_recording_bitplanes;
    };
  } // namespace MDR
} // namespace mgard_x
#endif