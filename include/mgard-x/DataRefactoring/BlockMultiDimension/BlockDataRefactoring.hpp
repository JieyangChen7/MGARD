/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_BLOCK_MULTI_DIMENSION_REFACTORING_TEMPLATE
#define MGARD_X_BLOCK_MULTI_DIMENSION_REFACTORING_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"
#include "GPKFunctor.h"

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class BlockDecomposition : public Functor<DeviceType> {
public:
  MGARDX_CONT BlockDecomposition() {}
  MGARDX_CONT BlockDecomposition(SubArray<D, T, DeviceType> v, SIZE l_target): v(v), l_target(l_target) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {

    sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    v_sm = sm;
    w_sm = sm + F * C * R * sizeof(T);
    r = R;
    c = C;
    f = F;

    r_gl = FunctorBase<DeviceType>::GetBlockIdZ() *
           FunctorBase<DeviceType>::GetBlockDimZ() +
           FunctorBase<DeviceType>::GetThreadIdZ();

    c_gl = FunctorBase<DeviceType>::GetBlockIdY() *
           FunctorBase<DeviceType>::GetBlockDimY() +
           FunctorBase<DeviceType>::GetThreadIdY();

    f_gl = FunctorBase<DeviceType>::GetBlockIdX() *
           FunctorBase<DeviceType>::GetBlockDimX() +
           FunctorBase<DeviceType>::GetThreadIdX();

    r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();

    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();

    // Check if out of bound
    if (r_gl >= v.shape(D-3) || c_gl >= v.shape(D-2) || f_gl >= v.shape(D-1)) {
      return;
    }
    w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = *v(r_gl, c_gl, f_gl);
  }

  MGARDX_EXEC void Operation2() {
    for (int l = l_target; l > 0; l--) {
      rr = r / 2 + 1;
      cc = c / 2 + 1;
      ff = f / 2 + 1;
      
      SIZE base = 0;
      // Coarse
      // 27
      if (base <= threadId && threadId < rr * cc * ff) {
        r_sm = (threadId / ff) / cc;
        c_sm = (threadId / ff) % cc;
        f_sm = threadId % ff;
        v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      }
      base += rr * cc * ff;

      // f - 18
      if (threadId < rr * cc * (ff-1)) {

      }

      // c - 18
      if (threadId < rr * (cc-1) * ff) {

      }

      // r - 18
      if (threadId < (rr-1) * cc * ff) {

      }

      // cf - 12
      if (threadId < rr * (cc-1) * (ff-1)) {

      }

      // rf - 12
      if (threadId < (rr-1) * cc * (ff-1)) {

      }

      // rc - 12
      if (threadId < (rr-1) * (cc-1) * ff) {

      }

      //rcf - 8
      if (threadId < (rr-1) * (cc-1) * (ff-1)) {

      }
    }

  }

private:
  // functor parameters
  SubArray<D, T, DeviceType> v
  SIZE l_target;
  // thread local variables
  SIZE r_sm, c_sm, f_sm;
  SIZE r_gl, c_gl, f_gl;
  SIZE threadId;
  T *sm;
  const SIZE ld1 = F;
  const SIZE ld2 = C;
  T *v_sm, *w_sm;
  SIZE r, c, f, rr, cc, ff;
};

template <DIM D, typename T, typename DeviceType>
class GpkReo3D : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  GpkReo3D() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<GpkReo3DFunctor<D, T, R, C, F, DeviceType>>
  GenTask(SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
          SubArray<1, T, DeviceType> ratio_r,
          SubArray<1, T, DeviceType> ratio_c,
          SubArray<1, T, DeviceType> ratio_f, SubArray<D, T, DeviceType> v,
          SubArray<D, T, DeviceType> w, SubArray<D, T, DeviceType> wf,
          SubArray<D, T, DeviceType> wc, SubArray<D, T, DeviceType> wr,
          SubArray<D, T, DeviceType> wcf, SubArray<D, T, DeviceType> wrf,
          SubArray<D, T, DeviceType> wrc, SubArray<D, T, DeviceType> wrcf,
          int queue_idx) {
    using FunctorType = GpkReo3DFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(nr, nc, nf, nr_c, nc_c, nf_c, ratio_r, ratio_c, ratio_f,
                        v, w, wf, wc, wr, wcf, wrf, wrc, wrcf);

    SIZE total_thread_z = std::max(nr - 1, (SIZE)1);
    SIZE total_thread_y = std::max(nc - 1, (SIZE)1);
    SIZE total_thread_x = std::max(nf - 1, (SIZE)1);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size;
    tbz = R;
    tby = C;
    tbx = F;
    sm_size = R * C * F * sizeof(T) * 2;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "GpkReo3D");
  }

  MGARDX_CONT
  void Execute(SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
               SubArray<1, T, DeviceType> ratio_r,
               SubArray<1, T, DeviceType> ratio_c,
               SubArray<1, T, DeviceType> ratio_f, SubArray<D, T, DeviceType> v,
               SubArray<D, T, DeviceType> w, SubArray<D, T, DeviceType> wf,
               SubArray<D, T, DeviceType> wc, SubArray<D, T, DeviceType> wr,
               SubArray<D, T, DeviceType> wcf, SubArray<D, T, DeviceType> wrf,
               SubArray<D, T, DeviceType> wrc, SubArray<D, T, DeviceType> wrcf,
               int queue_idx) {
    int range_l = std::min(6, (int)std::log2(nf) - 1);
    int prec = TypeToIdx<T>();
    int config =
        AutoTuner<DeviceType>::autoTuningTable.gpk_reo_3d[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define GPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = GPK_CONFIG[D - 1][CONFIG][0];                                \
    const int C = GPK_CONFIG[D - 1][CONFIG][1];                                \
    const int F = GPK_CONFIG[D - 1][CONFIG][2];                                \
    using FunctorType = GpkReo3DFunctor<D, T, R, C, F, DeviceType>;            \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<R, C, F>(nr, nc, nf, nr_c, nc_c, nf_c, ratio_r,    \
                                     ratio_c, ratio_f, v, w, wf, wc, wr, wcf,  \
                                     wrf, wrc, wrcf, queue_idx);               \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    GPK(6) if (!ret.success) config--;
    GPK(5) if (!ret.success) config--;
    GPK(4) if (!ret.success) config--;
    GPK(3) if (!ret.success) config--;
    GPK(2) if (!ret.success) config--;
    GPK(1) if (!ret.success) config--;
    GPK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err << "no suitable config for GpkReo3D.\n";
      exit(-1);
    }
#undef GPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("gpk_reo_3d", prec, range_l, min_config);
    }
  }
};


} // namespace mgard_x

#endif