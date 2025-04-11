/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeX.h"
#include <stdio.h>

#ifndef MGARD_X_MaxAbsIndexKernel
#define MGARD_X_MaxAbsIndexKernel

namespace mgard_x {

namespace data_refactoring {

namespace multi_dimension {

template <DIM D, typename T, typename DeviceType>
class MaxAbsIndexFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT MaxAbsIndexFunctor() {}
  MGARDX_CONT MaxAbsIndexFunctor(SubArray<1, double, DeviceType> x,
                                 SubArray<1, double, DeviceType> maxabs,
                                 SubArray<1, uint32_t, DeviceType> index)
      : x(x), maxabs(maxabs), index(index) {
    Functor<DeviceType>();
    n = x.shape(0);
  }

  MGARDX_EXEC void Operation1() {
    int tid = FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX() + FunctorBase<DeviceType>::GetThreadIdX();
    int stride = FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetGridDimX();

    for (int i = tid; i < n; i += stride){
      double value = (double)*x(i);
      if(value == (double)(*maxabs(0))) *index(0) = (uint32_t) i;
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SubArray<1, T, DeviceType> x;
  SubArray<1, double, DeviceType> maxabs;
  SubArray<1, uint32_t, DeviceType> index;
  uint32_t n;
};

template <DIM D, typename T, typename DeviceType>
class MaxAbsIndexKernel : public Kernel {
public:
  constexpr static std::string_view Name = "max abs index kernel";
  constexpr static double EnableAutoTuning() { return false; }
  MGARDX_CONT
  MaxAbsIndexKernel(SubArray<1, double, DeviceType> x,
                    SubArray<1, double, DeviceType> maxabs,
                    SubArray<1, uint32_t, DeviceType> index)
      : x(x), maxabs(maxabs), index(index) {}


  MGARDX_CONT Task<MaxAbsIndexFunctor<D, T, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = MaxAbsIndexFunctor<D, T, DeviceType>;
    FunctorType functor(x, maxabs, index);
    SIZE total_thread_x = x.shape(0);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = ceil((double)total_thread_x / tbx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<D, T, DeviceType> x;
  SubArray<1, double, DeviceType> maxabs;
  SubArray<1, uint32_t, DeviceType> index;
};

template <DIM D, typename T, typename DeviceType>
void Copy3D(SubArray<D, T, DeviceType> x,
            SubArray<1, double, DeviceType> maxabs,
            SubArray<1, uint32_t, DeviceType> out, int queue_idx) {
  
  DeviceLauncher<DeviceType>::Execute(MaxAbsIndexKernel<D, T, DeviceType>(x, maxabs, out), queue_idx);
  
}

} // namespace multi_dimension

} // namespace data_refactoring

} // namespace mgard_x

#endif