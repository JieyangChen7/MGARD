/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeX.h"


#ifndef MGARD_X_QoIKernel
#define MGARD_X_QoIKernel

namespace mgard_x {

namespace data_refactoring {

namespace multi_dimension {

template <DIM D, typename T, typename DeviceType>
class QoIFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT QoIFunctor() {}
  MGARDX_CONT QoIFunctor(SubArray<D, T, DeviceType> x,
                          SubArray<D, T, DeviceType> y,
                          SubArray<D, T, DeviceType> z,
                          SubArray<D, bool, DeviceType> out,
                          double eb_x,
                          double eb_y,
                          double eb_z,
                          double tolerance)
      : x(x), y(y), z(z), out(out), eb_x(eb_x), eb_y(eb_y), eb_z(eb_z), tolerance(tolerance) {
    Functor<DeviceType>();
    n = x.shape(0) * x.shape(1) * x.shape(2);
  }

  MGARDX_EXEC T compute_bound_x_square(T x, T eb){
    return 2 * fabs(x) * eb + eb * eb;
  }

  MGARDX_EXEC T compute_bound_square_root_x(T x, T eb){
    if (x == 0) {
      return sqrt(eb);
    }
    if (x > eb) {
      return eb / (sqrt(x - eb) + sqrt(x));
    } else {
      return eb / sqrt(x);
    }
  }

  MGARDX_EXEC void Operation1() {
    int tid = FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX() + FunctorBase<DeviceType>::GetThreadIdX();
    int stride = FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetGridDimX();

    for (int i = tid; i < n; i += stride){
      double Vx = (double)*x(i);
      double Vy = (double)*y(i);
      double Vz = (double)*z(i);
      // if (check_flag(tolerance_exceed_flag)) return;
      double e_V_TOT_2 = compute_bound_x_square(Vx, eb_x)
                      + compute_bound_x_square(Vy, eb_y)
                      + compute_bound_x_square(Vz, eb_z);
      double V_TOT_2 = Vx * Vx + Vy * Vy + Vz * Vz;
      double e_V_TOT = compute_bound_square_root_x(V_TOT_2, e_V_TOT_2);
      *out(i) = e_V_TOT > tolerance;
      // double V_TOT = sqrt(V_TOT_2);
      // if (e_V_TOT > tolerance){
      //   raise_flag(tolerance_exceed_flag);
      //   return;
      // }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SubArray<D, T, DeviceType> x;
  SubArray<D, T, DeviceType> y;
  SubArray<D, T, DeviceType> z;
  SubArray<D, bool, DeviceType> out;
  double eb_x;
  double eb_y;
  double eb_z;
  double tolerance;
  uint32_t n;
};

template <DIM D, typename T, typename DeviceType>
class QoIKernel : public Kernel {
public:
  constexpr static std::string_view Name = "qoi kernel";
  constexpr static bool EnableAutoTuning() { return false; }
  MGARDX_CONT
  QoIKernel(SubArray<D, T, DeviceType> x,
                          SubArray<D, T, DeviceType> y,
                          SubArray<D, T, DeviceType> z,
                          SubArray<D, bool, DeviceType> out,
                          double eb_x,
                          double eb_y,
                          double eb_z,
                          double tolerance)
      : x(x), y(y), z(z), out(out), eb_x(eb_x), eb_y(eb_y), eb_z(eb_z), tolerance(tolerance) {}


  MGARDX_CONT Task<QoIFunctor<D, T, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = QoIFunctor<D, T, DeviceType>;
    FunctorType functor(x, y, z, out, eb_x, eb_y, eb_z, tolerance);
    SIZE total_thread_x = x.shape(0) * x.shape(1) * x.shape(2);

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
  SubArray<D, T, DeviceType> y;
  SubArray<D, T, DeviceType> z;
  SubArray<D, bool, DeviceType> out;
  double eb_x;
  double eb_y;
  double eb_z;
  double tolerance;
};

template <DIM D, typename T, typename DeviceType>
void Copy3D(SubArray<D, T, DeviceType> x,
            SubArray<D, T, DeviceType> y,
            SubArray<D, T, DeviceType> z,
            SubArray<D, bool, DeviceType> out, int queue_idx) {
  
  DeviceLauncher<DeviceType>::Execute(QoIKernel<D, T, DeviceType>(x, y, z, out), queue_idx);
  
}

} // namespace multi_dimension

} // namespace data_refactoring

} // namespace mgard_x

#endif