#include "mgard-x/MDRHighLevel/qoi_kernel.hpp"
#include <iostream>

namespace mgard::MDR {

__device__ bool check_flag(int *flag) {
  return atomicAdd(flag, 0);
}
__device__ void raise_flag(int *flag) {
  atomicExch(flag, 1);
}

template <class T>
void V_TOT_computation(const T *Vx, const T *Vy, const T *Vz, T *V_TOT, size_t n){
  dim3 block(BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  compute_V_TOT<T><<<grid, block>>>(Vx, Vy, Vz, V_TOT, n);
  cudaDeviceSynchronize();
}

template <class T>
void V_TOT_computation(const T *Vx, T *V_TOT, size_t n){
  dim3 block(BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  compute_V_TOT<T><<<grid, block>>>(Vx, V_TOT, n);
  cudaDeviceSynchronize();
}

template <class T>
bool V_TOT_error_estimation(const T *Vx, const T *Vy, const T *Vz, size_t n, double eb_Vx, double eb_Vy, double eb_Vz, double tolerance){
  // std::cout << "From CUDA: eb_Vx: " << eb_Vx << ", eb_Vy: " << eb_Vy << ", eb_Vz: " << eb_Vz << ", requested QoI error: " << tolerance << std::endl;
  dim3 block(BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int tolerance_exceed_flag_h;
  int *tolerance_exceed_flag_d;
  cudaMalloc((void**) &tolerance_exceed_flag_d, sizeof(int));
  cudaMemset(tolerance_exceed_flag_d, 0, sizeof(int));
  estimate_error_V_TOT<T><<<grid, block>>>(Vx, Vy, Vz, n, eb_Vx, eb_Vy, eb_Vz, tolerance, tolerance_exceed_flag_d);
  cudaDeviceSynchronize();
  cudaMemcpy(&tolerance_exceed_flag_h, tolerance_exceed_flag_d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(tolerance_exceed_flag_d);
  // std::cout << "From CUDA: tolerance_exceed_flag_h = " << tolerance_exceed_flag_h << std::endl;
  if (tolerance_exceed_flag_h == 0){
    return false;
  }
  else{
    return true;
  }
}

template <class T>
void V_TOT_error_estimation(const T *Vx, size_t n, double eb_Vx, double eb_Vy, double eb_Vz, double tolerance){
  dim3 block(BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  // estimate_error_V_TOT<T><<<grid, block>>>(Vx, n, ebs, error_est_V_TOT, error_V_TOT, V_TOT_ori);
  cudaDeviceSynchronize();
}

template void V_TOT_computation<float>(const float*, const float*, const float*, float*, size_t);
template void V_TOT_computation<float>(const float*, float*, size_t);
template bool V_TOT_error_estimation<float>(const float*, const float*, const float*, size_t, double, double, double, double);
template void V_TOT_error_estimation<float>(const float*, size_t, double, double, double, double);

template void V_TOT_computation<double>(const double*, const double*, const double*, double*, size_t);
template void V_TOT_computation<double>(const double*, double*, size_t);
template bool V_TOT_error_estimation<double>(const double*, const double*, const double*, size_t, double, double, double, double);
template void V_TOT_error_estimation<double>(const double*, size_t, double, double, double, double);
} // namespace mgard::MDR
