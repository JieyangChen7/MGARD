#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


namespace mgard::MDR {

template <class T>
__host__ __device__ inline T compute_bound_x_square(T x, T eb){
  return 2 * fabs(x) * eb + eb * eb;
}

template <class T>
__host__ __device__ inline T compute_bound_square_root_x(T x, T eb){
  if (x == 0) {
    return sqrt(eb);
  }
  if (x > eb) {
    return eb / (sqrt(x - eb) + sqrt(x));
  } else {
    return eb / sqrt(x);
  }
}

__device__ bool check_flag(int *flag);
__device__ void raise_flag(int *flag);


template <class T>
__global__ void compute_V_TOT(const T *Vx, const T *Vy, const T *Vz, T *V_TOT, size_t n){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride){
    V_TOT[i] = sqrt(Vx[i]*Vx[i] + Vy[i]*Vy[i] + Vz[i]*Vz[i]);
  }
}

template <class T>
__global__ void compute_V_TOT(const T *Vx, T *V_TOT, size_t n){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride){
    V_TOT[i] = sqrt(Vx[i]*Vx[i]);
  }
}

template <class T>
__global__ void estimate_error_V_TOT(const T *Vx, const T *Vy, const T *Vz, size_t n, double eb_Vx, double eb_Vy, double eb_Vz, double tolerance, int *tolerance_exceed_flag) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += stride){
    if (check_flag(tolerance_exceed_flag)) return;
    double e_V_TOT_2 = compute_bound_x_square((double)Vx[i], eb_Vx)
                     + compute_bound_x_square((double)Vy[i], eb_Vy)
                     + compute_bound_x_square((double)Vz[i], eb_Vz);
    double V_TOT_2 = Vx[i]*Vx[i] + Vy[i]*Vy[i] + Vz[i]*Vz[i];
    double e_V_TOT = compute_bound_square_root_x(V_TOT_2, e_V_TOT_2);
    double V_TOT = sqrt(V_TOT_2);
    if (e_V_TOT > tolerance){
			raise_flag(tolerance_exceed_flag);
			return;
		}
  }
  return;
}

template <class T>
__global__ void estimate_error_V_TOT(const T *Vx, size_t n, double *ebs) {

  double eb_Vx = ebs[0];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += stride){
    double e_V_TOT_2 = compute_bound_x_square((double)Vx[i], eb_Vx);
    double V_TOT_2 = Vx[i]*Vx[i];
    double e_V_TOT = compute_bound_square_root_x(V_TOT_2, e_V_TOT_2);
    double V_TOT = sqrt(V_TOT_2);
  }
}

}  // namespace mgard::MDR
