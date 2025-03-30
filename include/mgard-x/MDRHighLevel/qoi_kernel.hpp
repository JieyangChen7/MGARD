#pragma once

#include <cstddef>

#define BLOCK_SIZE 256

namespace mgard::MDR {

// Host launcher declarations (OK in .hpp)
template <class T>
void V_TOT_computation(const T *Vx, const T *Vy, const T *Vz, T *V_TOT, size_t n);

template <class T>
void V_TOT_computation(const T *Vx, T *V_TOT, size_t n);

template <class T>
bool V_TOT_error_estimation(const T *Vx, const T *Vy, const T *Vz, size_t n, double eb_Vx, double eb_Vy, double eb_Vz, double tolerance);

template <class T>
void V_TOT_error_estimation(const T *Vx, size_t n, double eb_Vx, double eb_Vy, double eb_Vz, double tolerance);

}  // namespace mgard::MDR

#include "qoi_kernel.inl"
