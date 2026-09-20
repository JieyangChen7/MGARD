/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 */

#ifndef MGARD_X_UTILITIES_PROJECTION_MODE_H
#define MGARD_X_UTILITIES_PROJECTION_MODE_H

#include <limits>

#include "../RuntimeX/Utilities/Exceptions.h"
#include "Types.h"

namespace mgard_x {

// Resolves Auto against the error-bound norm s, then validates the result:
// Hierarchical only supports L-infinity error control (its reconstruction is
// a partition-of-unity prolongation, so it only bounds the max error, not the
// L_2 norm). This is the single choke point both Compressor and
// HybridHierarchyCompressor call, so one Config field controls both.
template <typename T>
inline compression_projection_mode_type
resolve_projection_mode(compression_projection_mode_type mode, T s) {
  if (mode == compression_projection_mode_type::Auto) {
    mode = s == std::numeric_limits<T>::infinity()
               ? compression_projection_mode_type::Hierarchical
               : compression_projection_mode_type::Orthogonal;
  }
  if (mode == compression_projection_mode_type::Hierarchical &&
      s != std::numeric_limits<T>::infinity()) {
    throw ProcessingException(
        "the hierarchical basis only supports L-infinity error control");
  }
  if (mode != compression_projection_mode_type::Orthogonal &&
      mode != compression_projection_mode_type::Hierarchical) {
    throw ProcessingException("unknown compression projection mode");
  }
  return mode;
}

// The boolean the (de)compose/quantize kernels actually consume.
template <typename T>
inline bool infer_orthogonal_projection(compression_projection_mode_type mode,
                                        T s) {
  return resolve_projection_mode(mode, s) ==
         compression_projection_mode_type::Orthogonal;
}

} // namespace mgard_x

#endif
