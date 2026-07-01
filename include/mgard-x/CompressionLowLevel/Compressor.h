/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_COMPRESSOR_H
#define MGARD_X_COMPRESSOR_H

#include <limits>

#include "../RuntimeX/RuntimeXPublic.h"

#include "../DataRefactoring/DataRefactor.hpp"

// #include "CompressionLowLevelWorkspace.hpp"

#include "NormCalculator.hpp"

#include "../Hierarchy/Hierarchy.h"

#include "../Lossless/Lossless.hpp"
#include "../Quantization/LinearQuantization.hpp"

#include "LossyCompressorInterface.hpp"

namespace mgard_x {

// L-infinity error control (s == inf) can use the cheaper hierarchical basis
// (no mass-matrix correction) instead of the orthogonal basis: the hierarchical
// reconstruction is a partition-of-unity prolongation whose per-level error
// amplification is 1, so the max error stays bounded while the correction step
// is skipped. Only D <= 3 is supported because the multi-dimensional
// decompose/recompose honor the flag only there; higher dimensions always apply
// the correction. Kept here so the low-level compressor and quantizer agree on
// exactly when correction is skipped.
template <DIM D, typename T> inline bool infer_orthogonal_projection(T s) {
  return !(s == std::numeric_limits<T>::infinity() && D <= 3);
}

template <DIM D, typename T, typename DeviceType>
class Compressor : public LossyCompressorInterface<D, T, DeviceType> {
public:
  using HierarchyType = Hierarchy<D, T, DeviceType>;
  using DataRefactorType = data_refactoring::DataRefactor<D, T, DeviceType>;
  using LosslessCompressorType =
      ComposedLosslessCompressor<QUANTIZED_INT, HUFFMAN_CODE, DeviceType>;
  using LinearQuantizerType = LinearQuantizer<D, T, QUANTIZED_INT, DeviceType>;

public:
  Compressor();

  Compressor(Hierarchy<D, T, DeviceType> &hierarchy, Config config);

  void Adapt(Hierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx);

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape, Config config);

  void CalculateNorm(Array<D, T, DeviceType> &original_data,
                     enum error_bound_type ebtype, T s, T &norm, int queue_idx);

  void Decompose(Array<D, T, DeviceType> &original_data,
                 bool orthogonal_projection, int queue_idx);

  void Quantize(Array<D, T, DeviceType> &original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                int queue_idx);

  void LosslessCompress(Array<1, Byte, DeviceType> &compressed_data,
                        int queue_idx);

  void Serialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx);

  void Deserialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx);

  void Recompose(Array<D, T, DeviceType> &decompressed_data,
                 bool orthogonal_projection, int queue_idx);

  void Dequantize(Array<D, T, DeviceType> &decompressed_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  int queue_idx);

  void LosslessDecompress(Array<1, Byte, DeviceType> &compressed_data,
                          int queue_idx);

  void Compress(Array<D, T, DeviceType> &original_data,
                enum error_bound_type ebtype, T tol, T s, T &norm,
                Array<1, Byte, DeviceType> &compressed_data, int queue_idx);
  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  enum error_bound_type ebtype, T tol, T s, T &norm,
                  Array<D, T, DeviceType> &decompressed_data, int queue_idx);

  bool initialized;
  Hierarchy<D, T, DeviceType> *hierarchy;
  Config config;
  // Whether the last (de)compose should use orthogonal projection. Derived from
  // s (see infer_orthogonal_projection) during Compress/Decompress/(De)quantize
  // and consumed by Recompose, which does not receive s. Defaults to true so
  // the orthogonal path is used unless s explicitly enables the hierarchical
  // fast path.
  bool orthogonal_projection = true;
  Array<1, T, DeviceType> norm_tmp_array;
  Array<1, T, DeviceType> norm_array;
  Array<D, QUANTIZED_INT, DeviceType> quantized_array;
  DataRefactorType refactor;
  LinearQuantizerType quantizer;
  LosslessCompressorType lossless_compressor;
};

} // namespace mgard_x

#endif