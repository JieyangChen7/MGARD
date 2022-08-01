/*
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jul 20, 2021
 */

#ifndef MGARD_X_SUBARRAY_HPP
#define MGARD_X_SUBARRAY_HPP
// #include "Common.h"
// #include "CommonInternal.h"
#include <iostream>
#include <vector>
#include <assert.h>

namespace mgard_x {

template <DIM D, typename T, typename DeviceType> class SubArray {
public:
  MGARDX_CONT_EXEC
  SubArray();

  MGARDX_CONT
  SubArray(Array<D, T, DeviceType> &array, bool get_host_pointer = false);

  MGARDX_CONT
  SubArray(std::vector<SIZE> shape, T *dv);

  MGARDX_CONT_EXEC
  void initialize();

  MGARDX_CONT_EXEC
  T *data() { return this->dv; }

  MGARDX_CONT_EXEC
  void setData(T *dv) { this->dv = dv; }

  MGARDX_CONT_EXEC
  bool hasHostData() { return has_host_pointer; };

  MGARDX_CONT
  T *dataHost() {
    if (!has_host_pointer) {
      std::cerr << log::log_err << "Host pointer not initialized!\n";
      exit(-1);
    }
    return v;
  }

  MGARDX_CONT
  void setDataHost(T *v) {
    this->has_host_pointer = true;
    this->v = v;
  }

  // shape
  MGARDX_CONT_EXEC
  SIZE getShape(DIM d) const { return this->_shape[d]; }

  MGARDX_CONT_EXEC
  SIZE *getShape() { return this->_shape; }

  MGARDX_CONT_EXEC
  void setShape(DIM d, SIZE n) {
    if (d >= D) {
      return;
    }
    this->_shape[d] = n;
  }

  MGARDX_CONT_EXEC
  void setShape(SIZE shape[D]) {
    for (DIM d = 0; d < D; d++) {
      this->_shape[d] = shape[d];
    }
  }

  MGARDX_CONT_EXEC
  SIZE shape(DIM d) const {
    if (d >= D) return 1;
    return __shape[d]; 
  }

  MGARDX_CONT_EXEC
  SIZE ld(DIM d) const {
    if (d >= D) return 1;
    return __ldvs[d]; 
  }

  MGARDX_CONT_EXEC
  SIZE getLd(DIM d) const { return this->_ldvs[d]; }

  MGARDX_CONT_EXEC
  SIZE *getLd() { return this->_ldvs; }

  MGARDX_CONT_EXEC
  void setLd(DIM d, SIZE ld) { this->_ldvs[d] = ld; }

  MGARDX_CONT_EXEC
  void setLd(SIZE ldvs[D]) {
    for (DIM d = 0; d < D; d++)
      this->_ldvs[d] = ldvs[d];
  }

  MGARDX_CONT_EXEC
  bool isPitched() { return this->pitched; }

  MGARDX_CONT_EXEC
  void setPitched(bool pitched) { this->pitched = pitched; }

  MGARDX_CONT_EXEC
  SIZE getLddv1() const { return this->lddv1; }

  MGARDX_CONT_EXEC
  SIZE getLddv2() const { return this->lddv2; }

  void offset(std::vector<SIZE> idx);

  MGARDX_CONT
  void resize(std::vector<SIZE> shape);

  MGARDX_CONT
  void offset(DIM dim, SIZE offset_value);

  MGARDX_CONT
  void resize(DIM dim, SIZE new_size);

  MGARDX_CONT
  void project(DIM dim_fastest, DIM dim_medium, DIM dim_slowest);


  void offset2(std::vector<SIZE> idx);

  MGARDX_CONT
  void resize2(std::vector<SIZE> shape);

  MGARDX_CONT
  void offset2(DIM dim, SIZE offset_value);

  MGARDX_CONT
  void resize2(DIM dim, SIZE new_size);

  MGARDX_CONT
  void project2(DIM dim_slowest, DIM dim_medium, DIM dim_fastest);

  MGARDX_CONT
  SubArray<1, T, DeviceType> Linearize();

  MGARDX_CONT
  SubArray<3, T, DeviceType> Slice3D(DIM d1, DIM d2, DIM d3);

  // TODO
  MGARDX_CONT_EXEC
  T *operator()(SIZE idx[D]) {
    LENGTH curr_stride = 1;
    LENGTH offset = 0;
    for (DIM i = 0; i < D; i++) {
      offset += idx[i] * curr_stride;
      curr_stride *= this->_ldvs[i];
    }
    return this->dv + offset;
  }

  MGARDX_CONT_EXEC
  T &operator[](SIZE idx[D]) {
    LENGTH curr_stride = 1;
    LENGTH offset = 0;
    for (int d = D-1; d >= 0; d--) {
      offset += idx[d] * curr_stride;
      curr_stride *= __ldvs[d];
    }
    return dv[offset];
  }

  MGARDX_CONT_EXEC
  T *operator()(IDX l, IDX z, IDX y, IDX x) {
    return dv + __ldvs[1] * __ldvs[2] * __ldvs[3] * l +
           __ldvs[2] * __ldvs[3] * z + __ldvs[3] * y + x;
  }
  MGARDX_CONT_EXEC
  T *operator()(IDX z, IDX y, IDX x) {
    return dv + lddv2 * lddv1 * z + lddv1 * y + x;
  }
  MGARDX_CONT_EXEC
  T *operator()(IDX y, IDX x) { 
    return dv + lddv1 * y + x; 
  }
  MGARDX_CONT_EXEC
  T *operator()(IDX x) { return dv + x; }

  MGARDX_EXEC void offset(SIZE idx[D]) {
    ptr_offset += calc_offset(idx);
    dv += calc_offset(idx);
  }

  MGARDX_EXEC void offset2(SIZE idx[D]) {
    ptr_offset += calc_offset2(idx);
    dv += calc_offset2(idx);
  }

  MGARDX_EXEC void offset(IDX z, IDX y, IDX x) {
    ptr_offset += this->lddv2 * this->lddv1 * z + this->lddv1 * y + x;
    this->dv += this->lddv2 * this->lddv1 * z + this->lddv1 * y + x;
  }
  MGARDX_EXEC void offset(IDX y, IDX x) {
    ptr_offset += this->lddv1 * y + x;
    this->dv += this->lddv1 * y + x;
  }
  MGARDX_EXEC void offset(IDX x) {
    ptr_offset += x;
    this->dv += x;
  }

  MGARDX_EXEC void reset_offset() {
    this->dv -= ptr_offset;
    ptr_offset = 0;
  }

  MGARDX_CONT_EXEC
  bool isNull() { return this->dv == NULL; }

  using DataType = T;
  using DevType = DeviceType;
  static const DIM NumDims = D;

private:
  T *dv; // device pointer
  T *v; // host pointer
  bool has_host_pointer;

  SIZE _ldvs[D];
  SIZE _shape[D];

  SIZE __ldvs[D];
  SIZE __shape[D];

  DIM projected_dim_fastest;
  DIM projected_dim_medium;
  DIM projected_dim_slowest;

  SIZE lddv1;
  SIZE lddv2;

  bool pitched;
  bool managed;

  LENGTH ptr_offset;

  MGARDX_CONT_EXEC
  SIZE calc_offset(SIZE idx[D]) {
    SIZE curr_stride = 1;
    SIZE offset = 0;
    for (DIM i = 0; i < D; i++) {
      offset += idx[i] * curr_stride;
      curr_stride *= this->_ldvs[i];
    }
    return offset;
  }

  MGARDX_CONT_EXEC
  SIZE calc_offset2(SIZE idx[D]) {
    SIZE curr_stride = 1;
    SIZE offset = 0;
    for (int d = D-1; d >= 0; d--) {
      offset += idx[d] * curr_stride;
      curr_stride *= __ldvs[d];
    }
    return offset;
  }
};

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT_EXEC void SubArray<D, T, DeviceType>::initialize() {
  dv = NULL;
  v = NULL;
  has_host_pointer = false;
  for (DIM d = 0; d < D; d++) {
    _ldvs[d] = 1;
    _shape[d] = 1;
    __ldvs[d] = 1;
    __shape[d] = 1;
  }

  projected_dim_fastest = D-1;
  projected_dim_medium = D-2;
  projected_dim_slowest = D-3;
  lddv1 = 1;
  lddv2 = 1;

  pitched = false;
  managed = false;

  ptr_offset = 0;
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT_EXEC SubArray<D, T, DeviceType>::SubArray() {
  initialize();
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT SubArray<D, T, DeviceType>::SubArray(Array<D, T, DeviceType> &array,
                                                 bool get_host_pointer) {
  initialize();
  dv = array.data();
  for (DIM d = 0; d < D; d++) {
    _shape[d] = array.shape()[d];
    _ldvs[d] = array.ld()[d];
    __shape[d] = array.shape(d);
    __ldvs[d] = array.ld(d);
  }
  lddv1 = __ldvs[D-1];
  if (D > 1) lddv2 = __ldvs[D-2];
  if (get_host_pointer) {
    v = array.hostCopy();
    has_host_pointer = true;
  }
  pitched = array.isPitched();
  managed = array.isManaged();
}

// TODO: update shape
template <DIM D, typename T, typename DeviceType>
MGARDX_CONT SubArray<D, T, DeviceType>::SubArray(std::vector<SIZE> shape,
                                                 T *dv) {
  initialize();
  this->dv = dv;
  for (DIM d = 0; d < D; d++) {
    this->_shape[d] = shape[d];
    this->_ldvs[d] = shape[d];
    this->__shape[D-1-d] = shape[d];
    this->__ldvs[D-1-d] = shape[d];
  }
  lddv1 = __ldvs[D-1];
  if (D > 1) lddv2 = __ldvs[D-2];
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT SubArray<1, T, DeviceType> SubArray<D, T, DeviceType>::Linearize() {
  SubArray<1, T, DeviceType> subArray;
  if (!pitched) {
    SIZE linearized_shape = 1;
    for (DIM d = 0; d < D; d++)
      linearized_shape *= this->__shape[d];
    subArray.setData(data());
    subArray.setShape(0, linearized_shape);
    subArray.setLd(0, linearized_shape);
    subArray.project(0, 1, 2);

    if (has_host_pointer) {
      subArray.setDataHost(dataHost());
    }
    subArray.setPitched(isPitched());
  } else {
    std::cout << log::log_err
              << "Linearized pitched SubArray not implemented!\n";
    exit(-1);
  }
  return subArray;
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT SubArray<3, T, DeviceType>
SubArray<D, T, DeviceType>::Slice3D(DIM d1, DIM d2, DIM d3) {

  if (D < 3) {
    std::cout << log::log_err << "calling Slice3D on SubArray with " << D
              << "D data.\n";
    exit(-1);
  }
  SubArray<3, T, DeviceType> subArray;
  subArray.setShape(0, this->_shape[d1]);
  subArray.setShape(1, this->_shape[d2]);
  subArray.setShape(2, this->_shape[d3]);
  subArray.setData(this->dv);
  subArray.setLd(0, this->_ldvs[d1]);
  subArray.setLd(1, this->_ldvs[d2]);
  subArray.setLd(2, this->_ldvs[d3]);
  subArray.project(d1, d2, d3);

  if (this->has_host_pointer) {
    subArray.setDataHost(this->v);
  }
  subArray.setPitched(this->pitched);
  return subArray;
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::offset(std::vector<SIZE> idx) {
  SIZE _idx[D];
  for (DIM d = 0; d < D; d++)
    _idx[d] = idx[d];
  dv += calc_offset(_idx);
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::resize(std::vector<SIZE> shape) {
  for (DIM d = 0; d < D; d++) {
    _shape[d] = shape[d];
    __shape[D-1-d] = shape[d];
  }
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::offset(DIM dim,
                                                    SIZE offset_value) {
  SIZE idx[D];
  for (DIM d = 0; d < D; d++) idx[d] = 0;
  idx[dim] = offset_value;
  dv += calc_offset(idx);
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::resize(DIM dim, SIZE new_size) {
  if (dim >= D) return;
  _shape[dim] = new_size;
  __shape[D-1-dim] = new_size;
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::offset2(std::vector<SIZE> idx) {
  if (idx.size() < D) {
    std::cerr << log::log_err << "SubArray::resize insufficient idx length.\n"; 
  }
  // In case shape.size > D;
  DIM skip_dim = idx.size() - D;
  SIZE _idx[D];
  for (DIM d = 0; d < D; d++)
    _idx[d] = idx[skip_dim + d];
  dv += calc_offset2(_idx);
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::resize2(std::vector<SIZE> shape) {
  if (shape.size() < D) {
    std::cerr << log::log_err << "SubArray::resize insufficient shape length.\n"; 
  }
  // In case shape.size > D;
  DIM skip_dim = shape.size() - D;
  for (DIM d = 0; d < D; d++) {
    _shape[D-1-d] = shape[skip_dim + d];
    __shape[d] = shape[skip_dim + d];
  }
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::offset2(DIM dim,
                                                    SIZE offset_value) {
  if (dim >= D) return;
  SIZE idx[D];
  for (DIM d = 0; d < D; d++) {
    idx[d] = 0;
  }
  idx[dim] = offset_value;
  dv += calc_offset2(idx);
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::resize2(DIM dim, SIZE new_size) {
  if (dim >= D) return;
  _shape[D-1-dim] = new_size;
  __shape[dim] = new_size;
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::project(DIM dim_fastest, DIM dim_medium,
                                                     DIM dim_slowest) {
  projected_dim_fastest = dim_fastest;
  projected_dim_medium = dim_medium;
  projected_dim_slowest = dim_slowest;
  lddv1 = 1, lddv2 = 1;
  for (DIM d = projected_dim_fastest; d < projected_dim_medium; d++) {
    lddv1 *= _ldvs[d];
  }
  for (DIM d = projected_dim_medium; d < projected_dim_slowest; d++) {
    lddv2 *= _ldvs[d];
  }
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::project2(DIM dim_slowest, DIM dim_medium,
                                                     DIM dim_fastest) {
  projected_dim_slowest = dim_slowest;
  projected_dim_medium = dim_medium;
  projected_dim_fastest = dim_fastest;
  if (projected_dim_slowest >= D) projected_dim_slowest = 0;
  if (projected_dim_medium >= D) projected_dim_medium = 0;
  lddv1 = 1, lddv2 = 1;
  for (int d = projected_dim_fastest; d > projected_dim_medium; d--) {
    lddv1 *= __ldvs[d];
  }
  for (int d = projected_dim_medium; d > projected_dim_slowest; d--) {
    lddv2 *= __ldvs[d];
  }
}


} // namespace mgard_x
#endif
