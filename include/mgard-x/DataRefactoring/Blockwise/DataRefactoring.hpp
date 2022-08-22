/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_BLOCKWISE_DATA_REFACTORING_TEMPLATE
#define MGARD_X_BLOCKWISE_DATA_REFACTORING_TEMPLATE

#include "../../RuntimeX/RuntimeX.h"
#include "../../Hierarchy/Hierarchy.hpp"
#include "../MultiDimension/Coefficient/GPKFunctor.h"
#include "../MultiDimension/Correction/LPKFunctor.h"
#include "../MultiDimension/Correction/IPKFunctor.h"

namespace mgard_x {

namespace blockwise {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class DataRefactoringFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DataRefactoringFunctor() {}
  MGARDX_CONT DataRefactoringFunctor(SubArray<D, T, DeviceType> v, 
                                     SubArray<1, SubArray<1, T, DeviceType>, DeviceType> am_r,
                                     SubArray<1, SubArray<1, T, DeviceType>, DeviceType> am_c,
                                     SubArray<1, SubArray<1, T, DeviceType>, DeviceType> am_f,
                                     SubArray<1, SubArray<1, T, DeviceType>, DeviceType> bm_r,
                                     SubArray<1, SubArray<1, T, DeviceType>, DeviceType> bm_c,
                                     SubArray<1, SubArray<1, T, DeviceType>, DeviceType> bm_f,
                                     SIZE l_target): v(v), am_r(am_r), am_c(am_c), am_f(am_f), 
                                      bm_r(bm_r), bm_c(bm_c), bm_f(bm_f), l_target(l_target) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {

    sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    v_sm = sm; sm += F * C * R;
    w_sm = sm; sm += F * C * R;
    am_r_sm = sm; sm += (R+1) * (l_target+1);
    am_c_sm = sm; sm += (C+1) * (l_target+1);
    am_f_sm = sm; sm += (F+1) * (l_target+1);
    bm_r_sm = sm; sm += (R+1) * (l_target+1);
    bm_c_sm = sm; sm += (C+1) * (l_target+1);
    bm_f_sm = sm; sm += (F+1) * (l_target+1);

    ld1 = F;
    ld2 = C;
    nr = R;
    nc = C;
    nf = F;

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
    // printf("load: %u %u %u -> %f [%u]\n", r_gl, c_gl, f_gl, *v(r_gl, c_gl, f_gl), get_idx(ld1, ld2, r_sm, c_sm, f_sm));
    v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = *v(r_gl, c_gl, f_gl);

    assert(am_r.shape(0) == l_target + 1);
    assert(am_c.shape(0) == l_target + 1);
    assert(am_f.shape(0) == l_target + 1);
    assert(bm_r.shape(0) == l_target + 1);
    assert(bm_c.shape(0) == l_target + 1);
    assert(bm_f.shape(0) == l_target + 1);
/*
    for (SIZE l = 0; l < l_target+1; l++) {
      SubArray<1, T, DeviceType>& ptr = *am_r(l); 
      if (threadId < ptr.shape(0)) {
        am_r_sm[l * (R+1) + threadId] = *ptr(threadId);
      }
    }
    for (SIZE l = 0; l < l_target+1; l++) {
      SubArray<1, T, DeviceType>& ptr = *am_c(l); 
      if (threadId < ptr.shape(0)) {
        am_c_sm[l * (C+1) + threadId] = *ptr(threadId);
      }
    }
    for (SIZE l = 0; l < l_target+1; l++) {
      SubArray<1, T, DeviceType>& ptr = *am_f(l); 
      if (threadId < ptr.shape(0)) {
        am_f_sm[l * (F+1) + threadId] = *ptr(threadId);
      }
    }
    for (SIZE l = 0; l < l_target+1; l++) {
      SubArray<1, T, DeviceType>& ptr = *bm_r(l); 
      if (threadId < ptr.shape(0)) {
        bm_r_sm[l * (R+1) + threadId] = *ptr(threadId);
      }
    }
    for (SIZE l = 0; l < l_target+1; l++) {
      SubArray<1, T, DeviceType>& ptr = *bm_c(l); 
      if (threadId < ptr.shape(0)) {
        bm_c_sm[l * (C+1) + threadId] = *ptr(threadId);
      }
    }
    for (SIZE l = 0; l < l_target+1; l++) {
      SubArray<1, T, DeviceType>& ptr = *bm_f(l); 
      if (threadId < ptr.shape(0)) {
        bm_f_sm[l * (F+1) + threadId] = *ptr(threadId);
      }
    }
  */
  }

  MGARDX_EXEC void CalculateCoefficientFixedAssignment() {
    r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();
    T a, b, c, d, e, f;
    if (r_sm % 2 == 0 && c_sm % 2 == 0 && f_sm % 2 == 0) {
      v_sm[get_idx(ld1, ld2, r_sm/2, c_sm/2, f_sm/2)] = w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)];
    } else if (r_sm % 2 == 0 && c_sm % 2 == 0 && f_sm % 2 != 0) {
      T left = w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm-1)];
      T right = w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm+1)];
      v_sm[get_idx(ld1, ld2, r_sm/2, c_sm/2, f_sm/2+nff)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(left, right, (T)0.5);
    } else if (r_sm % 2 == 0 && c_sm % 2 != 0 && f_sm % 2 == 0) {
      T back = w_sm[get_idx(ld1, ld2, r_sm, c_sm-1, f_sm)];
      T front = w_sm[get_idx(ld1, ld2, r_sm, c_sm+1, f_sm)];
      v_sm[get_idx(ld1, ld2, r_sm/2, c_sm/2+ncc, f_sm/2)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(back, front, (T)0.5);
    } else if (r_sm % 2 != 0 && c_sm % 2 == 0 && f_sm % 2 == 0) {
      T top = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm, f_sm)];
      T bottom = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm, f_sm)];
      v_sm[get_idx(ld1, ld2, r_sm/2+nrr, c_sm/2, f_sm/2)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(top, bottom, (T)0.5);
    } else if (r_sm % 2 == 0 && c_sm % 2 != 0 && f_sm % 2 != 0) {
      T back_left = w_sm[get_idx(ld1, ld2, r_sm, c_sm-1, f_sm-1)];
      T back_right = w_sm[get_idx(ld1, ld2, r_sm, c_sm-1, f_sm+1)];
      T front_left = w_sm[get_idx(ld1, ld2, r_sm, c_sm+1, f_sm-1)];
      T front_right = w_sm[get_idx(ld1, ld2, r_sm, c_sm+1, f_sm+1)];
      T back = lerp(back_left, back_right, (T)0.5);
      T front = lerp(front_left, front_right, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm/2, c_sm/2+ncc, f_sm/2+nff)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(back, front, (T)0.5);
    } else if (r_sm % 2 != 0 && c_sm % 2== 0 && f_sm % 2 != 0) {
      T bottom_left = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm, f_sm-1)];
      T bottom_right = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm, f_sm+1)];
      T top_left = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm, f_sm-1)];
      T top_right = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm, f_sm+1)];
      T bottom = lerp(bottom_left, bottom_right, (T)0.5);
      T top = lerp(top_left, top_right, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm/2+nrr, c_sm/2, f_sm/2+nff)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(bottom, top, (T)0.5);
    } else if (r_sm % 2 != 0 && c_sm % 2!= 0 && f_sm % 2 == 0) {
      T bottom_back = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm-1, f_sm)];
      T bottom_front = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm+1, f_sm)];
      T top_back = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm-1, f_sm)];
      T top_front = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm+1, f_sm)];
      T bottom = lerp(bottom_back, bottom_front, (T)0.5);
      T top = lerp(top_back, top_front, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm/2+nrr, c_sm/2+ncc, f_sm/2)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(bottom, top, (T)0.5);
    } else if (r_sm % 2 != 0 && c_sm % 2!= 0 && f_sm % 2 == 0) {
      T bottom_back_left = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm-1, f_sm-1)];
      T bottom_front_left = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm+1, f_sm-1)];
      T top_back_left = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm-1, f_sm-1)];
      T top_front_left = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm+1, f_sm-1)];
      T bottom_back_right = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm-1, f_sm+1)];
      T bottom_front_right = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm+1, f_sm+1)];
      T top_back_right = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm-1, f_sm+1)];
      T top_front_right = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm+1, f_sm+1)];
      
      T bottom_back = lerp(bottom_back_left, bottom_back_right, (T)0.5);
      T bottom_front = lerp(bottom_front_left, bottom_front_right, (T)0.5);
      T top_back = lerp(top_back_left, top_back_right, (T)0.5);
      T top_front = lerp(top_front_left, top_front_right, (T)0.5);

      T bottom = lerp(bottom_back, bottom_front, (T)0.5);
      T top = lerp(top_back, top_front, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm/2+nrr, c_sm/2+ncc, f_sm/2+nff)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(bottom, top, (T)0.5);
    }
  }

  MGARDX_EXEC void CalculateCoefficientCompactReassignment() {
    int base = 0;
    threadId -= base;
    if (0 <= threadId && threadId < nrr * ncc * nff) {
      r_sm = (threadId / nff) / ncc;
      c_sm = (threadId / nff) % ncc;
      f_sm = threadId % nff;
      v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
    }
    threadId -= nrr * ncc * nff;
    // f - 18
    if (0 <= threadId && threadId < nrr * ncc * (nff-1)) {
      r_sm = (threadId / (nff-1)) / ncc;
      c_sm = (threadId / (nff-1)) % ncc;
      f_sm = threadId % (nff-1);
      T left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
      v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm+nff)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+1)] - lerp(left, right, (T)0.5);
    }
    threadId -= nrr * ncc * (nff-1);
    // c - 18
    if (0 <= threadId && threadId < nrr * (ncc-1) * nff) {
      r_sm = (threadId / nff) / (ncc-1);
      c_sm = (threadId / nff) % (ncc-1);
      f_sm = threadId % nff;
      T back = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T front = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
      v_sm[get_idx(ld1, ld2, r_sm, c_sm+ncc, f_sm)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2)] - lerp(back, front, (T)0.5);
    }
    threadId -= nrr * (ncc-1) * nff;
    // r - 18
    if (0 <= threadId && threadId < (nrr-1) * ncc * nff) {
      r_sm = (threadId / nff) / ncc;
      c_sm = (threadId / nff) % ncc;
      f_sm = threadId % nff;
      T bottom = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T top = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
      v_sm[get_idx(ld1, ld2, r_sm+nrr, c_sm, f_sm)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2, f_sm*2)] - lerp(bottom, top, (T)0.5);
    }
    threadId -= (nrr-1) * ncc * nff;
    // cf - 12
    if (0 <= threadId && threadId < nrr * (ncc-1) * (nff-1)) {
      r_sm = (threadId / (nff-1)) / (ncc-1);
      c_sm = (threadId / (nff-1)) % (ncc-1);
      f_sm = threadId % (nff-1);
      T back_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T back_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
      T front_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
      T front_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2+2)];
      T back = lerp(back_left, back_right, (T)0.5);
      T front = lerp(front_left, front_right, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm, c_sm+ncc, f_sm+nff)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2+1)] - lerp(back, front, (T)0.5);
    }
    threadId -= nrr * (ncc-1) * (nff-1);
    // rf - 12
    if (0 <= threadId && threadId < (nrr-1) * ncc * (nff-1)) {
      r_sm = (threadId / (nff-1)) / ncc;
      c_sm = (threadId / (nff-1)) % ncc;
      f_sm = threadId % (nff-1);
      T bottom_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T bottom_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
      T top_left = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
      T top_right = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2+2)];
      T bottom = lerp(bottom_left, bottom_right, (T)0.5);
      T top = lerp(top_left, top_right, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm, c_sm+ncc, f_sm+nff)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2+1)] - lerp(bottom, top, (T)0.5);
    }
    threadId -= (nrr-1) * ncc * (nff-1);
    // rc - 12
    if (0 <= threadId && threadId < (nrr-1) * (ncc-1) * nff) {
      r_sm = (threadId / nff) / (ncc-1);
      c_sm = (threadId / nff) % (ncc-1);
      f_sm = threadId % nff;
      T bottom_back = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T bottom_front = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
      T top_back = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
      T top_front = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2+2, f_sm*2)];
      T bottom = lerp(bottom_back, bottom_front, (T)0.5);
      T top = lerp(top_back, top_front, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm+nrr, c_sm+ncc, f_sm)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2+1, f_sm*2)] - lerp(bottom, top, (T)0.5);
    }
    threadId -= (nrr-1) * (ncc-1) * nff;
    //rcf - 8
    if (0 <= threadId && threadId < (nrr-1) * (ncc-1) * (nff-1)) {
      r_sm = (threadId / (nff-1)) / (ncc-1);
      c_sm = (threadId / (nff-1)) % (ncc-1);
      f_sm = threadId % (nff-1);
      T bottom_back_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T bottom_back_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
      T bottom_front_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
      T bottom_front_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2+2)];
      T top_back_left = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
      T top_back_right = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2+2)];
      T top_front_left = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2+2, f_sm*2)];
      T top_front_right = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2+2, f_sm*2+2)];
      T bottom_back = lerp(bottom_back_left, bottom_back_right, (T)0.5);
      T bottom_front = lerp(bottom_front_left, bottom_front_right, (T)0.5);
      T top_back = lerp(top_back_left, top_back_right, (T)0.5);
      T top_front = lerp(top_front_left, top_front_right, (T)0.5);
      T bottom = lerp(bottom_back, bottom_front, (T)0.5);
      T top = lerp(top_back, top_front, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm+nrr, c_sm+ncc, f_sm+nff)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2+1, f_sm*2+1)] - lerp(bottom, top, (T)0.5);
    }
    threadId -= (nrr-1) * (ncc-1) * (nff-1);
  }

  MGARDX_EXEC void CalculateCoefficientWarpReassignment() {
    assert(R * C * F >= MGARDX_WARP_SIZE);
    // 5*5*5 cube
    if constexpr (R == 5 && C == 5 && F == 5) {
      // Total number of full warps: 5^3/32 = 3;
      // Ensure we have enough threads
      assert(MGARDX_WARP_SIZE > nrr * ncc * nff);
      SIZE warp_id = threadId / MGARDX_WARP_SIZE;
      SIZE lane_id = threadId % MGARDX_WARP_SIZE;
      if (warp_id == 0) { // 0+1+1+1+3 = 6 lerps
        if (lane_id < nrr * ncc * nff) {
          r_sm = (lane_id / nff) / ncc;
          c_sm = (lane_id / nff) % ncc;
          f_sm = lane_id % nff;
          v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
        }
        if (lane_id < nrr * ncc * (nff-1)) {
          r_sm = (lane_id / (nff-1)) / ncc;
          c_sm = (lane_id / (nff-1)) % ncc;
          f_sm = lane_id % (nff-1);
          T left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm+nff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+1)] - lerp(left, right, (T)0.5);
        }
        if (lane_id < nrr * (ncc-1) * nff) {
          r_sm = (lane_id / nff) / (ncc-1);
          c_sm = (lane_id / nff) % (ncc-1);
          f_sm = lane_id % nff;
          T back = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T front = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          v_sm[get_idx(ld1, ld2, r_sm, c_sm+ncc, f_sm)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2)] - lerp(back, front, (T)0.5);
        }
        if (lane_id < (nrr-1) * ncc * nff) {
          r_sm = (lane_id / nff) / ncc;
          c_sm = (lane_id / nff) % ncc;
          f_sm = lane_id % nff;
          T bottom = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T top = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          v_sm[get_idx(ld1, ld2, r_sm+nrr, c_sm, f_sm)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2, f_sm*2)] - lerp(bottom, top, (T)0.5);
        }
        if (lane_id < nrr * (ncc-1) * (nff-1)) {
          r_sm = (lane_id / (nff-1)) / (ncc-1);
          c_sm = (lane_id / (nff-1)) % (ncc-1);
          f_sm = lane_id % (nff-1);
          T back_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T back_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          T front_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          T front_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2+2)];
          T back = lerp(back_left, back_right, (T)0.5);
          T front = lerp(front_left, front_right, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm, c_sm+ncc, f_sm+nff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2+1)] - lerp(back, front, (T)0.5);
        }
      }
      if (warp_id == 1) { // 
        if (lane_id < (nrr-1) * ncc * (nff-1)) {
          r_sm = (lane_id / (nff-1)) / ncc;
          c_sm = (lane_id / (nff-1)) % ncc;
          f_sm = lane_id % (nff-1);
          T bottom_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T bottom_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          T top_left = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          T top_right = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2+2)];
          T bottom = lerp(bottom_left, bottom_right, (T)0.5);
          T top = lerp(top_left, top_right, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm, c_sm+ncc, f_sm+nff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2+1)] - lerp(bottom, top, (T)0.5);
        }
        if (lane_id < (nrr-1) * (ncc-1) * nff) {
          r_sm = (lane_id / nff) / (ncc-1);
          c_sm = (lane_id / nff) % (ncc-1);
          f_sm = lane_id % nff;
          T bottom_back = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T bottom_front = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          T top_back = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          T top_front = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2+2, f_sm*2)];
          T bottom = lerp(bottom_back, bottom_front, (T)0.5);
          T top = lerp(top_back, top_front, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm+nrr, c_sm+ncc, f_sm)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2+1, f_sm*2)] - lerp(bottom, top, (T)0.5);
        }
      }
      if (warp_id == 2) {
        if (lane_id < (nrr-1) * (ncc-1) * (nff-1)) {
          r_sm = (lane_id / (nff-1)) / (ncc-1);
          c_sm = (lane_id / (nff-1)) % (ncc-1);
          f_sm = lane_id % (nff-1);
          T bottom_back_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T bottom_back_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          T bottom_front_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          T bottom_front_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2+2)];
          T top_back_left = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          T top_back_right = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2+2)];
          T top_front_left = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2+2, f_sm*2)];
          T top_front_right = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2+2, f_sm*2+2)];
          T bottom_back = lerp(bottom_back_left, bottom_back_right, (T)0.5);
          T bottom_front = lerp(bottom_front_left, bottom_front_right, (T)0.5);
          T top_back = lerp(top_back_left, top_back_right, (T)0.5);
          T top_front = lerp(top_front_left, top_front_right, (T)0.5);
          T bottom = lerp(bottom_back, bottom_front, (T)0.5);
          T top = lerp(top_back, top_front, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm+nrr, c_sm+ncc, f_sm+nff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2+1, f_sm*2+1)] - lerp(bottom, top, (T)0.5);
        }
      }
    } else if constexpr (R == 9 && C == 9 && F == 9) {
      // Total number of full warps: 9^3/32 = 22;
      // coarse   - 5*5*5 - 4 warps * 1 type  = 4 warps
      // f/c/r    - 4*5*5 - 4 warps * 3 types = 12 warps  1 lerp
      // cf/rf/rc - 4*4*5 - 3 warps * 3 types = 9 warps   3 lerps
      // rcf      - 4*4*4 - 2 warps * 1 type  = 2 warps   6 lerps
      // 22 warps: 4 3 2
      SIZE warp_id = threadId / MGARDX_WARP_SIZE;
      if (0 <= warp_id && warp_id < 4) {
        SIZE hyper_lane_id = (threadId - 0*MGARDX_WARP_SIZE) % (MGARDX_WARP_SIZE * 4);
        if (hyper_lane_id < nrr * ncc * nff) {
          r_sm = (hyper_lane_id / nff) / ncc;
          c_sm = (hyper_lane_id / nff) % ncc;
          f_sm = hyper_lane_id % nff;
          v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
        }
        if (hyper_lane_id < nrr * ncc * (nff-1)) {
          r_sm = (hyper_lane_id / (nff-1)) / ncc;
          c_sm = (hyper_lane_id / (nff-1)) % ncc;
          f_sm = hyper_lane_id % (nff-1);
          T left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm+nff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+1)] - lerp(left, right, (T)0.5);
        }
      }
      if (4 <= warp_id && warp_id < 8) {
        SIZE hyper_lane_id = (threadId - 4*MGARDX_WARP_SIZE) % (MGARDX_WARP_SIZE * 4);
        if (hyper_lane_id < nrr * (ncc-1) * nff) {
          r_sm = (hyper_lane_id / nff) / (ncc-1);
          c_sm = (hyper_lane_id / nff) % (ncc-1);
          f_sm = hyper_lane_id % nff;
          T back = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T front = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          v_sm[get_idx(ld1, ld2, r_sm, c_sm+ncc, f_sm)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2)] - lerp(back, front, (T)0.5);
        }
        if (hyper_lane_id < (nrr-1) * ncc * nff) {
          r_sm = (hyper_lane_id / nff) / ncc;
          c_sm = (hyper_lane_id / nff) % ncc;
          f_sm = hyper_lane_id % nff;
          T bottom = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T top = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          v_sm[get_idx(ld1, ld2, r_sm+nrr, c_sm, f_sm)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2, f_sm*2)] - lerp(bottom, top, (T)0.5);
        }
      }
      if (8 <= warp_id && warp_id < 11) {
        SIZE hyper_lane_id = (threadId - 8*MGARDX_WARP_SIZE) % (MGARDX_WARP_SIZE * 3);
        if (hyper_lane_id < nrr * (ncc-1) * (nff-1)) {
          r_sm = (hyper_lane_id / (nff-1)) / (ncc-1);
          c_sm = (hyper_lane_id / (nff-1)) % (ncc-1);
          f_sm = hyper_lane_id % (nff-1);
          T back_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T back_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          T front_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          T front_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2+2)];
          T back = lerp(back_left, back_right, (T)0.5);
          T front = lerp(front_left, front_right, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm, c_sm+ncc, f_sm+nff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2+1)] - lerp(back, front, (T)0.5);
        }
      }
      if (11 <= warp_id && warp_id < 14) {
        SIZE hyper_lane_id = (threadId - 11*MGARDX_WARP_SIZE) % (MGARDX_WARP_SIZE * 3);
        if (hyper_lane_id < (nrr-1) * ncc * (nff-1)) {
          r_sm = (hyper_lane_id / (nff-1)) / ncc;
          c_sm = (hyper_lane_id / (nff-1)) % ncc;
          f_sm = hyper_lane_id % (nff-1);
          T bottom_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T bottom_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          T top_left = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          T top_right = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2+2)];
          T bottom = lerp(bottom_left, bottom_right, (T)0.5);
          T top = lerp(top_left, top_right, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm, c_sm+ncc, f_sm+nff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2+1)] - lerp(bottom, top, (T)0.5);
        }
      }
      if (14 <= warp_id && warp_id < 17) {
        SIZE hyper_lane_id = (threadId - 14*MGARDX_WARP_SIZE) % (MGARDX_WARP_SIZE * 3);
        if (hyper_lane_id < (nrr-1) * (ncc-1) * nff) {
          r_sm = (hyper_lane_id / nff) / (ncc-1);
          c_sm = (hyper_lane_id / nff) % (ncc-1);
          f_sm = hyper_lane_id % nff;
          T bottom_back = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T bottom_front = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          T top_back = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          T top_front = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2+2, f_sm*2)];
          T bottom = lerp(bottom_back, bottom_front, (T)0.5);
          T top = lerp(top_back, top_front, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm+nrr, c_sm+ncc, f_sm)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2+1, f_sm*2)] - lerp(bottom, top, (T)0.5);
        }
      }
      if (17 <= warp_id && warp_id < 19) {
        SIZE hyper_lane_id = (threadId - 17*MGARDX_WARP_SIZE) % (MGARDX_WARP_SIZE * 2);
        if (hyper_lane_id < (nrr-1) * (ncc-1) * (nff-1)) {
          r_sm = (hyper_lane_id / (nff-1)) / (ncc-1);
          c_sm = (hyper_lane_id / (nff-1)) % (ncc-1);
          f_sm = hyper_lane_id % (nff-1);
          T bottom_back_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T bottom_back_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          T bottom_front_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          T bottom_front_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2+2)];
          T top_back_left = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          T top_back_right = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2+2)];
          T top_front_left = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2+2, f_sm*2)];
          T top_front_right = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2+2, f_sm*2+2)];
          T bottom_back = lerp(bottom_back_left, bottom_back_right, (T)0.5);
          T bottom_front = lerp(bottom_front_left, bottom_front_right, (T)0.5);
          T top_back = lerp(top_back_left, top_back_right, (T)0.5);
          T top_front = lerp(top_front_left, top_front_right, (T)0.5);
          T bottom = lerp(bottom_back, bottom_front, (T)0.5);
          T top = lerp(top_back, top_front, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm+nrr, c_sm+ncc, f_sm+nff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2+1, f_sm*2+1)] - lerp(bottom, top, (T)0.5);
        }
      }
    }
  }

  MGARDX_EXEC void CalculateCoefficientInterpolateAndSubtract() {

  }

  MGARDX_EXEC void CalculateMassTrans(T h) {
    T a = 0, b = 0, c = 0, d = 0, e = 0;
    if (0 <= threadId && threadId < nr * nc * nff) {
      r_sm = (threadId / nff) / nc;
      c_sm = (threadId / nff) % nc;
      f_sm = threadId % nff;
      if (r_sm >= nrr && c_sm >= ncc) {
        if (f_sm > 1) a = v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm-1)];
        c = v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)];
        if (f_sm+1 < nff) e = v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm+1)];
      }
      if (f_sm > 1) b = v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm+nff-1)];
      if (f_sm+nff < nf) d = v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm+nff)];

      w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] =
         mass_trans(a, b, c, d, e, h, h, h, h, (T)0.5, (T)0.5, (T)0.5, (T)0.5);

    }
  }

  MGARDX_EXEC void CalculateCorrection() {
    T prev; 
    if (0 <= threadId && threadId < ncc * nrr) {
      r_sm = threadId / ncc;
      c_sm = threadId % ncc;
      prev = 0.0;
      for (f_sm = 0; f_sm < nff; f_sm++) {
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = 
                   tridiag_forward2(prev, am_f_sm[f_sm], bm_f_sm[f_sm],
                             w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)]);
        prev = w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)];
      }
      prev = 0.0;
      for (f_sm = nff-1; f_sm >= 0; f_sm--) {
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = 
                   tridiag_backward2(prev, am_f_sm[f_sm], bm_f_sm[f_sm],
                              w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)]);
        prev = w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)];
      }
    }
#ifdef MGARDX_COMPILE_CUDA
    // __syncthreads();
#endif
    
    if (0 <= threadId && threadId < nff * nrr) {
      r_sm = threadId / nff;
      f_sm = threadId % nff;
      prev = 0.0;
      for (c_sm = 0; c_sm < ncc; c_sm++) {
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = 
                   tridiag_forward2(prev, am_c_sm[c_sm], bm_c_sm[c_sm],
                             w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)]);
        prev = w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)];
      }
      prev = 0.0;
      for (c_sm = ncc-1; c_sm >= 0; c_sm--) {
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = 
                   tridiag_backward2(prev, am_c_sm[c_sm], bm_c_sm[c_sm],
                              w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)]);
        prev = w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)];
      }
    }
#ifdef MGARDX_COMPILE_CUDA
    // __syncthreads();
#endif
    
    if (0 <= threadId && threadId < nff * ncc) {
      c_sm = threadId / nff;
      f_sm = threadId % nff;
      prev = 0.0;
      for (r_sm = 0; r_sm < nrr; r_sm++) {
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = 
                   tridiag_forward2(prev, am_r_sm[r_sm], bm_r_sm[r_sm],
                             w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)]);
        prev = w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)];
      }
      prev = 0.0;
      for (r_sm = nrr-1; r_sm >= 0; r_sm--) {
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = 
                   tridiag_backward2(prev, am_r_sm[r_sm], bm_r_sm[r_sm],
                              w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)]);
        prev = w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)];
      }
    }
#ifdef MGARDX_COMPILE_CUDA
    // __syncthreads();
#endif
  }



  MGARDX_EXEC void Operation2() {
    T h = 1;
    for (int l = l_target; l > 0; l--) {
      nrr = nr / 2 + 1;
      ncc = nc / 2 + 1;
      nff = nf / 2 + 1;


      // Copy from v_sm to w_sm
      r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
      c_sm = FunctorBase<DeviceType>::GetThreadIdY();
      f_sm = FunctorBase<DeviceType>::GetThreadIdX();
      w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)];

      // w_sm --> v_sm (coefficients)
      CalculateCoefficientFixedAssignment();
      #ifdef MGARDX_COMPILE_CUDA
          // __syncthreads();
      #endif
      // CalculateCoefficientCompactReassignment();
      // CalculateCoefficientWarpReassignment();
      // v_sm --> w_sm (correction)
      CalculateMassTrans(h);
      #ifdef MGARDX_COMPILE_CUDA
          // __syncthreads();
      #endif
      // CalculateCorrection();

      h *= 2;
    }
  }

  MGARDX_EXEC void Operation3() {
    if (r_gl >= v.shape(D-3) || c_gl >= v.shape(D-2) || f_gl >= v.shape(D-1)) {
      return;
    }

    r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();
    *v(r_gl, c_gl, f_gl) = v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)];
  }

private:
  // functor parameters
  SubArray<D, T, DeviceType> v;
  SubArray<1, SubArray<1, T, DeviceType>, DeviceType> am_r;
  SubArray<1, SubArray<1, T, DeviceType>, DeviceType> am_c;
  SubArray<1, SubArray<1, T, DeviceType>, DeviceType> am_f;
  SubArray<1, SubArray<1, T, DeviceType>, DeviceType> bm_r;
  SubArray<1, SubArray<1, T, DeviceType>, DeviceType> bm_c;
  SubArray<1, SubArray<1, T, DeviceType>, DeviceType> bm_f;
  SIZE l_target;
  // thread local variables
  int r_sm, c_sm, f_sm;
  int r_gl, c_gl, f_gl;
  int threadId;
  T *sm;
  SIZE ld1;
  SIZE ld2;
  T *v_sm, *w_sm;
  T *am_r_sm, *am_c_sm, *am_f_sm;
  T *bm_r_sm, *bm_c_sm, *bm_f_sm;
  SIZE nr, nc, nf, nrr, ncc, nff;
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class DataRefactoringKernel : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  DataRefactoringKernel() : AutoTuner<DeviceType>() {}

  MGARDX_CONT Task<DataRefactoringFunctor<D, T, R, C, F, DeviceType>>
  GenTask(SubArray<D, T, DeviceType> v, 
          SubArray<1, SubArray<1, T, DeviceType>, DeviceType> am_r,
          SubArray<1, SubArray<1, T, DeviceType>, DeviceType> am_c,
          SubArray<1, SubArray<1, T, DeviceType>, DeviceType> am_f,
          SubArray<1, SubArray<1, T, DeviceType>, DeviceType> bm_r,
          SubArray<1, SubArray<1, T, DeviceType>, DeviceType> bm_c,
          SubArray<1, SubArray<1, T, DeviceType>, DeviceType> bm_f,
          SIZE l_target, int queue_idx) {
  
    using FunctorType = DataRefactoringFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(v, am_r, am_c, am_f, bm_r, bm_c, bm_f, l_target);

    SIZE total_thread_z = v.shape(D-3);
    SIZE total_thread_y = v.shape(D-2);
    SIZE total_thread_x = v.shape(D-1);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size;
    tbz = R;
    tby = C;
    tbx = F;
    sm_size = R * C * F * sizeof(T) * 2;
    sm_size += ((R+1) + (C+1) + (F+1)) * 2 * (l_target+1) * sizeof(T); 
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "blockwise::DataRefactoring");
  }

  MGARDX_CONT
  void Execute(SubArray<D, T, DeviceType> v, int queue_idx) {
    using FunctorType = DataRefactoringFunctor<D, T, R, C, F, DeviceType>;
    using TaskType = Task<FunctorType>;

    // For calculating am and bm
    Hierarchy<D, T, DeviceType> hierarchy({R, C, F});
    SIZE l_target = hierarchy.l_target();
    bool pitched = false, managed = true;
    // am
    Array<1, SubArray<1, T, DeviceType>, DeviceType> am_r_array({l_target+1}, pitched, managed);
    SubArray am_r(am_r_array);
    for (SIZE l = 0; l < l_target+1; l++) *am_r(l) = hierarchy.am(l, D-3);
    Array<1, SubArray<1, T, DeviceType>, DeviceType> am_c_array({l_target+1}, pitched, managed);
    SubArray am_c(am_c_array);
    for (SIZE l = 0; l < l_target+1; l++) *am_c(l) = hierarchy.am(l, D-2);
    Array<1, SubArray<1, T, DeviceType>, DeviceType> am_f_array({l_target+1}, pitched, managed);
    SubArray am_f(am_f_array);
    for (SIZE l = 0; l < l_target+1; l++) *am_f(l) = hierarchy.am(l, D-1);
    // bm
    Array<1, SubArray<1, T, DeviceType>, DeviceType> bm_r_array({l_target+1}, pitched, managed);
    SubArray bm_r(bm_r_array);
    for (SIZE l = 0; l < l_target+1; l++) *bm_r(l) = hierarchy.bm(l, D-3);
    Array<1, SubArray<1, T, DeviceType>, DeviceType> bm_c_array({l_target+1}, pitched, managed);
    SubArray bm_c(bm_c_array);
    for (SIZE l = 0; l < l_target+1; l++) *bm_c(l) = hierarchy.bm(l, D-2);
    Array<1, SubArray<1, T, DeviceType>, DeviceType> bm_f_array({l_target+1}, pitched, managed);
    SubArray bm_f(bm_f_array);
    for (SIZE l = 0; l < l_target+1; l++) *bm_f(l) = hierarchy.bm(l, D-1);
    TaskType task = GenTask(v, am_r, am_c, am_f, bm_r, bm_c, bm_f, l_target, queue_idx);
    DeviceAdapter<TaskType, DeviceType> adapter;
    Timer timer_each;
    DeviceRuntime<DeviceType>::SyncDevice();
    timer_each.start();
    timer_each.start();
    ExecutionReturn ret = adapter.Execute(task);
    DeviceRuntime<DeviceType>::SyncDevice();
    timer_each.end();
    timer_each.print("blockwise::Decomposition::Kernel");
    timer_each.clear();
  }
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
void decompose(SubArray<D, T, DeviceType> v, int queue_idx) {
  DataRefactoringKernel<D, T, R, C, F, DeviceType>().Execute(v, queue_idx);
}

} //name blockwise

} // namespace mgard_x

#endif