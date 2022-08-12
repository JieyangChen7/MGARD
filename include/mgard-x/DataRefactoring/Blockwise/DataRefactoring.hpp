/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_BLOCKWISE_DATA_REFACTORING_TEMPLATE
#define MGARD_X_BLOCKWISE_DATA_REFACTORING_TEMPLATE

#include "../../RuntimeX/RuntimeX.h"
#include "../MultiDimension/Coefficient/GPKFunctor.h"

namespace mgard_x {

namespace blockwise {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class DataRefactoringFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DataRefactoringFunctor() {}
  MGARDX_CONT DataRefactoringFunctor(SubArray<D, T, DeviceType> v, SIZE l_target): v(v), l_target(l_target) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {

    sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    v_sm = sm;
    w_sm = sm + F * C * R;
    ld1 = F;
    ld2 = C;
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
    // printf("load: %u %u %u -> %f [%u]\n", r_gl, c_gl, f_gl, *v(r_gl, c_gl, f_gl), get_idx(ld1, ld2, r_sm, c_sm, f_sm));
    v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = *v(r_gl, c_gl, f_gl);
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
      v_sm[get_idx(ld1, ld2, r_sm/2, c_sm/2, f_sm/2+ff)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(left, right, (T)0.5);
    } else if (r_sm % 2 == 0 && c_sm % 2 != 0 && f_sm % 2 == 0) {
      T back = w_sm[get_idx(ld1, ld2, r_sm, c_sm-1, f_sm)];
      T front = w_sm[get_idx(ld1, ld2, r_sm, c_sm+1, f_sm)];
      v_sm[get_idx(ld1, ld2, r_sm/2, c_sm/2+cc, f_sm/2)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(back, front, (T)0.5);
    } else if (r_sm % 2 != 0 && c_sm % 2 == 0 && f_sm % 2 == 0) {
      T top = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm, f_sm)];
      T bottom = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm, f_sm)];
      v_sm[get_idx(ld1, ld2, r_sm/2+rr, c_sm/2, f_sm/2)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(top, bottom, (T)0.5);
    } else if (r_sm % 2 == 0 && c_sm % 2 != 0 && f_sm % 2 != 0) {
      T back_left = w_sm[get_idx(ld1, ld2, r_sm, c_sm-1, f_sm-1)];
      T back_right = w_sm[get_idx(ld1, ld2, r_sm, c_sm-1, f_sm+1)];
      T front_left = w_sm[get_idx(ld1, ld2, r_sm, c_sm+1, f_sm-1)];
      T front_right = w_sm[get_idx(ld1, ld2, r_sm, c_sm+1, f_sm+1)];
      T back = lerp(back_left, back_right, (T)0.5);
      T front = lerp(front_left, front_right, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm/2, c_sm/2+cc, f_sm/2+ff)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(back, front, (T)0.5);
    } else if (r_sm % 2 != 0 && c_sm % 2== 0 && f_sm % 2 != 0) {
      T bottom_left = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm, f_sm-1)];
      T bottom_right = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm, f_sm+1)];
      T top_left = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm, f_sm-1)];
      T top_right = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm, f_sm+1)];
      T bottom = lerp(bottom_left, bottom_right, (T)0.5);
      T top = lerp(top_left, top_right, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm/2+rr, c_sm/2, f_sm/2+ff)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(bottom, top, (T)0.5);
    } else if (r_sm % 2 != 0 && c_sm % 2!= 0 && f_sm % 2 == 0) {
      T bottom_back = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm-1, f_sm)];
      T bottom_front = w_sm[get_idx(ld1, ld2, r_sm-1, c_sm+1, f_sm)];
      T top_back = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm-1, f_sm)];
      T top_front = w_sm[get_idx(ld1, ld2, r_sm+1, c_sm+1, f_sm)];
      T bottom = lerp(bottom_back, bottom_front, (T)0.5);
      T top = lerp(top_back, top_front, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm/2+rr, c_sm/2+cc, f_sm/2)] = 
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
      v_sm[get_idx(ld1, ld2, r_sm/2+rr, c_sm/2+cc, f_sm/2+ff)] = 
        w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] -  lerp(bottom, top, (T)0.5);
    }
  }

  MGARDX_EXEC void CalculateCoefficientCompactReassignment() {
    int base = 0;
    threadId -= base;
    if (0 <= threadId && threadId < rr * cc * ff) {
      r_sm = (threadId / ff) / cc;
      c_sm = (threadId / ff) % cc;
      f_sm = threadId % ff;
      v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
    }
    threadId -= rr * cc * ff;
    // f - 18
    if (0 <= threadId && threadId < rr * cc * (ff-1)) {
      r_sm = (threadId / (ff-1)) / cc;
      c_sm = (threadId / (ff-1)) % cc;
      f_sm = threadId % (ff-1);
      T left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
      v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm+ff)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+1)] - lerp(left, right, (T)0.5);
    }
    threadId -= rr * cc * (ff-1);
    // c - 18
    if (0 <= threadId && threadId < rr * (cc-1) * ff) {
      r_sm = (threadId / ff) / (cc-1);
      c_sm = (threadId / ff) % (cc-1);
      f_sm = threadId % ff;
      T back = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T front = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
      v_sm[get_idx(ld1, ld2, r_sm, c_sm+cc, f_sm)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2)] - lerp(back, front, (T)0.5);
    }
    threadId -= rr * (cc-1) * ff;
    // r - 18
    if (0 <= threadId && threadId < (rr-1) * cc * ff) {
      r_sm = (threadId / ff) / cc;
      c_sm = (threadId / ff) % cc;
      f_sm = threadId % ff;
      T bottom = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T top = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
      v_sm[get_idx(ld1, ld2, r_sm+rr, c_sm, f_sm)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2, f_sm*2)] - lerp(bottom, top, (T)0.5);
    }
    threadId -= (rr-1) * cc * ff;
    // cf - 12
    if (0 <= threadId && threadId < rr * (cc-1) * (ff-1)) {
      r_sm = (threadId / (ff-1)) / (cc-1);
      c_sm = (threadId / (ff-1)) % (cc-1);
      f_sm = threadId % (ff-1);
      T back_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T back_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
      T front_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
      T front_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2+2)];
      T back = lerp(back_left, back_right, (T)0.5);
      T front = lerp(front_left, front_right, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm, c_sm+cc, f_sm+ff)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2+1)] - lerp(back, front, (T)0.5);
    }
    threadId -= rr * (cc-1) * (ff-1);
    // rf - 12
    if (0 <= threadId && threadId < (rr-1) * cc * (ff-1)) {
      r_sm = (threadId / (ff-1)) / cc;
      c_sm = (threadId / (ff-1)) % cc;
      f_sm = threadId % (ff-1);
      T bottom_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T bottom_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
      T top_left = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
      T top_right = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2+2)];
      T bottom = lerp(bottom_left, bottom_right, (T)0.5);
      T top = lerp(top_left, top_right, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm, c_sm+cc, f_sm+ff)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2+1)] - lerp(bottom, top, (T)0.5);
    }
    threadId -= (rr-1) * cc * (ff-1);
    // rc - 12
    if (0 <= threadId && threadId < (rr-1) * (cc-1) * ff) {
      r_sm = (threadId / ff) / (cc-1);
      c_sm = (threadId / ff) % (cc-1);
      f_sm = threadId % ff;
      T bottom_back = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
      T bottom_front = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
      T top_back = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
      T top_front = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2+2, f_sm*2)];
      T bottom = lerp(bottom_back, bottom_front, (T)0.5);
      T top = lerp(top_back, top_front, (T)0.5);
      v_sm[get_idx(ld1, ld2, r_sm+rr, c_sm+cc, f_sm)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2+1, f_sm*2)] - lerp(bottom, top, (T)0.5);
    }
    threadId -= (rr-1) * (cc-1) * ff;
    //rcf - 8
    if (0 <= threadId && threadId < (rr-1) * (cc-1) * (ff-1)) {
      r_sm = (threadId / (ff-1)) / (cc-1);
      c_sm = (threadId / (ff-1)) % (cc-1);
      f_sm = threadId % (ff-1);
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
      v_sm[get_idx(ld1, ld2, r_sm+rr, c_sm+cc, f_sm+ff)] = 
        w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2+1, f_sm*2+1)] - lerp(bottom, top, (T)0.5);
    }
    threadId -= (rr-1) * (cc-1) * (ff-1);
  }

  MGARDX_EXEC void CalculateCoefficientWarpReassignment() {
    assert(R * C * F >= MGARDX_WARP_SIZE);
    // 5*5*5 cube
    if constexpr (R == 5 && C == 5 && F == 5) {
      // Total number of full warps: 5^3/32 = 3;
      // Ensure we have enough threads
      assert(MGARDX_WARP_SIZE > rr * cc * ff);
      SIZE warp_id = threadId / MGARDX_WARP_SIZE;
      SIZE lane_id = threadId % MGARDX_WARP_SIZE;
      if (warp_id == 0) { // 0+1+1+1+3 = 6 lerps
        if (lane_id < rr * cc * ff) {
          r_sm = (lane_id / ff) / cc;
          c_sm = (lane_id / ff) % cc;
          f_sm = lane_id % ff;
          v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
        }
        if (lane_id < rr * cc * (ff-1)) {
          r_sm = (lane_id / (ff-1)) / cc;
          c_sm = (lane_id / (ff-1)) % cc;
          f_sm = lane_id % (ff-1);
          T left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm+ff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+1)] - lerp(left, right, (T)0.5);
        }
        if (lane_id < rr * (cc-1) * ff) {
          r_sm = (lane_id / ff) / (cc-1);
          c_sm = (lane_id / ff) % (cc-1);
          f_sm = lane_id % ff;
          T back = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T front = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          v_sm[get_idx(ld1, ld2, r_sm, c_sm+cc, f_sm)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2)] - lerp(back, front, (T)0.5);
        }
        if (lane_id < (rr-1) * cc * ff) {
          r_sm = (lane_id / ff) / cc;
          c_sm = (lane_id / ff) % cc;
          f_sm = lane_id % ff;
          T bottom = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T top = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          v_sm[get_idx(ld1, ld2, r_sm+rr, c_sm, f_sm)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2, f_sm*2)] - lerp(bottom, top, (T)0.5);
        }
        if (lane_id < rr * (cc-1) * (ff-1)) {
          r_sm = (lane_id / (ff-1)) / (cc-1);
          c_sm = (lane_id / (ff-1)) % (cc-1);
          f_sm = lane_id % (ff-1);
          T back_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T back_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          T front_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          T front_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2+2)];
          T back = lerp(back_left, back_right, (T)0.5);
          T front = lerp(front_left, front_right, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm, c_sm+cc, f_sm+ff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2+1)] - lerp(back, front, (T)0.5);
        }
      }
      if (warp_id == 1) { // 
        if (lane_id < (rr-1) * cc * (ff-1)) {
          r_sm = (lane_id / (ff-1)) / cc;
          c_sm = (lane_id / (ff-1)) % cc;
          f_sm = lane_id % (ff-1);
          T bottom_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T bottom_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          T top_left = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          T top_right = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2+2)];
          T bottom = lerp(bottom_left, bottom_right, (T)0.5);
          T top = lerp(top_left, top_right, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm, c_sm+cc, f_sm+ff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2+1)] - lerp(bottom, top, (T)0.5);
        }
        if (lane_id < (rr-1) * (cc-1) * ff) {
          r_sm = (lane_id / ff) / (cc-1);
          c_sm = (lane_id / ff) % (cc-1);
          f_sm = lane_id % ff;
          T bottom_back = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T bottom_front = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          T top_back = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          T top_front = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2+2, f_sm*2)];
          T bottom = lerp(bottom_back, bottom_front, (T)0.5);
          T top = lerp(top_back, top_front, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm+rr, c_sm+cc, f_sm)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2+1, f_sm*2)] - lerp(bottom, top, (T)0.5);
        }
      }
      if (warp_id == 2) {
        if (lane_id < (rr-1) * (cc-1) * (ff-1)) {
          r_sm = (lane_id / (ff-1)) / (cc-1);
          c_sm = (lane_id / (ff-1)) % (cc-1);
          f_sm = lane_id % (ff-1);
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
          v_sm[get_idx(ld1, ld2, r_sm+rr, c_sm+cc, f_sm+ff)] = 
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
        if (hyper_lane_id < rr * cc * ff) {
          r_sm = (hyper_lane_id / ff) / cc;
          c_sm = (hyper_lane_id / ff) % cc;
          f_sm = hyper_lane_id % ff;
          v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
        }
        if (hyper_lane_id < rr * cc * (ff-1)) {
          r_sm = (hyper_lane_id / (ff-1)) / cc;
          c_sm = (hyper_lane_id / (ff-1)) % cc;
          f_sm = hyper_lane_id % (ff-1);
          T left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm+ff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+1)] - lerp(left, right, (T)0.5);
        }
      }
      if (4 <= warp_id && warp_id < 8) {
        SIZE hyper_lane_id = (threadId - 4*MGARDX_WARP_SIZE) % (MGARDX_WARP_SIZE * 4);
        if (hyper_lane_id < rr * (cc-1) * ff) {
          r_sm = (hyper_lane_id / ff) / (cc-1);
          c_sm = (hyper_lane_id / ff) % (cc-1);
          f_sm = hyper_lane_id % ff;
          T back = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T front = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          v_sm[get_idx(ld1, ld2, r_sm, c_sm+cc, f_sm)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2)] - lerp(back, front, (T)0.5);
        }
        if (hyper_lane_id < (rr-1) * cc * ff) {
          r_sm = (hyper_lane_id / ff) / cc;
          c_sm = (hyper_lane_id / ff) % cc;
          f_sm = hyper_lane_id % ff;
          T bottom = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T top = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          v_sm[get_idx(ld1, ld2, r_sm+rr, c_sm, f_sm)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2, f_sm*2)] - lerp(bottom, top, (T)0.5);
        }
      }
      if (8 <= warp_id && warp_id < 11) {
        SIZE hyper_lane_id = (threadId - 8*MGARDX_WARP_SIZE) % (MGARDX_WARP_SIZE * 3);
        if (hyper_lane_id < rr * (cc-1) * (ff-1)) {
          r_sm = (hyper_lane_id / (ff-1)) / (cc-1);
          c_sm = (hyper_lane_id / (ff-1)) % (cc-1);
          f_sm = hyper_lane_id % (ff-1);
          T back_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T back_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          T front_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          T front_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2+2)];
          T back = lerp(back_left, back_right, (T)0.5);
          T front = lerp(front_left, front_right, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm, c_sm+cc, f_sm+ff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2+1)] - lerp(back, front, (T)0.5);
        }
      }
      if (11 <= warp_id && warp_id < 14) {
        SIZE hyper_lane_id = (threadId - 11*MGARDX_WARP_SIZE) % (MGARDX_WARP_SIZE * 3);
        if (hyper_lane_id < (rr-1) * cc * (ff-1)) {
          r_sm = (hyper_lane_id / (ff-1)) / cc;
          c_sm = (hyper_lane_id / (ff-1)) % cc;
          f_sm = hyper_lane_id % (ff-1);
          T bottom_left = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T bottom_right = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2+2)];
          T top_left = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          T top_right = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2+2)];
          T bottom = lerp(bottom_left, bottom_right, (T)0.5);
          T top = lerp(top_left, top_right, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm, c_sm+cc, f_sm+ff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+1, f_sm*2+1)] - lerp(bottom, top, (T)0.5);
        }
      }
      if (14 <= warp_id && warp_id < 17) {
        SIZE hyper_lane_id = (threadId - 14*MGARDX_WARP_SIZE) % (MGARDX_WARP_SIZE * 3);
        if (hyper_lane_id < (rr-1) * (cc-1) * ff) {
          r_sm = (hyper_lane_id / ff) / (cc-1);
          c_sm = (hyper_lane_id / ff) % (cc-1);
          f_sm = hyper_lane_id % ff;
          T bottom_back = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2, f_sm*2)];
          T bottom_front = w_sm[get_idx(ld1, ld2, r_sm*2, c_sm*2+2, f_sm*2)];
          T top_back = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2, f_sm*2)];
          T top_front = w_sm[get_idx(ld1, ld2, r_sm*2+2, c_sm*2+2, f_sm*2)];
          T bottom = lerp(bottom_back, bottom_front, (T)0.5);
          T top = lerp(top_back, top_front, (T)0.5);
          v_sm[get_idx(ld1, ld2, r_sm+rr, c_sm+cc, f_sm)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2+1, f_sm*2)] - lerp(bottom, top, (T)0.5);
        }
      }
      if (17 <= warp_id && warp_id < 19) {
        SIZE hyper_lane_id = (threadId - 17*MGARDX_WARP_SIZE) % (MGARDX_WARP_SIZE * 2);
        if (hyper_lane_id < (rr-1) * (cc-1) * (ff-1)) {
          r_sm = (hyper_lane_id / (ff-1)) / (cc-1);
          c_sm = (hyper_lane_id / (ff-1)) % (cc-1);
          f_sm = hyper_lane_id % (ff-1);
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
          v_sm[get_idx(ld1, ld2, r_sm+rr, c_sm+cc, f_sm+ff)] = 
            w_sm[get_idx(ld1, ld2, r_sm*2+1, c_sm*2+1, f_sm*2+1)] - lerp(bottom, top, (T)0.5);
        }
      }
    }
  }

  MGARDX_EXEC void CalculateCoefficientInterpolateAndSubtract() {

  }

  MGARDX_EXEC void CalculateMassTrans() {
    
  }



  MGARDX_EXEC void Operation2() {
    for (int l = 1; l > 0; l--) {
      rr = r / 2 + 1;
      cc = c / 2 + 1;
      ff = f / 2 + 1;

      // Copy from v_sm to w_sm
      r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
      c_sm = FunctorBase<DeviceType>::GetThreadIdY();
      f_sm = FunctorBase<DeviceType>::GetThreadIdX();
      w_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)] = v_sm[get_idx(ld1, ld2, r_sm, c_sm, f_sm)];

      // w_sm --> v_sm (coefficients)
      CalculateCoefficientFixedAssignment();
      // CalculateCoefficientCompactReassignment();
      // CalculateCoefficientWarpReassignment();
      // v_sm --> w_sm (correction)
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
  SIZE l_target;
  // thread local variables
  SIZE r_sm, c_sm, f_sm;
  SIZE r_gl, c_gl, f_gl;
  int threadId;
  T *sm;
  SIZE ld1;
  SIZE ld2;
  T *v_sm, *w_sm;
  SIZE r, c, f, rr, cc, ff;
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class DataRefactoringKernel : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  DataRefactoringKernel() : AutoTuner<DeviceType>() {}

  MGARDX_CONT Task<DataRefactoringFunctor<D, T, R, C, F, DeviceType>>
  GenTask(SubArray<D, T, DeviceType> v, int queue_idx) {

    SIZE l_target;
    if constexpr (R == 3) {
      l_target = 1;
    }
    if constexpr (R == 5) {
      l_target = 2;
    }
    if constexpr (R == 9){
      l_target = 3;
    }
    using FunctorType = DataRefactoringFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(v, l_target);

    SIZE total_thread_z = v.shape(D-3);
    SIZE total_thread_y = v.shape(D-2);
    SIZE total_thread_x = v.shape(D-1);
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
                "blockwise::DataRefactoring");
  }

  MGARDX_CONT
  void Execute(SubArray<D, T, DeviceType> v, int queue_idx) {
    using FunctorType = DataRefactoringFunctor<D, T, R, C, F, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask(v, queue_idx);
    DeviceAdapter<TaskType, DeviceType> adapter;
    ExecutionReturn ret = adapter.Execute(task);
  }
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
void decompose(SubArray<D, T, DeviceType> v, int queue_idx) {
  DataRefactoringKernel<D, T, R, C, F, DeviceType>().Execute(v, queue_idx);
}

} //name blockwise

} // namespace mgard_x

#endif