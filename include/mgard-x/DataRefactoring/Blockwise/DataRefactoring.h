/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_BLOCKWISE_DATA_REFACTORING_H
#define MGARD_X_BLOCKWISE_DATA_REFACTORING_H

// #include "Common.h"
#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeXPublic.h"


namespace mgard_x {

namespace blockwise {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
void decompose(SubArray<D, T, DeviceType> v, int queue_idx);

}


} //namespace mgard_x

#endif
