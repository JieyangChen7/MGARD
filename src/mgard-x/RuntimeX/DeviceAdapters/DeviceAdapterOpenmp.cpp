/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

int openmp_dev_id = 0;
DeviceQueues<OPENMP> DeviceRuntime<OPENMP>::queues;
DeviceSpecification<OPENMP> DeviceRuntime<OPENMP>::DeviceSpecs;

bool DeviceRuntime<OPENMP>::SyncAllKernelsAndCheckErrors = false;
bool DeviceRuntime<OPENMP>::TimingAllKernels = false;
bool DeviceRuntime<OPENMP>::PrintKernelConfig = false;

template <> bool deviceAvailable<OPENMP>() {
  return DeviceRuntime<OPENMP>::GetDeviceCount() > 0;
}

} // namespace mgard_x