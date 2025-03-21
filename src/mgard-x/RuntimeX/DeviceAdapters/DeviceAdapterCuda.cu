/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

int cuda_dev_id = 0;
DeviceQueues<CUDA> DeviceRuntime<CUDA>::queues;
DeviceSpecification<CUDA> DeviceRuntime<CUDA>::DeviceSpecs;

bool DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors = false;
bool DeviceRuntime<CUDA>::TimingAllKernels = false;
bool DeviceRuntime<CUDA>::PrintKernelConfig = false;

template <> bool deviceAvailable<CUDA>() {
  return DeviceRuntime<CUDA>::GetDeviceCount() > 0;
}

} // namespace mgard_x