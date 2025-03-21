/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

int hip_dev_id = 0;
DeviceQueues<HIP> DeviceRuntime<HIP>::queues;
DeviceSpecification<HIP> DeviceRuntime<HIP>::DeviceSpecs;

bool DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors = false;
bool DeviceRuntime<HIP>::TimingAllKernels = false;
bool DeviceRuntime<HIP>::PrintKernelConfig = false;

template <> bool deviceAvailable<HIP>() {
  return DeviceRuntime<HIP>::GetDeviceCount() > 0;
}

} // namespace mgard_x