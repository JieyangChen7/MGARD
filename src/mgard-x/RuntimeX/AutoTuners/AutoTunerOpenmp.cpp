/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */
#include "mgard-x/RuntimeX/RuntimeX.h"
namespace mgard_x {

template void BeginAutoTuning<OPENMP>();
template void EndAutoTuning<OPENMP>();

AutoTuningTable<OPENMP> AutoTuner<OPENMP>::autoTuningTable;
bool AutoTuner<OPENMP>::ProfileKernels = false;
bool AutoTuner<OPENMP>::WriteToTable = false;

} // namespace mgard_x