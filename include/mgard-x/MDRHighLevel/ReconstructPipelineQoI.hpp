/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_MDR_RECONSTRUCT_PIPELINE_QOI_HPP
#define MGARD_X_MDR_RECONSTRUCT_PIPELINE_QOI_HPP

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, typename DeviceType, typename ReconstructorType>
void reconstruct_pipeline_qoi(
    DomainDecomposer<D, T, ReconstructorType, DeviceType> &domain_decomposer,
    Config &config, RefactoredMetadata &refactored_metadata,
    RefactoredData &refactored_data, ReconstructedData &reconstructed_data) {
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();

  using Cache = ReconstructorCache<D, T, DeviceType, ReconstructorType>;
  using HierarchyType = typename ReconstructorType::HierarchyType;

  ReconstructorType &reconstructor = *Cache::cache.reconstructor;
  Array<D, T, DeviceType> *device_subdomain_buffer =
      Cache::cache.device_subdomain_buffer;
  MDRData<DeviceType> *mdr_data = Cache::cache.mdr_data;

  if (config.mdr_qoi_num_variables != domain_decomposer.num_subdomains()) {
    log::err("QOI mode requires the number of variables to be equal to the "
             "number of subdomains");
    exit(-1);
  }

  if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(0),
                                     domain_decomposer.uniform)) {
    Cache::cache.ClearHierarchyCache();
  }
  for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
    if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(id),
                                       domain_decomposer.uniform)) {
      Cache::cache.InsertHierarchyCache(
          domain_decomposer.subdomain_hierarchy(id));
    }
    mdr_data[id].Resize(refactored_metadata.metadata[id], 0);
    device_subdomain_buffer[id].resize(domain_decomposer.subdomain_shape(id),
                                       0);
    // Reset all signs to 0 for the initial QOI reconstruction
    if (!reconstructed_data.qoi_in_progress) {
      mdr_data[id].ResetSigns(0);
    }
  }

  log::info("Adjust device buffers");
  int current_buffer = 0;
  int current_queue = 0;

  // Prefetch the first subdomain
  mdr_data[current_buffer].CopyFromRefactoredData(
      refactored_metadata.metadata[0], refactored_data.data[0], current_queue);

  SIZE total_size = 0;

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    SIZE next_subdomain_id;
    int next_buffer = current_buffer + 1;
    int next_queue = (current_queue + 1) % 2;
    HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
        domain_decomposer.subdomain_shape(curr_subdomain_id));
    log::info("Adapt Refactor to hierarchy");
    reconstructor.Adapt(hierarchy, config, current_queue);
    total_size += hierarchy.total_num_elems() * sizeof(T);
    if (curr_subdomain_id + 1 < domain_decomposer.num_subdomains()) {
      // Prefetch the next subdomain
      next_subdomain_id = curr_subdomain_id + 1;
      mdr_data[next_buffer].CopyFromRefactoredData(
          refactored_metadata.metadata[next_subdomain_id],
          refactored_data.data[next_subdomain_id], next_queue);
    }

    if (curr_subdomain_id == config.mdr_qoi_num_variables - 1) {
      // We are about to finish reconstructing all variables
      // so, we need to fetch more data
      //
      // We need to update the metadata for all variables
      // for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
      //   metadata.requested_tol = tol; //new tolerance
      //   reconstructor.GenerateRequest(refactored_metadata.metadata[id]);
      // }
      // initiate the bitplane transfer for the 1st variable which
      // should coorespond to the next_buffer
      // mdr_data[0].CopyFromRefactoredData(
      //     refactored_metadata.metadata[0],
      //     refactored_data.data[0], next_queue);
    }

    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
    }
    log::info("Reconstruct subdomain " + std::to_string(curr_subdomain_id) +
              " with shape: " + ss.str());

    // Reconstruct
    reconstructor.ProgressiveReconstruct(
        refactored_metadata.metadata[curr_subdomain_id],
        mdr_data[current_buffer], config.mdr_adaptive_resolution,
        device_subdomain_buffer[current_buffer], current_queue);

    if (curr_subdomain_id == config.mdr_qoi_num_variables - 1) {
      DeviceRuntime<DeviceType>::SyncQueue(current_queue);
      // We are done with reconstructing all variables now
      // Do error estimation here
      // Var0 can be accessed from device_subdomain_buffer[0].data()
      // Var1 can be accessed from device_subdomain_buffer[1].data()
      // Var2 can be accessed from device_subdomain_buffer[2].data()
      //  if (tol NOT met) {
      //    need to contine reconstructing. Device buffers will NOT be released
      //    reconstructed_data.qoi_in_progress = true;
      //  } else {
      //     will stop reconstructing. Device buffers will be released
      //     reconstructed_data.qoi_in_progress = false;
      //  }
      //  we set it true for testing only
      reconstructed_data.qoi_in_progress = true;
    }
    
    current_buffer = next_buffer;
    current_queue = next_queue;
  }

  // Copy final data out if we are done with reconstructing
  DeviceRuntime<DeviceType>::SyncDevice();
  // We should only copy out data when we are done. But we copy it now for testing purposes
  // if (!reconstructed_data.qoi_in_progress) {
    for (SIZE curr_subdomain_id = 0;
      curr_subdomain_id < domain_decomposer.num_subdomains();
      curr_subdomain_id++) {
      // Update reconstructed data
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[curr_subdomain_id], curr_subdomain_id,
          subdomain_copy_direction::SubdomainToOriginal, current_queue);
    // }
  }

  DeviceRuntime<DeviceType>::SyncDevice();
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Reconstruct pipeline", total_size);
    timer_series.clear();
  }
}

} // namespace MDR
} // namespace mgard_x
#endif