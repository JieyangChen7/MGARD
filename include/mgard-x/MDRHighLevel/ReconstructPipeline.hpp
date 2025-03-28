/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_MDR_RECONSTRUCT_PIPELINE_HPP
#define MGARD_X_MDR_RECONSTRUCT_PIPELINE_HPP

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, typename DeviceType, typename ReconstructorType>
void reconstruct_pipeline(
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
  }

  log::info("Adjust device buffers");
  int current_buffer = 0;
  int current_queue = 0;

  mdr_data[current_buffer].Resize(refactored_metadata.metadata[0],
                                  current_queue);
  device_subdomain_buffer[current_buffer].resize(
      domain_decomposer.subdomain_shape(0), current_queue);

  // Prefetch the first subdomain
  mdr_data[current_buffer].CopyFromRefactoredData(
      refactored_metadata.metadata[0], refactored_data.data[0], current_queue);
  mdr_data[current_buffer].CopyFromRefactoredSigns(
      refactored_metadata.metadata[0], refactored_data.level_signs[0],
      current_queue);

  SIZE total_size = 0;

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    int next_queue = (current_queue + 1) % 3;
    HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
        domain_decomposer.subdomain_shape(curr_subdomain_id));
    log::info("Adapt Refactor to hierarchy");
    reconstructor.Adapt(hierarchy, config, current_queue);
    total_size += hierarchy.total_num_elems() * sizeof(T);
    if (curr_subdomain_id + 1 < domain_decomposer.num_subdomains()) {
      // Prefetch the next subdomain
      next_subdomain_id = curr_subdomain_id + 1;
      mdr_data[next_buffer].Resize(
          refactored_metadata.metadata[next_subdomain_id], next_queue);
      device_subdomain_buffer[next_buffer].resize(
          domain_decomposer.subdomain_shape(next_subdomain_id), next_queue);

      mdr_data[next_buffer].CopyFromRefactoredData(
          refactored_metadata.metadata[next_subdomain_id],
          refactored_data.data[next_subdomain_id], next_queue);
      // Copy signs
      mdr_data[next_buffer].CopyFromRefactoredSigns(
          refactored_metadata.metadata[next_subdomain_id],
          refactored_data.level_signs[next_subdomain_id], next_queue);
    }

    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
    }
    log::info("Reconstruct subdomain " + std::to_string(curr_subdomain_id) +
              " with shape: " + ss.str());
    device_subdomain_buffer[current_buffer].resize(
        hierarchy.level_shape(hierarchy.l_target()), current_queue);
    // Load previously reconstructred data
    domain_decomposer.copy_subdomain(
        device_subdomain_buffer[current_buffer], curr_subdomain_id,
        subdomain_copy_direction::OriginalToSubdomain, current_queue);
    // Reconstruct
    reconstructor.ProgressiveReconstruct(
        refactored_metadata.metadata[curr_subdomain_id],
        mdr_data[current_buffer], config.mdr_adaptive_resolution,
        device_subdomain_buffer[current_buffer], current_queue);

    // Update level signs for future progressive reconstruction
    mdr_data[current_buffer].CopyToRefactoredSigns(
        refactored_metadata.metadata[curr_subdomain_id],
        refactored_data.level_signs[curr_subdomain_id], current_queue);

    // Update reconstructed data
    domain_decomposer.copy_subdomain(
        device_subdomain_buffer[current_buffer], curr_subdomain_id,
        subdomain_copy_direction::SubdomainToOriginal, current_queue);

    if (config.mdr_adaptive_resolution) {
      reconstructed_data.shape[curr_subdomain_id] =
          device_subdomain_buffer[current_buffer].shape();
      reconstructed_data.offset[curr_subdomain_id] =
          domain_decomposer.dim_subdomain_offset(curr_subdomain_id);
    }
    current_buffer = next_buffer;
    current_queue = next_queue;
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