#ifndef MGARD_X_MDR_RECONSTRUCT_PIPELINE_QOI_HPP
#define MGARD_X_MDR_RECONSTRUCT_PIPELINE_QOI_HPP

#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <sstream>

#include "mgard-x/Config/Config.h"
#include "mgard-x/MDRHighLevel/MDRDataHighLevel.hpp"
#include "mgard-x/MDRHighLevel/MDRHighLevel.hpp"
#include "mgard-x/MDRHighLevel/QoIKernel.hpp"

namespace mgard_x {
namespace MDR {

inline uint32_t read_file_tmp(){
  std::string path = "/home/linusli037/Polaris/MGARD/build-cuda-turing/mgard/miniNYX/requested_size.txt";
  FILE *pFile;
  pFile = fopen(path.c_str(), "r");
  if (pFile == NULL) {
    std::cout << mgard_x::log::log_err << "file open error!\n";
    exit(1);
  }
  uint32_t value;
  if (fscanf(pFile, "%u", &value) != 1) {
    std::cout << mgard_x::log::log_err << "file read error!\n";
    fclose(pFile);
    exit(1);
  }
  fclose(pFile);
  return value;
}

template <DIM D, typename T, typename DeviceType, typename ReconstructorType>
void reconstruct_pipeline_qoi(
    DomainDecomposer<D, T, ReconstructorType, DeviceType> &domain_decomposer,
    Config &config, RefactoredMetadata &refactored_metadata,
    RefactoredData &refactored_data, ReconstructedData &reconstructed_data) {
  Timer timer_series, qoi_timer;
  if (log::level & log::TIME)
    timer_series.start();

  using Cache = ReconstructorCache<D, T, DeviceType, ReconstructorType>;
  using HierarchyType = typename ReconstructorType::HierarchyType;

  ReconstructorType &reconstructor = *Cache::cache.reconstructor;
  Array<D, T, DeviceType> *device_subdomain_buffer =
      Cache::cache.device_subdomain_buffer;
  MDRData<DeviceType> *mdr_data = Cache::cache.mdr_data;

  Array<D, double, DeviceType> error_out({config.domain_decomposition_sizes[0], config.domain_decomposition_sizes[1], config.domain_decomposition_sizes[2]});
  Array<1, double, DeviceType> error_final_out({1});
  Array<1, Byte, DeviceType> workspace;

  for(int i=0; i<2; i++){
    error_final_out.resize({1}, i);
    DeviceCollective<DeviceType>::AbsMax( config.domain_decomposition_sizes[0] * config.domain_decomposition_sizes[1] * config.domain_decomposition_sizes[2],
        SubArray<1, double, DeviceType>(), SubArray<1, double, DeviceType>(),
        workspace, false, 0);
  }


  if (config.mdr_qoi_num_variables != domain_decomposer.num_subdomains()) {
    log::err("QOI mode requires the number of variables to be equal to the "
             "number of subdomains"); 
    exit(-1);
  }

   log::info("Adjust device buffers");
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

  HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
          domain_decomposer.subdomain_shape(0));
  reconstructor.Adapt(hierarchy, config, 0);

 
  int current_buffer = 0;
  int current_queue = 0;

  // Prefetch the first subdomain
  mdr_data[current_buffer].CopyFromRefactoredData(
      refactored_metadata.metadata[0], refactored_data.data[0], current_queue);

  SIZE total_size = 0;
  uint32_t max_iter = 20;
  uint32_t iter = 0;
  int buffer_for_variable[3];
  double eb_Vx, eb_Vy, eb_Vz;
  double tol = refactored_metadata.metadata[0].requested_tol;

  reconstructed_data.qoi_in_progress = true;

  while((reconstructed_data.qoi_in_progress) && (iter < max_iter) ){
    iter++;
    std::cout << "======= Iteration " << iter << " =======" << std::endl;
    for (SIZE curr_subdomain_id = 0;
          curr_subdomain_id < domain_decomposer.num_subdomains();
          curr_subdomain_id++) {

      SIZE next_subdomain_id;
      int next_buffer = (current_buffer + 1) % domain_decomposer.num_subdomains();
      int next_queue = (current_queue + 1) % domain_decomposer.num_subdomains();
      HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
          domain_decomposer.subdomain_shape(curr_subdomain_id));
      log::info("Adapt Refactor to hierarchy");
      reconstructor.Adapt(hierarchy, config, current_queue);
      total_size += hierarchy.total_num_elems() * sizeof(T);
      reconstructor.LoadMetadata(refactored_metadata.metadata[curr_subdomain_id], mdr_data[current_buffer], current_queue);
      reconstructor.Decompress(refactored_metadata.metadata[curr_subdomain_id], mdr_data[current_buffer], current_queue);
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
        eb_Vx = refactored_metadata.metadata[0].corresponding_error;
        eb_Vy = refactored_metadata.metadata[1].corresponding_error;
        eb_Vz = refactored_metadata.metadata[2].corresponding_error;
        // std::cout << "eb_Vx: " << eb_Vx << ", eb_Vy: " << eb_Vy << ", eb_Vz: " << eb_Vz << ", requested QoI error: " << tol << std::endl;
        uint32_t usr_def_requested_size = read_file_tmp();
        for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
          refactored_metadata.metadata[id].requested_size = usr_def_requested_size; //new tolerance
          reconstructor.GenerateRequest(refactored_metadata.metadata[id]);
        }
        // for (auto &metadata : refactored_metadata.metadata) {
        //   metadata.PrintStatus();
        // }
        // size_t size_read = read_mdr(refactored_metadata, refactored_data, "/home/linusli037/Polaris/MGARD/build-cuda-turing/mgard/miniNYX/XYZ", false, config);
        // refactored_metadata.total_size += size_read;*****
        // initiate the bitplane transfer for the 1st variable which
        // should coorespond to the next_buffer
        mdr_data[0].CopyFromRefactoredData(
            refactored_metadata.metadata[0],
            refactored_data.data[0], next_queue);
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

        // DeviceRuntime<DeviceType>::SyncQueue(current_queue);

        // for (int q = 0; q < 2; q++) {
        //   DeviceRuntime<DeviceType>::SyncQueue(q);
        // }        

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

        if (log::level & log::TIME) {
          DeviceRuntime<DeviceType>::SyncQueue(current_queue);
          qoi_timer.start();
        }
        DeviceLauncher<DeviceType>::Execute(
        mgard_x::data_refactoring::multi_dimension::QoIKernel<D, T, DeviceType>(
                                          SubArray(device_subdomain_buffer[0]), 
                                          SubArray(device_subdomain_buffer[1]), 
                                          SubArray(device_subdomain_buffer[2]), 
                                          SubArray(error_out), eb_Vx, eb_Vy, eb_Vz, tol), 
                                        current_queue);
        SubArray<1, double, DeviceType> out_1d({config.domain_decomposition_sizes[0]*config.domain_decomposition_sizes[1]*config.domain_decomposition_sizes[2]}, error_out.data());
        // std::vector<double> out_vec(refactored_metadata.metadata[0].num_elements);
        // std::cout << "num_elements = " << refactored_metadata.metadata[0].num_elements << std::endl;
        // std::cout << "out_vec.data() = " << out_vec.data() << std::endl;
        // MemoryManager<DeviceType>::Copy1D(out_vec.data(), out_1d.data(), refactored_metadata.metadata[0].num_elements,
        //                                   current_queue);
        // std::cout << "max est error = " << *std::max_element(out_vec.begin(), out_vec.end()) << std::endl;
        DeviceCollective<DeviceType>::AbsMax(config.domain_decomposition_sizes[0]*config.domain_decomposition_sizes[1]*config.domain_decomposition_sizes[2], out_1d, SubArray(error_final_out),
                                 workspace, true, current_queue);
        if (log::level || log::TIME) {
          DeviceRuntime<DeviceType>::SyncQueue(current_queue);
          qoi_timer.end();
          qoi_timer.print("QoI error estimation: ", total_size / 3);
          qoi_timer.clear();
        }
        double error_final_out_host;
        MemoryManager<DeviceType>::Copy1D(&error_final_out_host, error_final_out.data(), 1,
                                          current_queue);
        DeviceRuntime<DeviceType>::SyncQueue(current_queue);
        // reconstructed_data.qoi_in_progress = error_final_out_host ? true : false;
        std::cout << "==== maximal est error = " << error_final_out_host << " ====" << std::endl;
        reconstructed_data.qoi_in_progress = (error_final_out_host > tol) ? true : false;
        if(reconstructed_data.qoi_in_progress){
            refactored_metadata.total_size += refactored_metadata.metadata[0].retrieved_size
                                    + refactored_metadata.metadata[1].retrieved_size
                                    + refactored_metadata.metadata[2].retrieved_size;
        }
        // std::cout << "reconstructed_data.qoi_in_progress = " << reconstructed_data.qoi_in_progress << std::endl;   
      }

      DeviceRuntime<DeviceType>::SyncQueue(current_queue);
      
      current_buffer = next_buffer;
      current_queue = next_queue;
    }
  }

  refactored_metadata.metadata[0].corresponding_error = eb_Vx;
  refactored_metadata.metadata[1].corresponding_error = eb_Vy;
  refactored_metadata.metadata[2].corresponding_error = eb_Vz;
  // Copy final data out if we are done with reconstructing
  for (SIZE curr_subdomain_id = 0;
    curr_subdomain_id < domain_decomposer.num_subdomains();
    curr_subdomain_id++) {
    // Update reconstructed data
    domain_decomposer.copy_subdomain(
        device_subdomain_buffer[curr_subdomain_id], curr_subdomain_id,
        subdomain_copy_direction::SubdomainToOriginal, current_queue);
  }

  DeviceRuntime<DeviceType>::SyncDevice();
  if (log::level || log::TIME) {
    timer_series.end();
    timer_series.print("Reconstruct pipeline", total_size);
    timer_series.clear();
  }
  
  std::cout << "Iterations = " << iter << std::endl;
}

} // namespace MDR
} // namespace mgard_x

#endif  // MGARD_X_MDR_RECONSTRUCT_PIPELINE_QOI_HPP