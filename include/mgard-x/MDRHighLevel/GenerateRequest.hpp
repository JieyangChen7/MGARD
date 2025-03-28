/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_MDR_GENERATE_PIPELINE_HPP
#define MGARD_X_MDR_GENERATE_PIPELINE_HPP

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, typename DeviceType>
void generate_request(DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>,
                                       DeviceType> &domain_decomposer,
                      Config config, RefactoredMetadata &refactored_metadata) {
  
  for (int subdomain_id = 0; subdomain_id < domain_decomposer.num_subdomains();
       subdomain_id++) {
    Hierarchy<D, T, DeviceType> hierarchy =
        domain_decomposer.subdomain_hierarchy(subdomain_id);
    ComposedReconstructor<D, T, DeviceType> reconstructor(hierarchy, config);
    reconstructor.GenerateRequest(refactored_metadata.metadata[subdomain_id]);
  }
}

}
}
#endif