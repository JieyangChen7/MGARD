/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_MDR_HIGH_LEVEL_API_HPP
#define MGARD_X_MDR_HIGH_LEVEL_API_HPP

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeX.h"

#include "mdr_x_lowlevel.hpp"

#include "../DomainDecomposer/DomainDecomposer.hpp"
#include "../Metadata/Metadata.hpp"
#include "MDRHighLevel.h"

#include "../MDR-X/Reconstructor/ReconstructorCache.hpp"
#include "../MDR-X/Refactor/RefactorCache.hpp"

#include "GenerateRequest.hpp"
#include "ReconstructPipeline.hpp"
#include "ReconstructPipelineQoI.hpp"
#include "RefactorPipeline.hpp"

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, typename DeviceType>
SIZE get_max_output_data_size(
    DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>
        &domain_decomposer,
    Config &config) {

  SIZE size = 0;
  for (SIZE subdomain_id = 0; subdomain_id < domain_decomposer.num_subdomains();
       subdomain_id++) {
    Hierarchy<D, T, DeviceType> hierarchy =
        domain_decomposer.subdomain_hierarchy(subdomain_id);
    size += ComposedRefactor<D, T, DeviceType>::MaxOutputDataSize(
        hierarchy.level_shape(hierarchy.l_target()), config);
  }
  return size;
}

template <typename DeviceType>
void load(Config &config, Metadata<DeviceType> &metadata) {
  config.domain_decomposition = metadata.ddtype;
  config.decomposition = metadata.decomposition;
  config.lossless = metadata.ltype;
  config.huff_dict_size = metadata.huff_dict_size;
  config.huff_block_size = metadata.huff_block_size;
  config.reorder = metadata.reorder;
  config.total_num_bitplanes = metadata.number_bitplanes;
}

template <DIM D, typename T, typename DeviceType>
SIZE MDRMaxOutputDataSize(std::vector<SIZE> shape, Config config) {
  DeviceRuntime<DeviceType>::Initialize();
  DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>
      domain_decomposer;
  domain_decomposer =
      DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>(
          shape, config);
  SIZE size = get_max_output_data_size(domain_decomposer, config);
  DeviceRuntime<DeviceType>::Finalize();
  return size;
}

template <DIM D, typename T, typename DeviceType, typename RefactorType>
void MDRefactor(std::vector<SIZE> shape, const void *original_data,
                bool uniform, std::vector<T *> coords,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {
  DeviceRuntime<DeviceType>::Initialize();
  size_t total_num_elem = 1;
  for (int i = 0; i < D; i++)
    total_num_elem *= shape[i];
  config.apply();

  Timer timer_total, timer_each;
  if (log::level & log::TIME)
    timer_total.start();

  using Cache = RefactorCache<D, T, DeviceType, RefactorType>;
  Cache::cache.SafeInitialize();

  DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>
      domain_decomposer;
  if (uniform) {
    domain_decomposer =
        DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>(
            shape, config);
  } else {
    domain_decomposer =
        DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>(
            shape, config, coords);
  }
  domain_decomposer.set_original_data((T *)original_data);

  if (log::level & log::TIME)
    timer_each.start();
  bool input_previously_pinned =
      !MemoryManager<DeviceType>::IsDevicePointer((void *)original_data) &&
      MemoryManager<DeviceType>::CheckHostRegister((void *)original_data);
  if (!input_previously_pinned) {
    MemoryManager<DeviceType>::HostRegister((void *)original_data,
                                            total_num_elem * sizeof(T));
  }

  refactored_metadata.InitializeForRefactor(domain_decomposer.num_subdomains());
  refactored_data.InitializeForRefactor(domain_decomposer, config);

  log::info("Output preallocated: " + std::to_string(output_pre_allocated));
  log::info("Input previously pinned: " +
            std::to_string(input_previously_pinned));

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Prepare input and output buffer");
    timer_each.clear();
  }

  refactor_pipeline(domain_decomposer, config, refactored_metadata,
                    refactored_data);

  if (log::level & log::TIME)
    timer_each.start();
  Metadata<DeviceType> m;
  if (uniform) {
    m.FillForMDR(
        (T)0.0, config.decomposition, config.lossless, config.huff_dict_size,
        config.huff_block_size, shape, domain_decomposer.domain_decomposed(),
        config.domain_decomposition, domain_decomposer.domain_decomposed_dim(),
        domain_decomposer.domain_decomposed_size(), config.total_num_bitplanes);
  } else {
    m.FillForMDR(
        (T)0.0, config.decomposition, config.lossless, config.huff_dict_size,
        config.huff_block_size, shape, domain_decomposer.domain_decomposed(),
        config.domain_decomposition, domain_decomposer.domain_decomposed_dim(),
        domain_decomposer.domain_decomposed_size(), config.total_num_bitplanes,
        coords);
  }

  uint32_t metadata_size;
  Byte *metadata = m.Serialize(metadata_size);
  refactored_metadata.header.resize(metadata_size);
  MemoryManager<DeviceType>::Copy1D(refactored_metadata.header.data(), metadata,
                                    metadata_size);
  MemoryManager<DeviceType>::Free(metadata);

  if (!input_previously_pinned) {
    MemoryManager<DeviceType>::HostUnregister((void *)original_data);
  }

  if (config.auto_cache_release)
    Cache::cache.SafeRelease();
  DeviceRuntime<DeviceType>::Finalize();

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Serialization");
    timer_each.clear();
    timer_total.end();
    timer_total.print("High-level refactoring", total_num_elem * sizeof(T));
    timer_total.clear();
  }
}

template <DIM D, typename T, typename DeviceType>
void MDRefactor(std::vector<SIZE> shape, const void *original_data,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {

  MDRefactor<D, T, DeviceType, ComposedRefactor<D, T, DeviceType>>(
      shape, original_data, true, std::vector<T *>(0), refactored_metadata,
      refactored_data, config, output_pre_allocated);
}

template <DIM D, typename T, typename DeviceType>
void MDRefactor(std::vector<SIZE> shape, const void *original_data,
                std::vector<T *> coords,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {

  MDRefactor<D, T, DeviceType, ComposedRefactor<D, T, DeviceType>>(
      shape, original_data, false, coords, refactored_metadata, refactored_data,
      config, output_pre_allocated);
}

template <DIM D, typename T, typename DeviceType>
void MDRequest(std::vector<SIZE> shape, RefactoredMetadata &refactored_metadata,
               Config config) {
  DeviceRuntime<DeviceType>::Initialize();
  Metadata<DeviceType> m;
  m.Deserialize((SERIALIZED_TYPE *)refactored_metadata.header.data());
  load(config, m);
  DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>
      domain_decomposer;
  domain_decomposer =
      DomainDecomposer<D, T, ComposedRefactor<D, T, DeviceType>, DeviceType>(
          shape, m.domain_decomposed, m.domain_decomposed_dim,
          m.domain_decomposed_size, config);
  generate_request(domain_decomposer, config, refactored_metadata);
  DeviceRuntime<DeviceType>::Finalize();
}

template <DIM D, typename T, typename DeviceType, typename ReconstructorType>
void MDReconstruct(std::vector<SIZE> shape,
                   RefactoredMetadata &refactored_metadata,
                   RefactoredData &refactored_data,
                   ReconstructedData &reconstructed_data, Config config,
                   bool output_pre_allocated) {
  DeviceRuntime<DeviceType>::Initialize();
  config.apply();

  size_t total_num_elem = 1;
  for (int i = 0; i < D; i++)
    total_num_elem *= shape[i];

  Timer timer_total, timer_each;
  if (log::level & log::TIME)
    timer_total.start();

  if (log::level & log::TIME)
    timer_each.start();

  Metadata<DeviceType> m;
  m.Deserialize((SERIALIZED_TYPE *)refactored_metadata.header.data());
  load(config, m);

  std::vector<T *> coords(D);
  if (m.dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    for (DIM d = 0; d < D; d++) {
      coords[d] = new T[shape[d]];
      for (SIZE i = 0; i < shape[d]; i++) {
        coords[d][i] = (float)m.coords[d][i];
      }
    }
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Deserialization");
    timer_each.clear();
  }

  if (log::level & log::TIME)
    timer_each.start();

  using Cache = ReconstructorCache<D, T, DeviceType, ReconstructorType>;

  if (!config.mdr_qoi_mode) {
    Cache::cache.SafeInitialize(3);
  } else if (config.mdr_qoi_mode && !reconstructed_data.qoi_in_progress) {
    Cache::cache.SafeInitialize(config.mdr_qoi_num_variables);
  }

  // Initialize DomainDecomposer
  DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>, DeviceType>
      domain_decomposer;
  if (m.dstype == data_structure_type::Cartesian_Grid_Uniform) {
    domain_decomposer =
        DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>,
                         DeviceType>(shape, m.domain_decomposed,
                                     m.domain_decomposed_dim,
                                     m.domain_decomposed_size, config);
  } else {
    domain_decomposer =
        DomainDecomposer<D, T, ComposedReconstructor<D, T, DeviceType>,
                         DeviceType>(shape, m.domain_decomposed,
                                     m.domain_decomposed_dim,
                                     m.domain_decomposed_size, config, coords);
  }
  if (!config.mdr_adaptive_resolution) {
    // Should not re-allocate if the data is already allocated
    if (!reconstructed_data.IsInitialized()) {
      reconstructed_data.template ResizeToSingleDomain<D, T, DeviceType>(shape);
    }
    domain_decomposer.set_original_data((T *)reconstructed_data.data[0]);
  } else {
    // Should not re-allocate if the data is already allocated
    if (!reconstructed_data.IsInitialized()) {
      reconstructed_data.template ResizeToMultipleSubdomains<D, T, DeviceType>(
          domain_decomposer);
    }
    std::vector<T *> decomposed_original_data(
        domain_decomposer.num_subdomains());
    for (int subdomain_id = 0;
         subdomain_id < domain_decomposer.num_subdomains(); subdomain_id++) {
      decomposed_original_data[subdomain_id] =
          (T *)reconstructed_data.data[subdomain_id];
    }
    domain_decomposer.set_decomposed_original_data(decomposed_original_data);
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Prepare input and output buffer");
    timer_each.clear();
  }

  if (config.mdr_qoi_mode) {
    reconstruct_pipeline_qoi(domain_decomposer, config, refactored_metadata,
                             refactored_data, reconstructed_data);
  } else {
    reconstruct_pipeline(domain_decomposer, config, refactored_metadata,
                         refactored_data, reconstructed_data);
  }

  if (m.dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    for (DIM d = 0; d < D; d++)
      delete[] coords[d];
  }

  if (config.auto_cache_release &&
      (!config.mdr_qoi_mode || !reconstructed_data.qoi_in_progress)) {
    Cache::cache.SafeRelease();
  }
  DeviceRuntime<DeviceType>::Finalize();

  if (log::level & log::TIME) {
    timer_total.end();
    timer_total.print("High-level reconstruction", total_num_elem * sizeof(T));
    timer_total.clear();
  }
}

template <DIM D, typename T, typename DeviceType>
void MDReconstruct(std::vector<SIZE> shape,
                   RefactoredMetadata &refactored_metadata,
                   RefactoredData &refactored_data,
                   ReconstructedData &reconstructed_data, Config config,
                   bool output_pre_allocated) {

  MDReconstruct<D, T, DeviceType, ComposedReconstructor<D, T, DeviceType>>(
      shape, refactored_metadata, refactored_data, reconstructed_data, config,
      output_pre_allocated);
}

template <typename DeviceType>
void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape,
                const void *original_data,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {
  if (dtype == data_type::Float) {
    if (D == 1) {
      MDRefactor<1, float, DeviceType>(shape, original_data,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 2) {
      MDRefactor<2, float, DeviceType>(shape, original_data,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 3) {
      MDRefactor<3, float, DeviceType>(shape, original_data,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 4) {
      MDRefactor<4, float, DeviceType>(shape, original_data,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 5) {
      MDRefactor<5, float, DeviceType>(shape, original_data,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (D == 1) {
      MDRefactor<1, double, DeviceType>(shape, original_data,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 2) {
      MDRefactor<2, double, DeviceType>(shape, original_data,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 3) {
      MDRefactor<3, double, DeviceType>(shape, original_data,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 4) {
      MDRefactor<4, double, DeviceType>(shape, original_data,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 5) {
      MDRefactor<5, double, DeviceType>(shape, original_data,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

template <typename DeviceType>
void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape,
                const void *original_data, std::vector<const Byte *> coords,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {
  if (dtype == data_type::Float) {
    std::vector<float *> float_coords;
    for (auto &coord : coords)
      float_coords.push_back((float *)coord);
    if (D == 1) {
      MDRefactor<1, float, DeviceType>(shape, original_data, float_coords,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 2) {
      MDRefactor<2, float, DeviceType>(shape, original_data, float_coords,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 3) {
      MDRefactor<3, float, DeviceType>(shape, original_data, float_coords,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 4) {
      MDRefactor<4, float, DeviceType>(shape, original_data, float_coords,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else if (D == 5) {
      MDRefactor<5, float, DeviceType>(shape, original_data, float_coords,
                                       refactored_metadata, refactored_data,
                                       config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    std::vector<double *> double_coords;
    for (auto &coord : coords)
      double_coords.push_back((double *)coord);
    if (D == 1) {
      MDRefactor<1, double, DeviceType>(shape, original_data, double_coords,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 2) {
      MDRefactor<2, double, DeviceType>(shape, original_data, double_coords,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 3) {
      MDRefactor<3, double, DeviceType>(shape, original_data, double_coords,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 4) {
      MDRefactor<4, double, DeviceType>(shape, original_data, double_coords,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else if (D == 5) {
      MDRefactor<5, double, DeviceType>(shape, original_data, double_coords,
                                        refactored_metadata, refactored_data,
                                        config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

template <typename DeviceType>
void MDRequest(RefactoredMetadata &refactored_metadata, Config config) {
  Metadata<DeviceType> meta;
  meta.Deserialize((SERIALIZED_TYPE *)refactored_metadata.header.data());

  std::vector<SIZE> shape = std::vector<SIZE>(meta.total_dims);
  for (DIM d = 0; d < shape.size(); d++)
    shape[d] = (SIZE)meta.shape[d];
  data_type dtype = meta.dtype;

  if (dtype == data_type::Float) {
    if (shape.size() == 1) {
      MDRequest<1, float, DeviceType>(shape, refactored_metadata, config);
    } else if (shape.size() == 2) {
      MDRequest<2, float, DeviceType>(shape, refactored_metadata, config);
    } else if (shape.size() == 3) {
      MDRequest<3, float, DeviceType>(shape, refactored_metadata, config);
    } else if (shape.size() == 4) {
      MDRequest<4, float, DeviceType>(shape, refactored_metadata, config);
    } else if (shape.size() == 5) {
      MDRequest<5, float, DeviceType>(shape, refactored_metadata, config);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      MDRequest<1, double, DeviceType>(shape, refactored_metadata, config);
    } else if (shape.size() == 2) {
      MDRequest<2, double, DeviceType>(shape, refactored_metadata, config);
    } else if (shape.size() == 3) {
      MDRequest<3, double, DeviceType>(shape, refactored_metadata, config);
    } else if (shape.size() == 4) {
      MDRequest<4, double, DeviceType>(shape, refactored_metadata, config);
    } else if (shape.size() == 5) {
      MDRequest<5, double, DeviceType>(shape, refactored_metadata, config);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

template <typename DeviceType>
SIZE MDRMaxOutputDataSize(DIM D, data_type dtype, std::vector<SIZE> shape,
                          Config config) {
  if (dtype == data_type::Float) {
    if (shape.size() == 1) {
      return MDRMaxOutputDataSize<1, float, DeviceType>(shape, config);
    } else if (shape.size() == 2) {
      return MDRMaxOutputDataSize<2, float, DeviceType>(shape, config);
    } else if (shape.size() == 3) {
      return MDRMaxOutputDataSize<3, float, DeviceType>(shape, config);
    } else if (shape.size() == 4) {
      return MDRMaxOutputDataSize<4, float, DeviceType>(shape, config);
    } else if (shape.size() == 5) {
      return MDRMaxOutputDataSize<5, float, DeviceType>(shape, config);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      return MDRMaxOutputDataSize<1, double, DeviceType>(shape, config);
    } else if (shape.size() == 2) {
      return MDRMaxOutputDataSize<2, double, DeviceType>(shape, config);
    } else if (shape.size() == 3) {
      return MDRMaxOutputDataSize<3, double, DeviceType>(shape, config);
    } else if (shape.size() == 4) {
      return MDRMaxOutputDataSize<4, double, DeviceType>(shape, config);
    } else if (shape.size() == 5) {
      return MDRMaxOutputDataSize<5, double, DeviceType>(shape, config);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

template <DIM D, typename T, typename DeviceType> void release_cache() {
  using Cache1 =
      RefactorCache<D, T, DeviceType, ComposedRefactor<D, T, DeviceType>>;
  Cache1::cache.SafeRelease();
  using Cache2 = ReconstructorCache<D, T, DeviceType,
                                    ComposedReconstructor<D, T, DeviceType>>;
  Cache2::cache.SafeRelease();
}

template <typename T, typename DeviceType> void release_cache() {
  release_cache<1, T, DeviceType>();
  release_cache<2, T, DeviceType>();
  release_cache<3, T, DeviceType>();
  release_cache<4, T, DeviceType>();
  release_cache<5, T, DeviceType>();
}

template <typename DeviceType> void release_cache() {
  release_cache<float, DeviceType>();
  release_cache<double, DeviceType>();
}

template <typename DeviceType>
void MDReconstruct(RefactoredMetadata &refactored_metadata,
                   RefactoredData &refactored_data,
                   ReconstructedData &reconstructed_data, Config config,
                   bool output_pre_allocated) {

  Metadata<DeviceType> meta;
  meta.Deserialize((SERIALIZED_TYPE *)refactored_metadata.header.data());

  std::vector<SIZE> shape = std::vector<SIZE>(meta.total_dims);
  for (DIM d = 0; d < shape.size(); d++)
    shape[d] = (SIZE)meta.shape[d];
  data_type dtype = meta.dtype;

  if (dtype == data_type::Float) {
    if (shape.size() == 1) {
      MDReconstruct<1, float, DeviceType>(shape, refactored_metadata,
                                          refactored_data, reconstructed_data,
                                          config, output_pre_allocated);
    } else if (shape.size() == 2) {
      MDReconstruct<2, float, DeviceType>(shape, refactored_metadata,
                                          refactored_data, reconstructed_data,
                                          config, output_pre_allocated);
    } else if (shape.size() == 3) {
      MDReconstruct<3, float, DeviceType>(shape, refactored_metadata,
                                          refactored_data, reconstructed_data,
                                          config, output_pre_allocated);
    } else if (shape.size() == 4) {
      MDReconstruct<4, float, DeviceType>(shape, refactored_metadata,
                                          refactored_data, reconstructed_data,
                                          config, output_pre_allocated);
    } else if (shape.size() == 5) {
      MDReconstruct<5, float, DeviceType>(shape, refactored_metadata,
                                          refactored_data, reconstructed_data,
                                          config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      MDReconstruct<1, double, DeviceType>(shape, refactored_metadata,
                                           refactored_data, reconstructed_data,
                                           config, output_pre_allocated);
    } else if (shape.size() == 2) {
      MDReconstruct<2, double, DeviceType>(shape, refactored_metadata,
                                           refactored_data, reconstructed_data,
                                           config, output_pre_allocated);
    } else if (shape.size() == 3) {
      MDReconstruct<3, double, DeviceType>(shape, refactored_metadata,
                                           refactored_data, reconstructed_data,
                                           config, output_pre_allocated);
    } else if (shape.size() == 4) {
      MDReconstruct<4, double, DeviceType>(shape, refactored_metadata,
                                           refactored_data, reconstructed_data,
                                           config, output_pre_allocated);
    } else if (shape.size() == 5) {
      MDReconstruct<5, double, DeviceType>(shape, refactored_metadata,
                                           refactored_data, reconstructed_data,
                                           config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

} // namespace MDR
} // namespace mgard_x

#endif