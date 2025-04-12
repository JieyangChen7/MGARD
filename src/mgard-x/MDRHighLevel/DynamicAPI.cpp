/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "mgard-x/Config/Config.h"
#include "mgard-x/RuntimeX/AutoTuners/AutoTuner.h"
#include "mgard-x/RuntimeX/DataTypes.h"
#include "mgard-x/Utilities/Types.h"

#include "mgard-x/Metadata/Metadata.hpp"

#include "mgard-x/MDRHighLevel/MDRHighLevel.h"

namespace mgard_x {
namespace MDR {

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

enum device_type auto_detect_device() {
  enum device_type dev_type = device_type::NONE;
#if MGARD_ENABLE_SERIAL
  dev_type = device_type::SERIAL;
#endif
#if MGARD_ENABLE_OPENMP
  dev_type = device_type::OPENMP;
#endif
#if MGARD_ENABLE_CUDA
  if (deviceAvailable<CUDA>()) {
    dev_type = device_type::CUDA;
  }
#endif
#if MGARD_ENABLE_HIP
  if (deviceAvailable<HIP>()) {
    dev_type = device_type::HIP;
  }
#endif
#if MGARD_ENABLE_SYCL
  if (deviceAvailable<SYCL>()) {
    dev_type = device_type::SYCL;
  }
#endif
  if (dev_type == device_type::NONE) {
    log::err("MDR-X was not built with any backend.");
    exit(-1);
  }
  return dev_type;
}

void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape,
                const void *original_data,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    MDRefactor<SERIAL>(D, dtype, shape, original_data, refactored_metadata,
                       refactored_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    MDRefactor<OPENMP>(D, dtype, shape, original_data, refactored_metadata,
                       refactored_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    MDRefactor<CUDA>(D, dtype, shape, original_data, refactored_metadata,
                     refactored_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    MDRefactor<HIP>(D, dtype, shape, original_data, refactored_metadata,
                    refactored_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    MDRefactor<SYCL>(D, dtype, shape, original_data, refactored_metadata,
                     refactored_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
    exit(-1);
  }
}

void MDRefactor(DIM D, data_type dtype, std::vector<SIZE> shape,
                const void *original_data, std::vector<const Byte *> coords,
                RefactoredMetadata &refactored_metadata,
                RefactoredData &refactored_data, Config config,
                bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    MDRefactor<SERIAL>(D, dtype, shape, original_data, coords,
                       refactored_metadata, refactored_data, config,
                       output_pre_allocated);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    MDRefactor<OPENMP>(D, dtype, shape, original_data, coords,
                       refactored_metadata, refactored_data, config,
                       output_pre_allocated);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    MDRefactor<CUDA>(D, dtype, shape, original_data, coords,
                     refactored_metadata, refactored_data, config,
                     output_pre_allocated);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    MDRefactor<HIP>(D, dtype, shape, original_data, coords, refactored_metadata,
                    refactored_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    MDRefactor<SYCL>(D, dtype, shape, original_data, coords,
                     refactored_metadata, refactored_data, config,
                     output_pre_allocated);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
    exit(-1);
  }
}

void MDRequest(RefactoredMetadata &refactored_metadata, Config config) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    MDRequest<SERIAL>(refactored_metadata, config);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    MDRequest<OPENMP>(refactored_metadata, config);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    MDRequest<CUDA>(refactored_metadata, config);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    MDRequest<HIP>(refactored_metadata, config);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    MDRequest<SYCL>(refactored_metadata, config);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
    exit(-1);
  }
}

SIZE MDRMaxOutputDataSize(DIM D, data_type dtype, std::vector<SIZE> shape,
                          Config config) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    return MDRMaxOutputDataSize<SERIAL>(D, dtype, shape, config);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    return MDRMaxOutputDataSize<OPENMP>(D, dtype, shape, config);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    return MDRMaxOutputDataSize<CUDA>(D, dtype, shape, config);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    return MDRMaxOutputDataSize<HIP>(D, dtype, shape, config);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    return MDRMaxOutputDataSize<SYCL>(D, dtype, shape, config);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
    exit(-1);
  }
}

void MDReconstruct(RefactoredMetadata &refactored_metadata,
                   RefactoredData &refactored_data,
                   ReconstructedData &reconstructed_data, Config config,
                   bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    MDReconstruct<SERIAL>(refactored_metadata, refactored_data,
                          reconstructed_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    MDReconstruct<OPENMP>(refactored_metadata, refactored_data,
                          reconstructed_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    MDReconstruct<CUDA>(refactored_metadata, refactored_data,
                        reconstructed_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    MDReconstruct<HIP>(refactored_metadata, refactored_data, reconstructed_data,
                       config, output_pre_allocated);
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    MDReconstruct<SYCL>(refactored_metadata, refactored_data,
                        reconstructed_data, config, output_pre_allocated);
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
    exit(-1);
  }
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

void release_cache(Config config) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::AUTO) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::SERIAL) {
#if MGARD_ENABLE_SERIAL
    release_cache<SERIAL>();
#else
    log::err("MDR-X was not built with SERIAL backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::OPENMP) {
#if MGARD_ENABLE_OPENMP
    release_cache<OPENMP>();
#else
    log::err("MDR-X was not built with OPENMP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#if MGARD_ENABLE_CUDA
    release_cache<CUDA>();
#else
    log::err("MDR-X was not built with CUDA backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#if MGARD_ENABLE_HIP
    release_cache<HIP>();
#else
    log::err("MDR-X was not built with HIP backend.");
    exit(-1);
#endif
  } else if (dev_type == device_type::SYCL) {
#if MGARD_ENABLE_SYCL
    elease_cache<SYCL>();
#else
    log::err("MDR-X was not built with SYCL backend.");
    exit(-1);
#endif
  } else {
    log::err("Unsupported backend.");
    exit(-1);
  }
}

} // namespace MDR
} // namespace mgard_x
