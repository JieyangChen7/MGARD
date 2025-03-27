/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_MDR_HIGH_LEVEL_DATA_HPP
#define MGARD_X_MDR_HIGH_LEVEL_DATA_HPP

#include "../DomainDecomposer/DomainDecomposer.hpp"

#include "../RuntimeX/DataStructures/MDRMetadata.hpp"

namespace mgard_x {
namespace MDR {

class RefactoredMetadata {
public:
  void InitializeForRefactor(SIZE num_subdomains) {
    this->num_subdomains = num_subdomains;
    metadata.resize(num_subdomains);
  }

  void InitializeForReconstruction() {
    for (SIZE subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
      metadata[subdomain_id].InitializeForReconstruction();
    }
  }

  std::vector<Byte> header;
  std::vector<MDRMetadata> metadata;
  SIZE num_subdomains;

  template <typename T> void Serialize(Byte *&ptr, T *data, SIZE bytes) {
    memcpy(ptr, (Byte *)data, bytes);
    ptr += bytes;
  }

  template <typename T> void Deserialize(Byte *&ptr, T *data, SIZE bytes) {
    memcpy((Byte *)data, ptr, bytes);
    ptr += bytes;
  }

  std::vector<Byte> Serialize() {
    SIZE num_subdomains = metadata.size();
    size_t metadata_size = 0;
    metadata_size += sizeof(SIZE);
    for (SIZE subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
      metadata_size += sizeof(SIZE);
      metadata_size += metadata[subdomain_id].MetadataSize();
    }
    std::vector<Byte> serialize_metadata(metadata_size);
    Byte *ptr = serialize_metadata.data();
    Serialize(ptr, &num_subdomains, sizeof(SIZE));
    for (SIZE subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
      std::vector<Byte> serialized_MDRMetadata =
          metadata[subdomain_id].Serialize();
      SIZE serialized_MDRMetadata_size = serialized_MDRMetadata.size();
      Serialize(ptr, &serialized_MDRMetadata_size, sizeof(SIZE));
      Serialize(ptr, serialized_MDRMetadata.data(),
                serialized_MDRMetadata.size());
    }
    return serialize_metadata;
  }

  void Deserialize(std::vector<Byte> serialize_metadata) {
    Byte *ptr = serialize_metadata.data();
    Deserialize(ptr, &num_subdomains, sizeof(SIZE));
    this->num_subdomains = num_subdomains;
    metadata.resize(num_subdomains);
    for (SIZE subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
      SIZE serialized_MDRMetadata_size;
      Deserialize(ptr, &serialized_MDRMetadata_size, sizeof(SIZE));
      std::vector<Byte> serialized_MDRMetadata(serialized_MDRMetadata_size);
      Deserialize(ptr, serialized_MDRMetadata.data(),
                  serialized_MDRMetadata.size());
      metadata[subdomain_id].Deserialize(serialized_MDRMetadata);
    }
  }
};

class RefactoredData {
public:
  template <DIM D, typename T, typename DeviceType, typename RefactorType>
  void InitializeForRefactor(DomainDecomposer<D, T, RefactorType, DeviceType> &domain_decomposer, Config config) {
    num_subdomains = domain_decomposer.num_subdomains();
    data.resize(num_subdomains);
    data_allocation_size.resize(num_subdomains);
    for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
      Hierarchy<D, T, DeviceType> hierarchy(domain_decomposer.subdomain_shape(id), config);
      std::vector<std::vector<SIZE>> estimation = RefactorType::output_size_estimation(hierarchy);
      SIZE num_levels = estimation.size();
      SIZE num_bitplanes = estimation[0].size();
      data[id].resize(num_levels);
      data_allocation_size[id].resize(num_levels);
      for (int level_idx = 0; level_idx < num_levels; level_idx++) {
        data[id][level_idx].resize(num_bitplanes);
        data_allocation_size[id][level_idx].resize(num_bitplanes);
        for (int bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {
          MemoryManager<DeviceType>::MallocHost(data[id][level_idx][bitplane_idx],
            estimation[level_idx][bitplane_idx], 0);
          data_allocation_size[id][level_idx][bitplane_idx] =
            estimation[level_idx][bitplane_idx];
        }
      }
    }
  }
  void InitializeForReconstruction(RefactoredMetadata &refactored_metadata) {
    int num_subdomains = refactored_metadata.metadata.size();
    this->num_subdomains = num_subdomains;
    data.resize(num_subdomains);
    level_signs.resize(num_subdomains);
    for (int subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
      MDRMetadata metadata = refactored_metadata.metadata[subdomain_id];
      int num_levels = metadata.level_sizes.size();
      data[subdomain_id].resize(num_levels);
      level_signs[subdomain_id].resize(num_levels);
      for (int level_idx = 0; level_idx < num_levels; level_idx++) {
        int num_bitplanes = metadata.level_sizes[level_idx].size();
        data[subdomain_id][level_idx].resize(num_bitplanes);
      }
    }
  }

  std::vector<std::vector<std::vector<Byte *>>> data;
  std::vector<std::vector<std::vector<SIZE>>> data_allocation_size;
  std::vector<std::vector<bool *>> level_signs;
  SIZE num_subdomains;
};

class ReconstructedData {
public:
  void Initialize(SIZE num_subdomains) {
    this->num_subdomains = num_subdomains;
    offset.resize(num_subdomains);
    shape.resize(num_subdomains);
    data.resize(num_subdomains);
    initialized = true;
  }

  template <DIM D, typename T, typename DeviceType>
  void ResizeToSingleDomain(std::vector<SIZE> domain_shape) {
    // First time reconstruction
    Initialize(1);
    SIZE total_num_elem = 1;
    for (int i = 0; i < D; i++)
      total_num_elem *= domain_shape[i];
    MemoryManager<DeviceType>::MallocHost(
        data[0], total_num_elem * sizeof(T), 0);
    // Is memset necessary?
    memset(data[0], 0, total_num_elem * sizeof(T));
    offset[0] = std::vector<SIZE>(D, 0);
    shape[0] = domain_shape;
  }

  template <DIM D, typename T, typename DeviceType, typename DomainDecomposerType>
  void ResizeToMultipleSubdomains(DomainDecomposerType &domain_decomposer) {
    SIZE num_subdomains = domain_decomposer.num_subdomains();
    Initialize(num_subdomains);
    for (SIZE subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
      SIZE total_num_elem = 1;
      for (int i = 0; i < domain_decomposer.subdomain_shape(subdomain_id).size(); i++)
        total_num_elem *= domain_decomposer.subdomain_shape(subdomain_id)[i];
      MemoryManager<DeviceType>::MallocHost(
          data[subdomain_id], total_num_elem * sizeof(T), 0);
      // Is memset necessary?
      memset(data[subdomain_id], 0, total_num_elem * sizeof(T));
    }
  }
  bool IsInitialized() { return initialized; }
  std::vector<std::vector<SIZE>> offset;
  std::vector<std::vector<SIZE>> shape;
  std::vector<Byte *> data;
  SIZE num_subdomains;
  bool initialized = false;
};

} // namespace MDR
} // namespace mgard_x

#endif