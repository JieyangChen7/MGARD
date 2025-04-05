/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <chrono>
#include <cstring>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/types.h>

#include "compress_x.hpp"
#include "mdr_x.hpp"
#include "mgard-x/RuntimeX/Utilities/Log.h"
#include "mgard-x/Utilities/ErrorCalculator.h"

#include "ArgumentParser.h"
using namespace std::chrono;

void print_usage_message(std::string error) {
  if (error.compare("") != 0) {
    std::cout << mgard_x::log::log_err << error << std::endl;
  }
  printf("Options\n\
\t -z / --refactor: refactor data\n\
\t\t -i / --input <path to data file to be refactored>\n\
\t\t -o / --output <path to refactored data dir>\n\
\t\t -dt / --data-type <s/single|d/double>: data type (s: single; d:double)\n\
\t\t -dim / --dimension <ndim>: total number of dimensions\n\
\t\t\t [dim1]: slowest dimention\n\
\t\t\t [dim2]: 2nd slowest dimention\n\
\t\t\t  ...\n\
\t\t\t [dimN]: fastest dimention\n\
\t\t -d / --device <auto|serial|cuda|hip>: device type\n\
\t\t (optional) -v / --verbose <0|1|2|3> 0: error; 1: error+info; 2: error+timing; 3: all\n\
\t\t (optional) -m / --max-memory <max memory usage>  \n\
\t\t (optional) -dd / --domain-decomposition <max-dim|block>\n\
\t\t\t (optional) -dd-size / --domain-decomposition-size <integer> (for block domain decomposition only) \n\
\n\
\t -x / --reconstruct: reconstruct data\n\
\t\t -i / --input <path to refactored data dir>\n\
\t\t -o / --output <path to reconstructed data file>\n\
\t\t (optional)  -g / --orginal <path to original data file for error calculation> (optinal)\n\
\t\t -e / --error-bound <float>: error bound\n\
\t\t -me / --multi-error-bounds <num of error bounds> <float> <float>..: multiple error bounds\n\
\t\t -s / --smoothness <float>: smoothness parameter\n\
\t\t -d <auto|serial|cuda|hip>: device type\n\
\t\t (optional) -v / --verbose <0|1|2|3> 0: error; 1: error+info; 2: error+timing; 3: all\n");
  exit(0);
}

template <typename T> void min_max(size_t n, T *in_buff) {
  T min = std::numeric_limits<T>::infinity();
  T max = 0;
  for (size_t i = 0; i < n; i++) {
    if (min > in_buff[i]) {
      min = in_buff[i];
    }
    if (max < in_buff[i]) {
      max = in_buff[i];
    }
  }
  printf("Min: %f, Max: %f\n", min, max);
}

template <typename T> size_t readfile(std::string input_file, T *&in_buff) {
  // std::cout << mgard_x::log::log_info << "Loading file: " << input_file <<
  // "\n";

  FILE *pFile;
  pFile = fopen(input_file.c_str(), "rb");
  if (pFile == NULL) {
    std::cout << mgard_x::log::log_err << "file open error!\n";
    exit(1);
  }
  fseek(pFile, 0, SEEK_END);
  size_t lSize = ftell(pFile);
  rewind(pFile);
  in_buff = (T *)malloc(lSize);
  lSize = fread(in_buff, 1, lSize, pFile);
  fclose(pFile);
  // min_max(lSize/sizeof(T), in_buff);
  return lSize;
}

template <typename T>
size_t readfile_header_metadata(std::string input_file, std::vector<T> &in_buff) {
  // std::cout << mgard_x::log::log_info << "Loading file: " << input_file <<
  // "\n";

  FILE *pFile;
  pFile = fopen(input_file.c_str(), "rb");
  if (pFile == NULL) {
    std::cout << mgard_x::log::log_err << "file open error!\n";
    exit(1);
  }
  fseek(pFile, 0, SEEK_END);
  size_t lSize = ftell(pFile);
  rewind(pFile);
  in_buff.resize(lSize / sizeof(T));
  lSize = fread(in_buff.data(), 1, lSize, pFile);
  fclose(pFile);
  return lSize;
}

template <typename T>
void writefile(std::string output_file, T *out_buff, size_t num_bytes) {
  FILE *file = fopen(output_file.c_str(), "w");
  fwrite(out_buff, 1, num_bytes, file);
  fclose(file);
}

template <typename T>
void print_statistics(double s, enum mgard_x::error_bound_type mode,
                      std::vector<mgard_x::SIZE> shape, T *original_data,
                      T *decompressed_data, T tol, bool normalize_coordinates) {
  mgard_x::SIZE n = 1;
  for (mgard_x::DIM d = 0; d < shape.size(); d++)
    n *= shape[d];
  T actual_error = 0.0;
  std::cout << std::scientific;
  if (s == std::numeric_limits<T>::infinity()) {
    actual_error =
        mgard_x::L_inf_error(n, original_data, decompressed_data, mode);
    if (mode == mgard_x::error_bound_type::ABS) {
      std::cout << mgard_x::log::log_info
                << "Absoluate L_inf error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    } else if (mode == mgard_x::error_bound_type::REL) {
      std::cout << mgard_x::log::log_info
                << "Relative L_inf error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    }
  } else {
    actual_error = mgard_x::L_2_error(shape, original_data, decompressed_data,
                                      mode, normalize_coordinates);
    if (mode == mgard_x::error_bound_type::ABS) {
      std::cout << mgard_x::log::log_info
                << "Absoluate L_2 error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    } else if (mode == mgard_x::error_bound_type::REL) {
      std::cout << mgard_x::log::log_info
                << "Relative L_2 error: " << actual_error << " ("
                << (actual_error < tol ? "\e[32mSatisified\e[0m"
                                       : "\e[31mNot Satisified\e[0m")
                << ")"
                << "\n";
    }
  }

  std::cout << mgard_x::log::log_info
            << "MSE: " << mgard_x::MSE(n, original_data, decompressed_data)
            << "\n";
  std::cout << std::defaultfloat;
  std::cout << mgard_x::log::log_info
            << "PSNR: " << mgard_x::PSNR(n, original_data, decompressed_data)
            << "\n";

  // if (actual_error > tol)
    // exit(-1);
}

void create_dir(std::string name) {
  struct stat st = {0};
  if (stat(name.c_str(), &st) == -1) {
    mkdir(name.c_str(), 0700);
  }
}

void write_mdr(mgard_x::MDR::RefactoredMetadata &refactored_metadata,
               mgard_x::MDR::RefactoredData &refactored_data,
               std::string output) {
  size_t size_written = 0;
  create_dir(output);
  std::vector<mgard_x::Byte> serialized_metadata =
      refactored_metadata.Serialize();
  writefile(output + "/header", refactored_metadata.header.data(),
            refactored_metadata.header.size());
  writefile(output + "/metadata", serialized_metadata.data(),
            serialized_metadata.size());
  for (int subdomain_id = 0; subdomain_id < refactored_metadata.metadata.size();
       subdomain_id++) {
    for (int level_idx = 0;
         level_idx <
         refactored_metadata.metadata[subdomain_id].level_sizes.size();
         level_idx++) {
      for (int bitplane_idx = 0;
           bitplane_idx < refactored_metadata.metadata[subdomain_id]
                              .level_sizes[level_idx]
                              .size();
           bitplane_idx++) {
        std::string filename = "component_" + std::to_string(subdomain_id) +
                               "_" + std::to_string(level_idx) + "_" +
                               std::to_string(bitplane_idx);
        writefile(output + "/" + filename,
                  refactored_data.data[subdomain_id][level_idx][bitplane_idx],
                  refactored_metadata.metadata[subdomain_id]
                      .level_sizes[level_idx][bitplane_idx]);
        size_written += refactored_metadata.metadata[subdomain_id]
                            .level_sizes[level_idx][bitplane_idx];
      }
    }
  }
  std::cout << mgard_x::log::log_info << size_written << " bytes written\n";
}

size_t read_mdr_metadata(mgard_x::MDR::RefactoredMetadata &refactored_metadata,
                       mgard_x::MDR::RefactoredData &refactored_data,
                       std::string input) {
  
  size_t metadata_size = 0;
  metadata_size += readfile_header_metadata(input + "/header", refactored_metadata.header);
  std::vector<mgard_x::Byte> serialized_metadata;
  metadata_size += readfile_header_metadata(input + "/metadata", serialized_metadata);
  refactored_metadata.Deserialize(serialized_metadata);
  refactored_metadata.InitializeForReconstruction();
  refactored_data.InitializeForReconstruction(refactored_metadata);
  return metadata_size;
}

// size_t read_mdr(mgard_x::MDR::RefactoredMetadata &refactored_metadata,
//               mgard_x::MDR::RefactoredData &refactored_data, std::string input,
//               bool initialize_signs, mgard_x::Config config) {

//   size_t size_read = 0;
//   int num_subdomains = refactored_metadata.metadata.size();
//   for (int subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
//     mgard_x::MDR::MDRMetadata metadata =
//         refactored_metadata.metadata[subdomain_id];
//     int num_levels = metadata.level_sizes.size();
//     for (int level_idx = 0; level_idx < num_levels; level_idx++) {
//       int num_bitplanes = metadata.level_sizes[level_idx].size();
//       int loaded_bitplanes = metadata.loaded_level_num_bitplanes[level_idx];
//       int reqested_bitplanes =
//           metadata.requested_level_num_bitplanes[level_idx];
//       for (int bitplane_idx = loaded_bitplanes;
//            bitplane_idx < reqested_bitplanes; bitplane_idx++) {
//         std::string filename = "component_" + std::to_string(subdomain_id) +
//                                "_" + std::to_string(level_idx) + "_" +
//                                std::to_string(bitplane_idx);
//         mgard_x::SIZE level_size = readfile(
//             input + "/" + filename,
//             refactored_data.data[subdomain_id][level_idx][bitplane_idx]);
//         mgard_x::pin_memory(
//             refactored_data.data[subdomain_id][level_idx][bitplane_idx],
//             level_size, config);
//         if (level_size != refactored_metadata.metadata[subdomain_id]
//                               .level_sizes[level_idx][bitplane_idx]) {
//           std::cout << "mdr component size mismatch.";
//           exit(-1);
//         }
//         size_read += level_size;
//       }
//       if (initialize_signs) {
//         // level sign
//         refactored_data.level_signs[subdomain_id][level_idx] =
//             (bool *)malloc(sizeof(bool) * metadata.level_num_elems[level_idx]);
//         memset(refactored_data.level_signs[subdomain_id][level_idx], 0,
//                sizeof(bool) * metadata.level_num_elems[level_idx]);
//         mgard_x::pin_memory(
//             refactored_data.level_signs[subdomain_id][level_idx],
//             sizeof(bool) * metadata.level_num_elems[level_idx], config);
//       }
//     }
//   }
//   return size_read;
// }

size_t read_mdr(mgard_x::MDR::RefactoredMetadata &refactored_metadata,
              mgard_x::MDR::RefactoredData &refactored_data, std::string input,
              bool initialize_signs, mgard_x::Config config) {

  size_t size_read = 0;
  int num_subdomains = refactored_metadata.metadata.size();
  for (int subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
    mgard_x::MDR::MDRMetadata metadata =
        refactored_metadata.metadata[subdomain_id];
    int num_levels = metadata.level_sizes.size();
    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      int num_bitplanes = metadata.level_sizes[level_idx].size();
      for (int bitplane_idx = 0;
           bitplane_idx < num_bitplanes; bitplane_idx++) {
        std::string filename = "component_" + std::to_string(subdomain_id) +
                               "_" + std::to_string(level_idx) + "_" +
                               std::to_string(bitplane_idx);
        mgard_x::SIZE level_size = readfile(
            input + "/" + filename,
            refactored_data.data[subdomain_id][level_idx][bitplane_idx]);
        mgard_x::pin_memory(
            refactored_data.data[subdomain_id][level_idx][bitplane_idx],
            level_size, config);
        if (level_size != refactored_metadata.metadata[subdomain_id]
                              .level_sizes[level_idx][bitplane_idx]) {
          std::cout << "mdr component size mismatch.";
          exit(-1);
        }
        size_read += level_size;
      }
      if (initialize_signs) {
        // level sign
        refactored_data.level_signs[subdomain_id][level_idx] =
            (bool *)malloc(sizeof(bool) * metadata.level_num_elems[level_idx]);
        memset(refactored_data.level_signs[subdomain_id][level_idx], 0,
               sizeof(bool) * metadata.level_num_elems[level_idx]);
        mgard_x::pin_memory(
            refactored_data.level_signs[subdomain_id][level_idx],
            sizeof(bool) * metadata.level_num_elems[level_idx], config);
      }
    }
  }
  return size_read;
}

int verbose_to_log_level(int verbose) {
  if (verbose == 0) {
    return mgard_x::log::ERR;
  } else if (verbose == 1) {
    return mgard_x::log::ERR | mgard_x::log::INFO;
  } else if (verbose == 2) {
    return mgard_x::log::ERR | mgard_x::log::TIME;
  } else if (verbose == 3) {
    return mgard_x::log::ERR | mgard_x::log::INFO | mgard_x::log::TIME;
  }
}

template <typename T>
int launch_refactor(mgard_x::DIM D, enum mgard_x::data_type dtype,
                    std::string input_file, std::string output_file,
                    std::vector<mgard_x::SIZE> shape,
                    std::string domain_decomposition, mgard_x::SIZE block_size,
                    enum mgard_x::device_type dev_type, int verbose,
                    mgard_x::SIZE max_memory_footprint) {

  mgard_x::Config config;
  config.normalize_coordinates = false;
  config.log_level = verbose_to_log_level(verbose);
  config.decomposition = mgard_x::decomposition_type::MultiDim;
  if (domain_decomposition == "max-dim") {
    config.domain_decomposition = mgard_x::domain_decomposition_type::MaxDim;
  } else if (domain_decomposition == "block") {
    config.domain_decomposition = mgard_x::domain_decomposition_type::Block;
    config.block_size = block_size;
  } else if (domain_decomposition == "variable") {
    config.domain_decomposition = mgard_x::domain_decomposition_type::Variable;
  }

  config.domain_decomposition = mgard_x::domain_decomposition_type::Variable;
  config.domain_decomposition_dim = 0;
  config.domain_decomposition_sizes = {256, 256, 256};

  config.dev_type = dev_type;
  config.max_memory_footprint = max_memory_footprint;
  if (dtype == mgard_x::data_type::Float) {
    config.total_num_bitplanes = 32;
  } else if (dtype == mgard_x::data_type::Double) {
    config.total_num_bitplanes = 64;
  }

  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  T *original_data = (T *)malloc(original_size * sizeof(T));
  size_t in_size = 0;
  if (std::string(input_file).compare("random") == 0) {
    in_size = original_size * sizeof(T);
    srand(7117);
    T c = 0;
    for (size_t i = 0; i < original_size; i++) {
      original_data[i] = rand() % 10 + 1;
    }
  } else {
    T *file_data;
    in_size = readfile(input_file, file_data);

    size_t loaded_size = 0;
    while (loaded_size < original_size) {
      std::memcpy(original_data + loaded_size, file_data,
                  std::min(in_size / sizeof(T), original_size - loaded_size) *
                      sizeof(T));
      loaded_size += std::min(in_size / sizeof(T), original_size - loaded_size);
    }
    in_size = loaded_size * sizeof(T);
  }
  if (in_size != original_size * sizeof(T)) {
    std::cout << mgard_x::log::log_warn << "input file size mismatch "
              << in_size << " vs. " << original_size * sizeof(T) << "!\n";
  }

  std::cout << mgard_x::log::log_info << "Max output data size: "
            << mgard_x::MDR::MDRMaxOutputDataSize(D, dtype, shape, config)
            << " bytes\n";

  mgard_x::MDR::RefactoredMetadata refactored_metadata;
  mgard_x::MDR::RefactoredData refactored_data;
  mgard_x::pin_memory(original_data, original_size * sizeof(T), config);

  mgard_x::MDR::MDRefactor(D, dtype, shape, original_data, refactored_metadata,
                           refactored_data, config, false);

  write_mdr(refactored_metadata, refactored_data, output_file);

  mgard_x::unpin_memory(original_data, config);
  delete[](T *) original_data;

  return 0;
}

template <class T>
T compute_max_abs_error(const T *vec_ori, const T * vec_rec, size_t n){
  T error = fabs(vec_ori[0] - vec_rec[0]);
	T max = error;
	for(int i=1; i<n; i++){
    error = fabs(vec_ori[i] - vec_rec[i]);
		if(max < error) max = error;
	}
	return max;
}

template <class T>
T compute_value_range(const T * vec, size_t n){
	T min = vec[0];
	T max = vec[0];
	for(int i=0; i<n; i++){
		if(vec[i] < min) min = vec[i];
		if(vec[i] > max) max = vec[i];
	}
	return max - min;
}

template <class T>
void compute_VTOT(const T * Vx, const T * Vy, const T * Vz, size_t n, T * V_TOT_){
	for(int i=0; i<n; i++){
		double V_TOT_2 = Vx[i]*Vx[i] + Vy[i]*Vy[i] + Vz[i]*Vz[i];
		double V_TOT = sqrt(V_TOT_2);
		V_TOT_[i] = V_TOT;
	}
}

int launch_reconstruct(std::string input_file, std::string output_file,
                       std::string original_file, enum mgard_x::data_type dtype,
                       std::vector<mgard_x::SIZE> shape,
                       std::vector<double> tols, double s,
                       enum mgard_x::error_bound_type mode,
                       bool adaptive_resolution,
                       enum mgard_x::device_type dev_type, int verbose) {

  double bitrate = 0;
  mgard_x::Config config;
  config.normalize_coordinates = false;
  config.log_level = verbose_to_log_level(verbose);
  config.dev_type = dev_type;
  config.mdr_adaptive_resolution = adaptive_resolution;

  config.mdr_qoi_mode = true;
  config.mdr_qoi_num_variables = 3;
  config.domain_decomposition = mgard_x::domain_decomposition_type::Variable;
  config.domain_decomposition_dim = 0;
  config.domain_decomposition_sizes = {256, 256, 256};

  mgard_x::Byte *original_data;
  size_t in_size = 0;
  size_t original_size = 1;
  for (mgard_x::DIM i = 0; i < shape.size(); i++)
    original_size *= shape[i];
  if (original_file.compare("none") != 0 && !config.mdr_adaptive_resolution) {
    if (original_file.compare("random") == 0) {
      if (dtype == mgard_x::data_type::Float) {
        in_size = original_size * sizeof(float);
        original_data = (mgard_x::Byte *)new float[original_size];
        srand(7117);
        for (size_t i = 0; i < original_size; i++) {
          ((float *)original_data)[i] = rand() % 10 + 1;
        }
      } else if (dtype == mgard_x::data_type::Double) {
        in_size = original_size * sizeof(double);
        original_data = (mgard_x::Byte *)new double[original_size];
        srand(7117);
        for (size_t i = 0; i < original_size; i++) {
          ((double *)original_data)[i] = rand() % 10 + 1;
        }
      }
    } else {
      mgard_x::Byte *file_data;
      in_size = readfile<mgard_x::Byte>(original_file, file_data);

      if (dtype == mgard_x::data_type::Float) {
        original_size *= sizeof(float);
      } else if (dtype == mgard_x::data_type::Double) {
        original_size *= sizeof(double);
      }

      original_data = (mgard_x::Byte *)malloc(original_size);

      size_t loaded_size = 0;
      while (loaded_size < original_size) {

        std::memcpy(original_data + loaded_size, file_data,
                    std::min(in_size, original_size - loaded_size));
        loaded_size += std::min(in_size, original_size - loaded_size);
      }
      in_size = loaded_size;
    }
  }
  mgard_x::Byte * V_TOT_ori;
  size_t num_elements;
  double tau = 0;
  V_TOT_ori = (mgard_x::Byte *)malloc(in_size / config.mdr_qoi_num_variables);
  mgard_x::Byte* org_Vx_ptr = original_data + original_size/3 * 0;
  mgard_x::Byte* org_Vy_ptr = original_data + original_size/3 * 1;
  mgard_x::Byte* org_Vz_ptr = original_data + original_size/3 * 2;
  if (dtype == mgard_x::data_type::Float){
    num_elements = (in_size / config.mdr_qoi_num_variables) / sizeof(float);
    compute_VTOT<float>((float *) org_Vx_ptr, (float *) org_Vy_ptr, (float *) org_Vz_ptr, num_elements, (float *) V_TOT_ori);
    tau = compute_value_range((float *) V_TOT_ori, num_elements) * tols[0];
  } else if (dtype == mgard_x::data_type::Double){
    num_elements = (in_size / config.mdr_qoi_num_variables) / sizeof(double);
    compute_VTOT<double>((double *) org_Vx_ptr, (double *) org_Vy_ptr, (double *) org_Vz_ptr, num_elements, (double *) V_TOT_ori);
    tau = compute_value_range((double *) V_TOT_ori, num_elements) * tols[0];
  }

  mgard_x::MDR::RefactoredMetadata refactored_metadata;
  mgard_x::MDR::RefactoredData refactored_data;
  mgard_x::MDR::ReconstructedData reconstructed_data;
  size_t metadata_size = read_mdr_metadata(refactored_metadata, refactored_data, input_file);
  refactored_metadata.total_size += metadata_size;

  for (int i = 0; i < config.mdr_qoi_num_variables; i++) {
    refactored_metadata.metadata[i].num_elements = num_elements;
    refactored_metadata.metadata[i].requested_tol = tau;
    refactored_metadata.metadata[i].requested_size = 10000000;
    refactored_metadata.metadata[i].requested_s = s;
    refactored_metadata.metadata[i].segmented = true;
  }
  mgard_x::MDR::MDRequest(refactored_metadata, config);
  refactored_metadata.total_size += refactored_metadata.metadata[0].retrieved_size
                                    + refactored_metadata.metadata[1].retrieved_size
                                    + refactored_metadata.metadata[2].retrieved_size;
  // for (auto &metadata : refactored_metadata.metadata) {
  //   metadata.PrintStatus();
  // }
  size_t size_read = read_mdr(refactored_metadata, refactored_data, input_file,
            true, config);
  // refactored_metadata.total_size += size_read;

  mgard_x::MDR::MDReconstruct(refactored_metadata, refactored_data,
                              reconstructed_data, config, false);

  // we can check reconstructed_data.qoi_in_progress here

  std::cout << mgard_x::log::log_info << "Additional " << size_read
            << " bytes read for reconstruction\n";

  std::vector<mgard_x::Byte*> rec_var_ptrs;
  if (original_file.compare("none") != 0 && !config.mdr_adaptive_resolution) {
    for (int i = 0; i < config.mdr_qoi_num_variables; i++) {
      std::vector<mgard_x::SIZE> var_shape = shape;
      var_shape[0] /= config.mdr_qoi_num_variables;
      mgard_x::Byte* org_var_ptr = original_data + original_size/3 * i;
      mgard_x::Byte* rec_var_ptr = reconstructed_data.data[0] + original_size/3 * i;
      rec_var_ptrs.push_back(rec_var_ptr);
      if (dtype == mgard_x::data_type::Float) {
        print_statistics<float>(s, mode, var_shape, (float *)org_var_ptr,
                                (float *)rec_var_ptr, refactored_metadata.metadata[i].corresponding_error,
                                config.normalize_coordinates);
      } else if (dtype == mgard_x::data_type::Double) {
        print_statistics<double>(s, mode, var_shape, (double *)org_var_ptr,
                                (double *)rec_var_ptr, refactored_metadata.metadata[i].corresponding_error,
                                config.normalize_coordinates);
      }
    }
  }
  mgard_x::Byte* V_TOT_rec;
  V_TOT_rec = (mgard_x::Byte *)malloc(in_size / config.mdr_qoi_num_variables);
  if (dtype == mgard_x::data_type::Float){
    compute_VTOT<float>((float *) rec_var_ptrs[0], (float *) rec_var_ptrs[1], (float *) rec_var_ptrs[2], num_elements, (float *) V_TOT_rec);
  } else if (dtype == mgard_x::data_type::Double){
    compute_VTOT<double>((double *) rec_var_ptrs[0], (double *) rec_var_ptrs[1], (double *) rec_var_ptrs[2], num_elements, (double *) V_TOT_rec);
  }
  std::vector<mgard_x::SIZE> var_shape = shape;
  var_shape[0] /= config.mdr_qoi_num_variables;
  if (dtype == mgard_x::data_type::Float) {
    print_statistics<float>(s, mode, var_shape, (float *) V_TOT_ori,
                            (float *) V_TOT_rec, tau,
                            config.normalize_coordinates);
    bitrate = 32 / ((double) in_size / refactored_metadata.total_size);
  } else if (dtype == mgard_x::data_type::Double) {
    print_statistics<double>(s, mode, var_shape, (double *) V_TOT_ori,
                            (double *) V_TOT_rec, tau,
                            config.normalize_coordinates);
    bitrate = 64 / ((double) in_size / refactored_metadata.total_size);
  }
  // std::cout << "refactored_metadata.total_size = " << refactored_metadata.total_size << std::endl;
  // std::cout << "in_size = " << in_size << std::endl;
  std::cout << "Bitrate = " << bitrate << std::endl;
  // std::cout << "Original Vx[35345] = " << ((float*) org_Vx_ptr)[35345] << ", Reconstructed Vx[35345] = " << ((float*) rec_var_ptrs[0])[35345] << std::endl;
  std::cout << "Requested Tau = " << tau << std::endl;
  std::cout << "Real max error = " << compute_max_abs_error((float*) V_TOT_ori, (float*)V_TOT_rec, num_elements) << std::endl;
  return 0;
}

bool try_refactoring(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-z", "--refactor"))
    return false;
  mgard_x::log::info("Mode: refactor", true);

  std::string input_file =
      get_arg<std::string>(argc, argv, "Original data", "-i", "--input");
  std::string output_file =
      get_arg<std::string>(argc, argv, "Refactored data", "-o", "--output");
  enum mgard_x::data_type dtype = get_data_type(argc, argv);
  std::vector<mgard_x::SIZE> shape =
      get_args<mgard_x::SIZE>(argc, argv, "Dimensions", "-dim", "--dimension");
  // std::string lossless_level = get_arg<std::string>(argc, argv, "Lossless",
  // "-l", "--lossless");
  enum mgard_x::device_type dev_type = get_device_type(argc, argv);
  int verbose = 0;
  if (has_arg(argc, argv, "-v", "--verbose")) {
    verbose = get_arg<int>(argc, argv, "Verbose", "-v", "--verbose");
  }
  mgard_x::SIZE max_memory_footprint =
      std::numeric_limits<mgard_x::SIZE>::max();
  if (has_arg(argc, argv, "-m", "--max-memory")) {
    max_memory_footprint = (mgard_x::SIZE)get_arg<double>(
        argc, argv, "Max memory", "-m", "--max-memory");
  }
  std::string domain_decomposition = "max-dim";
  mgard_x::SIZE block_size = 0;
  if (has_arg(argc, argv, "-dd", "--domain-decomposition")) {
    domain_decomposition = get_arg<std::string>(
        argc, argv, "Domain decomposition", "-dd", "--domain-decomposition");
    if (domain_decomposition == "block") {
      block_size = get_arg<mgard_x::SIZE>(argc, argv, "Block size", "-dd-size",
                                          "--domain-decomposition-size");
    }
  }
  if (dtype == mgard_x::data_type::Double) {
    launch_refactor<double>(shape.size(), dtype, input_file.c_str(),
                            output_file.c_str(), shape, domain_decomposition,
                            block_size, dev_type, verbose,
                            max_memory_footprint);
  } else if (dtype == mgard_x::data_type::Float) {
    launch_refactor<float>(shape.size(), dtype, input_file.c_str(),
                           output_file.c_str(), shape, domain_decomposition,
                           block_size, dev_type, verbose, max_memory_footprint);
  }
  return true;
}

bool try_reconstruction(int argc, char *argv[]) {
  if (!has_arg(argc, argv, "-x", "--reconstruct"))
    return false;
  mgard_x::log::info("mode: reconstruct", true);
  std::string input_file =
      get_arg<std::string>(argc, argv, "Refactored data", "-i", "--input");
  std::string output_file =
      get_arg<std::string>(argc, argv, "Reconstructed data", "-o", "--output");
  // default is none (means original data not provided)
  std::string original_file = "none";
  enum mgard_x::data_type dtype;
  std::vector<mgard_x::SIZE> shape;
  if (has_arg(argc, argv, "-g", "--orignal")) {
    original_file =
        get_arg<std::string>(argc, argv, "Original data", "-g", "--orignal");
    dtype = get_data_type(argc, argv);
    shape = get_args<mgard_x::SIZE>(argc, argv, "Dimensions", "-dim",
                                    "--dimension");
  }
  // only abs mode is supported now
  enum mgard_x::error_bound_type mode =
      mgard_x::error_bound_type::ABS; // REL or ABS

  std::vector<double> tols;
  if (has_arg(argc, argv, "-e", "--error-bound")) {
    tols.push_back(
        get_arg<double>(argc, argv, "Error bound", "-e", "--error-bound"));
  } else if (has_arg(argc, argv, "-me", "--multi-error-bounds")) {
    tols = get_args<double>(argc, argv, "Multi error bounds", "-me",
                            "--multi-error-bounds");
  } else {
    throw std::runtime_error(
        "Missing option -e/--error-bound or -me/--multi-error-bounds");
  }
  double s = get_arg<double>(argc, argv, "Smoothness", "-s", "--smoothness");
  enum mgard_x::device_type dev_type = get_device_type(argc, argv);
  int verbose = 0;
  if (has_arg(argc, argv, "-v", "--verbose")) {
    verbose = get_arg<int>(argc, argv, "Verbose", "-v", "--verbose");
  }
  bool adaptive_resolution = false;
  if (has_arg(argc, argv, "-ar", "--adaptive-resolution")) {
    adaptive_resolution = get_arg<int>(argc, argv, "Adaptive resolution", "-ar",
                                       "--adaptive-resolution");
  }
  if (verbose)
    std::cout << mgard_x::log::log_info << "verbose: enabled.\n";
  launch_reconstruct(input_file, output_file, original_file, dtype, shape, tols,
                     s, mode, adaptive_resolution, dev_type, verbose);
  return true;
}

int main(int argc, char *argv[]) {

  if (!try_refactoring(argc, argv) && !try_reconstruction(argc, argv)) {
    print_usage_message("");
  }
  return 0;
}