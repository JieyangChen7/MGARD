/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_HUFFMAN_TEMPLATE_HPP
#define MGARD_X_HUFFMAN_TEMPLATE_HPP

static bool debug_print_huffman = false;

#include "../LosslessCompressorInterface.hpp"

#include "../../RuntimeX/Utilities/Serializer.hpp"
#include "Condense.hpp"
#include "Decode.hpp"
#include "Deflate.hpp"
#include "DictionaryShift.hpp"
#include "EncodeFixedLen.hpp"
#include "GetCodebook.hpp"
#include "Histogram.hpp"
#include "HuffmanWorkspace.hpp"
#include "OutlierSeparator.hpp"

#include <chrono>
using namespace std::chrono;

namespace mgard_x {

template <typename Q, typename S, typename H, typename DeviceType>
class Huffman {
public:
  Huffman() : initialized(false) {}

  Huffman(SIZE max_size, int dict_size, int chunk_size,
          double estimated_outlier_ratio)
      : initialized(true), max_size(max_size), dict_size(dict_size),
        chunk_size(chunk_size) {
    workspace = HuffmanWorkspace<Q, S, H, DeviceType>(
        max_size, dict_size, chunk_size, estimated_outlier_ratio);
  }

  void Resize(SIZE max_size, int dict_size, int chunk_size,
              double estimated_outlier_ratio, int queue_idx) {
    this->initialized = true;
    this->max_size = max_size;
    this->dict_size = dict_size;
    this->chunk_size = chunk_size;
    MemoryManager<DeviceType>::MallocHost(signature_verify, 7 * sizeof(char),
                                          queue_idx);
    workspace.resize(max_size, dict_size, chunk_size, estimated_outlier_ratio,
                     queue_idx);
  }

  static size_t EstimateMemoryFootprint(SIZE primary_count, SIZE dict_size,
                                        SIZE chunk_size,
                                        double estimated_outlier_ratio = 1) {
    return HuffmanWorkspace<Q, S, H, DeviceType>::EstimateMemoryFootprint(
        primary_count, dict_size, chunk_size, estimated_outlier_ratio);
  }

  double EstimateCR(Array<1, Q, DeviceType> &primary_data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    SubArray primary_subarray(primary_data);
    workspace.reset(queue_idx);

    primary_count = primary_subarray.shape(0);

    Histogram<Q, unsigned int, DeviceType>(primary_subarray,
                                           workspace.freq_subarray,
                                           primary_count, dict_size, queue_idx);
    auto type_bw = sizeof(H) * 8;

    SubArray<1, H, DeviceType> _d_first_subarray(
        {(SIZE)type_bw}, (H *)workspace.decodebook_subarray((IDX)0));
    SubArray<1, H, DeviceType> _d_entry_subarray(
        {(SIZE)type_bw},
        (H *)workspace.decodebook_subarray(sizeof(H) * type_bw));
    SubArray<1, Q, DeviceType> _d_qcode_subarray(
        {(SIZE)dict_size},
        (Q *)workspace.decodebook_subarray(sizeof(H) * 2 * type_bw));

    // Sort Qcodes by frequency
    DeviceLauncher<DeviceType>::Execute(
        FillArraySequenceKernel(_d_qcode_subarray), queue_idx);

    MemoryManager<DeviceType>::Copy1D(workspace._d_freq_copy_subarray.data(),
                                      workspace.freq_subarray.data(), dict_size,
                                      queue_idx);
    MemoryManager<DeviceType>::Copy1D(workspace._d_qcode_copy_subarray.data(),
                                      _d_qcode_subarray.data(), dict_size,
                                      queue_idx);
    DeviceCollective<DeviceType>::SortByKey(
        (SIZE)dict_size, workspace._d_freq_copy_subarray,
        workspace._d_qcode_copy_subarray, workspace.freq_subarray,
        _d_qcode_subarray, workspace.sort_by_key_workspace, true, queue_idx);

    DeviceLauncher<DeviceType>::Execute(
        GetFirstNonzeroIndexKernel<unsigned int, DeviceType>(
            workspace.freq_subarray, workspace.first_nonzero_index_subarray),
        queue_idx);

    unsigned int first_nonzero_index;
    MemoryManager<DeviceType>().Copy1D(
        &first_nonzero_index, workspace.first_nonzero_index_subarray(IDX(0)), 1,
        queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    int nz_dict_size = dict_size - first_nonzero_index;

    SubArray<1, unsigned int, DeviceType> _nz_d_freq_subarray(
        {(SIZE)nz_dict_size}, workspace.freq_subarray(first_nonzero_index));
    SubArray<1, H, DeviceType> _nz_d_codebook_subarray(
        {(SIZE)nz_dict_size}, workspace.codebook_subarray(first_nonzero_index));

    DeviceLauncher<DeviceType>::Execute(
        GenerateCLKernel<unsigned int, DeviceType>(
            _nz_d_freq_subarray, workspace.CL_subarray, nz_dict_size,
            _nz_d_freq_subarray, workspace.lNodesLeader_subarray,
            workspace.iNodesFreq_subarray, workspace.iNodesLeader_subarray,
            workspace.tempFreq_subarray, workspace.tempIsLeaf_subarray,
            workspace.tempIndex_subarray, workspace.copyFreq_subarray,
            workspace.copyIsLeaf_subarray, workspace.copyIndex_subarray,
            workspace.diagonal_path_intersections_subarray,
            workspace.status_subarray),
        queue_idx);

    unsigned int *_freq = new unsigned int[dict_size];
    unsigned int *_cl = new unsigned int[dict_size];
    MemoryManager<DeviceType>::Copy1D(_freq, workspace.freq_subarray.data(),
                                      dict_size, queue_idx);
    MemoryManager<DeviceType>::Copy1D(_cl, workspace.CL_subarray.data(),
                                      dict_size, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    double LC = 0;
    for (SIZE i = 0; i < dict_size; i++) {
      LC += (double)_freq[i] * _cl[i];
    }
    delete[] _freq;
    delete[] _cl;

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman estimate CR", primary_count * sizeof(Q));
      timer.clear();
    }

    double CR = (double)(sizeof(Q) * primary_count) / (LC / 8 + 2000);
    return CR;
  }

  bool CompressPrimary(Array<1, Q, DeviceType> &primary_data,
                       Array<1, Byte, DeviceType> &compressed_data,
                       float target_cr, int queue_idx) {

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SubArray primary_subarray(primary_data);
    workspace.reset(queue_idx);

    primary_count = primary_subarray.shape(0);

    Histogram<Q, unsigned int, DeviceType>(primary_subarray,
                                           workspace.freq_subarray,
                                           primary_count, dict_size, queue_idx);

    if (debug_print_huffman) {
      PrintSubarray("Histogram::freq_subarray", workspace.freq_subarray);
    }

    GetCodebook(dict_size, workspace.freq_subarray, workspace.codebook_subarray,
                workspace.decodebook_subarray, workspace, queue_idx);

    if (target_cr > 1.0) {
      workspace.freq_array.hostCopy(false, queue_idx);
      workspace.CL_array.hostCopy(false, queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      unsigned int *_freq = workspace.freq_array.dataHost();
      unsigned int *_cl = workspace.CL_array.dataHost();
      double LC = 0;
      for (SIZE i = 0; i < dict_size; i++) {
        LC += (double)_freq[i] * _cl[i];
      }
      double estimated_cr =
          (double)(sizeof(Q) * primary_count) / (LC / 8 + 2000);
      log::info("Huffman estimated CR: " + std::to_string(estimated_cr) +
                " (target: " + std::to_string(target_cr) + ")");
      if (estimated_cr < target_cr) {
        return false;
      }
    }

    if (debug_print_huffman) {
      PrintSubarray("GetCodebook::codebook_subarray",
                    workspace.codebook_subarray);
      PrintSubarray("GetCodebook::decodebook_subarray",
                    workspace.decodebook_subarray);
    }
    DeviceLauncher<DeviceType>::Execute(
        EncodeFixedLenKernel<Q, H, DeviceType>(primary_subarray,
                                               workspace.huff_subarray,
                                               workspace.codebook_subarray),
        queue_idx);

    if (debug_print_huffman) {
      PrintSubarray("EncodeFixedLen::huff_subarray", workspace.huff_subarray);
    }
    // deflate
    DeviceLauncher<DeviceType>::Execute(
        DeflateKernel<H, DeviceType>(workspace.huff_subarray,
                                     workspace.huff_bitwidths_subarray,
                                     chunk_size),
        queue_idx);
    if (debug_print_huffman) {
      PrintSubarray("Deflate::huff_subarray", workspace.huff_subarray);
      PrintSubarray("Deflate::huff_bitwidths_subarray",
                    workspace.huff_bitwidths_subarray);
    }

    // Serialize(compressed_data, queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman compress", primary_count * sizeof(Q));
      timer.clear();
    }
    return true;
  }

  void Serialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    auto nchunk = (primary_count - 1) / chunk_size + 1;
    size_t *h_meta = new size_t[nchunk * 3]();
    size_t *dH_uInt_meta = h_meta;
    size_t *dH_bit_meta = h_meta + nchunk;
    size_t *dH_uInt_entry = h_meta + nchunk * 2;

    MemoryManager<DeviceType>().Copy1D(dH_bit_meta,
                                       workspace.huff_bitwidths_subarray.data(),
                                       nchunk, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // transform in uInt
    memcpy(dH_uInt_meta, dH_bit_meta, nchunk * sizeof(size_t));
    std::for_each(dH_uInt_meta, dH_uInt_meta + nchunk,
                  [&](size_t &i) { i = (i - 1) / (sizeof(H) * 8) + 1; });
    // make it entries
    memcpy(dH_uInt_entry + 1, dH_uInt_meta, (nchunk - 1) * sizeof(size_t));
    for (auto i = 1; i < nchunk; i++)
      dH_uInt_entry[i] += dH_uInt_entry[i - 1];

    // sum bits from each chunk
    auto total_bits =
        std::accumulate(dH_bit_meta, dH_bit_meta + nchunk, (size_t)0);
    auto total_uInts =
        std::accumulate(dH_uInt_meta, dH_uInt_meta + nchunk, (size_t)0);

    // printf("huffman encode time: %.6f s\n", time_span.count());

    // out_meta: |outlier count|outlier idx|outlier data|primary count|dict
    // size|chunk size|huffmeta size|huffmeta|decodebook size|decodebook|
    // out_data: |huffman data|

    size_t type_bw = sizeof(H) * 8;
    size_t decodebook_size = workspace.decodebook_subarray.shape(0);
    size_t huffmeta_size = 2 * nchunk;

    size_t ddata_size = total_uInts;

    SIZE byte_offset = 0;
    advance_with_align<Byte>(byte_offset, 7); // signature
    advance_with_align<size_t>(byte_offset, 1);
    advance_with_align<int>(byte_offset, 1);
    advance_with_align<int>(byte_offset, 1);
    advance_with_align<size_t>(byte_offset, 1);
    advance_with_align<size_t>(byte_offset, huffmeta_size);
    advance_with_align<size_t>(byte_offset, 1);
    advance_with_align<uint8_t>(
        byte_offset, (sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size));
    advance_with_align<size_t>(byte_offset, 1);
    advance_with_align<H>(byte_offset, ddata_size);
    // outliter
    advance_with_align<ATOMIC_IDX>(byte_offset, 1);
    advance_with_align<ATOMIC_IDX>(byte_offset, outlier_count);
    advance_with_align<S>(byte_offset, outlier_count);

    compressed_data.resize({(SIZE)(byte_offset)});
    SubArray compressed_data_subarray(compressed_data);

    byte_offset = 0;
    SerializeArray<Byte>(compressed_data_subarray, signature, 7, byte_offset,
                         queue_idx);
    SerializeArray<size_t>(compressed_data_subarray, &primary_count, 1,
                           byte_offset, queue_idx);
    SerializeArray<int>(compressed_data_subarray, &dict_size, 1, byte_offset,
                        queue_idx);
    SerializeArray<int>(compressed_data_subarray, &chunk_size, 1, byte_offset,
                        queue_idx);
    SerializeArray<size_t>(compressed_data_subarray, &huffmeta_size, 1,
                           byte_offset, queue_idx);
    SerializeArray<size_t>(compressed_data_subarray, dH_bit_meta, huffmeta_size,
                           byte_offset, queue_idx);
    SerializeArray<size_t>(compressed_data_subarray, &decodebook_size, 1,
                           byte_offset, queue_idx);
    SerializeArray<uint8_t>(compressed_data_subarray,
                            workspace.decodebook_subarray.data(),
                            (sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size),
                            byte_offset, queue_idx);
    SerializeArray<size_t>(compressed_data_subarray, &ddata_size, 1,
                           byte_offset, queue_idx);

    align_byte_offset<H>(byte_offset);

    MemoryManager<DeviceType>::Copy1D(
        workspace.condense_write_offsets_subarray.data(), dH_uInt_entry, nchunk,
        queue_idx);
    MemoryManager<DeviceType>::Copy1D(
        workspace.condense_actual_lengths_subarray.data(), dH_uInt_meta, nchunk,
        queue_idx);
    SubArray<1, H, DeviceType> compressed_data_cast_subarray(
        {(SIZE)ddata_size}, (H *)compressed_data_subarray(byte_offset));
    DeviceLauncher<DeviceType>::Execute(
        CondenseKernel<H, DeviceType>(
            workspace.huff_subarray, workspace.condense_write_offsets_subarray,
            workspace.condense_actual_lengths_subarray,
            compressed_data_cast_subarray, chunk_size, nchunk),
        queue_idx);

    advance_with_align<H>(byte_offset, ddata_size);

    // outlier
    SerializeArray<ATOMIC_IDX>(compressed_data_subarray, &outlier_count, 1,
                               byte_offset, queue_idx);
    SerializeArray<ATOMIC_IDX>(compressed_data_subarray,
                               workspace.outlier_idx_subarray.data(),
                               outlier_count, byte_offset, queue_idx);
    SerializeArray<S>(compressed_data_subarray,
                      workspace.outlier_subarray.data(), outlier_count,
                      byte_offset, queue_idx);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    delete[] h_meta;

    log::dbg("Huffman block size: " + std::to_string(chunk_size));
    log::dbg("Huffman dictionary size: " + std::to_string(dict_size));
    log::dbg("Huffman compress ratio (primary): " +
             std::to_string(primary_count * sizeof(Q)) + "/" +
             std::to_string(ddata_size * sizeof(H)) + " (" +
             std::to_string((double)primary_count * sizeof(Q) /
                            (ddata_size * sizeof(H))) +
             ")");
    log::info(
        "Huffman compress ratio: " + std::to_string(primary_count * sizeof(Q)) +
        "/" + std::to_string(compressed_data.shape(0)) + " (" +
        std::to_string((double)primary_count * sizeof(Q) /
                       compressed_data.shape(0)) +
        ")");
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman serialize", compressed_data.shape(0));
      timer.clear();
    }
  }

  bool Verify(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    SubArray compressed_subarray(compressed_data);
    SIZE byte_offset = 0;
    DeserializeArray<Byte>(compressed_subarray, signature_verify, 7,
                           byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    for (int i = 0; i < 7; i++) {
      if (signature[i] != signature_verify[i]) {
        return false;
      }
    }
    return true;
  }

  void Deserialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    if (!Verify(compressed_data, queue_idx)) {
      throw std::runtime_error("Huffman signature mismatch.");
    }

    SubArray compressed_subarray(compressed_data);

    Byte *signature_ptr = nullptr;
    size_t *primary_count_ptr = &primary_count;
    int *dict_size_ptr = &dict_size;
    int *chunk_size_ptr = &chunk_size;
    size_t *huffmeta_size_ptr = &huffmeta_size;
    size_t *decodebook_size_ptr = &decodebook_size;
    size_t *ddata_size_ptr = &ddata_size;
    ATOMIC_IDX *outlier_count_ptr = &outlier_count;

    SIZE byte_offset = 0;
    DeserializeArray<Byte>(compressed_subarray, signature_ptr, 7, byte_offset,
                           true, queue_idx);
    DeserializeArray<size_t>(compressed_subarray, primary_count_ptr, 1,
                             byte_offset, false, queue_idx);
    DeserializeArray<int>(compressed_subarray, dict_size_ptr, 1, byte_offset,
                          false, queue_idx);
    DeserializeArray<int>(compressed_subarray, chunk_size_ptr, 1, byte_offset,
                          false, queue_idx);
    DeserializeArray<size_t>(compressed_subarray, huffmeta_size_ptr, 1,
                             byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    DeserializeArray<size_t>(compressed_subarray, huffmeta, huffmeta_size,
                             byte_offset, true, queue_idx);
    DeserializeArray<size_t>(compressed_subarray, decodebook_size_ptr, 1,
                             byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    DeserializeArray<uint8_t>(compressed_subarray, decodebook, decodebook_size,
                              byte_offset, true, queue_idx);
    DeserializeArray<size_t>(compressed_subarray, ddata_size_ptr, 1,
                             byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    DeserializeArray<H>(compressed_subarray, ddata, ddata_size, byte_offset,
                        true, queue_idx);

    // outlier
    DeserializeArray<ATOMIC_IDX>(compressed_subarray, outlier_count_ptr, 1,
                                 byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    DeserializeArray<ATOMIC_IDX>(compressed_subarray, outlier_idx,
                                 outlier_count, byte_offset, true, queue_idx);
    DeserializeArray<S>(compressed_subarray, outlier, outlier_count,
                        byte_offset, true, queue_idx);
    workspace.outlier_idx_subarray =
        SubArray<1, ATOMIC_IDX, DeviceType>({(SIZE)outlier_count}, outlier_idx);
    workspace.outlier_subarray =
        SubArray<1, S, DeviceType>({(SIZE)outlier_count}, outlier);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman deserialize", compressed_data.shape(0));
      timer.clear();
    }
  }

  void DecompressPrimary(Array<1, Byte, DeviceType> &compressed_data,
                         Array<1, Q, DeviceType> &primary_data, int queue_idx) {

    // Deserialize(compressed_data, queue_idx);

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SubArray<1, H, DeviceType> ddata_subarray({(SIZE)ddata_size}, ddata);
    SubArray<1, size_t, DeviceType> huffmeta_subarray({(SIZE)huffmeta_size},
                                                      huffmeta);
    primary_data.resize({(SIZE)primary_count});
    SubArray primary_subarray(primary_data);

    SubArray<1, uint8_t, DeviceType> decodebook_subarray(
        {(SIZE)decodebook_size}, decodebook);

    int nchunk = (primary_count - 1) / chunk_size + 1;
    Decode<Q, H, DeviceType>(
        ddata_subarray, huffmeta_subarray, primary_subarray, primary_count,
        chunk_size, nchunk, decodebook_subarray, decodebook_size, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman decompress", primary_count * sizeof(Q));
      timer.clear();
    }
  }

  bool Compress(Array<1, S, DeviceType> &original_data,
                Array<1, Byte, DeviceType> &compressed_data, float target_cr,
                int queue_idx) {

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    ATOMIC_IDX zero = 0;
    MemoryManager<DeviceType>::Copy1D(workspace.outlier_count_subarray.data(),
                                      &zero, 1, queue_idx);

    DeviceLauncher<DeviceType>::Execute(
        DictionaryShiftKernel<S, MGARDX_SHIFT_DICT, DeviceType>(
            SubArray(original_data), dict_size),
        queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        OutlierSeparatorKernel<S, MGARDX_SEPARATE_OUTLIER, DeviceType>(
            SubArray(original_data), dict_size,
            workspace.outlier_count_subarray, workspace.outlier_idx_subarray,
            workspace.outlier_subarray),
        queue_idx);
    MemoryManager<DeviceType>::Copy1D(
        &outlier_count, workspace.outlier_count_subarray.data(), 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    log::info(
        "Outlier ratio: " + std::to_string(outlier_count) + "/" +
        std::to_string(original_data.shape(0)) + " (" +
        std::to_string((double)100 * outlier_count / original_data.shape(0)) +
        "%)");
    if (outlier_count > workspace.outlier_subarray.shape(0)) {
      throw std::runtime_error("Not enough workspace for outliers.");
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman outlier", original_data.shape(0) * sizeof(S));
      timer.clear();
    }

    // Cast to unsigned type
    Array<1, Q, DeviceType> primary_data({original_data.shape(0)},
                                         (Q *)original_data.data());
    return CompressPrimary(primary_data, compressed_data, target_cr, queue_idx);
  }

  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  Array<1, S, DeviceType> &decompressed_data, int queue_idx) {

    // Cast to unsigned type.
    // We must use the correct size to avoid resize to new allocation
    Array<1, Q, DeviceType> primary_data({decompressed_data.shape(0)},
                                         (Q *)decompressed_data.data());
    DecompressPrimary(compressed_data, primary_data, queue_idx);

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    DeviceLauncher<DeviceType>::Execute(
        OutlierSeparatorKernel<S, MGARDX_RESTORE_OUTLIER, DeviceType>(
            decompressed_data, dict_size, workspace.outlier_count_subarray,
            workspace.outlier_idx_subarray, workspace.outlier_subarray),
        queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        DictionaryShiftKernel<S, MGARDX_RESTORE_DICT, DeviceType>(
            decompressed_data, dict_size),
        queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman outlier", decompressed_data.shape(0) * sizeof(S));
      timer.clear();
    }
  }

  bool initialized;
  SIZE max_size;
  size_t primary_count;
  ATOMIC_IDX outlier_count;
  int dict_size;
  int chunk_size;
  size_t huffmeta_size;
  size_t decodebook_size;
  size_t ddata_size;
  size_t *huffmeta;
  uint8_t *decodebook;
  ATOMIC_IDX *outlier_idx;
  S *outlier;
  H *ddata;
  Byte signature[7] = {'M', 'G', 'X', 'H', 'U', 'F', 'F'};
  Byte *signature_verify = nullptr;
  HuffmanWorkspace<Q, S, H, DeviceType> workspace;
};

} // namespace mgard_x

#endif
