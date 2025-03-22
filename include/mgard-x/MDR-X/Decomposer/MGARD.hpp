#ifndef _MDR_MGARD_DECOMPOSER_HPP
#define _MDR_MGARD_DECOMPOSER_HPP

#include "../../DataRefactoring/DataRefactor.hpp"
#include "DecomposerInterface.hpp"

#include <cstring>

namespace mgard_x {
namespace MDR {

struct DecompsitionBasis {};
struct Orthogonal : DecompsitionBasis {};
struct Hierarchical : DecompsitionBasis {};

template <DIM D, typename T, typename Basis, typename DeviceType>
class MGARDDecomposer : public concepts::DecomposerInterface<D, T, DeviceType> {
public:
  MGARDDecomposer() : initialized(false) {}
  MGARDDecomposer(Hierarchy<D, T, DeviceType> &hierarchy, Config config) {
    Adapt(hierarchy, config, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  void Adapt(Hierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    refactor.Adapt(hierarchy, config, queue_idx);
  }
  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 0;
    Hierarchy<D, T, DeviceType> hierarchy;
    size += hierarchy.EstimateMemoryFootprint(shape);
    size += data_refactoring::DataRefactor<
        D, T, DeviceType>::EstimateMemoryFootprint(shape);
    return size;
  }
  void decompose(Array<D, T, DeviceType> &v, int start_level, int stop_level,
                 int queue_idx) {
    if constexpr (std::is_same<Basis, Orthogonal>::value) {
      refactor.Decompose(v, start_level, stop_level, true, queue_idx);
    } else if constexpr (std::is_same<Basis, Hierarchical>::value) {
      refactor.Decompose(v, start_level, stop_level, false, queue_idx);
    }
  }
  void recompose(Array<D, T, DeviceType> &v, int start_level, int stop_level,
                 int queue_idx) {
    if constexpr (std::is_same<Basis, Orthogonal>::value) {
      refactor.Recompose(v, start_level, stop_level, true, queue_idx);
    } else if constexpr (std::is_same<Basis, Hierarchical>::value) {
      refactor.Recompose(v, start_level, stop_level, false, queue_idx);
    }
  }
  void print() const { std::cout << "MGARD decomposer" << std::endl; }

private:
  bool initialized;
  Hierarchy<D, T, DeviceType> *hierarchy;
  data_refactoring::DataRefactor<D, T, DeviceType> refactor;
};

} // namespace MDR
} // namespace mgard_x
#endif
