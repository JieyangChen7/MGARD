#ifndef _MDR_BITPLANE_ENCODER_INTERFACE_HPP
#define _MDR_BITPLANE_ENCODER_INTERFACE_HPP

#include <cassert>
namespace MDR {
namespace concepts {
#define UINT8_BITS 8
// concept of encoder which encodes T type data into bitstreams
template <typename T> class BitplaneEncoderInterface {
public:
  virtual ~BitplaneEncoderInterface() = default;

  virtual std::vector<uint8_t *>
  encode(T const *data, uint32_t n, int32_t exp, uint8_t num_bitplanes,
         std::vector<uint32_t> &streams_sizes) const = 0;

  virtual T *decode(const std::vector<uint8_t const *> &streams, uint32_t n,
                    int exp, uint8_t num_bitplanes) = 0;

  virtual T *progressive_decode(const std::vector<uint8_t const *> &streams,
                                uint32_t n, int exp, uint8_t starting_bitplane,
                                uint8_t num_bitplanes, int level) = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR

namespace mgard_x {
namespace MDR {
namespace concepts {
// concept of encoder which encodes T type data into bitstreams
template <DIM D, typename T_data, typename T_bitplane, typename T_error,
          bool CollectError, typename DeviceType>
class BitplaneEncoderInterface {
public:
  virtual ~BitplaneEncoderInterface() = default;

  virtual void encode(SIZE n, int num_bitplanes,
                      SubArray<1, T_data, DeviceType> abs_max,
                      SubArray<1, T_data, DeviceType> v,
                      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                      SubArray<1, T_error, DeviceType> level_errors,
                      int queue_idx) = 0;

  virtual void decode(SIZE n, int num_bitplanes,
                      SubArray<1, T_data, DeviceType> abs_max,
                      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                      int level, SubArray<1, T_data, DeviceType> v,
                      int queue_idx) = 0;

  virtual void
  progressive_decode(SIZE n, int starting_bitplane, int num_bitplanes,
                     SubArray<1, T_data, DeviceType> abs_max,
                     SubArray<2, T_bitplane, DeviceType> encoded_bitplanes,
                     SubArray<1, bool, DeviceType> level_signs, int level,
                     SubArray<1, T_data, DeviceType> v, int queue_idx) = 0;

  virtual void print() const = 0;
};

} // namespace concepts
} // namespace MDR
} // namespace mgard_x
#endif
