#include "catch2/catch_test_macros.hpp"

#include <climits>

#include <algorithm>
#include <random>

#include "testing_utilities.hpp"

#include "huffman.hpp"

namespace {

void test_encoding_regression(long int *const quantized, const std::size_t N) {
  long int *const quantized_new = new long int[N];
  std::copy(quantized, quantized + N, quantized_new);

  const mgard::HuffmanEncodedStream out = mgard::huffman_encoding(quantized, N);
  const mgard::HuffmanEncodedStream out_new =
      mgard::huffman_encoding_rewritten(quantized_new, N);

  unsigned char const *const hit = out.hit.data.get();
  REQUIRE(out_new.nbits == out.nbits);
  const std::size_t nbytes = (out.nbits + CHAR_BIT - 1) / CHAR_BIT;
  REQUIRE(std::equal(hit, hit + nbytes, out_new.hit.data.get()));

  unsigned char const *const missed = out.missed.data.get();
  const std::size_t nmissed = out.missed.size;
  REQUIRE(out_new.missed.size == nmissed);
  REQUIRE(std::equal(missed, missed + nmissed, out_new.missed.data.get()));

  unsigned char const *const frequencies = out.frequencies.data.get();
  const std::size_t nfrequencies = out.frequencies.size;
  REQUIRE(out_new.frequencies.size == nfrequencies);
  REQUIRE(std::equal(frequencies, frequencies + nfrequencies,
                     out_new.frequencies.data.get()));

  delete[] quantized_new;
}

void test_decoding_regression(long int *const quantized, const std::size_t N) {
  long int *const quantized_new = new long int[N];
  std::copy(quantized, quantized + N, quantized_new);

  const mgard::HuffmanEncodedStream encoded =
      mgard::huffman_encoding(quantized, N);
  const mgard::HuffmanEncodedStream encoded_new =
      mgard::huffman_encoding(quantized_new, N);

  delete[] quantized_new;

  const mgard::MemoryBuffer<long int> out = mgard::huffman_decoding(encoded);
  const mgard::MemoryBuffer<long int> out_new =
      mgard::huffman_decoding(encoded);

  REQUIRE(out.size == out_new.size);
  long int const *const p = out.data.get();
  long int const *const p_new = out_new.data.get();
  REQUIRE(std::equal(p, p + out.size, p_new));
}

void test_encoding_regression_constant(const std::size_t N, const long int q) {
  long int *const quantized = new long int[N];
  std::fill(quantized, quantized + N, q);
  test_encoding_regression(quantized, N);
  delete[] quantized;
}

void test_encoding_regression_periodic(const std::size_t N, const long int q,
                                       const std::size_t period) {
  long int *const quantized = new long int[N];
  std::generate(quantized, quantized + N, PeriodicGenerator(period, q));
  test_encoding_regression(quantized, N);
  delete[] quantized;
}

void test_encoding_regression_random(const std::size_t N, const long int a,
                                     const long int b,
                                     std::default_random_engine &gen) {
  std::uniform_int_distribution<long int> dis(a, b);
  long int *const quantized = new long int[N];
  std::generate(quantized, quantized + N, [&] { return dis(gen); });
  test_encoding_regression(quantized, N);
  delete[] quantized;
}

void test_decoding_regression_constant(const std::size_t N, const long int q) {
  long int *const quantized = new long int[N];
  std::fill(quantized, quantized + N, q);
  test_decoding_regression(quantized, N);
  delete[] quantized;
}

void test_decoding_regression_periodic(const std::size_t N, const long int q,
                                       const std::size_t period) {
  long int *const quantized = new long int[N];
  std::generate(quantized, quantized + N, PeriodicGenerator(period, q));
  test_decoding_regression(quantized, N);
  delete[] quantized;
}

void test_decoding_regression_random(const std::size_t N, const long int a,
                                     const long int b,
                                     std::default_random_engine &gen) {
  std::uniform_int_distribution<long int> dis(a, b);
  long int *const quantized = new long int[N];
  std::generate(quantized, quantized + N, [&] { return dis(gen); });
  test_decoding_regression(quantized, N);
  delete[] quantized;
}

} // namespace

TEST_CASE("encoding regression", "[huffman] [regression]") {
  SECTION("constant data") {
    test_encoding_regression_constant(10, 0);
    test_encoding_regression_constant(100, 732);
    test_encoding_regression_constant(1000, -10);
  }

  SECTION("periodic data") {
    test_encoding_regression_periodic(10, -3, 3);
    test_encoding_regression_periodic(100, 0, 10);
    test_encoding_regression_periodic(1000, 51, 17);
  }

  SECTION("random data") {
    std::default_random_engine gen(131051);
    test_encoding_regression_random(10, 0, 1, gen);
    test_encoding_regression_random(100, -15, -5, gen);
    test_encoding_regression_random(1000, std::numeric_limits<int>::min(),
                                    std::numeric_limits<int>::max(), gen);
    test_encoding_regression_random(10000, -100, 100, gen);
  }
}

TEST_CASE("decoding regression", "[huffman] [regression]") {
  SECTION("constant data") {
    test_decoding_regression_constant(10, -11);
    test_decoding_regression_constant(100, 79);
    test_decoding_regression_constant(1000, -7296);
  }

  SECTION("periodic data") {
    test_decoding_regression_periodic(10, 12, 4);
    test_decoding_regression_periodic(100, -71, 9);
    test_decoding_regression_periodic(1000, 3280, 23);
  }

  SECTION("random data") {
    std::default_random_engine gen(363022);
    test_decoding_regression_random(10, 0, 1, gen);
    test_decoding_regression_random(100, -15, -5, gen);
    test_decoding_regression_random(1000, std::numeric_limits<int>::min(),
                                    std::numeric_limits<int>::max(), gen);
    test_decoding_regression_random(10000, -100, 100, gen);
  }
}
