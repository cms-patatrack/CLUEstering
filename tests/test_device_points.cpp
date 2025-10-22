
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/internal/algorithm/algorithm.hpp"
#include "CLUEstering/utils/get_device.hpp"
#include "CLUEstering/utils/get_queue.hpp"

#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include "doctest.h"

template <std::size_t Ndim>
struct KernelCompareDevicePoints {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                clue::PointsView<Ndim> view,
                                float* d_input,
                                uint32_t size,
                                int* result) const {
    if (alpaka::oncePerGrid(acc))
      *result = 1;
    for (auto i : alpaka::uniformElements(acc, size)) {
      int comparison = 1;
      for (auto dim = 0u; dim < Ndim; ++dim) {
        comparison = (view.coords[dim][i] == d_input[i + dim * size]);
      }
      comparison = (view.weight[i] == d_input[i + Ndim * size]);

      alpaka::atomicAnd(acc, result, comparison);
    }
  }
};

template <std::ranges::range TRange, std::size_t Ndim>
ALPAKA_FN_HOST bool compareDevicePoints(clue::Queue queue,
                                        TRange&& h_coords,
                                        TRange&& h_weights,
                                        clue::PointsDevice<Ndim>& d_points,
                                        uint32_t size) {
  auto h_points = clue::PointsHost<Ndim>(queue, size);
  std::ranges::copy(h_coords, h_points.coords(0).begin());
  std::ranges::copy(h_weights, h_points.weights().begin());
  clue::copyToDevice(queue, d_points, h_points);

  // define buffers for comparison
  auto d_input = clue::make_device_buffer<float[]>(queue, (Ndim + 1) * size);
  alpaka::memcpy(
      queue, d_input, clue::make_host_view(h_points.coords(0).data(), (Ndim + 1) * size));

  auto d_comparison_result = clue::make_device_buffer<int>(queue);
  const auto blocksize = 512;
  const auto gridsize = clue::divide_up_by(size, blocksize);
  auto work_division = clue::make_workdiv<clue::internal::Acc>(gridsize, blocksize);
  alpaka::exec<clue::internal::Acc>(queue,
                                    work_division,
                                    KernelCompareDevicePoints<Ndim>{},
                                    d_points.view(),
                                    d_input.data(),
                                    size,
                                    d_comparison_result.data());
  int comparison = 1;
  alpaka::memcpy(queue, clue::make_host_view<int>(comparison), d_comparison_result);
  alpaka::wait(queue);

  return static_cast<bool>(comparison);
}

TEST_CASE("Test device points with internal allocation") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  clue::PointsDevice<2> d_points(queue, size);

  auto to_float = [](int i) -> float { return static_cast<float>(i); };
  CHECK(compareDevicePoints(
      queue,
      std::views::iota(0) | std::views::take(2 * size) | std::views::transform(to_float),
      std::views::iota(0) | std::views::take(size) | std::views::transform(to_float),
      d_points,
      size));
}

TEST_CASE("Test device points with external allocation of whole buffer") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  const auto bytes = clue::soa::device::computeSoASize<2>(size);
  auto buffer = clue::make_device_buffer<std::byte[]>(queue, bytes);

  clue::PointsDevice<2> d_points(queue, size, std::span(buffer.data(), bytes));

  auto to_float = [](int i) -> float { return static_cast<float>(i); };
  CHECK(compareDevicePoints(
      queue,
      std::views::iota(0) | std::views::take(2 * size) | std::views::transform(to_float),
      std::views::iota(0) | std::views::take(size) | std::views::transform(to_float),
      d_points,
      size));
}

TEST_CASE("Test device points with external allocation passing the two buffers as pointers") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  auto input = clue::make_device_buffer<float[]>(queue, 3 * size);
  auto output = clue::make_device_buffer<int[]>(queue, 2 * size);

  clue::PointsDevice<2> d_points(queue, size, input.data(), output.data());
  auto to_float = [](int i) -> float { return static_cast<float>(i); };
  CHECK(compareDevicePoints(
      queue,
      std::views::iota(0) | std::views::take(2 * size) | std::views::transform(to_float),
      std::views::iota(0) | std::views::take(size) | std::views::transform(to_float),
      d_points,
      size));
}

TEST_CASE("Test device points with external allocation passing four buffers as pointers") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  auto coords = clue::make_device_buffer<float[]>(queue, 2 * size);
  auto weights = clue::make_device_buffer<float[]>(queue, size);
  auto cluster_ids = clue::make_device_buffer<int[]>(queue, size);

  clue::PointsDevice<2> d_points(queue, size, coords.data(), weights.data(), cluster_ids.data());
  auto to_float = [](int i) -> float { return static_cast<float>(i); };
  CHECK(compareDevicePoints(
      queue,
      std::views::iota(0) | std::views::take(2 * size) | std::views::transform(to_float),
      std::views::iota(0) | std::views::take(size) | std::views::transform(to_float),
      d_points,
      size));
}

TEST_CASE("Test extrema functions on device points column") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<float> data(size);
  std::iota(data.begin(), data.end(), 0.0f);

  clue::PointsHost<2> h_points(queue, 1000);
  std::ranges::copy(data, h_points.coords(0).begin());
  std::ranges::copy(data, h_points.weights().begin());

  clue::PointsDevice<2> d_points(queue, size);
  clue::copyToDevice(queue, d_points, h_points);

  auto max_it =
      clue::internal::algorithm::max_element(d_points.weights().begin(), d_points.weights().end());
  auto max = 0.f;
  alpaka::memcpy(
      queue, clue::make_host_view(max), clue::make_device_view(alpaka::getDev(queue), *max_it));
  CHECK(max == static_cast<float>(size - 1));
}

TEST_CASE("Test reduction of device points column") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<float> data(size);
  std::iota(data.begin(), data.end(), 0.0f);

  clue::PointsHost<2> h_points(queue, 1000);
  std::ranges::copy(data, h_points.coords(0).begin());
  std::ranges::copy(data, h_points.weights().begin());

  clue::PointsDevice<2> d_points(queue, size);
  clue::copyToDevice(queue, d_points, h_points);

  CHECK(clue::internal::algorithm::reduce(d_points.weights().begin(), d_points.weights().end()) ==
        499500.0f);
}

TEST_CASE("Test constructor throwing conditions") {
  auto queue = clue::get_queue(0u);
  CHECK_THROWS(clue::PointsDevice<2>(queue, 0));
  CHECK_THROWS(clue::PointsDevice<2>(queue, -5));
}

TEST_CASE("Test coordinate getter throwing conditions") {
  SUBCASE("Const points") {
    const uint32_t size = 1000;
    auto queue = clue::get_queue(0u);
    clue::PointsHost<2> points(queue, size);
    CHECK_THROWS(points.coords(3));
    CHECK_THROWS(points.coords(10));
  }
  SUBCASE("Non-const points") {
    const uint32_t size = 1000;
    auto queue = clue::get_queue(0u);
    const clue::PointsDevice<2> points(queue, size);
    CHECK_THROWS(points.coords(3));
    CHECK_THROWS(points.coords(10));
  }
}
