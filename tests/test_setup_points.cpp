
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/core/detail/SetupPoints.hpp"

#include <alpaka/alpaka.hpp>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

inline constexpr bool is_cpu = std::same_as<clue::Device, alpaka::DevCpu>;

template <typename THostView, typename TDeviceView, std::size_t Ndim>
void check_aliasing(const THostView& h_view, const TDeviceView& d_view) {
  for (auto dim = 0u; dim < Ndim; ++dim) {
    CHECK(d_view.m_coords[dim] == h_view.m_coords[dim]);
    CHECK(d_view.m_sigmas[dim] == h_view.m_sigmas[dim]);
  }
  CHECK(d_view.m_weight == h_view.m_weight);
  CHECK(d_view.m_cluster_index == h_view.m_cluster_index);
  CHECK(d_view.m_density_uncertainty == h_view.m_density_uncertainty);
  CHECK(d_view.m_tags == h_view.m_tags);
  CHECK(d_view.m_n == h_view.m_n);
}

TEST_CASE("Test make_device_points aliases host memory on CPU") {
  auto queue = clue::get_queue(0u);
  const auto size = 1000;

  SUBCASE("One dimension") {
    clue::PointsHost<1> h_points(queue, size);
    auto d_points = clue::detail::make_device_points(queue, h_points);

    if constexpr (is_cpu) {
      check_aliasing<decltype(h_points.view()), decltype(d_points.view()), 1>(h_points.view(),
                                                                              d_points.view());
    } else {
      CHECK(d_points.view().m_coords[0] != h_points.view().m_coords[0]);
    }
  }

  SUBCASE("Two dimensions") {
    clue::PointsHost<2> h_points(queue, size);
    auto d_points = clue::detail::make_device_points(queue, h_points);

    if constexpr (is_cpu) {
      check_aliasing<decltype(h_points.view()), decltype(d_points.view()), 2>(h_points.view(),
                                                                              d_points.view());
    } else {
      CHECK(d_points.view().m_coords[0] != h_points.view().m_coords[0]);
    }
  }

  SUBCASE("Three dimensions") {
    clue::PointsHost<3> h_points(queue, size);
    auto d_points = clue::detail::make_device_points(queue, h_points);

    if constexpr (is_cpu) {
      check_aliasing<decltype(h_points.view()), decltype(d_points.view()), 3>(h_points.view(),
                                                                              d_points.view());
    } else {
      CHECK(d_points.view().m_coords[0] != h_points.view().m_coords[0]);
    }
  }

  SUBCASE("Const input") {
    std::vector<float> input(3 * size);
    std::vector<int> output(size);
    std::iota(input.begin(), input.end(), 0.f);

    clue::ConstPointsHost<2> h_points(queue, size, input.data(), output.data());
    auto d_points = clue::detail::make_device_points(queue, h_points);

    if constexpr (is_cpu) {
      check_aliasing<decltype(h_points.view()), decltype(d_points.view()), 2>(h_points.view(),
                                                                              d_points.view());
    } else {
      CHECK(d_points.view().m_coords[0] != h_points.view().m_coords[0]);
    }
  }

  SUBCASE("Optional sigmas, density uncertainty and tags") {
    clue::PointsHost<2> h_points(queue, size);
    std::vector<float> sigma_x(size, 1.f), sigma_y(size, 2.f), uncertainty(size, 1.f);
    std::vector<std::uint32_t> tags(size);
    std::iota(tags.begin(), tags.end(), 0u);
    h_points.set_sigmas(sigma_x, sigma_y);
    h_points.set_density_uncertainty(uncertainty);
    h_points.set_tags(tags);

    auto d_points = clue::detail::make_device_points(queue, h_points);

    if constexpr (is_cpu) {
      check_aliasing<decltype(h_points.view()), decltype(d_points.view()), 2>(h_points.view(),
                                                                              d_points.view());
    } else {
      CHECK(d_points.view().m_sigmas[0] != h_points.view().m_sigmas[0]);
      CHECK(d_points.view().m_density_uncertainty != h_points.view().m_density_uncertainty);
      CHECK(d_points.view().m_tags != h_points.view().m_tags);
    }
  }
}
