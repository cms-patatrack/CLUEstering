/// @file scores.hpp
/// @brief Provides fucntions for computing scores validating the quality of the clustering.
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/internal/alpaka/config.hpp"
#include <concepts>
#include <cstddef>

CLUE_NAMESPACE_BEGIN

/// @brief Compute the silhouette score for a specific point in the dataset.
///
/// @tparam Ndim The number of dimensions of the points
/// @tparam InputType The data type for the point coordinates and weights, which must be a floating-point type.
/// By default, it is set to `float`.
/// @tparam DistanceMetric The metric used to compute the distances between points. Its `value_type`
/// must match `InputType`. By default, it is set to `clue::EuclideanMetric<Ndim, InputType>`.
/// @param points The dataset containing the points
/// @param point The index of the point for which to compute the silhouette score
/// @param metric The distance metric used to compute the distances between points
/// @return The silhouette score of the specified point
/// @note To compute the score for points with periodic coordinates, pass a `clue::PeriodicEuclideanMetric`.
/// When using `clue::MahalanobisMetric`, the sigmas of the points must be set beforehand.
template <
    std::size_t Ndim,
    std::floating_point InputType = float,
    clue::concepts::distance_metric<Ndim> DistanceMetric = clue::EuclideanMetric<Ndim, InputType>>
  requires std::same_as<typename DistanceMetric::value_type, InputType>
auto silhouette(const clue::PointsHost<Ndim, InputType>& points,
                int point,
                const DistanceMetric& metric = DistanceMetric{});

/// @brief Compute the average silhouette score for the entire dataset.
///
/// @tparam Ndim The number of dimensions of the points
/// @tparam InputType The data type for the point coordinates and weights, which must be a floating-point type.
/// By default, it is set to `float`.
/// @tparam DistanceMetric The metric used to compute the distances between points. Its `value_type`
/// must match `InputType`. By default, it is set to `clue::EuclideanMetric<Ndim, InputType>`.
/// @param points The dataset containing the points
/// @param metric The distance metric used to compute the distances between points
/// @return The average silhouette score of the dataset
/// @note To compute the score for points with periodic coordinates, pass a `clue::PeriodicEuclideanMetric`.
/// When using `clue::MahalanobisMetric`, the sigmas of the points must be set beforehand.
template <
    std::size_t Ndim,
    std::floating_point InputType = float,
    clue::concepts::distance_metric<Ndim> DistanceMetric = clue::EuclideanMetric<Ndim, InputType>>
  requires std::same_as<typename DistanceMetric::value_type, InputType>
auto silhouette(const clue::PointsHost<Ndim, InputType>& points,
                const DistanceMetric& metric = DistanceMetric{});

CLUE_NAMESPACE_END

#include "CLUEstering/utils/detail/scores.hpp"
