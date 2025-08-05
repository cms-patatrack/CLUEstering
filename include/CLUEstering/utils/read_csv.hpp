
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/detail/concepts.hpp"

#include <string>

namespace clue {

  template <size_t NDim, concepts::queue TQueue>
  inline clue::PointsHost<NDim> read_csv(TQueue& queue, const std::string& file_path);

  template <size_t NDim, concepts::queue TQueue>
  inline clue::PointsHost<NDim> read_output(TQueue& queue, const std::string& file_path);

}  // namespace clue

#include "CLUEstering/utils/detail/read_csv.hpp"
