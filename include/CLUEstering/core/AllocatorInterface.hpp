
#pragma once

#include <cstddef>

namespace clue {

  class AllocatorInterface {
  public:
    virtual ~AllocatorInterface() = default;

    virtual void* allocate(std::size_t size, std::size_t /* alignment */) = 0;
    virtual void deallocate(void* ptr) = 0;
  };

}  // namespace clue
