#ifndef HeterogeneousCore_AlpakaCore_ContextState_h
#define HeterogeneousCore_AlpakaCore_ContextState_h

#include <memory>
#include <stdexcept>
#include <utility>

#include <alpaka/alpaka.hpp>

namespace cms::alpakatools {

  template <typename TQueue>
  class ScopedContextAcquire;

  template <typename TQueue>
  class ScopedContextProduce;

  template <typename TQueue>
  class ScopedContextTask;

  /**
     * The purpose of this class is to deliver the device and CUDA stream
     * information from ExternalWork's acquire() to producer() via a
     * member/StreamCache variable.
     */
  template <typename TQueue>
  class ContextState {
  public:
    using Queue = TQueue;
    using Device = alpaka::Dev<Queue>;

    ContextState() = default;
    ~ContextState() = default;

    ContextState(const ContextState&) = delete;
    ContextState& operator=(const ContextState&) = delete;
    ContextState(ContextState&&) = delete;
    ContextState& operator=(ContextState&& other) = delete;

  private:
    friend class ScopedContextAcquire<TQueue>;
    friend class ScopedContextProduce<TQueue>;
    friend class ScopedContextTask<TQueue>;

    void set(std::shared_ptr<Queue> stream) {
      throwIfStream();
      stream_ = std::move(stream);
    }

    Device device() const {
      throwIfNoStream();
      return alpaka::getDev(*stream_);
    }

    Queue stream() const {
      throwIfNoStream();
      return *stream_;
    }

    std::shared_ptr<Queue> const& streamPtr() const {
      throwIfNoStream();
      return stream_;
    }

    std::shared_ptr<Queue> releaseStreamPtr() {
      throwIfNoStream();
      // This function needs to effectively reset stream_ (i.e. stream_
      // must be empty after this function). This behavior ensures that
      // the std::shared_ptr<Queue> is not hold for inadvertedly long (i.e. to
      // the next event), and is checked at run time.
      return std::move(stream_);
    }

    void throwIfStream() const {
      if (stream_) {
        throw std::runtime_error("Trying to set ContextState, but it already had a valid state");
      }
    }

    void throwIfNoStream() const {
      if (not stream_) {
        throw std::runtime_error("Trying to get ContextState, but it did not have a valid state");
      }
    }

    std::shared_ptr<Queue> stream_;
  };

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaCore_ContextState_h
