#ifndef AlpakaCore_ProductBase_h
#define AlpakaCore_ProductBase_h

#include <atomic>
#include <memory>
#include <utility>

#include <alpaka/alpaka.hpp>

namespace cms::alpakatools {

  namespace impl {
    template <typename TQueue>
    class ScopedContextBase;
  }

  template <typename TQueue>
  class ScopedContextProduce;

  /**
     * Base class for all instantiations of Product<TQueue, T> to hold the
     * non-T-dependent members.
     */
  template <typename TQueue>
  class ProductBase {
  public:
    using Queue = TQueue;
    using Event = alpaka::Event<Queue>;
    using Device = alpaka::Dev<Queue>;

    ProductBase() = default;  // Needed only for ROOT dictionary generation

    ~ProductBase() {
      // Make sure that the production of the product in the GPU is
      // complete before destructing the product. This is to make sure
      // that the EDM stream does not move to the next event before all
      // asynchronous processing of the current is complete.

      // TODO: a callback notifying a WaitingTaskHolder (or similar)
      // would avoid blocking the CPU, but would also require more work.

      // FIXME: this may throw an execption if the underlaying call fails.
      if (event_) {
        alpaka::wait(*event_);
      }
    }

    ProductBase(const ProductBase&) = delete;
    ProductBase& operator=(const ProductBase&) = delete;
    ProductBase(ProductBase&& other)
        : stream_{std::move(other.stream_)},
          event_{std::move(other.event_)},
          mayReuseStream_{other.mayReuseStream_.load()} {}
    ProductBase& operator=(ProductBase&& other) {
      stream_ = std::move(other.stream_);
      event_ = std::move(other.event_);
      mayReuseStream_ = other.mayReuseStream_.load();
      return *this;
    }

    bool isValid() const { return stream_.get() != nullptr; }

    bool isAvailable() const {
      // if default-constructed, the product is not available
      if (not event_) {
        return false;
      }
      return alpaka::isComplete(*event_);
    }

    Device device() const { return alpaka::getDev(stream()); }

    // cudaStream_t is a pointer to a thread-safe object, for which a
    // mutable access is needed even if the ScopedContext itself
    // would be const. Therefore it is ok to return a non-const
    // pointer from a const method here.
    Queue& stream() const { return *stream_; }

    // cudaEvent_t is a pointer to a thread-safe object, for which a
    // mutable access is needed even if the ScopedContext itself
    // would be const. Therefore it is ok to return a non-const
    // pointer from a const method here.
    Event& event() const { return *event_; }

  protected:
    explicit ProductBase(std::shared_ptr<Queue> stream, std::shared_ptr<Event> event)
        : stream_{std::move(stream)}, event_{std::move(event)} {}

  private:
    friend class impl::ScopedContextBase<Queue>;
    friend class ScopedContextProduce<Queue>;

    // The following function is intended to be used only from ScopedContext
    const std::shared_ptr<Queue>& streamPtr() const { return stream_; }

    bool mayReuseStream() const {
      bool expected = true;
      bool changed = mayReuseStream_.compare_exchange_strong(expected, false);
      // If the current thread is the one flipping the flag, it may
      // reuse the stream.
      return changed;
    }

    // The cudaStream_t is really shared among edm::Event products, so
    // using shared_ptr also here
    std::shared_ptr<Queue> stream_;  //!
    // shared_ptr because of caching in EventCache
    std::shared_ptr<Event> event_;  //!

    // This flag tells whether the CUDA stream may be reused by a
    // consumer or not. The goal is to have a "chain" of modules to
    // queue their work to the same stream.
    mutable std::atomic<bool> mayReuseStream_ = true;  //!
  };

}  // namespace cms::alpakatools

#endif  // AlpakaCore_ProductBase_h
