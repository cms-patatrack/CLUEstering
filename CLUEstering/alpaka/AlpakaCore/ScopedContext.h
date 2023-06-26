#ifndef HeterogeneousCore_AlpakaCore_ScopedContext_h
#define HeterogeneousCore_AlpakaCore_ScopedContext_h

#include <memory>
#include <stdexcept>
#include <utility>

#include "AlpakaCore/ContextState.h"
#include "AlpakaCore/EventCache.h"
#include "AlpakaCore/Product.h"
#include "AlpakaCore/StreamCache.h"
#include "AlpakaCore/HostOnlyTask.h"
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/chooseDevice.h"
#include "Framework/EDGetToken.h"
#include "Framework/EDPutToken.h"
#include "Framework/Event.h"
#include "Framework/WaitingTaskWithArenaHolder.h"

namespace cms::alpakatest {
  class TestScopedContext;
}

namespace cms::alpakatools {

  namespace impl {
    // This class is intended to be derived by other ScopedContext*, not for general use
    template <typename TQueue>
    class ScopedContextBase {
    public:
      using Queue = TQueue;
      using Device = alpaka::Dev<Queue>;
      using Platform = alpaka::Pltf<Device>;

      Device device() const { return alpaka::getDev(*stream_); }

      // cudaStream_t is a pointer to a thread-safe object, for which a
      // mutable access is needed even if the ScopedContext itself
      // would be const. Therefore it is ok to return a non-const
      // pointer from a const method here.
      Queue& stream() const { return *stream_; }
      const std::shared_ptr<Queue>& streamPtr() const { return stream_; }

    protected:
      // The constructors set the current device, but the device
      // is not set back to the previous value at the destructor. This
      // should be sufficient (and tiny bit faster) as all CUDA API
      // functions relying on the current device should be called from
      // the scope where this context is. The current device doesn't
      // really matter between modules (or across TBB tasks).

      ScopedContextBase(ProductBase<Queue> const& data)
          : stream_{data.mayReuseStream() ? data.streamPtr() : getStreamCache<Queue>().get(data.device())} {}

      explicit ScopedContextBase(std::shared_ptr<Queue> stream) : stream_(std::move(stream)) {}

      explicit ScopedContextBase(edm::StreamID streamID)
          : stream_{getStreamCache<Queue>().get(cms::alpakatools::chooseDevice<Platform>(streamID))} {}

    private:
      std::shared_ptr<Queue> stream_;
    };

    template <typename TQueue>
    class ScopedContextGetterBase : public ScopedContextBase<TQueue> {
    public:
      using Queue = TQueue;

      template <typename T>
      const T& get(Product<Queue, T> const& data) {
        synchronizeStreams(data);
        return data.data_;
      }

      template <typename T>
      const T& get(const edm::Event& iEvent, edm::EDGetTokenT<Product<Queue, T>> token) {
        return get(iEvent.get(token));
      }

    protected:
      template <typename... Args>
      ScopedContextGetterBase<TQueue>(Args&&... args) : ScopedContextBase<Queue>(std::forward<Args>(args)...) {}

      void synchronizeStreams(ProductBase<Queue> const& data) {
        // If the product has been enqueued to a different queue, make sure that it is available before accessing it
        if (data.stream() != this->stream()) {
          // Different streams, check if the underlying device is the same
          if (data.device() != this->device()) {
            // Eventually replace with prefetch to current device (assuming unified memory works)
            // If we won't go to unified memory, need to figure out something else...
            throw std::runtime_error("Handling data from multiple devices is not yet supported");
          }
          // If the data product is not yet available, synchronize the two streams
          if (not data.isAvailable()) {
            // Event not yet occurred, so need to add synchronization
            // here. Sychronization is done by making the current queue
            // wait for an event, so all subsequent work in the stream
            // will run only after the event has "occurred" (i.e. data
            // product became available).
            alpaka::wait(this->stream(), data.event());
          }
        }
      }
    };

    class ScopedContextHolderHelper {
    public:
      ScopedContextHolderHelper(edm::WaitingTaskWithArenaHolder waitingTaskHolder)
          : waitingTaskHolder_{std::move(waitingTaskHolder)} {}

      template <typename F, typename TQueue>
      void pushNextTask(F&& f, ContextState<TQueue> const* state) {
        replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder{edm::make_waiting_task_with_holder(
            std::move(waitingTaskHolder_), [state, func = std::forward<F>(f)](edm::WaitingTaskWithArenaHolder h) {
              func(ScopedContextTask{state, std::move(h)});
            })});
      }

      void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
        waitingTaskHolder_ = std::move(waitingTaskHolder);
      }

      template <typename TQueue>
      void enqueueCallback(TQueue& stream) {
        alpaka::enqueue(stream, alpaka::HostOnlyTask([holder = std::move(waitingTaskHolder_)]() {
                          // The functor is required to be const, but the original waitingTaskHolder_
                          // needs to be notified...
                          const_cast<edm::WaitingTaskWithArenaHolder&>(holder).doneWaiting(nullptr);
                        }));
      }

    private:
      edm::WaitingTaskWithArenaHolder waitingTaskHolder_;
    };
  }  // namespace impl

  /**
     * The aim of this class is to do necessary per-event "initialization" in ExternalWork acquire():
     * - setting the current device
     * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
     * - synchronizing between CUDA streams if necessary
     * and enforce that those get done in a proper way in RAII fashion.
     */
  template <typename TQueue>
  class ScopedContextAcquire : public impl::ScopedContextGetterBase<TQueue> {
  public:
    using Queue = TQueue;
    using ScopedContextGetterBase = impl::ScopedContextGetterBase<Queue>;
    using ScopedContextGetterBase::stream;
    using ScopedContextGetterBase::streamPtr;

    /// Constructor to create a new CUDA stream (no need for context beyond acquire())
    explicit ScopedContextAcquire(edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : ScopedContextGetterBase(streamID), holderHelper_{std::move(waitingTaskHolder)} {}

    // /// Constructor to create a new CUDA stream, and the context is needed after acquire()
    explicit ScopedContextAcquire(edm::StreamID streamID,
                                  edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                                  ContextState<Queue>& state)
        : ScopedContextGetterBase(streamID), holderHelper_{std::move(waitingTaskHolder)}, contextState_{&state} {}

    // /// Constructor to (possibly) re-use a CUDA stream (no need for context beyond acquire())
    explicit ScopedContextAcquire(ProductBase<Queue> const& data, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : ScopedContextGetterBase(data), holderHelper_{std::move(waitingTaskHolder)} {}

    // /// Constructor to (possibly) re-use a CUDA stream, and the context is needed after acquire()
    explicit ScopedContextAcquire(ProductBase<Queue> const& data,
                                  edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                                  ContextState<Queue>& state)
        : ScopedContextGetterBase(data), holderHelper_{std::move(waitingTaskHolder)}, contextState_{&state} {}

    ~ScopedContextAcquire() {
      holderHelper_.enqueueCallback(stream());
      if (contextState_) {
        contextState_->set(streamPtr());
      }
    }

    template <typename F>
    void pushNextTask(F&& f) {
      if (contextState_ == nullptr)
        throwNoState();
      holderHelper_.pushNextTask(std::forward<F>(f), contextState_);
    }

    void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
      holderHelper_.replaceWaitingTaskHolder(std::move(waitingTaskHolder));
    }

  private:
    void throwNoState() {
      throw std::runtime_error(
          "Calling ScopedContextAcquire::insertNextTask() requires ScopedContextAcquire to be constructed with "
          "ContextState, but that was not the case");
    }

    impl::ScopedContextHolderHelper holderHelper_;
    ContextState<Queue>* contextState_ = nullptr;
  };

  /**
     * The aim of this class is to do necessary per-event "initialization" in ExternalWork produce() or normal produce():
     * - setting the current device
     * - synchronizing between CUDA streams if necessary
     * and enforce that those get done in a proper way in RAII fashion.
     */
  template <typename TQueue>
  class ScopedContextProduce : public impl::ScopedContextGetterBase<TQueue> {
  public:
    using Queue = TQueue;
    using Event = alpaka::Event<Queue>;
    using ScopedContextGetterBase = impl::ScopedContextGetterBase<Queue>;
    using ScopedContextGetterBase::device;
    using ScopedContextGetterBase::stream;
    using ScopedContextGetterBase::streamPtr;

    /// Constructor to re-use the CUDA stream of acquire() (ExternalWork module)
    explicit ScopedContextProduce(ContextState<Queue>& state)
        : ScopedContextGetterBase(state.releaseStreamPtr()), event_{getEventCache<Event>().get(device())} {}

    explicit ScopedContextProduce(ProductBase<Queue> const& data)
        : ScopedContextGetterBase(data), event_{getEventCache<Event>().get(device())} {}

    explicit ScopedContextProduce(edm::StreamID streamID)
        : ScopedContextGetterBase(streamID), event_{getEventCache<Event>().get(device())} {}

    /// Record the event, all asynchronous work must have been queued before the destructor
    ~ScopedContextProduce() {
      // FIXME: this may throw an execption if the underlaying call fails.
      alpaka::enqueue(stream(), *event_);
    }

    template <typename T>
    std::unique_ptr<Product<Queue, T>> wrap(T data) {
      // make_unique doesn't work because of private constructor
      return std::unique_ptr<Product<Queue, T>>(new Product<Queue, T>(streamPtr(), std::move(data)));
    }

    template <typename T, typename... Args>
    auto emplace(edm::Event& iEvent, edm::EDPutTokenT<Product<Queue, T>> token, Args&&... args) {
      return iEvent.emplace(token, streamPtr(), event_, std::forward<Args>(args)...);
    }

  private:
    friend class ::cms::alpakatest::TestScopedContext;

    explicit ScopedContextProduce(std::shared_ptr<Queue> stream)
        : ScopedContextGetterBase(std::move(stream)), event_{getEventCache<Event>().get(device())} {}

    std::shared_ptr<Event> event_;
  };

  /**
     * The aim of this class is to do necessary per-task "initialization" tasks created in ExternalWork acquire():
     * - setting the current device
     * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
     * and enforce that those get done in a proper way in RAII fashion.
     */
  template <typename TQueue>
  class ScopedContextTask : public impl::ScopedContextBase<TQueue> {
  public:
    using Queue = TQueue;
    using ScopedContextBase = impl::ScopedContextBase<Queue>;
    using ScopedContextBase::stream;
    using ScopedContextBase::streamPtr;

    /// Constructor to re-use the CUDA stream of acquire() (ExternalWork module)
    explicit ScopedContextTask(ContextState<Queue> const* state, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : ScopedContextBase(state->streamPtr()),  // don't move, state is re-used afterwards
          holderHelper_{std::move(waitingTaskHolder)},
          contextState_{state} {}

    ~ScopedContextTask() { holderHelper_.enqueueCallback(stream()); }

    template <typename F>
    void pushNextTask(F&& f) {
      holderHelper_.pushNextTask(std::forward<F>(f), contextState_);
    }

    void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
      holderHelper_.replaceWaitingTaskHolder(std::move(waitingTaskHolder));
    }

  private:
    impl::ScopedContextHolderHelper holderHelper_;
    ContextState<Queue> const* contextState_;
  };

  /**
     * The aim of this class is to do necessary per-event "initialization" in analyze()
     * - setting the current device
     * - synchronizing between CUDA streams if necessary
     * and enforce that those get done in a proper way in RAII fashion.
     */
  template <typename TQueue>
  class ScopedContextAnalyze : public impl::ScopedContextGetterBase<TQueue> {
  public:
    using Queue = TQueue;
    using ScopedContextGetterBase = impl::ScopedContextGetterBase<Queue>;
    using ScopedContextGetterBase::stream;
    using ScopedContextGetterBase::streamPtr;

    /// Constructor to (possibly) re-use a CUDA stream
    explicit ScopedContextAnalyze(ProductBase<Queue> const& data) : ScopedContextGetterBase(data) {}
  };

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaCore_ScopedContext_h
