#ifndef Points_Alpaka_h
#define Points_Alpaka_h

#include <cstdint>
#include <memory>

#include "../../AlpakaCore/alpakaConfig.h"
#include "../../AlpakaCore/alpakaMemory.h"
#include "AlpakaVecArray.h"
#include "../Points.h"

using cms::alpakatools::VecArray;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <uint8_t Ndim>
  class PointsAlpaka {
  public:
    PointsAlpaka() = delete;
    explicit PointsAlpaka(Queue stream, int n_points)
        : coords{cms::alpakatools::make_device_buffer<VecArray<float, Ndim>[]>(stream,
                                                                               n_points)},
          weight{cms::alpakatools::make_device_buffer<float[]>(stream, n_points)},
          rho{cms::alpakatools::make_device_buffer<float[]>(stream, n_points)},
          delta{cms::alpakatools::make_device_buffer<float[]>(stream, n_points)},
          nearest_higher{cms::alpakatools::make_device_buffer<int[]>(stream, n_points)},
          cluster_index{cms::alpakatools::make_device_buffer<int[]>(stream, n_points)},
          is_seed{cms::alpakatools::make_device_buffer<int[]>(stream, n_points)},
          view_dev{cms::alpakatools::make_device_buffer<PointsAlpakaView>(stream)} {
      auto view_host = cms::alpakatools::make_host_buffer<PointsAlpakaView>(stream);
      view_host->coords = coords.data();
      view_host->weight = weight.data();
      view_host->rho = rho.data();
      view_host->delta = delta.data();
      view_host->nearest_higher = nearest_higher.data();
      view_host->cluster_index = cluster_index.data();
      view_host->is_seed = is_seed.data();

      // Copy memory inside the host view to device
      alpaka::memcpy(stream, view_dev, view_host);
    }
    // Copy constructor/assignment operator
    PointsAlpaka(const PointsAlpaka&) = delete;
    PointsAlpaka& operator=(const PointsAlpaka&) = delete;
    // Move constructor/assignment operator
    PointsAlpaka(PointsAlpaka&&) = default;
    PointsAlpaka& operator=(PointsAlpaka&&) = default;
    // Destructor
    ~PointsAlpaka() = default;

    cms::alpakatools::device_buffer<Device, VecArray<float, Ndim>[]> coords;
    cms::alpakatools::device_buffer<Device, float[]> weight;
    cms::alpakatools::device_buffer<Device, float[]> rho;
    cms::alpakatools::device_buffer<Device, float[]> delta;
    cms::alpakatools::device_buffer<Device, int[]> nearest_higher;
    cms::alpakatools::device_buffer<Device, int[]> cluster_index;
    cms::alpakatools::device_buffer<Device, int[]> is_seed;

    class PointsAlpakaView {
    public:
      VecArray<float, Ndim>* coords;
      float* weight;
      float* rho;
      float* delta;
      int* nearest_higher;
      int* cluster_index;
      int* is_seed;
    };

    PointsAlpakaView* view() { return view_dev.data(); }

  private:
    cms::alpakatools::device_buffer<Device, PointsAlpakaView> view_dev;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
