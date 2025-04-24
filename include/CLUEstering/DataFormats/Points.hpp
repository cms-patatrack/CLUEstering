
#pragma once

#include <array>
#include <memory>
#include <vector>

template <uint8_t Ndim>
struct PointsSoA {
public:
  struct View {
    float* coords;
    float* weights;
    int* clusterIndexes;
    int* isSeed;
  };

  struct DebugInfo {
#ifdef CLUE_DEBUG
    std::vector<float> rho;
    std::vector<float> delta;
    std::vector<int> nearestHigher;
#endif

#ifdef CLUE_DEBUG
    DebugInfo(uint32_t n) : rho(n), delta(n), nearestHigher(n) {}
#else
    DebugInfo(uint32_t) {}
#endif
  };

  PointsSoA(float* floatBuffer, int* intBuffer, const uint32_t nPoints)
      : m_coordsBuffer{floatBuffer},
        m_resultsBuffer{intBuffer},
        m_view{std::make_unique<View>()},
        m_debugInfo{nPoints},
        m_nPoints{nPoints} {
    m_view->coords = floatBuffer;
    m_view->weights = floatBuffer + nPoints * Ndim;
    m_view->clusterIndexes = intBuffer;
    m_view->isSeed = intBuffer + nPoints;
  }

  PointsSoA(const PointsSoA&) = delete;
  PointsSoA& operator=(const PointsSoA&) = delete;
  PointsSoA(PointsSoA&&) = default;
  PointsSoA& operator=(PointsSoA&&) = default;
  ~PointsSoA() = default;

  uint32_t nPoints() const { return m_nPoints; }

  const float* coords() const { return m_view->coords; }
  const float* weights() const { return m_view->weights; }

  int* clusterIndexes() { return m_view->clusterIndexes; }
  const int* clusterIndexes() const { return m_view->clusterIndexes; }
  int* isSeed() { return m_view->isSeed; }
  const int* isSeed() const { return m_view->isSeed; }

  const View* view() const { return m_view.get(); }

private:
  float* m_coordsBuffer;
  int* m_resultsBuffer;
  std::unique_ptr<View> m_view;
  DebugInfo m_debugInfo;
  uint32_t m_nPoints;
};
