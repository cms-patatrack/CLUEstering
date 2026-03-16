/// @file CLUEstering.hpp
/// @brief Header file for the CLUEstering library.
/// @author Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#if not defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) and  \
    not defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED) and \
    not defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED) and  \
    not defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#if defined(__CUDACC__) and not defined(ALPAKA_HOST_ONLY)
#define ALPAKA_ACC_GPU_CUDA_ENABLED
#elif defined(__HIPCC__) and not defined(ALPAKA_HOST_ONLY)
#define ALPAKA_ACC_GPU_HIP_ENABLED
#elif defined(_OPENMP)
#define ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#else
#define ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#endif
#endif

#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/PointsConversion.hpp"
#include "CLUEstering/utils/read_csv.hpp"
#include "CLUEstering/utils/cluster_centroid.hpp"
#include "CLUEstering/utils/get_clusters.hpp"
#include "CLUEstering/utils/get_device.hpp"
#include "CLUEstering/utils/get_queue.hpp"
#include "CLUEstering/utils/scores.hpp"
