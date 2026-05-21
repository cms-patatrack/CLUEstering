
#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#define backend serial
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define backend cuda
#endif
