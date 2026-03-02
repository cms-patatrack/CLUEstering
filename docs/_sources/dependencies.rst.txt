Supported Backends and Dependencies
===================================

Supported Backends
------------------

CLUEstering uses the Alpaka library to achieve performance portability across many backends without code duplication. Backends are activated by passing specific flags. The currently supported flags are:

- **Serial CPU**: ``ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED``
- **Parallel CPU with TBB**: ``ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED``
- **Parallel CPU with OpenMP**: ``ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED``
- **CUDA GPUs**: ``ALPAKA_ACC_GPU_CUDA_ENABLED``
- **AMD GPUs**: ``ALPAKA_ACC_GPU_HIP_ENABLED``
- **Intel CPU with SYCL**: ``ALPAKA_ACC_SYCL_ENABLED`` + ``ALPAKA_SYCL_ONEAPI_CPU``
- **Intel GPU with SYCL**: ``ALPAKA_ACC_SYCL_ENABLED`` + ``ALPAKA_SYCL_ONEAPI_GPU``

In addition to passing Alpaka flags, **non-CPU backends** require the appropriate compiler and compilation flags.

Dependencies
------------

CLUEstering requires:

- **Alpaka** version 1.2.0  
- **C++20 compiler**  
- **Boost** version 1.78.0 or later  

Additional dependencies depending on the chosen backend:

- **Serial backend**: no additional dependencies  
- **TBB backend**: TBB 2.2 or later  
- **OpenMP backend**: OpenMP 2.0 or later  
- **CUDA backend**: CUDA 12.0 or later  
- **HIP backend**: ROCm 6.0 or later  
- **SYCL backends**: oneAPI 2024.2 or later  

When installing CLUEstering as a Python library, shared libraries for each supported backend are automatically compiled. For building, the following tools are also required:

- **CMake** 3.16.0 or later  
- **setuptools** 42 or later  
- **wheel**  
- **pathlib**  

Additionally, the following Python libraries are required for backend functionality:

- **numpy**  
- **scikit-learn**  
- **pandas**  
- **matplotlib**  
