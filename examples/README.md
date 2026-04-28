# CLUEstering Examples

All examples cluster the same 2-D dataset (`data/sissa_1000.csv`, relative to the repository root) and print the cluster index assigned to each point.
They all share the same three algorithm parameters: `dc = 20`, `rhoc = 10`, `outlier = 20`.

## Prerequisites

CLUEstering must be installed before building any example.
Follow the [installation instructions](../README.md#installation) in the top-level README.
alpaka must also be available on your system.

---

## `basic` — CMake + alpaka backends

The simplest starting point.
Shows how to use CLUEstering through alpaka's portable backend abstraction.
The backend is selected at configure time via a CMake flag; the same `main.cpp` compiles for every target.

| Backend | CMake flag |
|---------|-----------|
| Serial CPU | `-DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED=ON` |
| TBB | `-DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED=ON` |
| OpenMP | `-DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED=ON` |
| CUDA | `-DALPAKA_ACC_GPU_CUDA_ENABLED=ON` |

```bash
cd examples/basic
cmake -B build -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED=ON
cmake --build build
./build/serial.out
```

Replace the flag (and the executable name) with any backend from the table above.

---

## `conan` — Conan package manager

Identical to the `basic` example in behaviour, but CLUEstering and alpaka are fetched through [Conan](https://conan.io/) instead of a manual system install.

```bash
cd examples/conan

# Install dependencies declared in conanfile.txt
conan install . --output-folder=build --build=missing

cmake -B build \
      -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake \
      -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED=ON
cmake --build build
./build/serial.out
```

The same backend flags listed in the `basic` table apply here.

---

## `cuda_native` — Native CUDA stream

Demonstrates how to wrap an existing `cudaStream_t` so that CLUEstering uses it directly, rather than creating its own internal stream.
This is useful when CLUEstering is embedded inside a larger CUDA application that already manages its own streams.

**Requires:** CUDA Toolkit.

```bash
cd examples/cuda_native
cmake -B build -DALPAKA_ACC_GPU_CUDA_ENABLED=ON
cmake --build build
./build/native_cuda.out
```

---

## `hip_native` — Native HIP stream

Same concept as `cuda_native`, but for AMD GPUs via HIP.
A `hipStream_t` is created externally and passed to CLUEstering.

**Requires:** ROCm / HIP.

```bash
cd examples/hip_native
cmake -B build -DALPAKA_ACC_GPU_HIP_ENABLED=ON
cmake --build build
./build/native_hip.out
```
