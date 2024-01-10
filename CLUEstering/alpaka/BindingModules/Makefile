
# Compilers
export CXX := g++
export CUDA := nvcc
export HIP := hipcc

export CUDA_ARCH := 50 60 61 62 70

# Compiler flags
CXX_FLAGS = -std=c++17 -g -O2 
export CUDA_FLAGS = -x cu --expt-relaxed-constexpr -gencode arch=compute_61,code=[sm_61,compute_61] -G 
# -Wno-deprecated-gpu-targets -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr --expt-extended-lambda --generate-line-info --source-in-ptx --display-error-number --cudart=shared --threads 1

# CUDA_FLAGS := $$(foreach ARCH,-gencode arch=compute_$$(CUDA_ARCH),code=[sm_$$(CUDA_ARCH),compute_$$(CUDA_ARCH)]) -Wno-deprecated-gpu-targets -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr --expt-extended-lambda --generate-line-info --source-in-ptx --display-error-number --cudart=shared
# $(2)NVCC_COMMON := -std=c++17 -O3 -g $$($(2)NVCC_FLAGS) -ccbin $(CXX) --compiler-options '$(HOST_CXXFLAGS) $(USER_CXXFLAGS)'
export HIP_BASE = /opt/rocm/
export HIP_FLAGS := -I$(HIP_BASE)/include \
					-I$(HIP_BASE)/hiprand/include \
					-I$(HIP_BASE)/rocrand/include

TBB_FLAGS = -ltbb

export CUDA_CXXFLAGS := -I$(CUDA_BASE)/include
# export CUDA_LDFLAGS := -L$(CUDA_LIBDIR) -lcudart -lcudadevrt
# export CUDA_NVCC := $(CUDA_BASE)/bin/nvcc

# Dependencies flags
ALPAKA_PATH = ../../../extern/alpaka/include
BOOST_PATH = /usr/include/boost

# Alpaka backend compilation flags
ALPAKA_SERIAL_FLAGS = -DALPAKA_HOST_ONLY -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND
ALPAKA_TBB_FLAGS = -DALPAKA_ACC_CPU_B_TBB_T_SEQ_PRESENT -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND
ALPAKA_CUDA_FLAGS = -DALPAKA_ACC_GPU_CUDA_PRESENT -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND 
ALPAKA_HIP_FLAGS = -DALPAKA_ACC_GPU_HIP_PRESENT -DALPAKA_ACC_GPU_HIP_ENABLED -DALPAKA_ACC_GPU_HIP_ASYNC_BACKEND

# Binding flags
PYTHON_VERS := $(shell python3 -V | awk -F '' '{print $$8$$9$$10$$11}' | sed 's/\.//g'  )
PYTHON_VERS_DOT := $(shell python3 -V | awk -F '' '{print $$8$$9$$10$$11}' )
PYTHON_PATH := /usr/include/python$(PYTHON_VERS_DOT)
# PYBIND_PATH := ${HOME}/.local/lib/python$(PYTHON_VERS_DOT)/site-packages/pybind11/include
PYBIND_PATH := ../../../extern/pybind11/include
BINDING_FLAGS := -Wall -shared -fPIC -I$(PYTHON_PATH) -I$(PYBIND_PATH)
CUDA_BINDING_FLAGS := -shared --compiler-options '-fPIC' -I$(PYTHON_PATH) -I$(PYBIND_PATH)

# Backend module names
KERNELS_MODULE_NAME := CLUE_Convolutional_Kernels.cpython-$(PYTHON_VERS)-x86_64-linux-gnu.so
SERIAL_MODULE_NAME := CLUE_CPU_Serial.cpython-$(PYTHON_VERS)-x86_64-linux-gnu.so
TBB_MODULE_NAME := CLUE_CPU_TBB.cpython-$(PYTHON_VERS)-x86_64-linux-gnu.so
CUDA_MODULE_NAME := CLUE_GPU_CUDA.cpython-$(PYTHON_VERS)-x86_64-linux-gnu.so

all:
	$(CXX) $(CXX_FLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_SERIAL_FLAGS) $(BINDING_FLAGS) binding_cpu.cc -o $(SERIAL_MODULE_NAME)
	$(CXX) $(CXX_FLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_SERIAL_FLAGS) $(BINDING_FLAGS) binding_kernels.cc -o $(KERNELS_MODULE_NAME)
	$(CXX) $(CXX_FLAGS) $(TBB_FLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_TBB_FLAGS) $(BINDING_FLAGS) binding_cpu_tbb.cc -o $(TBB_MODULE_NAME)
	$(CUDA) $(CUDA_FLAGS) $(CXX_FLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_CUDA_FLAGS) $(CUDA_BINDING_FLAGS) binding_gpu_cuda.cc -o $(CUDA_MODULE_NAME)
	$(HIP) $(HIP_FLAGS) $(CXX_FLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_HIP_FLAGS) $(BINDING_FLAGS) binding_gpu_hip.cc -o CLUE_GPU_HIP.cpython-$(PYTHON_VERS)-x86_64-linux-gnu.so

serial:
	$(CXX) $(CXX_FLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_SERIAL_FLAGS) $(BINDING_FLAGS) binding_cpu.cc -o $(SERIAL_MODULE_NAME)

tbb:
	$(CXX) $(CXX_FLAGS) $(TBB_FLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_TBB_FLAGS) $(BINDING_FLAGS) binding_cpu_tbb.cc -o $(TBB_MODULE_NAME)

cuda:
	$(CUDA) $(CUDA_FLAGS) $(CXX_FLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_CUDA_FLAGS) $(CUDA_BINDING_FLAGS) binding_gpu_cuda.cc -o $(CUDA_MODULE_NAME)

hip:
	$(HIP) $(HIP_FLAGS) $(CXX_FLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_HIP_FLAGS) $(BINDING_FLAGS) binding_gpu_hip.cc -o CLUE_GPU_HIP.cpython-$(PYTHON_VERS)-x86_64-linux-gnu.so

kernel:
	$(CXX) $(CXX_FLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_SERIAL_FLAGS) $(BINDING_FLAGS) binding_kernels.cc -o $(KERNELS_MODULE_NAME)
