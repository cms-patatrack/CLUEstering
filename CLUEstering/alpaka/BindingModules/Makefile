
export CXX := g++
export CUDA := nvcc
CXXFLAGS = -std=c++17 -g
ALPAKA_SERIAL_FLAGS = -DALPAKA_HOST_ONLY -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND
ALPAKA_TBB_FLAGS = -DALPAKA_ACC_CPU_B_TBB_T_SEQ_PRESENT -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND
ALPAKA_PATH = /install/include
BOOST_PATH = /usr/install/boost
TBB_FLAGS = -ltbb

# Binding flags
PYTHON_VERS := $(shell python3 -V | awk -F '' '{print $$8$$9$$10$$11}' | sed 's/\.//g'  )
PYTHON_VERS_DOT := $(shell python3 -V | awk -F '' '{print $$8$$9$$10$$11}' )
PYTHON_PATH := /usr/include/python$(PYTHON_VERS_DOT)
PYBIND_PATH := ${HOME}/.local/lib/python$(PYTHON_VERS_DOT)/site-packages/pybind11/include
BINDING_FLAGS := -Wall -shared -fPIC -I$(PYTHON_PATH) -I$(PYBIND_PATH)
SERIAL_MODULE_NAME := CLUE_CPU_Serial.cpython-$(PYTHON_VERS)-x86_64-linux-gnu.so
TBB_MODULE_NAME := CLUE_CPU_TBB.cpython-$(PYTHON_VERS)-x86_64-linux-gnu.so

all:
	$(CXX) $(CXXFLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_SERIAL_FLAGS) $(BINDING_FLAGS) binding_cpu.cc -o $(SERIAL_MODULE_NAME)
	$(CXX) $(CXXFLAGS) $(TBB_FLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_TBB_FLAGS) $(BINDING_FLAGS) binding_cpu_tbb.cc -o $(TBB_MODULE_NAME)

serial:
	$(CXX) $(CXXFLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_SERIAL_FLAGS) $(BINDING_FLAGS) binding_cpu.cc -o $(SERIAL_MODULE_NAME)

tbb:
	$(CXX) $(CXXFLAGS) $(TBB_FLAGS) -I$(BOOST_PATH) -I$(ALPAKA_PATH) $(ALPAKA_TBB_FLAGS) $(BINDING_FLAGS) binding_cpu_tbb.cc -o $(TBB_MODULE_NAME)

# nvcc -x cu --compiler-options '-fPIC' --expt-relaxed-constexpr -O3 -shared -std=c++17 -DALPAKA_ACC_GPU_CUDA_ENABLED -I/usr/local/boost/include -I/usr/local/alpaka/include $(python3 -m pybind11 --includes) helloWorld.cu -o 
