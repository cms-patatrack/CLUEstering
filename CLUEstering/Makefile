
# C++ compiler
export CXX := g++
CXXFLAGS = -std=c++17 -O3

# Binding flags
PYTHON_VERS := $(shell python3 -V | awk -F '' '{print $$8$$9$$10$$11}' | sed 's/\.//g'  )
PYTHON_VERS_DOT := $(shell python3 -V | awk -F '' '{print $$8$$9$$10$$11}' )
PYTHON_PATH := /usr/include/python$(PYTHON_VERS_DOT)
PYBIND_PATH := ${HOME}/.local/lib/python$(PYTHON_VERS_DOT)/site-packages/pybind11/include
BIND_FLAGS := -Wall -shared -fPIC -I$(PYTHON_PATH) -I$(PYBIND_PATH)
MODULE_NAME := CLUEsteringCPP.cpython-$(PYTHON_VERS)-x86_64-linux-gnu.so

serial: binding.cc ./include/*
	$(CXX) $(CXXFLAGS) $(BIND_FLAGS) ./binding.cc -o $(MODULE_NAME) && clear

clean:
	rm -f *.so
	clear
