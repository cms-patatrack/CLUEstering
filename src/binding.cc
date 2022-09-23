#include <pybind11/pybind11.h>
#include "include/run.h"
#include <pybind11/stl.h> 

PYBIND11_MODULE(CLUEsteringCPP, m) {
    m.doc() = "Binding for CLUE";

	m.def("mainRun", &mainRun, "mainRun");
}
