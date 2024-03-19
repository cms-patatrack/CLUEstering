
# Profiling
## How to compile
The profiling code can be compiled automatically for all the supported backends with CMake
using the command:
```bash
cmake -B build && make -C build
```
In the code there are some print-outs which give the execution times of different parts of
the execution before the algorithm starts. This can be enabled by adding a flag to the
previous command:
```bash
cmake -B build -DANNOTATE=ON && make -C build
```
