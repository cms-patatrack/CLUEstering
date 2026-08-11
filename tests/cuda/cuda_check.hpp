
#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <cuda.h>
#include <cuda_runtime.h>

namespace test {

  [[noreturn]] inline void abortOnCudaError(const char* file,
                                            int line,
                                            const char* cmd,
                                            const char* error,
                                            const char* message,
                                            std::string_view description = std::string_view()) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "CUDA_CHECK(" << cmd << ");\n";
    out << error << ": " << message << "\n";
    if (!description.empty())
      out << description << "\n";
    throw std::runtime_error(out.str());
  }

  inline bool cudaCheck(const char* file,
                        int line,
                        const char* cmd,
                        CUresult result,
                        std::string_view description = std::string_view()) {
    if (result == CUDA_SUCCESS)
      return true;

    const char* error = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &error);
    cuGetErrorString(result, &message);
    abortOnCudaError(file, line, cmd, error, message, description);
    return false;
  }

  inline bool cudaCheck(const char* file,
                        int line,
                        const char* cmd,
                        cudaError_t result,
                        std::string_view description = std::string_view()) {
    if (result == cudaSuccess)
      return true;

    const char* error = cudaGetErrorName(result);
    const char* message = cudaGetErrorString(result);
    abortOnCudaError(file, line, cmd, error, message, description);
    return false;
  }
}  // namespace test

#define CUDA_CHECK(ARG, ...) (test::cudaCheck(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))
