
#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <hip/hip_runtime.h>

namespace test {

  [[noreturn]] inline void abortOnError(const char* file,
                                        int line,
                                        const char* cmd,
                                        const char* error,
                                        const char* message,
                                        std::string_view description = std::string_view()) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "HIP_CHECK(" << cmd << ");\n";
    out << error << ": " << message << "\n";
    if (!description.empty())
      out << description << "\n";
    throw std::runtime_error(out.str());
  }

  inline bool hipCheck(const char* file,
                       int line,
                       const char* cmd,
                       hipError_t result,
                       std::string_view description = std::string_view()) {
    if (result == hipSuccess)
      return true;

    const char* error = hipGetErrorName(result);
    const char* message = hipGetErrorString(result);
    abortOnError(file, line, cmd, error, message, description);
    return false;
  }
}  // namespace test

#define HIP_CHECK(ARG, ...) (test::hipCheck(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))
