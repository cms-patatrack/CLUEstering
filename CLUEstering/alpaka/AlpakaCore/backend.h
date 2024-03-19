#ifndef AlpakaCore_backend_h
#define AlpakaCore_backend_h

enum class Backend { SERIAL, TBB, CUDA, HIP };

inline std::string const& name(Backend backend) {
  static const std::string names[] = {
      "serial_sync", "tbb_async", "cuda_async", "rocm_async"};
  return names[static_cast<int>(backend)];
}

template <typename T>
inline T& operator<<(T& out, Backend backend) {
  out << name(backend);
  return out;
}

#endif  // AlpakaCore_backend_h
