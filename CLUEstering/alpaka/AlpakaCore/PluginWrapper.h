#ifndef AlpakaDataFormats_Plugin_Wrapper_h
#define AlpakaDataFormats_Plugin_Wrapper_h

namespace cms {
  namespace alpakatools {
    template <typename T, typename P>
    class PluginWrapper {
    public:
      template <typename... Args>
      explicit PluginWrapper(Args&&... args) : obj_{std::forward<Args>(args)...} {}
      T const& get() const { return obj_; }

    private:
      T obj_;
    };

  }  // namespace alpakatools
}  // namespace cms

#endif