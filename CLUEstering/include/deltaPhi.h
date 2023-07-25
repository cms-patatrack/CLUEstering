#include <algorithm>

template <typename T>
T deltaPhi(T phi1, T phi2, T x_min, T x_max) {
    T delta_phi{std::abs(phi2 - phi1)};
    return std::min(delta_phi, x_max - x_min - delta_phi);
}