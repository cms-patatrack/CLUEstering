
#pragma once

#include <concepts>
#include <tuple>
#include <iterator>
#include <type_traits>

namespace clue {
  namespace nostd {

    template <typename... Iterators>
    class zip_iterator {
    public:
      using value_type = std::tuple<typename std::iter_value_t<Iterators>...>;
      using difference_type = int;
      using pointer = void;
      using reference_type = std::tuple<typename std::iter_reference_t<Iterators>...>;
      using iterator_category = std::random_access_iterator_tag;

    private:
      std::tuple<Iterators...> m_iterators;

      template <std::size_t... Indexes>
      reference_type accessor_impl(std::size_t idx, std::index_sequence<Indexes...>) {
        return std::tie(std::get<Indexes>(m_iterators)[idx]...);
      }

      template <std::size_t... Indexes>
      value_type accessor_impl(std::size_t idx, std::index_sequence<Indexes...>) const {
        return std::make_tuple(std::get<Indexes>(m_iterators)[idx]...);
      }

      template <std::size_t... Indexes>
      reference_type dereference_impl(std::index_sequence<Indexes...>) {
        return std::tie(*std::get<Indexes>(m_iterators)...);
      }

      template <std::size_t... Indexes>
      value_type dereference_impl(std::index_sequence<Indexes...>) const {
        return std::make_tuple(*std::get<Indexes>(m_iterators)...);
      }

      template <std::size_t... Indexes>
      void advance_impl(int n, std::index_sequence<Indexes...>) {
        ((std::get<Indexes>(m_iterators) += n), ...);
      }

      template <std::size_t... Indexes>
      auto advance_impl(int n, std::index_sequence<Indexes...>) const {
        return std::make_tuple((std::get<Indexes>(m_iterators) + n)...);
      }

      template <std::size_t... Indexes>
      void increment_impl(std::index_sequence<Indexes...>) {
        (++std::get<Indexes>(m_iterators), ...);
      }

      template <std::size_t... Indexes>
      void decrement_impl(std::index_sequence<Indexes...>) {
        (--std::get<Indexes>(m_iterators), ...);
      }

    public:
      zip_iterator(Iterators... iterators) : m_iterators{std::make_tuple(iterators...)} {}
      zip_iterator(const std::tuple<Iterators...>& tuple) : m_iterators{tuple} {}

      reference_type operator[](std::size_t idx) {
        return accessor_impl(idx, std::index_sequence_for<Iterators...>());
      }
      value_type operator[](std::size_t idx) const {
        return accessor_impl(idx, std::index_sequence_for<Iterators...>());
      }
      reference_type operator*() {
        return dereference_impl(std::index_sequence_for<Iterators...>());
      }
      value_type operator*() const {
        return dereference_impl(std::index_sequence_for<Iterators...>());
      }

      zip_iterator& operator++() {
        increment_impl(std::index_sequence_for<Iterators...>());
        return *this;
      }
      zip_iterator& operator--() {
        decrement_impl(std::index_sequence_for<Iterators...>());
        return *this;
      }
      zip_iterator& operator+=(std::size_t increment) {
        advance_impl(increment, std::index_sequence_for<Iterators...>());
        return *this;
      }
      zip_iterator& operator-=(std::size_t decrement) {
        advance_impl(-decrement, std::index_sequence_for<Iterators...>());
        return *this;
      }
      template <typename... Tn>
      friend zip_iterator<Tn...> operator+(const zip_iterator<Tn...>& it, std::size_t increment);
      template <typename... Tn>
      friend zip_iterator<Tn...> operator-(const zip_iterator<Tn...>& it, std::size_t increment);
      template <typename... Tn>
      friend bool operator==(const zip_iterator<Tn...>& lhs, const zip_iterator<Tn...>& rhs);
    };

    template <typename... Iterators>
    zip_iterator<Iterators...> operator+(const zip_iterator<Iterators...>& it,
                                         std::size_t increment) {
      return zip_iterator<Iterators...>(
          it.advance_impl(increment, std::index_sequence_for<Iterators...>()));
    }
    template <typename... Iterators>
    zip_iterator<Iterators...> operator-(const zip_iterator<Iterators...>& it,
                                         std::size_t decrement) {
      return zip_iterator<Iterators...>(
          it.advance_impl(-decrement, std::index_sequence_for<Iterators...>()));
    }

    template <typename... Iterators>
    bool operator==(const zip_iterator<Iterators...>& lhs, const zip_iterator<Iterators...>& rhs) {
      return lhs.m_iterators == rhs.m_iterators;
    }

    template <typename... Iterators>
    auto zip(Iterators... iterators) {
      return zip_iterator<Iterators...>{iterators...};
    }

    auto zip_view();

  }  // namespace nostd
}  // namespace clue
