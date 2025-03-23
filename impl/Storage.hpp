#pragma once

#include "Typing.hpp"
#include <concepts>
#include <memory>
#include <type_traits>
#include <vector>

template <class T>
concept IMatrixStorage = requires(T &t, T const &constT, u32 index) {
  typename T::value_type;
  { constT[index] } -> std::same_as<std::add_lvalue_reference_t<std::add_const_t<typename T::value_type>>>;
  { t[index] } -> std::same_as<std::add_lvalue_reference_t<typename T::value_type>>;
};

template <class T> struct VecStorage {
  std::vector<T> m_data;
  explicit VecStorage(u32 n) : m_data() { m_data.resize(n); }
  using value_type = T;
  T const &operator[](u32 i) const { return m_data[i]; }
  T &operator[](u32 i) { return m_data[i]; }

  static std::shared_ptr<VecStorage> create(u32 size) { return std::make_shared<VecStorage>(size); }
};

template <IMatrixStorage Storage> struct Slice {
  std::shared_ptr<Storage> storage_;
  u32 begin_;
  u32 size_;
  explicit Slice(std::shared_ptr<Storage> storage, u32 begin, u32 size)
      : storage_(storage), begin_(begin), size_(size) {}
  using value_type = typename Storage::value_type;
  value_type const &operator[](u32 i) const { return (*storage_)[i + begin_]; }
  value_type &operator[](u32 i) { return (*storage_)[i + begin_]; }
  static std::shared_ptr<Slice> create(std::shared_ptr<Storage> storage, u32 begin, u32 size) {
    return std::make_shared<Slice>(storage, begin, size);
  }
};

static_assert(IMatrixStorage<Slice<VecStorage<int>>>, "");
