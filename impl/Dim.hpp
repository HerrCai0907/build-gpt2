#pragma once

#include "Typing.hpp"
#include <concepts>

template <class T>
concept IDim = requires(T const &t) {
  { t.dim() } -> std::same_as<u32>;
};
template <class T>
concept IStaticDim = requires() {
  { T::static_dim() } -> std::same_as<u32>;
};

template <u32 N> struct StaticDim {
  static_assert(N != 0, "dim must be greater than 0");
  constexpr u32 dim() const { return N; }
  static constexpr u32 static_dim() { return N; }
};
static_assert(IDim<StaticDim<1>>, "");
static_assert(IStaticDim<StaticDim<1>>, "");
struct DynDim {
  u32 m_dim;
  u32 dim() const { return m_dim; }
};
static_assert(IDim<DynDim>, "");
