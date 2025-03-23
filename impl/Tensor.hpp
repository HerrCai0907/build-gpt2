#pragma once

#include "Dim.hpp"
#include "Storage.hpp"
#include "Typing.hpp"
#include <iostream>
#include <memory>

template <IMatrixStorage Storage, IDim... N> struct Tensor;
template <IMatrixStorage Storage, IDim N0> struct Tensor<Storage, N0> {
  using value_type = typename Storage::value_type;
  std::shared_ptr<Storage> m_stroage;

  N0 m_n0;
  Tensor(std::shared_ptr<Storage> stroage, N0 n0) : m_stroage(stroage), m_n0(n0) {}

  u32 get_total_size() const { return m_n0.dim(); }

  value_type const &get(u32 offset) const { return (*m_stroage)[offset]; }
  value_type &get(u32 offset) { return (*m_stroage)[offset]; }
  void dump_value() const {
    bool isDotDotDot0 = false;
    for (u32 i = 0; i < m_n0.dim(); i++) {
      if (i < 3 || i + 3 >= m_n0.dim()) {
        std::cout << get(i) << " ";
      } else if (!isDotDotDot0) {
        std::cout << "... ";
        isDotDotDot0 = true;
      }
    }
    std::cout << "\n";
  }
  void dump() const {
    std::cout << "size: (" << m_n0.dim() << ")\n";
    dump_value();
  }
};
template <IMatrixStorage Storage, IDim N0, IDim N1> struct Tensor<Storage, N0, N1> {
  using value_type = typename Storage::value_type;
  std::shared_ptr<Storage> m_stroage;

  N0 m_n0;
  N1 m_n1;

  Tensor(std::shared_ptr<Storage> stroage, N0 n0, N1 n1) : m_stroage(stroage), m_n0(n0), m_n1(n1) {}

  u32 get_total_size() const { return m_n0.dim() * m_n1.dim(); }

  value_type const &get(u32 offset1, u32 offset2) const { return (*m_stroage)[offset1 * m_n0.dim() + offset2]; }
  value_type &get(u32 offset1, u32 offset2) { return (*m_stroage)[offset1 * m_n0.dim() + offset2]; }
  void dump() const {
    std::cout << "size: (" << m_n0.dim() << "," << m_n1.dim() << ")\n";
    bool isDotDotDot0 = false;
    for (u32 i = 0; i < m_n0.dim(); i++) {
      if (i < 3 || i + 3 >= m_n0.dim()) {
        Tensor<Slice<Storage>, N1> const v{Slice<Storage>::create(m_stroage, i * m_n1.dim(), m_n1.dim()), m_n1};
        v.dump_value();
      } else if (!isDotDotDot0) {
        std::cout << "...\n";
        isDotDotDot0 = true;
      }
    }
  }
};
template <IMatrixStorage Storage, IDim N0, IDim N1, IDim N2> struct Tensor<Storage, N0, N1, N2> {
  using value_type = typename Storage::value_type;
  std::shared_ptr<Storage> m_stroage;

  N0 m_n0;
  N1 m_n1;
  N2 m_n2;

  Tensor(std::shared_ptr<Storage> stroage, N0 n0, N1 n1, N2 n2) : m_stroage(stroage), m_n0(n0), m_n1(n1), m_n2(n2) {}

  u32 get_total_size() const { return m_n0.dim() * m_n1.dim() * m_n2.dim(); }

  value_type const &get(u32 offset1, u32 offset2, u32 offset3) const {
    return (*m_stroage)[(offset1 * m_n0.dim() + offset2) * m_n1.dim() + offset3];
  }
  value_type &get(u32 offset1, u32 offset2, u32 offset3) {
    return (*m_stroage)[(offset1 * m_n0.dim() + offset2) * m_n1.dim() + offset3];
  }
};
template <IMatrixStorage Storage, IDim N0, IDim N1, IDim N2, IDim N3> struct Tensor<Storage, N0, N1, N2, N3> {
  using value_type = typename Storage::value_type;
  std::shared_ptr<Storage> m_stroage;

  N0 m_n0;
  N1 m_n1;
  N2 m_n2;
  N3 m_n3;

  Tensor(std::shared_ptr<Storage> stroage, N0 n0, N1 n1, N2 n2, N3 n3)
      : m_stroage(stroage), m_n0(n0), m_n1(n1), m_n2(n2), m_n3(n3) {}

  u32 get_total_size() const { return m_n0.dim() * m_n1.dim() * m_n2.dim() * m_n3.dim(); }

  value_type const &get(u32 offset1, u32 offset2, u32 offset3, u32 offset4) const {
    return (*m_stroage)[((offset1 * m_n0.dim() + offset2) * m_n1.dim() + offset3) * m_n2.dim() + offset4];
  }
  value_type &get(u32 offset1, u32 offset2, u32 offset3, u32 offset4) {
    return (*m_stroage)[((offset1 * m_n0.dim() + offset2) * m_n1.dim() + offset3) * m_n2.dim() + offset4];
  }
};
