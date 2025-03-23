#include "Dim.hpp"
#include "MMapFile.hpp"
#include "Tensor.hpp"
#include "Typing.hpp"

constexpr u32 n_embd = 768;
constexpr u32 n_head = 12;
constexpr u32 n_layer = 12;

constexpr u32 vocab_size = 50257;
constexpr u32 block_size = 1024;

template <IStaticDim Num, IStaticDim EmbededDim> struct EmbeddedOp {
  Tensor<MMapFile, Num, EmbededDim> m_weight;
  explicit EmbeddedOp(std::string const &name) : m_weight(MMapFile::load(name + ".weight"), Num{}, EmbededDim{}) {}

  template <IMatrixStorage Storage, IDim N0>
  Tensor<VecStorage<DT>, N0, EmbededDim> execute(Tensor<Storage, N0> const &input) const {
    Tensor<VecStorage<DT>, N0, EmbededDim> output{
        VecStorage<DT>::create(input.get_total_size() * EmbededDim::static_dim()),
        input.m_n0,
        EmbededDim{},
    };
    for (u32 i = 0; i < input.get_total_size(); ++i)
      for (u32 j = 0; j < EmbededDim::static_dim(); ++j)
        output.get(i, j) = m_weight.get(input.get(i), j);
    return output;
  }
};

struct AddOp {
  template <IMatrixStorage S1, IMatrixStorage S2, IDim N0, IDim N1>
  static Tensor<VecStorage<DT>, N0, N1> execute(Tensor<S1, N0, N1> const &a, Tensor<S2, N0, N1> const &b) {
    u32 const size = a.get_total_size();
    Tensor<VecStorage<DT>, N0, N1> output{VecStorage<DT>::create(size), a.m_n0, a.m_n1};
    for (u32 i = 0; i < a.m_n0.dim(); i++)
      for (u32 j = 0; j < a.m_n1.dim(); j++)
        output.get(i, j) = a.get(i, j) + b.get(i, j);
    return output;
  }
};

// // norm on last dim
// struct LayerNorm {
//   std::unique_ptr<StaticMatrix> m_weight;
//   std::unique_ptr<StaticMatrix> m_bias;
//   constexpr static DT esp = 1e-5;
//   LayerNorm(u32 embd, std::string const &name)
//       : m_weight(load({embd}, name + ".weight")),
//         m_bias(load({embd}, name + ".bias")) {}

//   template <u32 Dim>
//   Matrix<DT, Dim> execute(Matrix<DT, Dim> const &inputs) const {
//     Matrix<DT, Dim> output{};
//     output.m_size = inputs.m_size;
//     output.m_data.resize(inputs.m_data.size());
//     const u32 D = m_weight->m_size[0];
//     u32 k = inputs.m_data.size() / D;
//     for (u32 i = 0; i < k; i++) {
//       DT sum = 0.0;
//       for (u32 j = 0; j < D; j++) {
//         sum += inputs.get(i, j);
//       }
//       DT const mean = sum / D;
//       DT var_sum = 0.0;
//       for (u32 j = 0; j < D; j++) {
//         float const diff = inputs.get(i, j) - mean;
//         var_sum += diff * diff;
//       }
//       DT const var = var_sum / D;

//       for (u32 j = 0; j < D; j++) {
//         float normalized = (inputs.get(i, j) - mean) / sqrtf(var + esp);
//         output.get(i, j) = normalized * m_weight->get(j) + m_bias->get(j);
//       }
//     }
//     return output;
//   }
// };

// struct Linear {
//   std::unique_ptr<StaticMatrix> m_weight;
//   std::unique_ptr<StaticMatrix> m_bias;
//   constexpr static DT esp = 1e-5;
//   Linear(u32 embd, std::string const &name)
//       : m_weight(load({embd}, name + ".weight")),
//         m_bias(load({embd}, name + ".bias")) {}

//   template <u32 Dim> void execute(Matrix<DT, Dim> const &inputs) const {}
// };

// struct Block {};

int main() {
  Tensor<VecStorage<u32>, DynDim> inputs{VecStorage<u32>::create(7), DynDim{7}};
  inputs.m_stroage->m_data = {15496, 11, 314, 1101, 257, 3303, 2746, 11};
  Tensor<VecStorage<u32>, DynDim> positions{VecStorage<u32>::create(7), DynDim{7}};
  positions.m_stroage->m_data = {0, 1, 2, 3, 4, 5, 6, 7};

  EmbeddedOp<StaticDim<block_size>, StaticDim<n_embd>> wte{"models/transformer.wte"};
  EmbeddedOp<StaticDim<vocab_size>, StaticDim<n_embd>> wpe{"models/transformer.wpe"};
  wte.execute(inputs).dump();
  // LayerNorm ln_1{n_embd, "models/transformer.h.0.ln_1"};
  // ln_1.execute(R).dump();
}
