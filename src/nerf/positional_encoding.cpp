#include "positional_encoding.h"

namespace nerf {
torch::Tensor Embedder::Embedding(
    const torch::Tensor& position,
    const int64_t length) {  // TODO: reduce precision & unsigned
    const auto batch_size = position.size(0);
    const auto idx = torch::arange(torch::Scalar(length))
                         .to(torch::kFloat32)
                         .unsqueeze(0)
                         .unsqueeze(0)
                         .to(torch::kCUDA);
    const auto position_tanh = torch::tanh(position).to(torch::kCUDA);
    const auto a = 2 * idx * std::numbers::pi * position_tanh.unsqueeze(2);
    const auto s = torch::sin(a);
    const auto c = torch::cos(a);
    return torch::cat({s, c}, 2).view({batch_size, -1});
}
}  // namespace nerf