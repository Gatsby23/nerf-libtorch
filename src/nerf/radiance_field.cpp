#include "radiance_field.h"

namespace nerf {
void InitWeights(torch::nn::Module& module) {
    torch::NoGradGuard no_grad;
    if (auto* linear = module.as<torch::nn::Linear>()) {
        torch::nn::init::kaiming_normal_(linear->weight);
        torch::nn::init::zeros_(linear->bias);
    }
}
RadianceFieldImpl::RadianceFieldImpl(const int64_t length_x,
                                     const int64_t length_dim)
    : length_x_(length_x),
      length_dim_(length_dim),
      layer_0_(torch::nn::LinearOptions(6 * length_x, 256)),
      layer_1_(torch::nn::LinearOptions(256, 256)),
      layer_2_(torch::nn::LinearOptions(256, 256)),
      layer_3_(torch::nn::LinearOptions(256, 256)),
      layer_4_(torch::nn::LinearOptions(256, 256)),
      layer_5_(torch::nn::LinearOptions(256 + 6 * length_x, 256)),
      layer_6_(torch::nn::LinearOptions(256, 256)),
      layer_7_(torch::nn::LinearOptions(256, 256)),
      layer_8_(torch::nn::LinearOptions(256, 256)),
      layer_9_(torch::nn::LinearOptions(256 + 6 * length_dim, 128)),
      layer_10_(torch::nn::LinearOptions(128, 128)),
      layer_11_(torch::nn::LinearOptions(128, 128)),
      layer_12_(torch::nn::LinearOptions(128, 128)),
      sigma_(torch::nn::LinearOptions(256, 1)),
      rgb_(torch::nn::LinearOptions(128, 3)) {
    register_module("layer_0_", layer_0_);
    register_module("layer_1_", layer_1_);
    register_module("layer_2_", layer_2_);
    register_module("layer_3_", layer_3_);
    register_module("layer_4_", layer_4_);
    register_module("layer_5_", layer_5_);
    register_module("layer_6_", layer_6_);
    register_module("layer_7_", layer_7_);
    register_module("layer_8_", layer_8_);
    register_module("layer_9_", layer_9_);
    register_module("layer_10_", layer_10_);
    register_module("layer_11_", layer_11_);
    register_module("layer_12_", layer_12_);
    register_module("sigma_", sigma_);
    register_module("rgb_", rgb_);

    this->apply(InitWeights);
}

std::tuple<torch::Tensor, torch::Tensor> RadianceFieldImpl::forward(
    const torch::Tensor& x, const torch::Tensor& d) {
    const auto e_x = embedder_.Embedding(x, length_x_);
    const auto e_d = embedder_.Embedding(d, length_dim_);
    auto h = torch::relu(layer_0_(e_x));
    h = torch::relu(layer_1_(h));
    h = torch::relu(layer_2_(h));
    h = torch::relu(layer_3_(h));
    h = torch::relu(layer_4_(h));
    h = torch::cat({h, e_x}, 1);
    h = torch::relu(layer_5_(h));
    h = torch::relu(layer_6_(h));
    h = torch::relu(layer_7_(h));
    auto sigma = torch::relu(sigma_(h));
    h = torch::relu(layer_8_(h));
    h = torch::cat({h, e_d}, 1);
    h = torch::relu(layer_9_(h));
    h = torch::relu(layer_10_(h));
    h = torch::relu(layer_11_(h));
    h = torch::relu(layer_12_(h));
    auto rgb = torch::sigmoid(rgb_(h));
    return {rgb, sigma};
}
}  // namespace nerf
