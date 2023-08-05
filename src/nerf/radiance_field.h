#ifndef RADIANCE_FIELD_H
#define RADIANCE_FIELD_H

#include "stdafx.h"
//
#include "positional_encoding.h"

namespace nerf {
class RadianceFieldImpl : public torch::nn::Module {
   public:
    RadianceFieldImpl(const int64_t length_x = 10,
                      const int64_t length_dim = 4);
    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x,
                                                     const torch::Tensor& d);

   private:
    nerf::Embedder embedder_;

    // torch nn modules
    const int64_t length_x_, length_dim_;
    torch::nn::Linear layer_0_, layer_1_, layer_2_, layer_3_, layer_4_,
        layer_5_, layer_6_, layer_7_, layer_8_, layer_9_, layer_10_, layer_11_,
        layer_12_, rgb_, sigma_;
};
TORCH_MODULE(RadianceField);
}  // namespace nerf

#endif