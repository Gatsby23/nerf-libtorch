#include "volume_rendering.h"

#include "ray.h"
using namespace torch::indexing;
namespace nerf {

std::tuple<torch::Tensor, torch::Tensor> VolumeRenderingWithRadianceField(
    VolumeRenderingWithRadianceFieldParams& params) {
    const auto batch_size = params.o.size(0);
    auto partitions = SplitRay(params.t_n, params.t_f, params.n_c, batch_size);
    const auto bg =
        torch::from_blob(reinterpret_cast<void*>(params.c_bg.data()), {1, 3},
                         torch::kFloat)
            .to(torch::kCUDA);
    const auto t_c = SampleCoarse(partitions);
    auto [rgb_c, w_c] =
        RgbAndWeight(params.module_c, params.o, params.d, t_c, params.n_c);
    auto c_c = torch::sum(w_c.index({"...", None}) * rgb_c, 1);
    c_c += (1.0f - torch::sum(w_c, 1, true)) * bg;

    const auto t_f = SampleFine(partitions, w_c, t_c, params.n_f);
    auto [rgb_f, w_f] = RgbAndWeight(params.module_f, params.o, params.d, t_f,
                                     params.n_f + params.n_c);
    auto c_f = torch::sum(w_f.index({"...", None}) * rgb_f, 1);
    c_f += (1.0f - torch::sum(w_f, 1, true)) * bg;
    return {c_c, c_f};
}

}  // namespace nerf