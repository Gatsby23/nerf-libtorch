#include "ray.h"
using namespace torch::indexing;

static const torch::Tensor epsilon = torch::tensor(1e-16f).to(torch::kCUDA);

namespace nerf {
torch::Tensor PcPdf(torch::Tensor& partitions, torch::Tensor& weights,
                    const int64_t n_s) {
    const auto batch_size = weights.size(0);
    const auto n_p = weights.size(1);
    weights = weights.clamp(epsilon, weights.max());
    weights /= weights.sum(1, true);
    auto sample =
        std::get<0>(torch::rand({batch_size, n_s},
                                torch::TensorOptions().device(torch::kCUDA))
                        .sort(1));
    auto a = (partitions.index({Slice(), Slice(1)}) -
              partitions.index({Slice(), Slice(None, -1)})) /
             weights;
    auto cum_weights = torch::cumsum(weights, 1);
    cum_weights = torch::pad(cum_weights, {1, 0, 0, 0});
    const auto b = partitions.index({Slice(), Slice(None, -1)}) -
                   a * cum_weights.index({Slice(), Slice(None, -1)});
    auto ret_sample =
        torch::zeros_like(sample, torch::TensorOptions().device(torch::kCUDA));

    for (size_t j = 0; j < n_p; j++) {
        const auto min_j = cum_weights.index({Slice(), Slice(j, j + 1)});
        const auto max_j = cum_weights.index({Slice(), Slice(j + 1, j + 2)});
        const auto a_j = a.index({Slice(), Slice(j, j + 1)});
        const auto b_j = b.index({Slice(), Slice(j, j + 1)});
        const auto mask =
            torch::where((min_j <= sample) & (sample < max_j), 1.0f, 0.0f);
        ret_sample += (a_j * sample + b_j) * mask;
    }
    return ret_sample;
}

torch::Tensor Ray(const torch::Tensor& o, const torch::Tensor& d,
                  const torch::Tensor& t) {
    return o.index({Slice(), None}) +
           t.index({"...", None}) * d.index({Slice(), None});
}

std::tuple<torch::Tensor, torch::Tensor> RgbAndWeight(RadianceField module,
                                                      const torch::Tensor& o,
                                                      const torch::Tensor& d,
                                                      const torch::Tensor& t,
                                                      const int64_t n) {
    const auto batch_size = o.size(0);
    const auto x_tmp = Ray(o, d, t).view({batch_size * n, -1});
    const auto d_tmp =
        d.index({Slice(), None}).repeat({1, n, 1}).view({batch_size * n, -1});
    auto [rgb, sigma] = module->forward(x_tmp, d_tmp);
    rgb = rgb.view({batch_size, n, -1});
    sigma = sigma.view({batch_size, n, -1});
    auto delta = torch::pad(
        t.index({Slice(), Slice(1)}) - t.index({Slice(), Slice(None, -1)}),
        {0, 1}, "constant", 1e8f);
    auto mass = sigma.index({"...", 0}) * delta;
    mass = torch::pad(mass, {1, 0}, "constant", 0.0f);
    const auto alpha = 1.0f - torch::exp(-mass.index({Slice(), Slice(1)}));
    const auto t_tmp =
        torch::exp(-torch::cumsum(mass.index({Slice(), Slice(None, -1)}), 1));
    const auto w = t_tmp * alpha;
    return {rgb, w};
}

torch::Tensor SampleCoarse(const torch::Tensor& partitions) {
    const auto batch_size = partitions.size(0);
    const auto& min = partitions.index({Slice(), Slice(None, -1)});
    const auto& max = partitions.index({Slice(), Slice(1)});
    auto samples = torch::rand({batch_size, min.size(1)},
                               torch::TensorOptions().device(torch::kCUDA));
    samples = min + samples * (max - min);
    return samples;
}

torch::Tensor SampleFine(torch::Tensor& partitions, torch::Tensor& weights,
                         const torch::Tensor& t_c, const int64_t n_f) {
    const auto t_f = PcPdf(partitions, weights, n_f);
    auto concat_sample = std::get<0>(torch::cat({t_c, t_f}, 1).sort(1));
    return concat_sample;
}

torch::Tensor SplitRay(const float t_n, const float t_f, const int64_t n,
                       int64_t batch_size) {
    auto partitions =
        torch::linspace(t_n, t_f, n + 1,
                        torch::TensorOptions().device(torch::kCUDA))
            .unsqueeze(0);
    partitions = partitions.repeat({batch_size, 1});
    return partitions;
}
}  // namespace nerf