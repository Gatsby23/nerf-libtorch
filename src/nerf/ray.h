#ifndef RAY_H
#define RAY_H
#include "stdafx.h"
//
#include "radiance_field.h"

namespace nerf {
torch::Tensor PcPdf(torch::Tensor& partitions, torch::Tensor& weights,
                    const int64_t n_s);
torch::Tensor Ray(const torch::Tensor& o, const torch::Tensor& d,
                  const torch::Tensor& t);
std::tuple<torch::Tensor, torch::Tensor> RgbAndWeight(
    RadianceField module, const torch::Tensor& o, const torch::Tensor& d,
    const torch::Tensor& t,
    const int64_t n);  // TODO: function name
torch::Tensor SampleCoarse(const torch::Tensor& partitions);
torch::Tensor SampleFine(torch::Tensor& partitions, torch::Tensor& weights,
                         const torch::Tensor& t_c, const int64_t n_f);
torch::Tensor SplitRay(const float t_n, const float t_f, const int64_t n,
                       int64_t batch_size);
}  // namespace nerf

#endif