#ifndef VOLUME_RENDERING_H
#define VOLUME_RENDERING_H

#include "stdafx.h"
//
#include "radiance_field.h"

namespace nerf {
struct VolumeRenderingWithRadianceFieldParams {
    RadianceField module_c;
    RadianceField module_f;
    torch::Tensor& o;
    torch::Tensor& d;
    float t_n;
    float t_f;
    const int64_t n_c;
    const int64_t n_f;
    std::array<float, 3>& c_bg;
};

std::tuple<torch::Tensor, torch::Tensor> VolumeRenderingWithRadianceField(
    VolumeRenderingWithRadianceFieldParams& params);
}  // namespace nerf

#endif