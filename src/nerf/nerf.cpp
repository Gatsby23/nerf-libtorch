#include "nerf.h"

#include "volume_rendering.h"

using namespace torch::indexing;

namespace nerf {
std::tuple<torch::Tensor, torch::Tensor> CameraParamsToRays(const View& view) {
    auto o = torch::zeros({view.width, view.height, 4},
                          torch::TensorOptions().requires_grad(false));
    o.index({Slice(), Slice(), 3}) = 1.0f;
    const auto grid = torch::meshgrid(
        {torch::arange(view.height), torch::arange(view.width)}, "xy");
    const auto u = grid[1];
    const auto v = grid[0];
    const auto x = (u - view.cx) / view.f;
    const auto y = (v - view.cy) / view.f;
    const auto z = torch::ones_like(x);
    const auto w = torch::ones_like(x);
    auto d = torch::stack({x, y, z, w}, 2);
    o = torch::matmul(view.pose, o.unsqueeze(-1))
            .index({"...", Slice(0, 3), 0});
    d = torch::matmul(view.pose.unsqueeze(0), d.unsqueeze(-1))
            .index({"...", Slice(0, 3), 0});
    d -= o;
    d /= torch::norm(d, 2, 2, true);
    return {o, d};
}

NeRF::NeRF(const int width, const int height, const int64_t batch_size,
           const float t_n, const float t_f, const int64_t length_x,
           const int64_t length_dim, const std::array<float, 3>& c_bg)
    : t_n_(t_n),
      t_f_(t_f),
      length_x_(length_x),
      length_dim_(length_dim),
      c_bg_(c_bg),
      rf_c_(length_x, length_dim),
      rf_f_(length_x, length_dim),
      width_(width),
      height_(height),
      batch_size_(batch_size) {}

std::tuple<torch::Tensor, torch::Tensor> NeRF::forward(const View& view) {
    auto [o, d] = CameraParamsToRays(view);
    o = o.reshape({-1, 3}).to(torch::kCUDA);
    d = d.reshape({-1, 3}).to(torch::kCUDA);
    torch::Tensor c_c;
    torch::Tensor c_f;
    {
        torch::NoGradGuard no_grad;
        for (int64_t i = 0; i < o.size(0); i += batch_size_) {
            auto o_i = o.index({torch::indexing::Slice({i, i + batch_size_})});
            auto d_i = d.index({torch::indexing::Slice({i, i + batch_size_})});
            auto params = VolumeRenderingWithRadianceFieldParams{
                .module_c = rf_c_,
                .module_f = rf_f_,
                .o = o_i,
                .d = d_i,
                .t_n = t_n_,
                .t_f = t_f_,
                .n_c = kNc,
                .n_f = kNf,
                .c_bg = c_bg_,
            };
            auto [c_c_tmp, c_f_tmp] = VolumeRenderingWithRadianceField(params);
            if (c_c.size(0) == 0) {
                c_c = c_c_tmp.detach().cpu();
                c_f = c_f_tmp.detach().cpu();
            } else {
                c_c = torch::cat({c_c, c_c_tmp.detach().cpu()}, 0);
                c_f = torch::cat({c_f, c_f_tmp.detach().cpu()}, 0);
            }
        }
    }
    PrintInfo(c_c, "c_c");
    PrintInfo(c_f, "c_f");
    c_c = c_c.reshape({width_, height_, 3}).clamp(0.0f, 1.0f);
    c_f = c_f.reshape({width_, height_, 3}).clamp(0.0f, 1.0f);
    return {c_c, c_f};
}
}  // namespace nerf