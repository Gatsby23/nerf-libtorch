#ifndef NERF_H
#define NERF_H
#include "stdafx.h"
//
#include "radiance_field.h"

namespace nerf {
struct View {
    float f;
    float cx;
    float cy;
    int width;
    int height;
    torch::Tensor pose;
};

std::tuple<torch::Tensor, torch::Tensor> CameraParamsToRays(const View& view);

class NeRF {
   public:
    NeRF() = delete;
    NeRF(const int width, const int height, const int64_t batch_size,
         const float t_n = 0.0f, const float t_f = 2.5f,
         const int64_t length_x = 10, const int64_t length_dim = 4,
         const std::array<float, 3>& c_bg = {1.0f, 1.0f, 1.0f});
    std::tuple<torch::Tensor, torch::Tensor> forward(const View& view);
    RadianceField RadianceFieldCoarse() { return rf_c_; }
    RadianceField RadianceFieldFine() { return rf_f_; }
    const float GetTFar() const { return t_f_; }
    const float GetTNear() const { return t_n_; }
    const int64_t GetNCoarse() const { return kNc; }
    const int64_t GetNFine() const { return kNf; }
    std::array<float, 3>& GetBackColor() { return c_bg_; }
    void SaveModel(const std::filesystem::path& log_dir, const int epoch) {
        torch::save(
            rf_c_,
            (log_dir / ("coarse_" + std::to_string(epoch) + ".pth")).string());
        torch::save(
            rf_f_,
            (log_dir / ("fine_" + std::to_string(epoch) + ".pth")).string());
    }
    void LoadModel(const std::filesystem::path& log_dir, const int epoch) {
        const auto coarse_path =
            log_dir / ("coarse_" + std::to_string(epoch) + ".pth");
        const auto fine_path =
            log_dir / ("fine_" + std::to_string(epoch) + ".pth");

        std::cout << "Use checkpoints\ncoarse: " << coarse_path << "\n";
        std::cout << "fine: " << fine_path << "\n";
        assert(std::filesystem::exists(coarse_path));
        assert(std::filesystem::exists(fine_path));
        torch::load(rf_c_, coarse_path.string(), torch::kCUDA);
        torch::load(rf_f_, fine_path.string(), torch::kCUDA);
        std::cout << "Finished loading\n";
    }

   private:
    static constexpr int64_t kNc = 64;
    static constexpr int64_t kNf = 128;
    std::array<float, 3> c_bg_;
    const int width_;
    const int height_;

    const float t_n_;
    const float t_f_;
    const int64_t length_x_;
    const int64_t length_dim_;
    const int64_t batch_size_;

    RadianceField rf_c_;
    RadianceField rf_f_;
};
}  // namespace nerf
#endif