#include "nerf/stdafx.h"
//
#include "nerf/dataset.h"
#include "nerf/logger.h"
#include "nerf/nerf.h"
#include "nerf/volume_rendering.h"
#include "utils.h"

using namespace nerf;

int main() {
    torch::NoGradGuard no_grad;

    const auto log_dir =
        std::filesystem::absolute(std::filesystem::path("log"));
    const auto output_dir =
        std::filesystem::absolute(std::filesystem::path("output"));
    std::filesystem::create_directories(output_dir);

    std::cout << "Using log from: " << log_dir.string() << std::endl;
    auto logger = Logger(log_dir);
    constexpr int64_t kBatchSize = 2048;
    auto nerf = NeRF(512, 512, kBatchSize);
    nerf.LoadModel(log_dir, 9);

    nerf.RadianceFieldCoarse()->to(torch::kCUDA);
    nerf.RadianceFieldCoarse()->eval();
    nerf.RadianceFieldFine()->to(torch::kCUDA);
    nerf.RadianceFieldFine()->eval();

    float f, cx, cy;
    auto dataset_path = GetDatasetPath();
    LoadIntrinsics(f, cx, cy, dataset_path / "intrinsics.txt");

    // Load exmaple pose
    auto pose_path = dataset_path / "pose" / "000000.txt";
    assert(std::filesystem::exists(pose_path));
    std::ifstream fin(pose_path);
    std::array<float, 16> pose_arr;
    for (auto& val : pose_arr) {
        fin >> val;
    }
    fin.close();
    auto pose = torch::from_blob(pose_arr.data(), {4, 4}, torch::kFloat);
    auto view = View{
        .f = f, .cx = cx, .cy = cy, .width = 512, .height = 512, .pose = pose};
    const auto a_vector =
        torch::linspace(-std::numbers::pi, std::numbers::pi, 65);

    std::cout << "output to: " << output_dir << std::endl;
    for (auto a_i = 0; a_i < a_vector.size(0); a_i++) {
        std::cout << "a_i: " << a_i << std::endl;
        const auto a = a_vector[a_i];
        const auto c = torch::cos(a);
        const auto s = torch::sin(a);
        const auto rot =
            torch::tensor({c.item<float>(), -s.item<float>(), 0.0f, 0.0f,
                           s.item<float>(), c.item<float>(), 0.0f, 0.0f,
                           s.item<float>(), 0.0f, c.item<float>(), 0.0f, 0.0f,
                           0.0f, 0.0f, 1.0f})
                .view({4, 4})
                .to(torch::kFloat);
        view.pose = torch::mm(rot, view.pose);
        auto [c_c, c_f] = nerf.forward(view);
        auto save_coarse_path =
            output_dir / ("coarse_" + std::to_string(a_i) + ".png");
        auto save_fine_path =
            output_dir / ("fine_" + std::to_string(a_i) + ".png");
        SaveImageFromTensor(c_c, save_coarse_path);
        SaveImageFromTensor(c_f, save_fine_path);
    }

    return 0;
}