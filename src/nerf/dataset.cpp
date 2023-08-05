#include "dataset.h"

#include "nerf.h"

namespace nerf {
torch::Tensor MatToTensor(const cv::Mat& image) {
    cv::Mat image_conv;
    cv::cvtColor(image, image_conv, cv::COLOR_BGR2RGB);
    auto tensor =
        torch::from_blob(image_conv.data, {image_conv.rows, image_conv.cols, 3},
                         torch::kUInt8)
            .clone();
    assert(!torch::sum(tensor.min() < 0).item<bool>() &&
           !torch::sum(tensor.max() > 255).item<bool>());
    return tensor;
}

void LoadIntrinsics(float& f, float& cx, float& cy,
                    const std::filesystem::path& path) {
    assert(std::filesystem::exists(path));
    std::ifstream fin(path);
    float null;
    fin >> f >> cx >> cy >> null;
    float origin_x, origin_y, origin_z;
    fin >> origin_x >> origin_y >> origin_z;
    float near_plane, scale, width, height;
    fin >> near_plane >> scale >> height >> width;

    // dataset width and height
    constexpr float kWidth = 512;
    constexpr float kHeight = 512;
    f = f * kHeight / height;
    cx = cx * kWidth / width;
    cy = cy * kHeight / height;
    fin.close();
}
Dataset::Dataset(const std::filesystem::path& dataset_path)
    : dataset_path_(dataset_path) {
    std::cout << "Start loading dataset\n";
    assert(std::filesystem::exists(dataset_path));

    // intrinsics
    const auto intrinsics_path = dataset_path / "intrinsics.txt";
    float f, cx, cy, width, height;
    LoadIntrinsics(f, cx, cy, intrinsics_path);
    auto pose_paths = std::vector<std::filesystem::path>();
    for (const auto& entry :
         std::filesystem::directory_iterator(dataset_path / "pose")) {
        pose_paths.emplace_back(entry.path());
    }
    std::sort(pose_paths.begin(), pose_paths.end());

    auto rgb_paths = std::vector<std::filesystem::path>();
    for (const auto& entry :
         std::filesystem::directory_iterator(dataset_path / "rgb")) {
        rgb_paths.emplace_back(entry.path());
    }
    std::sort(rgb_paths.begin(), rgb_paths.end());

    assert(pose_paths.size() == rgb_paths.size());

    // dataset width and height
    constexpr float kWidth = 512;
    constexpr float kHeight = 512;

    std::vector<torch::Tensor> data_vector;
    std::vector<torch::Tensor> target_vector;
    for (size_t pose_i = 0; pose_i < pose_paths.size(); pose_i++) {
        torch::NoGradGuard no_grad;
        std::cout << "\r" << pose_i << "/" << pose_paths.size() << std::flush;
        auto& pose_path = pose_paths[pose_i];
        assert(std::filesystem::exists(pose_path));
        std::ifstream fin(pose_path);
        std::array<float, 16> pose_arr;
        for (auto& val : pose_arr) {
            fin >> val;
        }
        fin.close();
        auto pose_tensor =
            torch::from_blob(pose_arr.data(), {4, 4}, torch::kFloat);

        auto& rgb_path = rgb_paths[pose_i];
        assert(std::filesystem::exists(rgb_path));
        auto image = cv::imread(rgb_path);  // {512, 512, 3}
        auto rgb_tensor = MatToTensor(image);
        auto [o, d] = CameraParamsToRays(View{
            .f = f,
            .cx = cx,
            .cy = cy,
            .width = static_cast<int>(kWidth),
            .height = static_cast<int>(kHeight),
            .pose = pose_tensor,
        });
        auto c = (rgb_tensor / 255.0f)
                     .index({"...",
                             torch::indexing::Slice(torch::indexing::None, 3)});
        o = o.reshape({-1, 3});
        d = d.reshape({-1, 3});
        c = c.reshape({-1, 3});

        auto data = torch::stack({o, d}, 1);
        assert(data.size(0) == c.size(0));

        if (data_.size(0) == 0) {
            data_ = data;
            target_ = c;
        } else {
            data_ = torch::cat({data_, data}, 0);
            target_ = torch::cat({target_, c}, 0);
        }
    }
    std::cout << "\n";
    std::cout << "dataset size: " << data_.size(0) << "\n";
    std::cout << "Loaded dataset\n";
}

torch::data::Example<> Dataset::get(size_t index) {
    return {data_[index], target_[index]};
}
}  // namespace nerf