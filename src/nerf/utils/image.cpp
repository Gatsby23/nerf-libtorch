#include "image.h"

#include <cassert>
#include <opencv2/opencv.hpp>

namespace nerf {
uint8_t PixelFloatToUint8(const float v) {
    return static_cast<uint8_t>(v * 255.0);
}
void SaveImageFromTensor(torch::Tensor tensor,
                         const std::filesystem::path& path) {
    assert(torch::sum(0.0f > tensor).item<int>() == 0);
    assert(torch::sum(1.0f < tensor).item<int>() == 0);
    tensor.to(torch::kCPU);
    const auto width = tensor.size(0);
    const auto height = tensor.size(1);
    const auto channel = tensor.size(2);
    assert(channel == 3);

    auto image = cv::Mat(cv::Size(width, height), CV_8UC3);
    for (auto w = 0; w < width; w++) {
        for (auto h = 0; h < height; h++) {
            image.at<cv::Vec3b>(w, h) = cv::Vec3b(
                PixelFloatToUint8(tensor.index({w, h, 0}).item<float>()),
                PixelFloatToUint8(tensor.index({w, h, 1}).item<float>()),
                PixelFloatToUint8(tensor.index({w, h, 2}).item<float>()));
        }
    }

    cv::Mat image_conv;
    cv::cvtColor(image, image_conv, cv::COLOR_RGB2BGR);
    cv::imwrite(path.string(), image_conv);
}
}  // namespace nerf