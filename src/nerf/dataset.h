#ifndef DATASET_H
#define DATASET_H

#include "stdafx.h"
//
namespace nerf {
torch::Tensor MatToTensor(const cv::Mat& image);
void LoadIntrinsics(float& f, float& cx, float& cy,
                    const std::filesystem::path& path);
class Dataset : public torch::data::Dataset<Dataset> {
   public:
    Dataset(const std::filesystem::path& dataset_path);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const { return data_.size(0); }

   private:
    const std::filesystem::path& dataset_path_;
    torch::Tensor data_;  // {N, 2, 3}
    torch::Tensor target_;
};
}  // namespace nerf
#endif