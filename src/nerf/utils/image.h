#ifndef IMAGE_H
#define IMAGE_H

#include "torch/torch.h"

namespace nerf {
void SaveImageFromTensor(torch::Tensor tensor,
                         const std::filesystem::path& path);
}  // namespace nerf
#endif