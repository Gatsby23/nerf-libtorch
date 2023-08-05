#ifndef DEBUG_H
#define DEBUG_H
#include <iostream>

#include "torch/torch.h"

namespace nerf {
inline void PrintSize(const torch::Tensor& tensor) {
    std::cout << "(";
    for (const auto& s : tensor.sizes()) {
        std::cout << s << ",";
    }
    std::cout << ")\n";
}

inline void PrintInfo(const torch::Tensor& tensor,
                      const std::string_view name = "") {
    std::cout << name << "------------\n";
    std::cout << "size: ";
    PrintSize(tensor);
    std::cout << "is_cuda(): " << tensor.is_cuda() << "\n";
    std::cout << "type(): " << tensor.type() << "\n";
    std::cout << "requires_grad(): " << tensor.requires_grad() << "\n";
    std::cout << "mean(): " << tensor.mean().item<float>() << "\n";
    std::cout << "max(): " << tensor.max().item<float>() << "\n";
    std::cout << "min(): " << tensor.min().item<float>() << "\n";
    std::cout << "---------------------\n";
}

inline void PrintTensor2D(const torch::Tensor& tensor,
                          const std::string_view prefix = "") {
    std::cout << prefix.data() << ":\n";
    for (size_t i = 0; i < tensor.size(0); i++) {
        for (size_t j = 0; j < tensor.size(1); j++) {
            std::cout << tensor[i][j].item<float>() << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
}  // namespace nerf

#endif DEBUG_H