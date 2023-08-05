#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include "stdafx.h"

namespace nerf {
class Embedder {
   public:
    Embedder() = default;
    torch::Tensor Embedding(const torch::Tensor& position,
                            const int64_t length);
};
}  // namespace nerf
#endif