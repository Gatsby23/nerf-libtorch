#include "torch/torch.h"

namespace nerf {
template <typename T>
class AverageMeter {
   public:
    AverageMeter() = default;
    AverageMeter(const AverageMeter&) = delete;
    AverageMeter& operator=(const AverageMeter&) = delete;
    Add(const T& value, const uint32_t num = 1) {
        value_ += num * value;
        num_ += num;
    }
    T GetAverage() const { return value_ / num_; }

   private:
    uint32_t num_;
    T value_;
};

inline torch::Tensor ScaleTensor(const torch::Tensor& value, const float min,
                                 const float max) {
    assert(max - min > 0);
    return min + (max - min) * value;
}
}  // namespace nerf