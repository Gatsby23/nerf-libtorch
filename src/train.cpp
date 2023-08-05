#include "nerf/stdafx.h"
//
#include "nerf/dataset.h"
#include "nerf/logger.h"
#include "nerf/nerf.h"
#include "nerf/volume_rendering.h"
#include "utils.h"

using namespace nerf;

int main() {
#if defined(_DEBUG)
    torch::autograd::AnomalyMode::set_enabled(true);
#endif
    const auto log_dir =
        std::filesystem::absolute(std::filesystem::path("log"));
    std::cout << "logged at: " << log_dir.string() << std::endl;
    auto logger = Logger(log_dir);

    constexpr int kNumEpoch = 10;
    constexpr int64_t kBatchSize = 2048;
    const auto dataset_path = GetDatasetPath();
    auto dataset =
        Dataset(dataset_path)
            .map(torch::data::transforms::Stack<>());  // size == 125566976

    const auto train_dataloader =
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            dataset,
            torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(4));
    auto nerf = NeRF(512, 512, kBatchSize);

    nerf.RadianceFieldCoarse()->to(torch::kCUDA);
    nerf.RadianceFieldCoarse()->train();
    nerf.RadianceFieldFine()->to(torch::kCUDA);
    nerf.RadianceFieldFine()->train();

    auto optimizer = torch::optim::Adam(
        {nerf.RadianceFieldCoarse()->parameters(),
         nerf.RadianceFieldFine()->parameters()},
        torch::optim::AdamOptions().lr(3e-4).betas({0.9, 0.999}).eps(1e-7));

    for (auto epoch_i = 0; epoch_i < kNumEpoch; epoch_i++) {
        std::cout << "epoch: " << epoch_i << "\n";
        auto batch_i = 0;
        auto epoch_loss = AverageMeter<float>();
        for (auto &batch : *train_dataloader) {
            std::cout << batch_i << " / "
                      << static_cast<int64_t>(dataset.size().value()) /
                                 kBatchSize -
                             1
                      << "\n";
            auto data = batch.data;
            auto o = data.index({torch::indexing::Slice(), 0}).to(torch::kCUDA);
            auto d = data.index({torch::indexing::Slice(), 1}).to(torch::kCUDA);
            auto c = batch.target.to(torch::kCUDA).detach();
            auto params = VolumeRenderingWithRadianceFieldParams{
                .module_c = nerf.RadianceFieldCoarse(),
                .module_f = nerf.RadianceFieldFine(),
                .o = o,
                .d = d,
                .t_n = nerf.GetTNear(),
                .t_f = nerf.GetTFar(),
                .n_c = nerf.GetNCoarse(),
                .n_f = nerf.GetNFine(),
                .c_bg = nerf.GetBackColor(),
            };
            auto [c_c, c_f] = VolumeRenderingWithRadianceField(params);
            auto loss = torch::nn::functional::mse_loss(c_c, c) +
                        torch::nn::functional::mse_loss(c_f, c);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            batch_i++;
            std::cout << loss.item<float>() << std::endl;
            epoch_loss.Add(loss.item<float>());
        }
        const auto loss = epoch_loss.GetAverage();
        std::cout << "loss: " << loss << std::endl;

        const auto log = std::unordered_map<LoggerData, LoggerType>{
            {
                LoggerData::kEpoch,
                epoch_i,
            },
            {
                LoggerData::kLoss,
                loss,
            },
        };
        logger.Log(log);
        nerf.SaveModel(log_dir, epoch_i);
    }
    logger.Write();
    return 0;
}