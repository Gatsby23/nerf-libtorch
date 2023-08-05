#include <filesystem>
#include <iostream>
#include <source_location>

std::filesystem::path GetDatasetPath() {
    auto dataset_path =
        std::filesystem::path(std::source_location::current().file_name());
    dataset_path = dataset_path.parent_path().parent_path().parent_path() /
                   "data" / "synthetic_scenes" / "train" / "greek";
    std::cout << "dataset path: " << dataset_path << " ... ";
    if (std::filesystem::exists(dataset_path)) {
        std::cout << "exists\n";
    } else {
        std::cout << "does not exist\n";
        std::terminate();
    }
    return dataset_path;
}