#include "logger.h"

namespace nerf {
Logger::Logger(const std::filesystem::path& log_dir) : log_dir_(log_dir) {
    std::filesystem::create_directories(log_dir);
}
void Logger::Log(const std::unordered_map<LoggerData, LoggerType>& data) {
    data_.emplace_back(data);
}
void Logger::Write() {
    std::ofstream ofs((log_dir_ / "log.csv").string());
    const auto num_type =
        static_cast<std::underlying_type<LoggerData>::type>(LoggerData::kCount);
    for (auto type_i = 0; type_i < num_type; type_i++) {
        const auto type = static_cast<LoggerData>(type_i);
        const auto str = LoggerDataToString.at(type);
        if (type_i == num_type - 1) {
            ofs << str << std::endl;
        } else {
            ofs << str << ",";
        }
    }
    for (auto& data : data_) {
        WriteData(data, ofs);
    }
    ofs.close();
}
void Logger::WriteData(const std::unordered_map<LoggerData, LoggerType>& data,
                       std::ofstream& ofs) {
    const auto num_type =
        static_cast<std::underlying_type<LoggerData>::type>(LoggerData::kCount);
    for (auto type_i = 0; type_i < num_type; type_i++) {
        const auto type = static_cast<LoggerData>(type_i);
        if (!data.contains(type)) {
            ofs << ",";
        } else {
            const auto str = std::visit(LoggerTypeToString{}, data.at(type));
            ofs << str;
            if (type_i < num_type - 1) {
                ofs << ",";
            }
        }
    }
    ofs << "\n";
}
}  // namespace nerf