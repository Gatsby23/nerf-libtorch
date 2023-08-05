#ifndef LOGGER_H
#define LOGGER_H
#include "stdafx.h"

namespace nerf {
enum class LoggerData : int { kEpoch = 0, kLoss, kCount };
static const std::unordered_map<LoggerData, std::string> LoggerDataToString{
    {LoggerData::kEpoch, "epoch"},
    {LoggerData::kLoss, "loss"},
};
typedef std::variant<int, float, std::string> LoggerType;
struct LoggerTypeToString {
    std::string operator()(int value) { return std::to_string(value); }
    std::string operator()(float value) { return std::to_string(value); }
    std::string operator()(std::string_view value) {
        return std::string(value);
    }
};
class Logger {
   public:
    Logger(const std::filesystem::path& log_dir);
    void Log(const std::unordered_map<LoggerData, LoggerType>& data);
    void Write();

   private:
    const std::filesystem::path log_dir_;
    std::vector<std::unordered_map<LoggerData, LoggerType>> data_;

    void WriteData(const std::unordered_map<LoggerData, LoggerType>& data,
                   std::ofstream&);
};
}  // namespace nerf
#endif