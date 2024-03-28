#include "hint_builder.hpp"

namespace Hints
{
    std::unordered_map<std::string, std::unique_ptr<HintBuilder>> HintBuilder::builders;

    std::unique_ptr<HintBuilder> HintBuilder::create(const std::string &hintName)
    {
        auto it = builders.find(hintName);
        if (it != builders.end())
        {
            return std::move(it->second);
        }
        throw std::invalid_argument("Unknown hint name: " + hintName);
    }

    void HintBuilder::registerBuilder(const std::string &hintName, std::unique_ptr<HintBuilder> builder)
    {

        builders.emplace(hintName, std::move(builder));
    }
} // namespace Hints