#include "hint_handler_builder.hpp"

namespace Hints
{
    std::unordered_map<std::string, std::shared_ptr<HintHandlerBuilder>> HintHandlerBuilder::builders;

    std::shared_ptr<HintHandlerBuilder> HintHandlerBuilder::create(const std::string &hintName)
    {
        auto it = builders.find(hintName);
        if (it != builders.end())
        {
            return it->second;
        }
        throw std::invalid_argument("HintHandlerBuilder not found for hint: " + hintName);
    }

    void HintHandlerBuilder::registerBuilder(const std::string &hintName, std::shared_ptr<HintHandlerBuilder> builder)
    {
        auto result = builders.insert(std::make_pair(hintName, std::move(builder)));
        if (!result.second)
        {
            throw std::runtime_error("Builder for hint " + hintName + " already registered.");
        }
    }
} // namespace Hints