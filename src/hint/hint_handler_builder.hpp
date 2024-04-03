#ifndef HINTBUILDER_H
#define HINTBUILDER_H

#include <string>
#include <memory>
#include <unordered_map>
#include "hint_handler.hpp"

namespace Hints
{
    class HintHandlerBuilder
    {
    public:
        virtual ~HintHandlerBuilder() {}
        virtual std::shared_ptr<HintHandler> build() const = 0;
        static std::shared_ptr<HintHandlerBuilder> create(const std::string &hintName);
        static void registerBuilder(const std::string &hintName, std::shared_ptr<HintHandlerBuilder> builder);

    private:
        static std::unordered_map<std::string, std::shared_ptr<HintHandlerBuilder>> builders;
    };
}

#endif // HINTBUILDER_H
