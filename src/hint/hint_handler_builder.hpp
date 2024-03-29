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
        virtual std::unique_ptr<HintHandler> build() const = 0;
        static std::unique_ptr<HintHandlerBuilder> create(const std::string &hintName);
        static void registerBuilder(const std::string &hintName, std::unique_ptr<HintHandlerBuilder> builder);

    private:
        static std::unordered_map<std::string, std::unique_ptr<HintHandlerBuilder>> builders;
    };
}

#endif // HINTBUILDER_H
