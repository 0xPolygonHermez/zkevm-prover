#ifndef HINTBUILDER_H
#define HINTBUILDER_H

#include <string>
#include <memory>
#include <unordered_map>
#include "hint.hpp"

namespace Hints
{
    class HintBuilder
    {
    public:
        virtual ~HintBuilder() {}
        virtual std::unique_ptr<Hint> build() const = 0;
        static std::unique_ptr<HintBuilder> create(const std::string &hintName);
        static void registerBuilder(const std::string &hintName, std::unique_ptr<HintBuilder> builder);

    private:
        static std::unordered_map<std::string, std::unique_ptr<HintBuilder>> builders;
    };
}

#endif // HINTBUILDER_H
