#ifndef GPROD_HINT_H
#define GPROD_HINT_H

#include "hint.hpp"
#include "hint_builder.hpp"

namespace Hints
{
    class GProdHint : public Hint
    {
    public:
        static std::string getName();
        static void resolveHint(int N, const std::map<std::string, Polinomial *> &polynomials);
        static std::vector<std::string> getFields();
        static std::vector<std::string> getDestination();
    };

    class GProdHintBuilder : public HintBuilder
    {
    public:
        std::unique_ptr<Hint> build() const override;
    };
}

#endif
