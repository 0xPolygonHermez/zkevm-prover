#ifndef H1H2_HINT_H
#define H1H2_HINT_H

#include "hint.hpp"
#include "hint_builder.hpp"

namespace Hints
{
    class H1H2Hint : public Hint {
    public:
        static std::string getName();
        static void resolveHint(int N, const std::map<std::string, Polinomial *> &polynomials);
        static std::vector<std::string> getFields();
        static std::vector<std::string> getDestination();
    };

    class H1H2HintBuilder : public HintBuilder
    {
    public:
        std::unique_ptr<Hint> build() const override;
    };
}

#endif
