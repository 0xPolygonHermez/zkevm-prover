#ifndef GPROD_HINT_H
#define GPROD_HINT_H

#include "hint_handler.hpp"
#include "hint_handler_builder.hpp"

namespace Hints
{
    class GProdHintHandler : public HintHandler
    {
    public:
        static std::string getName();
        static void resolveHint(int N, Hint hint, const std::map<std::string, Polinomial *> &polynomials);
        static std::vector<std::string> getSources();
        static std::vector<std::string> getDestinations();
    };

    class GProdHintHandlerBuilder : public HintHandlerBuilder
    {
    public:
        std::unique_ptr<HintHandler> build() const override;
    };
}

#endif
