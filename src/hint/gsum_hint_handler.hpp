#ifndef GSUM_HINT_H
#define GSUM_HINT_H

#include "hint_handler.hpp"
#include "hint_handler_builder.hpp"

namespace Hints
{
    class GSumHintHandler : public HintHandler
    {
    public:
        // Return the name of the hint
        static std::string getName();

        // Return the source names of the hint, so the fields needed to resolve the hint
        virtual std::vector<std::string> getSources() const override;

        // Return the destination names of the hint, so the fields that will be updated
        virtual std::vector<std::string> getDestinations() const override;

        // Resolve the hint
        virtual void resolveHint(int N, StepsParams &params, Hint hint, const std::map<std::string, Polinomial *> &polynomials) const override;

        void calculateS(Polinomial &s, Polinomial &den, Goldilocks::Element multiplicity) const;
    };

    class GSumHintHandlerBuilder : public HintHandlerBuilder
    {
    public:
        std::shared_ptr<HintHandler> build() const override;
    };
}

#endif
