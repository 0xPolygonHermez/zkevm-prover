#ifndef SUBPROOF_VALUE_HINT_H
#define SUBPROOF_VALUE_HINT_H

#include "hint_handler.hpp"
#include "hint_handler_builder.hpp"

namespace Hints
{
    class SubproofValueHintHandler : public HintHandler
    {
    public:
        // Return the name of the hint
        static std::string getName();

        // Return the source names of the hint, so the fields needed to resolve the hint
        static std::vector<std::string> getSources();

        // Return the destination names of the hint, so the fields that will be updated
        static std::vector<std::string> getDestinations();

        // Returns the extra memory needed in bytes to resolve the hint
        static size_t getMemoryNeeded(uint64_t N);

        // Resolve the hint
        static void resolveHint(int N, Hint hint, const std::map<std::string, Polinomial *> &polynomials, void *ptr_extra_mem);
    };

    class SubproofValueHintHandlerBuilder : public HintHandlerBuilder
    {
    public:
        std::shared_ptr<HintHandler> build() const override;
    };
}

#endif
