#include "subproof_value_hint_handler.hpp"

namespace Hints
{
    std::string SubproofValueHintHandler::getName()
    {
        return "subproofvalue";
    }

    std::vector<std::string> SubproofValueHintHandler::getSources()
    {
        return {"expression"};
    }

    std::vector<std::string> SubproofValueHintHandler::getDestinations()
    {
        return {"reference"};
    }

    size_t SubproofValueHintHandler::getMemoryNeeded(uint64_t N)
    {
        return 0;
    }

    void SubproofValueHintHandler::resolveHint(int N, Hint hint, const std::map<std::string, Polinomial *> &polynomials, void *ptr_extra_mem)
    {
        // TODO!
    }

    std::shared_ptr<HintHandler> SubproofValueHintHandlerBuilder::build() const
    {
        return std::make_unique<SubproofValueHintHandler>();
    }
}