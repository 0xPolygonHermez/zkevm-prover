#include "public_values_hint_handler.hpp"

namespace Hints
{
    std::string PublicValuesHintHandler::getName()
    {
        return "public";
    }

    std::vector<std::string> PublicValuesHintHandler::getSources()
    {
        return {"expression"};
    }

    std::vector<std::string> PublicValuesHintHandler::getDestinations()
    {
        return {"reference"};
    }

    void PublicValuesHintHandler::resolveHint(int N, Hint hint, const std::map<std::string, Polinomial *> &polynomials)
    {
        // TODO!
    }

    std::unique_ptr<HintHandler> PublicValuesHintHandlerBuilder::build() const
    {
        return std::make_unique<PublicValuesHintHandler>();
    }
}