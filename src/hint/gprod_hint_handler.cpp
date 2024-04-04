#include "gprod_hint_handler.hpp"

namespace Hints
{
    std::string GProdHintHandler::getName()
    {
        return "gprod";
    }

    std::vector<std::string> GProdHintHandler::getSources() const
    {
        return {"numerator", "denominator"};
    }

    std::vector<std::string> GProdHintHandler::getDestinations() const
    {
        return {"reference"};
    }

    size_t GProdHintHandler::getMemoryNeeded(uint64_t N) const
    {
        return 0;
    }

    void GProdHintHandler::resolveHint(int N, StepsParams &params, Hint hint, const std::map<std::string, Polinomial *> &polynomials, void *ptr_extra_mem) const
    {
        assert(polynomials.size() == 3);

        auto num = polynomials.find("numerator");
        auto den = polynomials.find("denominator");
        auto z = polynomials.find("reference");

        assert(num != polynomials.end());
        assert(den != polynomials.end());
        assert(z != polynomials.end());

        auto numPol = *num->second;
        auto denPol = *den->second;
        auto zPol = *z->second;

        assert(numPol.dim() == denPol.dim());
        assert(numPol.dim() == zPol.dim());

        Polinomial::calculateZ(zPol, numPol, denPol);
    }

    std::shared_ptr<HintHandler> GProdHintHandlerBuilder::build() const
    {
        return std::make_unique<GProdHintHandler>();
    }
}