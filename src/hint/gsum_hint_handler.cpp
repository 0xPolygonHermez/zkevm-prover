#include "gsum_hint_handler.hpp"

namespace Hints
{
    std::string GSumHintHandler::getName()
    {
        return "gsum";
    }

    std::vector<std::string> GSumHintHandler::getSources()
    {
        return {"numerator", "denominator"};
    }

    std::vector<std::string> GSumHintHandler::getDestinations()
    {
        return {"reference"};
    }

    size_t GSumHintHandler::getMemoryNeeded(uint64_t N)
    {
        return 0;
    }

    void GSumHintHandler::resolveHint(int N, Hint hint, const std::map<std::string, Polinomial *> &polynomials, void *ptr_extra_mem)
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

        //Polinomial::calculateS(zPol, numPol, denPol);
    }

    std::shared_ptr<HintHandler> GSumHintHandlerBuilder::build() const
    {
        return std::make_unique<GSumHintHandler>();
    }
}