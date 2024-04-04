#include "gsum_hint_handler.hpp"

namespace Hints
{
    std::string GSumHintHandler::getName()
    {
        return "gsum";
    }

    std::vector<std::string> GSumHintHandler::getSources() const
    {
        return {"denominator"};
    }

    std::vector<std::string> GSumHintHandler::getDestinations() const
    {
        return {"reference"};
    }

    size_t GSumHintHandler::getMemoryNeeded(uint64_t N) const
    {
        return 0;
    }

    void GSumHintHandler::resolveHint(int N, StepsParams &params, Hint hint, const std::map<std::string, Polinomial *> &polynomials, void *ptr_extra_mem) const
    {
        assert(polynomials.size() == 2);

        auto den = polynomials.find("denominator");
        auto s = polynomials.find("reference");

        assert(den != polynomials.end());
        assert(s != polynomials.end());

        auto denPol = *den->second;
        auto sPol = *s->second;

        assert(denPol.dim() == sPol.dim());

        Goldilocks::Element numeratorValue = Goldilocks::fromS64(hint.fields["numerator"].value);
        numeratorValue = Goldilocks::negone(); // TODO: NOT HARDCODE THIS!

        Polinomial::calculateS(sPol, denPol, numeratorValue);
    }

    std::shared_ptr<HintHandler> GSumHintHandlerBuilder::build() const
    {
        return std::make_unique<GSumHintHandler>();
    }
}