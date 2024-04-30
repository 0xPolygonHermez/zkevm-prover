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

        calculateS(sPol, denPol, numeratorValue);
    }

    void GSumHintHandler::calculateS(Polinomial &s, Polinomial &den, Goldilocks::Element multiplicity) const
    {
        uint64_t size = den.degree();

        Polinomial denI(size, 3);
        Polinomial checkVal(1, 3);

        Polinomial::batchInverse(denI, den);
        
        Polinomial::mulElement(s, 0, denI, 0, multiplicity);
        for (uint64_t i = 1; i < size; i++)
        {
            Polinomial tmp(1, 3);
            Polinomial::mulElement(tmp, 0, denI, i, multiplicity);
            Polinomial::addElement(s, i, s, i - 1, tmp, 0);
        }
        Polinomial tmp(1, 3);
        Polinomial::mulElement(tmp, 0, denI, size - 1, multiplicity);
        Polinomial::addElement(checkVal, 0, s, size - 1, tmp, 0);

        zkassert(Goldilocks3::isZero((Goldilocks3::Element &)*checkVal[0]));
    }

    std::shared_ptr<HintHandler> GSumHintHandlerBuilder::build() const
    {
        return std::make_unique<GSumHintHandler>();
    }
}