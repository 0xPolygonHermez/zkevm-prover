#include <iostream>
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
        return {"reference", "result"};
    }

    void GSumHintHandler::resolveHint(int N, StepsParams &params, Hint hint, const std::map<std::string, Polinomial *> &polynomials) const
    {
        assert(polynomials.size() == 2);

        auto den = polynomials.find("denominator");
        auto s = polynomials.find("reference");

        assert(den != polynomials.end());
        assert(s != polynomials.end());

        auto denPol = *den->second;
        auto sPol = *s->second;

        assert(denPol.dim() == sPol.dim());

        Goldilocks::Element numeratorValue = Goldilocks::fromU64(hint.fields["numerator"].value);

        calculateS(sPol, denPol, numeratorValue);
        
        uint64_t subproofValueId = hint.fields["result"].id;
        
        params.subproofValues[subproofValueId * FIELD_EXTENSION] = sPol[N - 1][0];
        params.subproofValues[subproofValueId * FIELD_EXTENSION + 1] = sPol[N - 1][1];
        params.subproofValues[subproofValueId * FIELD_EXTENSION + 2] = sPol[N - 1][2];
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