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

        // Calculate z
        calculateZ(zPol, numPol, denPol);
    }

    void GProdHintHandler::calculateZ(Polinomial &z, Polinomial &num, Polinomial &den) const
    {
        uint64_t size = num.degree();
        
        Polinomial denI(size, 3);
        Polinomial checkVal(1, 3);
        Goldilocks::Element *pZ = z[0];
        Goldilocks3::copy((Goldilocks3::Element *)&pZ[0], &Goldilocks3::one());
        
        Polinomial::batchInverse(denI, den);
        for (uint64_t i = 1; i < size; i++)
        {
            Polinomial tmp(1, 3);
            Polinomial::mulElement(tmp, 0, num, i - 1, denI, i - 1);
            Polinomial::mulElement(z, i, z, i - 1, tmp, 0);
        }
        Polinomial tmp(1, 3);
        Polinomial::mulElement(tmp, 0, num, size - 1, denI, size - 1);
        Polinomial::mulElement(checkVal, 0, z, size - 1, tmp, 0);

        zkassert(Goldilocks3::isOne((Goldilocks3::Element &)*checkVal[0]));
    }

    std::shared_ptr<HintHandler> GProdHintHandlerBuilder::build() const
    {
        return std::make_unique<GProdHintHandler>();
    }
}