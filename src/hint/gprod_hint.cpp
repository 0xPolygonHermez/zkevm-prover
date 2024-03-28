#include "gprod_hint.hpp"

namespace Hints
{
    std::string GProdHint::getName()
    {
        return "gprod";
    }

    void GProdHint::resolveHint(int N, const std::map<std::string, Polinomial *> &polynomials)
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

    std::vector<std::string> GProdHint::getFields()
    {
        return {"numerator", "denominator"};
    }

    std::vector<std::string> GProdHint::getDestination()
    {
        return {"reference"};
    }

    std::unique_ptr<Hint> GProdHintBuilder::build() const
    {
        return std::make_unique<GProdHint>();
    }
}