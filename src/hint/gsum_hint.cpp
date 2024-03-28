#include "gsum_hint.hpp"

namespace Hints
{
    std::string GSumHint::getName()
    {
        return "gsum";
    }

    void GSumHint::resolveHint(int N, const std::map<std::string, Polinomial *> &polynomials)
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

        Polinomial::calculateS(zPol, numPol, denPol);
    }

    std::vector<std::string> GSumHint::getFields()
    {
        return {"numerator", "denominator"};
    }

    std::vector<std::string> GSumHint::getDestination()
    {
        return {"reference"};
    }

    std::unique_ptr<Hint> GSumHintBuilder::build() const
    {
        return std::make_unique<GSumHint>();
    }
}