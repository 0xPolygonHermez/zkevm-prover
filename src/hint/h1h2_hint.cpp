#include "h1h2_hint.hpp"

namespace Hints
{
    std::string H1H2Hint::getName() {
        return "h1h2";
    }

    void H1H2Hint::resolveHint(int N, const std::map<std::string, Polinomial *> &polynomials)
    {
        assert(polynomials.size() == 4);

        auto h1 = polynomials.find("h1");
        auto h2 = polynomials.find("h2");
        auto f = polynomials.find("f");
        auto t = polynomials.find("t");

        assert(h1 != polynomials.end());
        assert(h2 != polynomials.end());
        assert(f != polynomials.end());
        assert(t != polynomials.end());

        auto h1Pol = *h1->second;
        auto h2Pol = *h2->second;
        auto fPol = *f->second;
        auto tPol = *t->second;

        assert(h1Pol.dim() == 1 || h1Pol.dim() == 3);
        assert(h1Pol.dim() == h2Pol.dim());
        assert(h1Pol.dim() == fPol.dim());
        assert(h1Pol.dim() == tPol.dim());

        if (h1Pol.dim() == 1)
        {
            // Polinomial::calculateH1H2_opt1(h1Pol, h2Pol, fPol, tPol, i, &pbufferH[omp_get_thread_num() * sizeof(Goldilocks::Element) * N], (sizeof(Goldilocks::Element) - 3) * N);
        }
        else if (h1Pol.dim() == 3)
        {
            // Polinomial::calculateH1H2_opt3(h1Pol, h2Pol, fPol, tPol, i, &pbufferH[omp_get_thread_num() * sizeof(Goldilocks::Element) * N], (sizeof(Goldilocks::Element) - 5) * N);
        }
    }

    std::vector<std::string> H1H2Hint::getFields()
    {
        return {"f", "t"};
    }

    std::vector<std::string> H1H2Hint::getDestination()
    {
        return {"h1", "h2"};
    }

    std::unique_ptr<Hint> H1H2HintBuilder::build() const
    {
        return std::make_unique<H1H2Hint>();
    }
}