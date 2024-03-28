#include "public_values_hint.hpp"

namespace Hints
{
    std::string PublicValuesHint::getName()
    {
        return "Public Values Hint";
    }

    void PublicValuesHint::resolveHint(int N, const std::map<std::string, Polinomial *> &polynomials)
    {
        // TODO!
    }

    std::vector<std::string> PublicValuesHint::getFields()
    {
        return {"numerator", "denominator"};
    }

    std::vector<std::string> PublicValuesHint::getDestination()
    {
        return {"reference"};
    }

    std::unique_ptr<Hint> PublicValuesHintBuilder::build() const
    {
        return std::make_unique<PublicValuesHint>();
    }
}