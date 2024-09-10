#ifndef HINT_H
#define HINT_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "polinomial.hpp"
#include "constant_pols_starks.hpp"
#include "stark_info.hpp"
#include "expressions_bin.hpp"
#include "steps.hpp"

namespace Hints
{
    class HintHandler
    {
    public:
        virtual ~HintHandler() {}

        // Return the name of the hint
        static std::string getName();

        // Return the source names of the hint, so the fields needed to resolve the hint
        virtual std::vector<std::string> getSources() const = 0;

        // Return the destination names of the hint, so the fields that will be updated
        virtual std::vector<std::string> getDestinations() const = 0;

        // Resolve the hint
        virtual void resolveHint(int N, StepsParams &params, Hint hint, const std::map<std::string, Polinomial *> &polynomials) const = 0;
    };

}

#endif // HINT_H