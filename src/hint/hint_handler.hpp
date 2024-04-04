#ifndef HINT_H
#define HINT_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "polinomial.hpp"
#include "chelpers.hpp"
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

        // Returns the extra memory needed in bytes to resolve the hint
        virtual size_t getMemoryNeeded(uint64_t N) const = 0;

        // Resolve the hint
        virtual void resolveHint(int N, StepsParams &params, Hint hint, const std::map<std::string, Polinomial *> &polynomials, void *mem) const = 0;
    };

}

#endif // HINT_H