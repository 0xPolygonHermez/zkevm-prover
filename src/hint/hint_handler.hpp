#ifndef HINT_H
#define HINT_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "../starkpil/polinomial.hpp"
#include "../starkpil/chelpers.hpp"

namespace Hints
{
    class HintHandler
    {
    public:
        virtual ~HintHandler() {}

        // Return the name of the hint
        static std::string getName();

        // Return the source names of the hint, so the fields needed to resolve the hint
        static std::vector<std::string> getSources();

        // Return the destination names of the hint, so the fields that will be updated
        static std::vector<std::string> getDestinations();

        // Returns the extra memory needed in bytes to resolve the hint
        static size_t getMemoryNeeded(uint64_t N);

        // Resolve the hint
        static void resolveHint(int N, Hint hint, const std::map<std::string, Polinomial *> &polynomials, void* mem);
    };

}

#endif // HINT_H