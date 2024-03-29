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

        static std::string getName();
        static void resolveHint(int N, Hint hint, const std::map<std::string, Polinomial *> &polynomials);
        static std::vector<std::string> getSources();
        static std::vector<std::string> getDestinations();
    };

}

#endif // HINT_H