#ifndef HINT_H
#define HINT_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "../starkpil/polinomial.hpp"

namespace Hints
{
    class Hint
    {
    public:
        virtual ~Hint() {}

        static std::string getName();
        static void resolveHint(int N, const std::map<std::string, Polinomial *> &polynomials);
        static std::vector<std::string> getFields();
        static std::vector<std::string> getDestination();
    };

}

#endif // HINT_H