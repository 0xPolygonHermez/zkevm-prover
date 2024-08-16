#ifndef HINT_H
#define HINT_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "polinomial.hpp"
#include "constant_pols_starks.hpp"
#include "steps.hpp"


typedef enum
{
    const_ = 0,
    cm = 1,
    tmp = 2,
    public_ = 3,
    subproofvalue = 4,
    challenge = 5,
    number = 6,
} opType;

typedef enum
{
    Field = 0,
    Extended_Field = 1,
    Column = 2,
    Extended_Column = 3,
} hintFieldType;

class HintField
{
    
public:
    opType operand;
    uint64_t id;
    uint64_t dim;
    uint64_t value;
};


class Hint 
{
public:
    std::string name;
    std::map<string,HintField> fields;
};

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