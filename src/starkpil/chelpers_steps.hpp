#ifndef CHELPERS_STEPS_HPP
#define CHELPERS_STEPS_HPP
#include "chelpers.hpp"
#include "steps.hpp"

class CHelpersSteps {
    public:
        virtual void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {};
};

#endif