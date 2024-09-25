#ifndef STEPS_HPP
#define STEPS_HPP

#pragma once 

struct StepsParams
{
    Goldilocks::Element *pols;
    Goldilocks::Element *publicInputs;
    Goldilocks::Element *challenges;
    Goldilocks::Element *subproofValues;
    Goldilocks::Element *evals;
    Goldilocks::Element *xDivXSub;
};

#endif