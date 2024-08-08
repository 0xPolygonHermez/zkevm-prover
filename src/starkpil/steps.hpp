#ifndef STEPS_HPP
#define STEPS_HPP

struct StepsParams
{
    Goldilocks::Element *pols;
    ConstantPolsStarks * pConstPols;
    ConstantPolsStarks * pConstPols2ns;
    Goldilocks::Element *challenges;
    Goldilocks::Element *subproofValues;
    Goldilocks::Element *evals;
    Goldilocks::Element *zi;
    Goldilocks::Element *xDivXSubXi;
    Goldilocks::Element *publicInputs;
};

#endif