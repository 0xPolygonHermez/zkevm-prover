#ifndef STEPS_HPP
#define STEPS_HPP

struct StepsParams
{
    Goldilocks::Element *pols;
    Goldilocks::Element *constPols;
    Goldilocks::Element *constPolsExtended;
    Goldilocks::Element *challenges;
    Goldilocks::Element *subproofValues;
    Goldilocks::Element *evals;
    Goldilocks::Element *zi;
    Goldilocks::Element *publicInputs;
};

#endif