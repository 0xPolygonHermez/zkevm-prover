#ifndef STEPS_HPP
#define STEPS_HPP

struct StepsParams
{
    Goldilocks::Element *pols;
    ConstantPolsStarks *pConstPols;
    ConstantPolsStarks *pConstPols2ns;
    Goldilocks::Element *challenges;
    Goldilocks::Element *subproofValues;
    Goldilocks::Element *evals;
    Goldilocks::Element *x_n;
    Goldilocks::Element *x_2ns;
    Goldilocks::Element *zi;
    Goldilocks::Element *xDivXSubXi;
    Goldilocks::Element *publicInputs;
    Goldilocks::Element *q_2ns;
    Goldilocks::Element *f_2ns;
};

#endif