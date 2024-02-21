#ifndef STEPS_HPP
#define STEPS_HPP

struct StepsParams
{
    Goldilocks::Element *pols;
    ConstantPolsStarks *pConstPols;
    ConstantPolsStarks *pConstPols2ns;
    Polinomial &challenges;
    Polinomial &x_n;
    Polinomial &x_2ns;
    Polinomial &zi;
    Polinomial &evals;
    Polinomial &xDivXSubXi;
    Goldilocks::Element *publicInputs;
    Goldilocks::Element *q_2ns;
    Goldilocks::Element *f_2ns;
};

#endif