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

void *ffi_create_steps_params(Goldilocks::Element *pols, Goldilocks::Element *constPols, Goldilocks::Element *challenges, Goldilocks::Element* subproofValues, Goldilocks::Element *publicInputs) {
    StepsParams *params = new StepsParams{
        pols : pols,
        constPols : constPols,
        constPolsExtended : nullptr,
        challenges : challenges,
        subproofValues : subproofValues,
        evals : nullptr,
        zi : nullptr,
        publicInputs : publicInputs,
    };

    return params;
};

#endif