#ifndef STARKS_C12_A_STEPS_HPP
#define STARKS_C12_A_STEPS_HPP

#include "goldilocks_cubic_extension.hpp"
#include "zhInv.hpp"

#include "starks.hpp"
#include "constant_pols_starks.hpp"

class C12aSteps : public Steps
{
public:
    void step2prev_first(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;
    void step2prev_i(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;
    void step2prev_last(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;

    void step3prev_first(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;
    void step3prev_i(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;
    void step3prev_last(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;

    void step4_first(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;
    void step4_i(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;
    void step4_last(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;

    void step42ns_first(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;
    void step42ns_i(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;
    void step42ns_last(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;

    void step52ns_first(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;
    void step52ns_i(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;
    void step52ns_last(Goldilocks::Element *pols, ConstantPolsStarks *pConstPols, ConstantPolsStarks *pConstPols2ns, Polinomial &challenges, Polinomial &x_n, Polinomial &x_2ns, ZhInv &zi, Polinomial &evals, Polinomial &xDivXSubXi, Polinomial &xDivXSubWXi, const Goldilocks::Element *publicInputs, uint64_t i) ;
};

#endif // STARKS_C12_STEPS_HPP
