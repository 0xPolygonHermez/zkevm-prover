#ifndef STARK_C12_A_HPP
#define STARK_C12_A_HPP

#include "stark_info.hpp"
#include "starkpil/src/commit_pols_c12a.hpp"
#include "starkpil/src/constant_pols_c12a.hpp"
#include "transcript.hpp"
#include "zhInv.hpp"
#include "merklehash_goldilocks.hpp"
#include "polinomial.hpp"
#include "ntt_goldilocks.hpp"
#include "friProof.hpp"
#include "friProve.hpp"
#include "proof2zkinStark.hpp"
#include "compare_fe.hpp"
#include <fstream>
#include <iostream>
#include <math.h> /* floor */

#define STARK_C12_A_NUM_TREES 5

class StarkC12a
{
    const Config &config;

public:
    StarkInfo starkInfo;

private:
    void *pConstPolsAddress;
    ConstantPolsC12a *pConstPols;
    void *pConstPolsAddress2ns;
    ConstantPolsC12a *pConstPols2ns;
    void *pConstTreeAddress;
    ZhInv zi;
    uint64_t numCommited;
    uint64_t N;
    uint64_t NExtended;
    NTT_Goldilocks ntt;
    Polinomial x_n;
    Polinomial x_2ns;
    Polinomial challenges;
    Polinomial xDivXSubXi;
    Polinomial xDivXSubWXi;
    Polinomial evals;
    Goldilocks::Element *trees[STARK_C12_A_NUM_TREES];

public:
    StarkC12a(const Config &config);
    ~StarkC12a();

    /* Resets the attributes before every proof generation */
    void reset(void)
    {
        std::memset(challenges.address(), 0, challenges.size());
        std::memset(xDivXSubXi.address(), 0, xDivXSubXi.size());
        std::memset(xDivXSubWXi.address(), 0, xDivXSubWXi.size());
        std::memset(evals.address(), 0, evals.size());
        numCommited = starkInfo.nCm1;
    }

    /* Returns the size of all the polynomials: committed, constant, etc. */
    uint64_t getTotalPolsSize(void) { return starkInfo.mapTotalN * sizeof(Goldilocks::Element); }

    /* Returns the size of the committed polynomials */
    uint64_t getCommitPolsSize(void) { return starkInfo.mapOffsets.section[cm2_n] * sizeof(Goldilocks::Element); }

    /* Generates a proof from the address to all polynomials memory area, and the committed pols */
    void genProof(void *pAddress, FRIProof &proof, Goldilocks::Element publicInputs[43]);

    void step2prev_first(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step2prev_i(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step2prev_last(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);

    void step3prev_first(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step3prev_i(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step3prev_last(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);

    void step4_first(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step4_i(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step4_last(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);

    void step42ns_first(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step42ns_i(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step42ns_last(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);

    void step52ns_first(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step52ns_i(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step52ns_last(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
};
#endif