#ifndef STARK_HPP
#define STARK_HPP

#include "stark_info.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "sm/pols_generated/constant_pols.hpp"
#include "proof.hpp"
#include "transcript.hpp"
#include "zhInv.hpp"
#include "merklehash_goldilocks.hpp"
#include "polinomial.hpp"
#include "ntt_goldilocks.hpp"
class Stark
{
    const Config &config;
    StarkInfo starkInfo;
    void *pConstPolsAddress;
    ConstantPols *pConstPols;
    void *pConstPolsAddress2ns;
    ConstantPols *pConstPols2ns;
    void *pConstTreeAddress;
    ZhInv zi;
    uint64_t numCommited;
    uint64_t N;
    uint64_t NExtended;
    NTT_Goldilocks ntt;
    Transcript transcript;
    Polinomial x_n;
    Polinomial x_2ns;
    Polinomial challenges;
    Polinomial xDivXSubXi;
    Polinomial xDivXSubWXi;
    Polinomial evals;
    Goldilocks::Element *trees[5];

private:
    void calculateH1H2(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol);
    void calculateZ(Polinomial &z, Polinomial &num, Polinomial &den);
    static inline void batchInverse(Polinomial &res, Polinomial &src);

public:
    Stark(const Config &config);
    ~Stark();

    /* Returns the size of all the polynomials: committed, constant, etc. */
    uint64_t getTotalPolsSize(void) { return starkInfo.mapTotalN * sizeof(Goldilocks::Element); }

    /* Returns the size of the committed polynomials */
    uint64_t getCommitPolsSize(void) { return starkInfo.mapOffsets.section[cm2_n] * sizeof(Goldilocks::Element); }

    /* Generates a proof from the address to all polynomials memory area, and the committed pols */
    void genProof(void *pAddress, CommitPols &cmPols, const PublicInputs &publicInputs, Proof &proof);

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