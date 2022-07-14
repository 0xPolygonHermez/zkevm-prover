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
    const ConstantPols *pConstPols;
    void *pConstPolsAddress2ns;
    const ConstantPols *pConstPols2ns;
    void *pConstTreeAddress;
    uint64_t N;
    uint64_t NExtended;

    Transcript transcript;
    ZhInv zi;
    uint64_t numCommited;

    Polinomial x_n;
    Polinomial x_2ns;
    Polinomial challenges;

    NTT_Goldilocks ntt;

public:
    Stark(const Config &config);
    ~Stark();

    /* Returns the size of all the polynomials: committed, constant, etc. */
    uint64_t getTotalPolsSize(void) { return starkInfo.mapTotalN * sizeof(Goldilocks::Element); }

    /* Returns the size of the committed polynomials */
    uint64_t getCommitPolsSize(void) { return starkInfo.mapOffsets.section[cm2_n] * sizeof(Goldilocks::Element); }

    /* Generates a proof from the address to all polynomials memory area, and the committed pols */
    void genProof(void *pAddress, CommitPols &cmPols, const PublicInputs &publicInputs, Proof &proof);
};

#endif