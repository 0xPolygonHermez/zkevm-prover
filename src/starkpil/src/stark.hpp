#ifndef STARK_HPP
#define STARK_HPP

#include "stark_info.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "sm/pols_generated/constant_pols.hpp"
#include "proof.hpp"

class Stark
{
    const Config &config;
    StarkInfo starkInfo;
    void * pConstPolsAddress;
    const ConstantPols * pConstPols;
    void * pConstTreeAddress;
public:
    Stark (const Config &config);
    ~Stark ();

    /* Returns the size of all the polynomials: committed, constant, etc. */
    uint64_t getTotalPolsSize (void) { return starkInfo.mapTotalN*sizeof(Goldilocks::Element); }

    /* Returns the size of the committed polynomials */
    uint64_t getCommitPolsSize (void) { return starkInfo.mapOffsets.section[cm2_n]*sizeof(Goldilocks::Element); }

    /* Generates a proof from the address to all polynomials memory area, and the committed pols */
    void genProof (void *pAddress, CommitPols &cmPols, const PublicInputs &publicInputs, Proof &proof);
};

#endif