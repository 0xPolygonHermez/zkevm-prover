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
    uint64_t getTotalPolsSize (void) { return starkInfo.mapTotalN*sizeof(Goldilocks::Element); }
    uint64_t getCommitPolsSize (void) { return starkInfo.mapOffsets.cm2_n*sizeof(Goldilocks::Element); }
    void genProof (void *pAddress, CommitPols &cmPols, Proof &proof);
};

#endif