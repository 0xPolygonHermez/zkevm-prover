#ifndef STARK_HPP
#define STARK_HPP

#include "stark_info.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "sm/pols_generated/constant_pols.hpp"
#include "proof.hpp"

class Stark
{
    StarkInfo &starkInfo;
public:
    Stark(StarkInfo &starkInfo) : starkInfo(starkInfo) {};
    void genProof(void *pAddress, CommitPols &cmPols, const ConstantPols &constPols, Proof &proof);
};

#endif