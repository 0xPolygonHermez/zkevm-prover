#ifndef STARK_HPP
#define STARK_HPP

#include "stark_info.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "sm/pols_generated/constant_pols.hpp"
#include "proof.hpp"

class Stark
{
    const StarkInfo &starkInfo;
    const ConstantPols &constPols;
public:
    Stark(const StarkInfo &starkInfo, const ConstantPols &constPols) : starkInfo(starkInfo), constPols(constPols) {};
    void genProof(void *pAddress, CommitPols &cmPols, Proof &proof);
};

#endif