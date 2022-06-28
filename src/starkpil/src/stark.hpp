#ifndef STARK_HPP
#define STARK_HPP

#include "sm/pols_generated/commit_pols.hpp"
#include "sm/pols_generated/constant_pols.hpp"
#include "proof.hpp"

class Stark
{
public:
    void genProof(CommitPols &cmPols, const ConstantPols &constPols, Proof &proof);
};

#endif