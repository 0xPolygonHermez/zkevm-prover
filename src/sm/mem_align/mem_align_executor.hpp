#ifndef MEM_ALIGN_SM_HPP
#define MEM_ALIGN_SM_HPP

#include <gmpxx.h>
#include "definitions.hpp"
#include "config.hpp"
#include "goldilocks_base_field.hpp"
#include "sm/pols_generated/commit_pols.hpp"

USING_PROVER_FORK_NAMESPACE;

class MemAlignAction
{
public:
    mpz_class m0;
    mpz_class m1;
    mpz_class v;
    mpz_class w0;
    mpz_class w1;
    uint8_t mode;
    uint8_t wr;
};

class MemAlignExecutor
{
    Goldilocks &fr;
    const Config &config;
    const uint64_t N;

public:
    MemAlignExecutor (Goldilocks &fr, const Config &config) :
        fr(fr),
        config(config),
        N(PROVER_FORK_NAMESPACE::MemAlignCommitPols::pilDegree()) {}
    void execute (vector<MemAlignAction> &input, PROVER_FORK_NAMESPACE::MemAlignCommitPols &pols);
};

#endif