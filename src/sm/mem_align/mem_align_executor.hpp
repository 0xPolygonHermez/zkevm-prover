#ifndef MEM_ALIGN_SM_HPP
#define MEM_ALIGN_SM_HPP

#include <gmpxx.h>
#include "definitions.hpp"
#include "config.hpp"
#include "goldilocks_base_field.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include <fstream>
#ifdef __ZKEVM_SM__
#include "zkevm_sm.h"
#endif

USING_PROVER_FORK_NAMESPACE;

class MemAlignAction
{
public:
    mpz_class m0;
    mpz_class m1;
    mpz_class v;
    mpz_class w0;
    mpz_class w1;
    uint8_t offset;
    uint8_t wr8;
    uint8_t wr256;
};

class MemAlignExecutor
{
    Goldilocks &fr;
    const Config &config;
    const uint64_t N;

#ifdef __ZKEVM_SM__
    void *ZkevmSMMemAlignPtr;
#endif

public:
    MemAlignExecutor (Goldilocks &fr, const Config &config) :
        fr(fr),
        config(config),
        N(PROVER_FORK_NAMESPACE::MemAlignCommitPols::pilDegree()) {
#ifdef __ZKEVM_SM__
            ZkevmSMMemAlignPtr = sm_mem_align_new(N);
#endif
        }
        
    void execute (vector<MemAlignAction> &input, PROVER_FORK_NAMESPACE::MemAlignCommitPols &pols);
};

#endif