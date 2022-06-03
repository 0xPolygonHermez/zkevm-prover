#ifndef MEM_ALIGN_SM_HPP
#define MEM_ALIGN_SM_HPP

#include "config.hpp"
#include "ff/ff.hpp"
#include "sm/pols_generated/commit_pols.hpp"

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
    FiniteField &fr;
    const Config &config;
public:
    MemAlignExecutor (FiniteField &fr, const Config &config) : fr(fr), config(config) {;}
    void execute (vector<MemAlignAction> &input, MemAlignCommitPols &pols);
};

#endif