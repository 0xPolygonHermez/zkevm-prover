#ifndef MEMORY_SM_HPP
#define MEMORY_SM_HPP

#include "definitions.hpp"
#include "config.hpp"
#include "goldilocks_base_field.hpp"
#include "sm/pols_generated/commit_pols.hpp"

USING_PROVER_FORK_NAMESPACE;

class MemoryAccess
{
public:
    bool bIsWrite;
    uint64_t address;
    uint64_t pc;
    Goldilocks::Element fe0;
    Goldilocks::Element fe1;
    Goldilocks::Element fe2;
    Goldilocks::Element fe3;
    Goldilocks::Element fe4;
    Goldilocks::Element fe5;
    Goldilocks::Element fe6;
    Goldilocks::Element fe7;
};

class MemoryExecutor
{
    Goldilocks &fr;
    const Config &config;
    const uint64_t N;
public:
    MemoryExecutor (Goldilocks &fr, const Config &config) :
        fr(fr),
        config(config),
        N(PROVER_FORK_NAMESPACE::MemCommitPols::pilDegree()) {}

    void execute (vector<MemoryAccess> &input, PROVER_FORK_NAMESPACE::MemCommitPols &pols);

    /* Reorder access list by the following criteria:
        - In order of incremental address
        - If addresses are the same, in order ov incremental pc
    */
    void reorder (const vector<MemoryAccess> &input, vector<MemoryAccess> &output);
    
    /* Prints access list contents, for debugging purposes */
    void print (const vector<MemoryAccess> &action, Goldilocks &fr);
};

#endif