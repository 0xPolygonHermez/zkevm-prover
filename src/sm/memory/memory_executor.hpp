#ifndef MEMORY_SM_HPP
#define MEMORY_SM_HPP

#include "config.hpp"
#include "ff/ff.hpp"
#include "sm/pols_generated/commit_pols.hpp"

class MemoryAccess
{
public:
    bool bIsWrite;
    uint64_t address;
    uint64_t pc;
    FieldElement fe0;
    FieldElement fe1;
    FieldElement fe2;
    FieldElement fe3;
    FieldElement fe4;
    FieldElement fe5;
    FieldElement fe6;
    FieldElement fe7;
};

class MemoryExecutor
{
    FiniteField &fr;
    const Config &config;
    const uint64_t N;
public:
    MemoryExecutor (FiniteField &fr, const Config &config) :
        fr(fr),
        config(config),
        N(MemCommitPols::degree()) {;}

    void execute (vector<MemoryAccess> &input, MemCommitPols &pols);

    /* Reorder access list by the following criteria:
        - In order of incremental address
        - If addresses are the same, in order ov incremental pc
    */
    void reorder (const vector<MemoryAccess> &input, vector<MemoryAccess> &output);
    
    /* Prints access list contents, for debugging purposes */
    void print (const vector<MemoryAccess> &action, FiniteField &fr);
};

#endif