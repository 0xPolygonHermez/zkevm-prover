#ifndef MEMORY_SM_HPP
#define MEMORY_SM_HPP

#include "config.hpp"
#include "sm/memory/memory_access_list.hpp"
#include "ff/ff.hpp"
#include "sm/pil/commit_pols.hpp"

class MemoryExecutor
{
    FiniteField &fr;
    const Config &config;
public:
    MemoryExecutor (FiniteField &fr, const Config &config) : fr(fr), config(config) {;}
    void execute (vector<MemoryAccess> &action, RamCommitPols &pols);
};

#endif