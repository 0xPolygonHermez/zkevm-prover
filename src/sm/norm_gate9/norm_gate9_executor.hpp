#ifndef NORM_GATE9_EXECUTOR_HPP
#define NORM_GATE9_EXECUTOR_HPP

#include <vector>
#include "commit_pols.hpp"

using namespace std;

class NormGate9ExecutorInput
{
public:
    Goldilocks::Element type; // 0=XORN, 1=ANDP
    Goldilocks::Element a; // Pin a of gate
    Goldilocks::Element b; // Pin b of gate
};

class NormGate9Executor
{
    Goldilocks &fr;
public:
    const uint64_t N;
    const uint64_t nBlocks;
    NormGate9Executor(Goldilocks &fr) :
        fr(fr),
        N(NormGate9CommitPols::degree()),
        nBlocks(N/3) {}
    void execute (vector<NormGate9ExecutorInput> &input, NormGate9CommitPols & pols);
};

#endif