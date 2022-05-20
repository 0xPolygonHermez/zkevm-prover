#ifndef NORM_GATE9_EXECUTOR_HPP
#define NORM_GATE9_EXECUTOR_HPP

#include <vector>
#include "commit_pols.hpp"

using namespace std;

class NormGate9ExecutorInput
{
public:
    FieldElement type; // 0=XORN, 1=ANDP
    FieldElement a; // Pin a of gate
    FieldElement b; // Pin b of gate
};

class NormGate9Executor
{
public:
    void execute (vector<NormGate9ExecutorInput> &input, NormGate9CommitPols & pols);
};

#endif