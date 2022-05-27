#ifndef NINE2ONE_EXECUTOR_HPP
#define NINE2ONE_EXECUTOR_HPP

#include <vector>
#include "commit_pols.hpp"

using namespace std;

class Nine2OneExecutorInput
{
public:
    uint64_t st[2][5][5][2];
    Nine2OneExecutorInput() {};
};

class Nine2OneExecutor
{
private:
    FiniteField &fr;
    const uint64_t slotSize;
    const uint64_t N;
    const uint64_t nSlots9;
public:
    Nine2OneExecutor(FiniteField &fr) :
        fr(fr),
        slotSize(158418),
        N(Nine2OneCommitPols::degree()),
        nSlots9((N-1)/slotSize) {};
    void execute (vector<Nine2OneExecutorInput> &input, Nine2OneCommitPols &pols, vector<vector<FieldElement>> &required);
private:
    FieldElement bitFromState (uint64_t (&st)[5][5][2], uint64_t i);
    FieldElement getBit (vector<Nine2OneExecutorInput> &input, uint64_t block, bool isOut, uint64_t pos);

};

#endif