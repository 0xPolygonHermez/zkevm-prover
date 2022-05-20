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
public:
    Nine2OneExecutor(FiniteField &fr) : fr(fr), slotSize(158418) {};
    void execute (vector<Nine2OneExecutorInput> &input, Nine2OneCommitPols &pols);
private:
    FieldElement bitFromState (uint64_t (&st)[5][5][2], uint64_t i);
    FieldElement getBit (vector<Nine2OneExecutorInput> &input, uint64_t block, bool isOut, uint64_t pos);

};

#endif