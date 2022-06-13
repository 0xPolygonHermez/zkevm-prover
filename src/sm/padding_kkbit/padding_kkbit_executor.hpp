#ifndef PADDING_KKBIT_EXECUTOR_HPP
#define PADDING_KKBIT_EXECUTOR_HPP

#include <vector>
#include "commit_pols.hpp"
#include "sm/nine2one/nine2one_executor.hpp"

using namespace std;

class PaddingKKBitExecutorInput
{
public:
    bool connected;
    uint8_t r[136];
    PaddingKKBitExecutorInput() : connected(false) {};
};

class PaddingKKBitExecutor
{
private:
    Goldilocks &fr;
    const uint64_t N;
    const uint64_t slotSize;
    const uint64_t nSlots;
public:
    PaddingKKBitExecutor(Goldilocks &fr) :
        fr(fr),
        N(PaddingKKBitCommitPols::degree()),
        slotSize(158418),
        nSlots(9*((N-1)/slotSize)) {};
    void execute (vector<PaddingKKBitExecutorInput> &input, PaddingKKBitCommitPols &pols, vector<Nine2OneExecutorInput> &required);
};

#endif