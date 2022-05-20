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
    const uint64_t slotSize;
public:
    PaddingKKBitExecutor() : slotSize(158418) {};
    void execute (vector<PaddingKKBitExecutorInput> &input, PaddingKKBitCommitPols &pols, vector<Nine2OneExecutorInput> &required);
};

#endif