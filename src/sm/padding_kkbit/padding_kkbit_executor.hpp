#ifndef PADDING_KKBIT_EXECUTOR_HPP
#define PADDING_KKBIT_EXECUTOR_HPP

#include <vector>
#include "commit_pols.hpp"

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
    void executor (vector<PaddingKKBitExecutorInput> &input, PaddingKKBitCommitPols & pols);
};

#endif