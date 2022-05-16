#ifndef PADDING_KKBIT_EXECUTOR_HPP
#define PADDING_KKBIT_EXECUTOR_HPP

#include <vector>
#include "commit_pols.hpp"

using namespace std;

class PaddingKKBitExecutor
{
private:
    const uint64_t slotSize;
public:
    PaddingKKBitExecutor() : slotSize(158418) {};
    void executor (vector<uint64_t[3]> &input, PaddingKKBitCommitPols & pols);
};

#endif