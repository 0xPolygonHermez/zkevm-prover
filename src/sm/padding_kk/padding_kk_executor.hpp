#ifndef PADDING_KKT_EXECUTOR_HPP
#define PADDING_KKT_EXECUTOR_HPP

#include <vector>
#include "commit_pols.hpp"
#include "ff/ff.hpp"

using namespace std;

class PaddingKKExecutorInput
{
public:
    string data;
    vector<uint8_t> dataBytes;
    uint64_t realLen;
    vector<uint64_t> reads;
    mpz_class hash;
};

class PaddingKKExecutor
{
private:
    FiniteField &fr;
    const uint64_t blockSize;
    const uint64_t bytesPerBlock;

void prepareInput (vector<PaddingKKExecutorInput> &input);

public:
    PaddingKKExecutor(FiniteField &fr) : fr(fr), blockSize(158418), bytesPerBlock(136) {};
    void executor (vector<PaddingKKExecutorInput> &input, PaddingKKCommitPols & pols);
};


#endif