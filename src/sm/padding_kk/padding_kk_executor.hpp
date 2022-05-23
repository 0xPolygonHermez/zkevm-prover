#ifndef PADDING_KK_EXECUTOR_HPP
#define PADDING_KK_EXECUTOR_HPP

#include <vector>
#include "commit_pols.hpp"
#include "ff/ff.hpp"
#include "sm/padding_kkbit/padding_kkbit_executor.hpp"

using namespace std;

class PaddingKKExecutorInput
{
public:
    string data;
    vector<uint8_t> dataBytes;
    uint64_t realLen;
    vector<uint64_t> reads;
    mpz_class hash;
    PaddingKKExecutorInput() : realLen(0) {};
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
    void execute (vector<PaddingKKExecutorInput> &input, PaddingKKCommitPols &pols, vector<PaddingKKBitExecutorInput> &required);
};


#endif