#ifndef PADDING_PG_EXECUTOR_HPP
#define PADDING_PG_EXECUTOR_HPP

#include <vector>
#include "commit_pols.hpp"
#include "ff/ff.hpp"
#include "poseidon_opt/poseidon_goldilocks.hpp"

using namespace std;

class PaddingPGExecutorInput
{
public:
    string data;
    vector<uint8_t> dataBytes;
    uint64_t realLen;
    vector<uint64_t> reads;
    mpz_class hash;
    PaddingPGExecutorInput() : realLen(0) {};
};

class PaddingPGExecutor
{
private:
    FiniteField &fr;
    Poseidon_goldilocks &poseidon;
    const uint64_t bytesPerElement;
    const uint64_t nElements;
    const uint64_t bytesPerBlock;

void prepareInput (vector<PaddingPGExecutorInput> &input);

public:
    PaddingPGExecutor(FiniteField &fr, Poseidon_goldilocks &poseidon) : fr(fr), poseidon(poseidon), bytesPerElement(7), nElements(8), bytesPerBlock(bytesPerElement*nElements) {};
    void execute (vector<PaddingPGExecutorInput> &input, PaddingPGCommitPols &pols/*, vector<PaddingKKBitExecutorInput> &required*/);
};


#endif