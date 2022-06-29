#ifndef PADDING_PG_EXECUTOR_HPP
#define PADDING_PG_EXECUTOR_HPP

#include <vector>
#include <array>
#include <gmpxx.h>
#include "commit_pols.hpp"
#include "goldilocks/goldilocks_base_field.hpp"
#include "goldilocks/poseidon_goldilocks.hpp"

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
    Goldilocks &fr;
    PoseidonGoldilocks &poseidon;
    const uint64_t bytesPerElement;
    const uint64_t nElements;
    const uint64_t bytesPerBlock;
    const uint64_t N;

void prepareInput (vector<PaddingPGExecutorInput> &input);

public:
    PaddingPGExecutor(Goldilocks &fr, PoseidonGoldilocks &poseidon) :
        fr(fr),
        poseidon(poseidon),
        bytesPerElement(7),
        nElements(8),
        bytesPerBlock(bytesPerElement*nElements),
        N(PaddingPGCommitPols::pilDegree()) {};
    void execute (vector<PaddingPGExecutorInput> &input, PaddingPGCommitPols &pols, vector<array<Goldilocks::Element, 16>> &required);
};


#endif