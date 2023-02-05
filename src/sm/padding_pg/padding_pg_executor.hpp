#ifndef PADDING_PG_EXECUTOR_HPP
#define PADDING_PG_EXECUTOR_HPP

#include <vector>
#include <array>
#include <gmpxx.h>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class PaddingPGExecutorInput
{
public:
    string data;
    vector<uint8_t> dataBytes;
    uint64_t realLen;
    vector<uint64_t> reads;
    mpz_class hash;
    bool digestCalled;
    bool lenCalled;
    PaddingPGExecutorInput() : realLen(0), digestCalled(false), lenCalled(false) {};
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

uint64_t prepareInput (vector<PaddingPGExecutorInput> &input);

public:
    PaddingPGExecutor(Goldilocks &fr, PoseidonGoldilocks &poseidon) :
        fr(fr),
        poseidon(poseidon),
        bytesPerElement(7),
        nElements(8),
        bytesPerBlock(bytesPerElement*nElements),
        N(PROVER_FORK_NAMESPACE::PaddingPGCommitPols::pilDegree()) {};
    void execute (vector<PaddingPGExecutorInput> &input, PROVER_FORK_NAMESPACE::PaddingPGCommitPols &pols, vector<array<Goldilocks::Element, 17>> &required);
};


#endif