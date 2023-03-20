#ifndef PADDING_KK_EXECUTOR_HPP
#define PADDING_KK_EXECUTOR_HPP

#include <vector>
#include <gmpxx.h>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "goldilocks_base_field.hpp"
#include "sm/padding_kkbit/padding_kkbit_executor.hpp"
#include "scalar.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class PaddingKKExecutorInput
{
public:
    string data;
    vector<uint8_t> dataBytes;
    uint64_t realLen;
    vector<uint64_t> reads;
    mpz_class hash;
    bool digestCalled;
    bool lenCalled;
    PaddingKKExecutorInput() : realLen(0), digestCalled(false), lenCalled(false) {};
};

class PaddingKKExecutor
{
private:

    /* Goldilocks reference */
    Goldilocks &fr;

    /* Constant values */
    const uint64_t blockSize;
    const uint64_t bytesPerBlock;
    const uint64_t N;

    /* Hash of an empty/zero message */
    mpz_class hashZeroScalar;
    Goldilocks::Element hash0[8];

uint64_t prepareInput (vector<PaddingKKExecutorInput> &input);

public:

    /* Constructor */
    PaddingKKExecutor(Goldilocks &fr) :
        fr(fr),
        blockSize(155286),
        bytesPerBlock(136),
        N(PROVER_FORK_NAMESPACE::PaddingKKCommitPols::pilDegree())
    {
        keccak256(NULL, 0, hashZeroScalar);
        scalar2fea(fr, hashZeroScalar, hash0);
    };

    /* Executor */
    void execute (vector<PaddingKKExecutorInput> &input, PROVER_FORK_NAMESPACE::PaddingKKCommitPols &pols, vector<PaddingKKBitExecutorInput> &required);
};


#endif