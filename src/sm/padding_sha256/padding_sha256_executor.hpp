#ifndef PADDING_SHA256_EXECUTOR_HPP
#define PADDING_SHA256_EXECUTOR_HPP

#include <vector>
#include <gmpxx.h>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "goldilocks_base_field.hpp"
#include "padding_sha256bit_executor.hpp"
#include "scalar.hpp"
#include "sha256.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class PaddingSha256ExecutorInput
{
public:
    string data;
    vector<uint8_t> dataBytes;
    uint64_t realLen;
    vector<uint64_t> reads;
    mpz_class hash;
    bool digestCalled;
    bool lenCalled;
    PaddingSha256ExecutorInput() : realLen(0), digestCalled(false), lenCalled(false) {};
};

class PaddingSha256Executor
{
private:

    /* Goldilocks reference */
    Goldilocks &fr;

    /* Constant values */
    const uint64_t blockSize;
    const uint64_t bytesPerBlock;
    const uint64_t bitsPerElement;
    const uint64_t N;

    /* Hash of an empty/zero message */
    mpz_class hashZeroScalar;
    Goldilocks::Element hash0[8];

uint64_t prepareInput (vector<PaddingSha256ExecutorInput> &input);

public:

    /* Constructor */
    PaddingSha256Executor(Goldilocks &fr) :
        fr(fr),
        blockSize(31488),
        bytesPerBlock(64),
        bitsPerElement(7),
        N(PROVER_FORK_NAMESPACE::PaddingSha256CommitPols::pilDegree())
    {
        SHA256(NULL, 0, hashZeroScalar);
        scalar2fea(fr, hashZeroScalar, hash0);
    };

    /* Executor */
    void execute (vector<PaddingSha256ExecutorInput> &input, PROVER_FORK_NAMESPACE::PaddingSha256CommitPols &pols, vector<PaddingSha256BitExecutorInput> &required);
};


#endif