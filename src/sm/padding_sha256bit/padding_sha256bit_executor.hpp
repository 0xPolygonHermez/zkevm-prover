#ifndef PADDING_SHA256BIT_EXECUTOR_HPP
#define PADDING_SHA256BIT_EXECUTOR_HPP

#include <vector>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "sm/bits2field_sha256/bits2field_sha256_executor.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class PaddingSha256BitExecutorInput
{
public:

    /* If connected, this means that this is not the first block of the sha256 message to hash,
       and then the state of the previous block is used as initial state of this block */
    bool connected;

    /* Input sha256 block data: 64 bytes = 512 bits */
    uint8_t data[64];

    /* Constructor */
    PaddingSha256BitExecutorInput() : connected(false) {};
};

class PaddingSha256BitExecutor
{
private:

    /* Goldilocks reference */
    Goldilocks &fr;

    /* Constant values */
    const uint64_t N;
    const uint64_t slotSize;
    const uint64_t bitsPerElement;
    const uint64_t nSlots;

public:

    /* Constructor */
    PaddingSha256BitExecutor(Goldilocks &fr) :
        fr(fr),
        N(PROVER_FORK_NAMESPACE::PaddingSha256BitCommitPols::pilDegree()),
        slotSize(31488),
        bitsPerElement(7),
        nSlots(bitsPerElement*((N-1)/slotSize)) {};

    /* Executor */
    void execute (vector<PaddingSha256BitExecutorInput> &input, PROVER_FORK_NAMESPACE::PaddingSha256BitCommitPols &pols, vector<Bits2FieldSha256ExecutorInput> &required);
};

#endif