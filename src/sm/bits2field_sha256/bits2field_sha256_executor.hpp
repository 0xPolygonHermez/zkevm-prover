#ifndef BITS2FIELD_SHA256_EXECUTOR_HPP
#define BITS2FIELD_SHA256_EXECUTOR_HPP

#include <vector>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "sm/sha256_f/sha256_f_executor.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class Bits2FieldSha256ExecutorInput
{
public:

    /* Input block and state, and output state */
    uint8_t  inBlock[64];
    uint32_t inputState[8];
    uint32_t outputState[8];
};
class Bits2FieldSha256Executor
{
private:

    /* Goldilocks reference */
    Goldilocks &fr;

    /* Constant values */
    const uint64_t slotSize;
    const uint64_t N;
    const uint64_t bitsPerElement;
    const uint64_t nSlots;

public:

    /* Constructor */
    Bits2FieldSha256Executor(Goldilocks &fr) :
        fr(fr),
        slotSize(31488),
        N(Bits2FieldSha256CommitPols::pilDegree()),
        bitsPerElement(7),
        nSlots((N-1)/slotSize) {};

    /* Executor */
    void execute (vector<Bits2FieldSha256ExecutorInput> &input, Bits2FieldSha256CommitPols &pols, vector<Sha256FExecutorInput> &required);

private:
    enum BitType {
        STATE_IN,
        STATE_OUT,
        BLOCK_IN
    };

    /* Gets bit "pos" from input vector position "block" */
    Goldilocks::Element getBit (vector<Bits2FieldSha256ExecutorInput> &input, uint64_t block, BitType type, uint64_t pos);

};

#endif