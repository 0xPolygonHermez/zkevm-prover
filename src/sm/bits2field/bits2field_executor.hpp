#ifndef BITS2FIELD_EXECUTOR_HPP
#define BITS2FIELD_EXECUTOR_HPP

#include <vector>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class Bits2FieldExecutorInput
{
public:

    /* Input and output states */
    uint8_t inputState[200];
    uint8_t outputState[200];
};

class Bits2FieldExecutor
{
private:

    /* Goldilocks reference */
    Goldilocks &fr;

    /* Constant values */
    const uint64_t slotSize;
    const uint64_t N;
    const uint64_t nSlots;

public:

    /* Constructor */
    Bits2FieldExecutor(Goldilocks &fr) :
        fr(fr),
        slotSize(155286),
        N(Bits2FieldCommitPols::pilDegree()),
        nSlots((N-1)/slotSize) {};

    /* Executor */
    void execute (vector<Bits2FieldExecutorInput> &input, Bits2FieldCommitPols &pols, vector<vector<Goldilocks::Element>> &required);

private:

    /* Gets bit "pos" from input vector position "block" */
    Goldilocks::Element getBit (vector<Bits2FieldExecutorInput> &input, uint64_t block, bool isOutput, uint64_t pos);

};

#endif