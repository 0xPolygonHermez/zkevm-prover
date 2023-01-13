#ifndef NINE2ONE_EXECUTOR_HPP
#define NINE2ONE_EXECUTOR_HPP

#include <vector>
#include "commit_pols.hpp"

using namespace std;

class Nine2OneExecutorInput
{
public:

    /* Input and output states */
    uint8_t inputState[200];
    uint8_t outputState[200];
};

class Nine2OneExecutor
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
    Nine2OneExecutor(Goldilocks &fr) :
        fr(fr),
        slotSize(155286),
        N(Nine2OneCommitPols::pilDegree()),
        nSlots((N-1)/slotSize) {};

    /* Executor */
    void execute (vector<Nine2OneExecutorInput> &input, Nine2OneCommitPols &pols, vector<vector<Goldilocks::Element>> &required);

private:

    /* Gets bit "pos" from input vector position "block" */
    Goldilocks::Element getBit (vector<Nine2OneExecutorInput> &input, uint64_t block, bool isOutput, uint64_t pos);

};

#endif