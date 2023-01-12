#ifndef NINE2ONE_EXECUTOR_HPP
#define NINE2ONE_EXECUTOR_HPP

#include <vector>
#include "commit_pols.hpp"

using namespace std;

class Nine2OneExecutorInput
{
public:
    /* Input and output states: st[I/O][x][y][z/2]
       x, y, z as per Keccak-f spec
       st[0] is the input state
       st[1] is the output state
       x = 0...4
       y = 0...4
       z = 0...63
       z values are split into 2 chunks of 32bits, stored in the 32 lower bits of uint64_t
    */
    uint64_t st[2][5][5][2];
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
    /* Gets bit "i" from 1600-bits state */
    Goldilocks::Element bitFromState (uint64_t (&st)[5][5][2], uint64_t i);

    /* Gets bit "pos" from input vector position "block" */
    Goldilocks::Element getBit (vector<Nine2OneExecutorInput> &input, uint64_t block, bool isOut, uint64_t pos);

};

#endif