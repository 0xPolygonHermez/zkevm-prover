#include "smkeccak_state.hpp"

/*
Steps:
1. For all triples (x, y, z) such that 0 ≤ x <5, 0 ≤ y < 5, and 0 ≤ z < w, let
A′[x, y, z]= A[(x + 3y) mod 5, x, z].
2. Return A′.
*/

void SMKeccakPi (SMKeccakState &S)
{
    // A′[x, y, z]= A[(x + 3y) mod 5, x, z]
    for (uint64_t x=0; x<5; x++)
    {
        for (uint64_t y=0; y<5; y++)
        {
            for (uint64_t z=0; z<64; z++)
            {
                //Sout.setBit(x, y, z, Sin.getBit((x+3*y)%5, x, z));
                S.SoutRefs[S.getBit(x, y, z)] = Sin + S.getBit((x+3*y)%5, x, z);
            }
        }
    }
}