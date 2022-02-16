#include "keccak2_state.hpp"

/*
Steps:
1. For all triples (x, y, z) such that 0 ≤ x <5, 0 ≤ y < 5, and 0 ≤ z < w
    A′ [x, y, z] = A[x, y, z] ⊕ ((A[(x+1) mod 5, y, z] ⊕ 1) ⋅ A[(x+2) mod 5, y, z])
2. Return A′
*/

void Keccak2Chi (Keccak2State &Sin, Keccak2State &Sout)
{
    Sout.copyCounters(Sin);

    // A′ [x, y, z] = A[x, y, z] ⊕ ( (A[(x+1) mod 5, y, z] ⊕ 1) ⋅ A[(x+2) mod 5, y, z] )
    for (uint64_t x=0; x<5; x++)
    {
        for (uint64_t y=0; y<5; y++)
        {
            for (uint64_t z=0; z<64; z++)
            {
                Sout.setBit(x, y, z, Sin.getBit(x, y, z)^((Sin.getBit((x+1)%5, y, z)^1)&Sin.getBit((x+2)%5, y, z)));
                Sout.xors += 2;
                Sout.ands ++;
                Sout.adds += 2;
                Sout.mods += 2;
            }
        }
    }
}