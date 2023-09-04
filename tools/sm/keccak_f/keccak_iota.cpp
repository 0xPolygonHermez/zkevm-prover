#include "keccak.hpp"
#include "keccak_rc.hpp"
#include "gate_state.hpp"

/*
Keccak-f Iota permutation.
Steps:
1. For all triples (x, y, z) such that 0 ≤ x <5, 0 ≤ y < 5, and 0 ≤ z < w, let A′[x, y, z] = A[x, y, z]
2. Let RC = 0w.
3. For j from 0 to l, let RC[2^j – 1] = rc(j + 7ir).
4. For all z such that 0 ≤ z <w, let A′ [0, 0, z] = A′[0, 0, z] ⊕ RC[z].
5. Return A′
*/

void KeccakIota (GateState &S, uint64_t ir)
{
    // Init KeccakRC, if required
    KeccakRCInit();

    // A′[x, y, z] = A[x, y, z]
    for (uint64_t x=0; x<5; x++)
    {
        for (uint64_t y=0; y<5; y++)
        {
            for (uint64_t z=0; z<64; z++)
            {
                S.SoutRefs[Bit(x, y, z)] = S.SinRefs[Bit(x, y, z)];
            }
        }
    }

    // For all z such that 0 ≤ z <w, let A′ [0, 0, z] = A′[0, 0, z] ⊕ RC[z]
    for (uint64_t z=0; z<64; z++)
    {
        if (KeccakRC[ir][z]==0)
        {
            continue;
        }
        uint64_t aux;
        aux = S.getFreeRef();
        if (KeccakRC[ir][z] == 1)
        {
            if (ir==23)
            {
                S.XOR(S.gateConfig.zeroRef, pin_b, S.SoutRefs[Bit(0, 0, z)], pin_r, aux );
            }
            else
            {
                S.XOR(S.gateConfig.zeroRef, pin_b, S.SoutRefs[Bit(0, 0, z)], pin_r, aux );
            }
        }
        else
        {
            if (ir==23)
            {
                S.XOR(S.gateConfig.zeroRef, pin_a, S.SoutRefs[Bit(0, 0, z)], pin_r, aux );
            }
            else
            {
                S.XOR(S.gateConfig.zeroRef, pin_a, S.SoutRefs[Bit(0, 0, z)], pin_r, aux );
            }
        }
        S.SoutRefs[Bit(0, 0, z)] = aux;
    }
}