#include "gate_state.hpp"
#include "keccak.hpp"

/*
Keccak-f Rho permutation.
Steps:
1. For all z such that 0 ≤ z <w, let A′ [0, 0, z] = A[0, 0, z]
2. Let (x, y) = (1, 0)
3. For t from 0 to 23:
    a. for all z such that 0 ≤ z <w, let A′[x, y, z] = A[x, y, (z – (t +1)(t + 2)/2) mod w]
    b. let (x, y) = (y, (2x + 3y) mod 5)
4. Return A′
*/

void KeccakRho (GateState &S)
{
    
    // For all z such that 0 ≤ z <w, let A′ [0, 0, z] = A[0, 0, z]
    for (uint64_t z=0; z<64; z++)
    {
        S.SoutRefs[Bit(0, 0, z)] = S.SinRefs[Bit(0, 0, z)];
    }

    // Let (x, y) = (1, 0)
    uint64_t x = 1;
    uint64_t y = 0;

    // For t from 0 to 23:
    for (uint64_t t=0; t<24; t++)
    {
        // for all z such that 0 ≤ z <w, let A′[x, y, z] = A[x, y, (z – (t +1)(t + 2)/2) mod w]
        for (uint64_t z=0; z<64; z++)
        {
            S.SoutRefs[Bit(x, y, z)] = S.SinRefs[Bit(x, y, (z - (t + 1)*(t + 2)/2)%64)];
        }

        // let (x, y) = (y, (2x + 3y) mod 5)
        uint64_t aux = y;
        y = (2*x + 3*y)%5;
        x = aux;
    }
}