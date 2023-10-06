#include "keccak.hpp"
#include "zkassert.hpp"
#include "gate_state.hpp"

/*
Keccak-f Theta permutation.
Steps:
1. For all pairs (x, z) such that 0 ≤ x < 5 and 0 ≤ z < w
    C[x, z] = A[x, 0, z] ⊕ A[x, 1, z] ⊕ A[x, 2, z] ⊕ A[x, 3, z] ⊕ A[x, 4, z]
2. For all pairs (x, z) such that 0 ≤ x < 5 and 0≤ z < w
    D[x, z] = C[(x-1) mod 5, z] ⊕ C[(x+1) mod 5, (z –1) mod w]
3. For all triples (x, y, z) such that 0 ≤ x <5, 0 ≤ y < 5, and 0 ≤ z < w
    A′[x, y, z] = A[x, y, z] ⊕ D[x, z]
*/

void KeccakTheta (GateState &S, uint64_t ir)
{
    // C[x, z] = A[x, 0, z] ⊕ A[x, 1, z] ⊕ A[x, 2, z] ⊕ A[x, 3, z] ⊕ A[x, 4, z]
    uint64_t C[5][64];
    for (uint64_t x=0; x<5; x++)
    {
        for (uint64_t z=0; z<64; z++)
        {
            uint64_t aux1;
            uint64_t aux2;
            uint64_t aux3;

            // aux1 = A[x, 0, z] ⊕ A[x, 1, z]
            aux1 = S.getFreeRef();
            if (ir==0) // First time we use pin_a input directly
            {
                zkassert(S.SinRefs[Bit(x, 0, z)] == S.gateConfig.sinRef0 + 44*Bit(x, 0, z));
                zkassert(S.SinRefs[Bit(x, 1, z)] == S.gateConfig.sinRef0 + 44*Bit(x, 1, z));
                S.XOR(S.SinRefs[Bit(x, 0, z)], pin_a, S.SinRefs[Bit(x, 1, z)], pin_a, aux1);
            }
            else
            {
                S.XOR(S.SinRefs[Bit(x, 0, z)], S.SinRefs[Bit(x, 1, z)], aux1);
            }

            // aux2 = aux1 ⊕ A[x, 2, z]
            aux2 = S.getFreeRef();
            if (ir==0) // First time we use pin_a input directly
            {
                zkassert(S.SinRefs[Bit(x, 2, z)] == S.gateConfig.sinRef0 + 44*Bit(x, 2, z));
                S.XOR(S.SinRefs[Bit(x, 2, z)], pin_a, aux1, pin_r, aux2);
            }
            else
            {
                S.XOR(aux1, S.SinRefs[Bit(x, 2, z)], aux2);
            }

            // aux3 = aux2 ⊕ A[x, 3, z]
            aux3 = S.getFreeRef();
            if (ir==0) // First time we use pin_a input directly
            {
                zkassert(S.SinRefs[Bit(x, 3, z)] == S.gateConfig.sinRef0 + 44*Bit(x, 3, z));
                S.XOR(S.SinRefs[Bit(x, 3, z)], pin_a, aux2, pin_r, aux3);
            }
            else
            {
                S.XOR(aux2, S.SinRefs[Bit(x, 3, z)], aux3);
            }
            // C[x][z] = aux ⊕ A[x, 4, z]
            // Calling XORN instead of XOR in order to keep the C[x][z] value=1,
            // since these gates have a big fan-out and impact a lot the overal number of XORNs
            C[x][z] = S.getFreeRef();
            if (ir==0) // First time we use pin_a input directly
            {
                zkassert(S.SinRefs[Bit(x, 4, z)] == S.gateConfig.sinRef0 + 44*Bit(x, 4, z));
                S.XOR(S.SinRefs[Bit(x, 4, z)], pin_a, aux3, pin_r, C[x][z]);
            }
            else
            {
                S.XOR(aux3, S.SinRefs[Bit(x, 4, z)], C[x][z]);
            }
        }
    }

    // D[x, z] = C[(x-1) mod 5, z] ⊕ C[(x+1) mod 5, (z–1) mod w]
    uint64_t D[5][64];
    for (uint64_t x=0; x<5; x++)
    {
        for (uint64_t z=0; z<64; z++)
        {
            D[x][z] = S.getFreeRef();
            S.XOR( C[(x+4)%5][z], C[(x+1)%5][(z+63)%64], D[x][z] ); // D[x][z] has a fan out = 5, value=2
        }
    }

    // A′[x, y, z] = A[x, y, z] ⊕ D[x, z]
    for (uint64_t x=0; x<5; x++)
    {
        for (uint64_t y=0; y<5; y++)
        {
            for (uint64_t z=0; z<64; z++)
            {
                uint64_t aux;
                if (ir==0) // First time we use the first 1600 Sin bit slots to store these gates
                {
                    aux = S.gateConfig.sinRef0 + 44*Bit(x, y, z);
                    zkassert(S.SinRefs[Bit(x, y, z)] == aux);
                    S.XOR(aux, pin_a, D[x][z], pin_r, aux);
                }
                else
                {
                    aux = S.getFreeRef();
                    S.XOR( S.SinRefs[Bit(x, y, z)], D[x][z], aux );
                }
                S.SoutRefs[Bit(x, y, z)] = aux;
            }
        }
    }
}