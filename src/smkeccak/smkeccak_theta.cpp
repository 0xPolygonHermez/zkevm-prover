#include "smkeccak_state.hpp"

/*
Steps:
1. For all pairs (x, z) such that 0 ≤ x < 5 and 0 ≤ z < w
    C[x, z] = A[x, 0, z] ⊕ A[x, 1, z] ⊕ A[x, 2, z] ⊕ A[x, 3, z] ⊕ A[x, 4, z]
2. For all pairs (x, z) such that 0 ≤ x < 5 and 0≤ z < w
    D[x, z] = C[(x-1) mod 5, z] ⊕ C[(x+1) mod 5, (z –1) mod w]
3. For all triples (x, y, z) such that 0 ≤ x <5, 0 ≤ y < 5, and 0 ≤ z < w
    A′[x, y, z] = A[x, y, z] ⊕ D[x, z]
*/

void SMKeccakTheta (SMKeccakState &S)
{
    uint64_t aux1 = S.getFreePos();
    uint64_t aux2 = S.getFreePos();

    // C[x, z] = A[x, 0, z] ⊕ A[x, 1, z] ⊕ A[x, 2, z] ⊕ A[x, 3, z] ⊕ A[x, 4, z]
    uint64_t C[5][64];
    for (uint64_t x=0; x<5; x++)
    {
        for (uint64_t z=0; z<64; z++)
        {
            C[x][z] = S.getFreePos();

            // C[x][z] = Sin.getBit(x, 0, z) ^ Sin.getBit(x, 1, z) ^ Sin.getBit(x, 2, z) ^ Sin.getBit(x, 3, z) ^ Sin.getBit(x, 4, z);
            S.XOR(Sin + S.getBit(x, 0, z), Sin + S.getBit(x, 1, z), aux1);
            S.XOR(aux1, Sin + S.getBit(x, 2, z), aux2);
            S.XOR(aux2, Sin + S.getBit(x, 3, z), aux1);
            S.XOR(aux1, Sin + S.getBit(x, 4, z), C[x][z]);
        }
    }

    // D[x, z] = C[(x-1) mod 5, z] ⊕ C[(x+1) mod 5, (z –1) mod w]
    uint64_t D[5][64];
    for (uint64_t x=0; x<5; x++)
    {
        for (uint64_t z=0; z<64; z++)
        {
            D[x][z] = S.getFreePos();

            //D[x][z] = C[(x+4)%5][z] ^ C[(x+1)%5][(z+63)%64];
            S.XOR(C[(x+4)%5][z], C[(x+1)%5][(z+63)%64], D[x][z]);
        }
    }

    // A′[x, y, z] = A[x, y, z] ⊕ D[x, z]
    for (uint64_t x=0; x<5; x++)
    {
        for (uint64_t y=0; y<5; y++)
        {
            for (uint64_t z=0; z<64; z++)
            {
                //Sout.setBit(x, y, z, Sin.getBit(x, y, z)^D[x][z]);
                S.XOR(Sin + S.getBit(x, y, z), D[x][z], Sout + S.getBit(x, y, z));
            }
        }
    }
}