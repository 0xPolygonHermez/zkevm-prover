#include "keccak2.hpp"

void KeccakF (const KeccakState &Sin, KeccakState &Sout)
{
    KeccakState Saux1 = Sin;
    KeccakState Saux2;
    for (uint64_t ir=0; ir<24; ir++ )
    {
        KeccakRound(Saux1, Saux2, ir);
        Saux1 = Saux2;
    }
    Sout = Saux2;
}