#include "keccak2.hpp"

void KeccakF (KeccakState &Sin, KeccakState &Sout)
{
    KeccakState Saux;

    // Rnd(A, ir) = ι( χ( π( ρ( θ(A) ) ) ), ir)
    for (uint64_t ir=0; ir<24; ir++ )
    {
        KeccakTheta(Sin, Sout);
        KeccakRho(Sout, Saux);
        KeccakPi(Saux, Sout);
        KeccakChi(Sout, Saux);
        KeccakIota(Saux, Sout, ir);
        Sin = Sout;
    }
}