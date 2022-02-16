#include "keccak_sm.hpp"

void KeccakSMF (KeccakSMState &S)
{
    // Rnd(A, ir) = ι( χ( π( ρ( θ(A) ) ) ), ir)
    for (uint64_t ir=0; ir<24; ir++ )
    {
        KeccakSMTheta(S);
        S.copySoutToSin();
        KeccakSMRho(S);
        S.copySoutToSin();
        KeccakSMPi(S);
        S.copySoutToSin();
        KeccakSMChi(S);
        S.copySoutToSin();
        KeccakSMIota(S, ir);
        S.copySoutToSin();
    }
}