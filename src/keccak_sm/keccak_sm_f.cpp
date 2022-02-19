#include "keccak_sm.hpp"

void KeccakSMF (KeccakSMState &S)
{
    // Rnd(A, ir) = ι( χ( π( ρ( θ(A) ) ) ), ir)
    for (uint64_t ir=0; ir<24; ir++ )
    {
        KeccakSMTheta(S);
        S.copySoutToSin();
        S.resetSoutRefs();
        KeccakSMRho(S);
        S.copySoutToSin();
        S.resetSoutRefs();
        KeccakSMPi(S);
        S.copySoutToSin();
        S.resetSoutRefs();
        KeccakSMChi(S);
        S.copySoutToSin();
        S.resetSoutRefs();
        KeccakSMIota(S, ir);
        S.copySoutToSin();
        if (ir!=23) S.resetSoutRefs();
    }
}