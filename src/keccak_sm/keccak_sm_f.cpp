#include "keccak_sm.hpp"

void KeccakSMF (KeccakSMState &S)
{
    // Rnd(A, ir) = ι( χ( π( ρ( θ(A) ) ) ), ir)
    for (uint64_t ir=0; ir<24; ir++ )
    {
        //if (ir==0) S.printRefs(S.SinRefs, "Before theta");
        KeccakSMTheta(S);
        S.copySoutRefsToSinRefs();
        //if (ir==0) S.printRefs(S.SinRefs, "After theta");
        KeccakSMRho(S);
        S.copySoutRefsToSinRefs();
        //if (ir==0) S.printRefs(S.SinRefs, "After rho");
        KeccakSMPi(S);
        S.copySoutRefsToSinRefs();
        //if (ir==0) S.printRefs(S.SinRefs, "After pi");
        KeccakSMChi(S);
        S.copySoutRefsToSinRefs();
        //if (ir==0) S.printRefs(S.SinRefs, "After chi");
        KeccakSMIota(S, ir);
        S.copySoutRefsToSinRefs();
        //if (ir==0) S.printRefs(S.SinRefs, "After iota");
    }
}