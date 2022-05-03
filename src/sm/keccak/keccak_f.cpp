#include "keccak.hpp"

void KeccakF (KeccakState &S)
{
    // Rnd(A, ir) = ι( χ( π( ρ( θ(A) ) ) ), ir)
    for (uint64_t ir=0; ir<24; ir++ )
    {
        //if (ir==0) S.printRefs(S.SinRefs, "Before theta");
        KeccakTheta(S, ir);
        S.copySoutRefsToSinRefs();
        //if (ir==0) S.printRefs(S.SinRefs, "After theta");
        KeccakRho(S);
        S.copySoutRefsToSinRefs();
        //if (ir==0) S.printRefs(S.SinRefs, "After rho");
        KeccakPi(S);
        S.copySoutRefsToSinRefs();
        //if (ir==0) S.printRefs(S.SinRefs, "After pi");
        KeccakChi(S, ir==23);
        S.copySoutRefsToSinRefs();
        //if (ir==0) S.printRefs(S.SinRefs, "After chi");
        KeccakIota(S, ir, ir==23);
        S.copySoutRefsToSinRefs();
        //if (ir==0) S.printRefs(S.SinRefs, "After iota");
    }
}