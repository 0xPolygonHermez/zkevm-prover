#include "smkeccak.hpp"

void SMKeccakF (SMKeccakState &S)
{
    // Rnd(A, ir) = ι( χ( π( ρ( θ(A) ) ) ), ir)
    for (uint64_t ir=0; ir<24; ir++ )
    {
        SMKeccakTheta(S);
        S.copySoutToSin();
        SMKeccakRho(S);
        S.copySoutToSin();
        SMKeccakPi(S);
        S.copySoutToSin();
        SMKeccakChi(S);
        S.copySoutToSin();
        SMKeccakIota(S, ir);
        S.copySoutToSin();
    }
}