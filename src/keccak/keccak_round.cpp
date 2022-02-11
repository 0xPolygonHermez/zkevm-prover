#include "keccak_state.hpp"
#include "keccak2.hpp"

// Rnd(A, ir) = ι( χ( π( ρ( θ(A) ) ) ), ir)

void KeccakRound (const KeccakState &Sin, KeccakState &Sout, uint64_t ir)
{
    KeccakState Saux;
    KeccakTheta(Sin, Sout);
    KeccakRho(Sout, Saux);
    KeccakPi(Saux, Sout);
    KeccakChi(Sout, Saux);
    KeccakIota(Saux, Sout, ir);
}