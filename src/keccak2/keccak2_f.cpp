#include "keccak2.hpp"

void Keccak2F (Keccak2State &Sin, Keccak2State &Sout)
{
    Keccak2State Saux;

    // Rnd(A, ir) = ι( χ( π( ρ( θ(A) ) ) ), ir)
    for (uint64_t ir=0; ir<24; ir++ )
    {
        Keccak2Theta(Sin, Sout);
        Keccak2Rho(Sout, Saux);
        Keccak2Pi(Saux, Sout);
        Keccak2Chi(Sout, Saux);
        Keccak2Iota(Saux, Sout, ir);
        Sin = Sout;
    }
}