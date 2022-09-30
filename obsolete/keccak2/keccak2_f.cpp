#include "keccak2.hpp"

void Keccak2F (Keccak2State &Sin, Keccak2State &Sout)
{
    Keccak2State Saux;

    // Rnd(A, ir) = ι( χ( π( ρ( θ(A) ) ) ), ir)
    for (uint64_t ir=0; ir<24; ir++ )
    {
        //if (ir==0) printBa(Sin.byte, 200, "Before theta");
        Keccak2Theta(Sin, Sout);
        //if (ir==0) printBa(Sout.byte, 200, "After theta");
        Keccak2Rho(Sout, Saux);
        //if (ir==0) printBa(Saux.byte, 200, "After rho");
        Keccak2Pi(Saux, Sout);
        //if (ir==0) printBa(Sout.byte, 200, "After pi");
        Keccak2Chi(Sout, Saux);
        //if (ir==0) printBa(Saux.byte, 200, "After chi");
        Keccak2Iota(Saux, Sout, ir);
        //if (ir==0) printBa(Sout.byte, 200, "After iota");
        Sin = Sout;
    }
}