#ifndef KECCAK_SM_INSTRUCTION_HPP
#define KECCAK_SM_INSTRUCTION_HPP

#include <array>
#include "gate.hpp"

using namespace std;

class KeccakInstruction
{
public:
    GateOperation op;
    uint64_t refa;
    uint64_t refb;
    uint64_t refr;
    uint64_t pina;
    uint64_t pinb;
    KeccakInstruction () {
        op = gop_xor;
        refa = 0;
        refb = 0;
        refr = 0;
        pina = 0;
        pinb = 0;
    }
};

#endif