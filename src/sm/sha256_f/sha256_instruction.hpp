#ifndef SHA256_SM_INSTRUCTION_HPP
#define SHA256_SM_INSTRUCTION_HPP

#include <array>
#include "gate_operation.hpp"

using namespace std;

enum TypeSha256Gate
{
    type_unknown = 0,
    type_wired = 1,
    type_input = 2,
    type_inputState = 3
};

class Sha256Instruction
{
public:

    GateOperation op;
    uint64_t ref;

    bool in[4];
    uint64_t pin[4];
    TypeSha256Gate type[4];
    uint64_t gate[4];
    uint64_t bit[4];

    Sha256Instruction () {
        op = gop_xor;
        ref = 0;
        in[0]=false;
        in[1]=false;
        in[2]=false;
        in[3]=false;
        memset(pin, 0, sizeof(pin));
        memset(type, 0, sizeof(type));
        memset(gate, 0, sizeof(gate));
        memset(bit, 0, sizeof(bit));
    }
};

#endif