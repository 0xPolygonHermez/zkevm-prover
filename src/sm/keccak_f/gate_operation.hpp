#ifndef GATE_OPERATION_HPP
#define GATE_OPERATION_HPP

enum GateOperation
{
    gop_unknown = 0,
    gop_xor     = 1, // r = XOR(a,b)
    gop_andp    = 2, // r = ANDP(a,b) = AND(NOT(a),b)
};

#endif