#ifndef GATE_OPERATION_HPP
#define GATE_OPERATION_HPP

#include <string>

using namespace std;

enum GateOperation
{
    gop_unknown = 0,
    gop_xor     = 1, // r = XOR(a,b)
    gop_andp    = 2, // r = ANDP(a,b) = AND(NOT(a),b)
    gop_or      = 3, // r = OR(a,b)
    gop_and     = 4, // r = AND(a,b)
};

// Map an operation code into a string
string gateop2string (GateOperation op);

#endif