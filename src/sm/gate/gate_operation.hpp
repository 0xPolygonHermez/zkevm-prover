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
    gop_ch      = 5, // r = ch(a,b,c)
    gop_maj     = 6, // r = maj(a,b,c)
    gop_add     = 7  // r = add(a,b,c)

};

// Map an operation code into a string
string gateop2string (GateOperation op);

#endif