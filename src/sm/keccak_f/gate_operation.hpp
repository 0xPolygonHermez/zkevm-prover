#ifndef GATE_OPERATION_HPP
#define GATE_OPERATION_HPP

enum GateOperation
{
    gop_unknown = 0,
    gop_xor     = 1, // r = XOR(a,b)
    gop_andp    = 2, // r = AND(NOT(a),b)
    gop_xorn    = 3  // r = XOR(a,b) with carry reset
};

#endif