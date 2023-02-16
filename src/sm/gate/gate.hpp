#ifndef GATE_HPP
#define GATE_HPP

#include "gate_operation.hpp"
#include "pin.hpp"

class Gate
{
public:
    GateOperation op;
    Pin pin[3];
    Gate () : pin{pin_a, pin_b, pin_r} { reset(); };
    void reset (void)
    {
        // The default value of a gate is r = XOR(0,0), where 0 is externally set
        op=gop_xor;
        pin[0].reset();
        pin[0].source = external;
        pin[0].bit = 0;
        pin[1].reset();
        pin[1].source = external;
        pin[1].bit = 0;
        pin[2].reset();
        pin[2].source = gated;
        pin[2].bit = pin[0].bit ^ pin[1].bit;
    }
};

#endif