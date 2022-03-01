#ifndef GATE_HPP
#define GATE_HPP

#include <vector>
#include <stdint.h>

using namespace std;

enum GateOperation
{
    gop_unknown = 0,
    gop_xor     = 1,
    gop_andp    = 2,
    gop_xorn    = 3
};

enum GatePin
{
    pin_input_a = 0,
    pin_input_b = 1,
    pin_output  = 2
};

class Gate
{
public:
    GateOperation op;
    uint64_t refA;
    uint64_t refB;
    uint64_t refR;
    uint64_t pinA;
    uint64_t pinB;
    uint8_t bit[3];
    uint64_t fanOut;
    uint64_t value;
    uint64_t maxValue;
    vector<uint64_t> connectionsToA;
    vector<uint64_t> connectionsToB;
    Gate () { reset(); };
    void reset (void)
    {
        op=gop_unknown;
        refA=0;
        refB=0;
        refR=0;
        pinA=0;
        pinB=0;
        bit[pin_input_a]=0;
        bit[pin_input_b]=0;
        bit[pin_output]=0;
        fanOut=0;
        value=1;
        maxValue=1;
        connectionsToA.clear();
        connectionsToB.clear();
    }
};

#endif