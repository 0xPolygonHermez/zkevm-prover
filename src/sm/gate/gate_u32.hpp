#ifndef GATE_U32_HPP
#define GATE_U32_HPP

#include "gate_state.hpp"
#include "gate_bit.hpp"

class GateU32
{
public:
    GateState &S;
    GateBit bit[32];
    GateU32(GateState &S) : S(S) { fromU32(0); };

    void     fromU32     (uint32_t value);
    uint32_t toU32       (void);
    string   toString    (void);
    void     rotateRight (uint64_t pos);
    void     shiftRight  (uint64_t pos);

    GateU32 & operator =(const uint32_t & value)
    {
        fromU32(value);
        return *this;
    }

    GateU32 & operator =(const GateU32 & other)
    {
        for (uint64_t i=0; i<32; i++)
        {
            bit[i] = other.bit[i];
        }
        return *this;
    }
};

void GateU32_xor (GateState &S, const GateU32 &a, const GateU32 &b, GateU32 &r);
void GateU32_and (GateState &S, const GateU32 &a, const GateU32 &b, GateU32 &r);
void GateU32_not (GateState &S, const GateU32 &a,                   GateU32 &r);
void GateU32_add (GateState &S, const GateU32 &a, const GateU32 &b, GateU32 &r);

#endif