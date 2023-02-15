#ifndef GATE_U64_HPP
#define GATE_U64_HPP

#include "gate_state.hpp"
#include "gate_bit.hpp"

class GateU64
{
public:
    GateState &S;
    GateBit bit[64];
    GateU64(GateState &S) : S(S) { fromU64(0); };

    void     fromU64     (uint64_t value);
    uint64_t toU64       (void);
    string   toString    (void);
    void     rotateRight (uint64_t pos);
    void     shiftRight  (uint64_t pos);

    GateU64 & operator =(const uint64_t & value)
    {
        fromU64(value);
        return *this;
    }

    GateU64 & operator =(const GateU64 & other)
    {
        for (uint64_t i=0; i<64; i++)
        {
            bit[i] = other.bit[i];
        }
        return *this;
    }
};

void GateU64_xor (GateState &S, const GateU64 &a, const GateU64 &b, GateU64 &r);
void GateU64_and (GateState &S, const GateU64 &a, const GateU64 &b, GateU64 &r);
void GateU64_not (GateState &S, const GateU64 &a,                   GateU64 &r);
void GateU64_add (GateState &S, const GateU64 &a, const GateU64 &b, GateU64 &r);


void GateU64_xor (GateState &S, const GateU64 &a, const GateBit &b, GateU64 &r);

#endif