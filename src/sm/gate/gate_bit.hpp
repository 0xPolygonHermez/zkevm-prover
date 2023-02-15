#ifndef GATE_BIT_HPP
#define GATE_BIT_HPP

class GateBit
{
public:
    uint64_t ref;
    PinId pin;

    GateBit & operator =(const GateBit & other)
    {
        ref = other.ref;
        pin = other.pin;
        return *this;
    }
};

#endif