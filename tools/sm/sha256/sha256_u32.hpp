#ifndef SHA256_U32_HPP
#define SHA256_U32_HPP

#include "sha256_state.hpp"

class SHA256_Bit
{
public:
    uint64_t ref;
    PinId pin;

    SHA256_Bit & operator =(const SHA256_Bit & other)
    {
        ref = other.ref;
        pin = other.pin;
        return *this;
    }
};

class SHA256_U32
{
public:
    SHA256_Bit bit[32];
    SHA256_U32(uint32_t value) { fromU32(value); };
    SHA256_U32() { fromU32(0); };
    void fromU32 (uint32_t value);
    uint32_t toU32 (SHA256_State &S);
    string toString (SHA256_State &S);
    void rotateRight (uint64_t pos);
    void shiftRight (uint64_t pos);

    SHA256_U32 & operator =(const uint32_t & value)
    {
        fromU32(value);
        return *this;
    }

    SHA256_U32 & operator =(const SHA256_U32 & other)
    {
        for (uint64_t i=0; i<32; i++)
        {
            bit[i] = other.bit[i];
        }
        return *this;
    }
};

void SHA256_xor (SHA256_State &S, const SHA256_U32 &a, const SHA256_U32 &b, SHA256_U32 &r);
void SHA256_and (SHA256_State &S, const SHA256_U32 &a, const SHA256_U32 &b, SHA256_U32 &r);
void SHA256_not (SHA256_State &S, const SHA256_U32 &a,                      SHA256_U32 &r);
void SHA256_add (SHA256_State &S, const SHA256_U32 &a, const SHA256_U32 &b, SHA256_U32 &r);

#endif