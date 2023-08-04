#include <iostream>
#include "gate_u64.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "exit_process.hpp"
#include "zklog.hpp"

using namespace std;

void GateU64::fromU64 (uint64_t value)
{
    // Convert U64 value to an array of bits
    vector<uint8_t> bits;
    u642bits(value, bits);
    zkassert(bits.size() == 64);

    // Assign the proper reference
    for (uint64_t i=0; i<64; i++)
    {
        bit[i].ref = S.gateConfig.zeroRef;
        if (bits[i] == 0)
        {
            bit[i].pin = pin_a;
        }
        else if (bits[i] == 1)
        {
            bit[i].pin = pin_b;
        }
        else
        {
            zklog.error("GateU64::fromU64() got invalid bit value=" + to_string(value) + " i=" + to_string(i) + " bits[i]=" + to_string(bits[i]));
            exitProcess();
        }
    }
}

uint64_t GateU64::toU64 (void)
{
    // Collect bits
    vector<uint8_t> bits;
    for (uint64_t i=0; i<64; i++)
    {
        bits.push_back(S.gate[bit[i].ref].pin[bit[i].pin].bit);
    }

    // Convert from bits to U64
    return bits2u64(bits);
}

string GateU64::toString (void)
{
    mpz_class aux = toU64();
    return aux.get_str(16);
}

void GateU64::rotateRight (uint64_t pos)
{
    // Store the rotated value on a temporary variable
    GateBit auxBit[64];
    for (uint64_t i=0; i<64; i++)
    {
        auxBit[i] = bit[(i+pos)%64];
    }

    // Copy the temporary variable into ref
    for (uint64_t i=0; i<64; i++)
    {
        bit[i] = auxBit[i];
    }
}

void GateU64::shiftRight (uint64_t pos)
{
    // Store the shifted value on a temporary variable
    GateBit auxBit[64] = {0};
    uint64_t i = 0;
    for (; i<(64-pos); i++)
    {
        auxBit[i] = bit[(i+pos)%64];
    }

    // Set rest of pins to zero
    for (; i<64; i++)
    {
        auxBit[i].ref = S.gateConfig.zeroRef;
        auxBit[i].pin = pin_a;
    }

    // Copy the temporary variable into ref
    for (uint64_t i=0; i<64; i++)
    {
        bit[i] = auxBit[i];
    }
}

/* r = xor(a,b), for every individual bit */

void GateU64_xor (GateState &S, const GateU64 &a, const GateU64 &b, GateU64 &r)
{
    for (uint64_t i=0; i<64; i++)
    {
        r.bit[i].ref = S.getFreeRef();
        S.XOR(a.bit[i].ref, a.bit[i].pin, b.bit[i].ref, b.bit[i].pin, r.bit[i].ref);
        r.bit[i].pin = pin_r;
    }
}

/* r = and(a,b), for every individual bit */

void GateU64_and (GateState &S, const GateU64 &a, const GateU64 &b, GateU64 &r)
{
    for (uint64_t i=0; i<64; i++)
    {
        r.bit[i].ref = S.getFreeRef();
        S.AND(a.bit[i].ref, a.bit[i].pin, b.bit[i].ref, b.bit[i].pin, r.bit[i].ref);
        r.bit[i].pin = pin_r;
    }
}

/* r = not(a), for every individual bit */

void GateU64_not (GateState &S, const GateU64 &a, GateU64 &r)
{
    for (uint64_t i=0; i<64; i++)
    {
        r.bit[i].ref = S.getFreeRef();
        // NOT(a) is the same operation as XOR(a,1)
        S.XOR(a.bit[i].ref, a.bit[i].pin, S.gateConfig.zeroRef, pin_b, r.bit[i].ref);
        r.bit[i].pin = pin_r;
    }
}

/*
   Add 2 numbers of 64 bits, taking into account the carry bit (c)

   a b c --> r c'
   0 0 0     0 0
   0 0 1     1 0
   0 1 0     1 0
   0 1 1     0 1
   1 0 0     1 0
   1 0 1     0 1
   1 1 0     0 1
   1 1 1     1 1

   bit  0:  r = 1 (a) + 1 (b)             = 0 = xor(a,b)              carry = 1 = and(a,b)
   bit  1:  r = 1 (a) + 1 (b) + 1 (carry) = 1 = xor(xor(a,b),carry))  carry = 1 = or(and(a,b),and(b,carry),and(a,carry))
   bit  2:  r = 1 (a) + 1 (b) + 1 (carry) = 1 = xor(xor(a,b),carry))  carry = 1 = or(and(a,b),and(b,carry),and(a,carry))
   ...
   bit 62:  r = 1 (a) + 1 (b) + 1 (carry) = 1 = xor(xor(a,b),carry))  carry = 1 = or(and(a,b),and(b,carry),and(a,carry))
   bit 63:  r = 1 (a) + 1 (b) + 1 (carry) = 1 = xor(xor(a,b),carry))  carry is not needed any more
*/

void GateU64_add (GateState &S, const GateU64 &a, const GateU64 &b, GateU64 &r)
{
    // Carry bit; set default to zero
    GateBit carry;
    carry.ref = S.gateConfig.zeroRef;
    carry.pin = pin_a;

    for (uint64_t i=0; i<64; i++)
    {
        // Calculate the result bit

        // If bit number is 0, then result does not include the carry: r = sor(a,b)
        if (i == 0)
        {
            r.bit[i].ref = S.getFreeRef();
            S.XOR(a.bit[i].ref, a.bit[i].pin, b.bit[i].ref, b.bit[i].pin, r.bit[i].ref);
            r.bit[i].pin = pin_r;
        }
        // Otherwise, result includes carry: r = xor(xor(a,b),carry))
        else
        {
            uint64_t auxRef = S.getFreeRef();
            S.XOR(a.bit[i].ref, a.bit[i].pin, b.bit[i].ref, b.bit[i].pin, auxRef);
            r.bit[i].ref = S.getFreeRef();
            S.XOR(auxRef, pin_r, carry.ref, carry.pin, r.bit[i].ref);
            r.bit[i].pin = pin_r;
        }

        // Calculate the carry bit

        // If bit number is 0, then carry does not include the carry: c = and(a,b)
        if (i == 0)
        {
            carry.ref = S.getFreeRef();
            S.AND(a.bit[i].ref, a.bit[i].pin, b.bit[i].ref, b.bit[i].pin, carry.ref);
            carry.pin = pin_r;
        }

        // If bit number is 1..62, then carry includes carry: c = or(and(a,b),and(b,carry),and(a,carry))
        else if (i < 63)
        {
            // andRef1 = and(a,b)
            uint64_t andRef1 = S.getFreeRef();
            S.AND(a.bit[i].ref, a.bit[i].pin, b.bit[i].ref, b.bit[i].pin, andRef1);
            
            // andRef2 = and(b,carry)
            uint64_t andRef2 = S.getFreeRef();
            S.AND(carry.ref, carry.pin, b.bit[i].ref, b.bit[i].pin, andRef2);
            
            // andRef3 = and(a,carry)
            uint64_t andRef3 = S.getFreeRef();
            S.AND(a.bit[i].ref, a.bit[i].pin, carry.ref, carry.pin, andRef3);

            // orRef = or(andRef1, andRef2)
            uint64_t orRef = S.getFreeRef();
            S.OR(andRef1, pin_r, andRef2, pin_r, orRef);

            // carry = or(orRef, andRef3)
            carry.ref = S.getFreeRef();
            S.OR(orRef, pin_r, andRef3, pin_r, carry.ref);
            carry.pin = pin_r;
        }
    }
}

/* r = xor(a,b), for every individual bit, agains bit b */

void GateU64_xor (GateState &S, const GateU64 &a, const GateBit &b, GateU64 &r)
{
    for (uint64_t i=0; i<64; i++)
    {
        r.bit[i].ref = S.getFreeRef();
        S.XOR(a.bit[i].ref, a.bit[i].pin, b.ref, b.pin, r.bit[i].ref);
        r.bit[i].pin = pin_r;
    }
}