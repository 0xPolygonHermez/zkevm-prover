#include <iostream>
#include "sha256_u32.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "exit_process.hpp"

using namespace std;

void SHA256_U32::fromU32 (uint32_t value)
{
    // Convert U32 value to an array of bits
    vector<uint8_t> bits;
    u322bits(value, bits);
    zkassert(bits.size() == 32);

    // Assign the proper reference
    for (uint64_t i=0; i<32; i++)
    {
        bit[i].ref = SHA256_ZeroRef;
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
            cerr << "Error: SHA256_U32::init() got invalid bit value=" << value << " i=" << i << " bits[i]=" << bits[i] << endl;
            exitProcess();
        }
    }
}

uint32_t SHA256_U32::toU32 (SHA256_State &S)
{
    // Collect bits
    vector<uint8_t> bits;
    for (uint64_t i=0; i<32; i++)
    {
        bits.push_back(S.gate[bit[i].ref].pin[bit[i].pin].bit);
    }

    // Convert from bits to U32
    return bits2u32(bits);
}

string SHA256_U32::toString (SHA256_State &S)
{
    mpz_class aux = toU32(S);
    return aux.get_str(16);
}

void SHA256_U32::rotateRight (uint64_t pos)
{
    // Store the rotated value on a temporary variable
    SHA256_Bit auxBit[32];
    for (uint64_t i=0; i<32; i++)
    {
        auxBit[i] = bit[(i+pos)%32];
    }

    // Copy the temporary variable into ref
    for (uint64_t i=0; i<32; i++)
    {
        bit[i] = auxBit[i];
    }
}

void SHA256_U32::shiftRight (uint64_t pos)
{
    // Store the shifted value on a temporary variable
    SHA256_Bit auxBit[32] = {0};
    uint64_t i = 0;
    for (; i<(32-pos); i++)
    {
        auxBit[i] = bit[(i+pos)%32];
    }

    // Set rest of pins to zero
    for (; i<32; i++)
    {
        auxBit[i].ref = SHA256_ZeroRef;
        auxBit[i].pin = pin_a;
    }

    // Copy the temporary variable into ref
    for (uint64_t i=0; i<32; i++)
    {
        bit[i] = auxBit[i];
    }
}

/* r = xor(a,b), for every individual bit */

void SHA256_xor (SHA256_State &S, const SHA256_U32 &a, const SHA256_U32 &b, SHA256_U32 &r)
{
    for (uint64_t i=0; i<32; i++)
    {
        r.bit[i].ref = S.getFreeRef();
        S.XOR(a.bit[i].ref, a.bit[i].pin, b.bit[i].ref, b.bit[i].pin, r.bit[i].ref);
        r.bit[i].pin = pin_r;
    }
}

/* r = and(a,b), for every individual bit */

void SHA256_and (SHA256_State &S, const SHA256_U32 &a, const SHA256_U32 &b, SHA256_U32 &r)
{
    for (uint64_t i=0; i<32; i++)
    {
        r.bit[i].ref = S.getFreeRef();
        S.AND(a.bit[i].ref, a.bit[i].pin, b.bit[i].ref, b.bit[i].pin, r.bit[i].ref);
        r.bit[i].pin = pin_r;
    }
}

/* r = not(a), for every individual bit */

void SHA256_not (SHA256_State &S, const SHA256_U32 &a, SHA256_U32 &r)
{
    for (uint64_t i=0; i<32; i++)
    {
        r.bit[i].ref = S.getFreeRef();
        // NOT(a) is the same operation as XOR(a,1)
        S.XOR(a.bit[i].ref, a.bit[i].pin, SHA256_ZeroRef, pin_b, r.bit[i].ref);
        r.bit[i].pin = pin_r;
    }
}

/*
   Add 2 numbers of 32 bits, taking into account the carry bit (c)

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
   bit 30:  r = 1 (a) + 1 (b) + 1 (carry) = 1 = xor(xor(a,b),carry))  carry = 1 = or(and(a,b),and(b,carry),and(a,carry))
   bit 31:  r = 1 (a) + 1 (b) + 1 (carry) = 1 = xor(xor(a,b),carry))  carry is not needed any more
*/

void SHA256_add (SHA256_State &S, const SHA256_U32 &a, const SHA256_U32 &b, SHA256_U32 &r)
{
    // Carry bit 
    SHA256_Bit carry;

    for (uint64_t i=0; i<32; i++)
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

        // If bit number is 1..30, then carry includes carry: c = or(and(a,b),and(b,carry),and(a,carry))
        else if (i < 31)
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