#ifndef FINITE_FIELD_NATIVE_HPP
#define FINITE_FIELD_NATIVE_HPP

#include <iostream>
#include <cstdint>
#include <gmpxx.h>

using namespace std;

#define DEFAULT_FF_PRIME (0xFFFFFFFF00000001)

typedef uint64_t FieldElement;

class FiniteField
{
private:
    uint64_t p;
    
public:

    FiniteField (uint64_t prime)
    {
        p = prime;
    }
    FiniteField ()
    {
        p = DEFAULT_FF_PRIME; 
    }

    FieldElement add (FieldElement a, FieldElement b);
    FieldElement sub (FieldElement a, FieldElement b);    
    FieldElement neg (FieldElement a);
    FieldElement mul (FieldElement a, FieldElement b);
    FieldElement inv (FieldElement a);
    FieldElement div (FieldElement a, FieldElement b);

    string toString (FieldElement a, uint64_t radix=10);
    void fromString (FieldElement &a, const string &s, uint64_t radix=10);


    /* Backwards compatible methods */
    inline FieldElement zero (void) { return 0; }
    inline FieldElement one (void) { return 1; }
    inline bool isZero (FieldElement a) { return a==0; }
    inline bool eq (FieldElement a, FieldElement b) { return a==b; }
    inline void add (FieldElement &r, FieldElement &a, FieldElement &b) { r = add(a, b); }
    inline void mul (FieldElement &r, FieldElement &a, FieldElement &b) { r = mul(a, b); }
    inline void inv (FieldElement &r, FieldElement &a) { r = inv(a); }
    inline void neg (FieldElement &r, FieldElement &a) { r = neg(a); }
    inline void square (FieldElement &r, FieldElement &a) { r = mul(a, a); }
    inline void fromUI (FieldElement &r, uint64_t ui) { r = ui; }
    inline uint64_t prime (void) { return p; }
    void test (void);

private:
    void check (bool condition);
    
};

#endif