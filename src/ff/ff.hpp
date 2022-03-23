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
public:
    FieldElement p;

    FiniteField (FieldElement prime)
    {
        p = prime;
    }
    FiniteField () { FiniteField(DEFAULT_FF_PRIME); }

    FieldElement add (FieldElement a, FieldElement b);
    FieldElement sub (FieldElement a, FieldElement b);    
    FieldElement neg (FieldElement a);
    FieldElement mul (FieldElement a, FieldElement b);
    FieldElement inv (FieldElement a);
    FieldElement div (FieldElement a, FieldElement b);
    void test (void);

private:
    void check (bool condition);
    
};

#endif