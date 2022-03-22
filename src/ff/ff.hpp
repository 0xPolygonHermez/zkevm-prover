#ifndef FINITE_FIELD_NATIVE_HPP
#define FINITE_FIELD_NATIVE_HPP

#include <iostream>
#include <cstdint>
#include <gmpxx.h>

using namespace std;

#define DEFAULT_FF_PRIME (0xFFFFFFFF00000001)

class FiniteField
{
public:
    uint64_t p;

    FiniteField (uint64_t prime)
    {
        p = prime;
    }
    FiniteField () { FiniteField(DEFAULT_FF_PRIME); }

    uint64_t add (uint64_t a, uint64_t b);
    uint64_t sub (uint64_t a, uint64_t b);    
    uint64_t neg (uint64_t a);
    uint64_t mul (uint64_t a, uint64_t b);
    uint64_t inv (uint64_t a);
    uint64_t div (uint64_t a, uint64_t b);
    void test (void);

private:
    void check (bool condition);
    
};

#endif