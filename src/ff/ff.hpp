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
    FieldElement mod (FieldElement a, FieldElement b);

    string toString (FieldElement a, uint64_t radix=10);
    void fromString (FieldElement &a, const string &s, uint64_t radix=10);


    /* Backwards compatible methods */
    inline FieldElement zero (void) { return 0; }
    inline FieldElement one (void) { return 1; }
    inline FieldElement negone (void) { return neg(one()); }
    inline bool isZero (const FieldElement a) { return a==0; }
    inline bool isZero (const FieldElement (&fea)[4]) { return isZero(fea[0]) && isZero(fea[1]) && isZero(fea[2]) && isZero(fea[3]); }
    inline bool isZero (const vector<FieldElement> &fea)
    {
        for (uint64_t i=0; i<fea.size(); i++)
            if (!isZero(fea[i]))
                return false;
        return true;
    }
    inline bool eq (const FieldElement a, const FieldElement b) { return a==b; }
    inline bool eq (const FieldElement (&fea1)[4], const FieldElement (&fea2)[4])
    {
        return eq(fea1[0], fea2[0]) && eq(fea1[1], fea2[1]) && eq(fea1[2], fea2[2]) && eq(fea1[3], fea2[3]);
    }
    inline void add (FieldElement &r, const FieldElement &a, const FieldElement &b) { r = add(a, b); }
    inline void mul (FieldElement &r, const FieldElement &a, const FieldElement &b) { r = mul(a, b); }
    inline void inv (FieldElement &r, const FieldElement &a) { r = inv(a); }
    inline void neg (FieldElement &r, const FieldElement &a) { r = neg(a); }
    inline void square (FieldElement &r, const FieldElement &a) { r = mul(a, a); }
    inline void fromUI (FieldElement &r, uint64_t ui) { r = ui; }
    inline uint64_t prime (void) { return p; }
    void test (void);

private:
    void check (bool condition);
    
};

#endif