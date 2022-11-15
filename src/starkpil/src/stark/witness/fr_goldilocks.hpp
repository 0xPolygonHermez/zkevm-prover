#ifndef __FR_GOLDILOCKS_H
#define __FR_GOLDILOCKS_H

#include <stdint.h>
#include <string>
#include <gmp.h>

#define FrG_N64 1
#define FrG_SHORT 0x00000000
#define FrG_LONG 0x80000000
#define FrG_LONGMONTGOMERY 0xC0000000
typedef uint64_t FrGRawElement[FrG_N64];
typedef struct __attribute__((__packed__)) {
    int32_t shortVal;
    uint32_t type;
    FrGRawElement longVal;
} FrGElement;
typedef FrGElement *PFrGElement;
extern FrGElement FrG_q;
extern FrGElement FrG_R3;
extern FrGRawElement FrG_rawq;
extern FrGRawElement FrG_rawR3;

extern "C" void FrG_copy(PFrGElement r, PFrGElement a);
extern "C" void FrG_copyn(PFrGElement r, PFrGElement a, int n);
extern "C" void FrG_add(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_sub(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_neg(PFrGElement r, PFrGElement a);
extern "C" void FrG_mul(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_square(PFrGElement r, PFrGElement a);
extern "C" void FrG_band(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_bor(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_bxor(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_bnot(PFrGElement r, PFrGElement a);
extern "C" void FrG_shl(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_shr(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_eq(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_neq(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_lt(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_gt(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_leq(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_geq(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_land(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_lor(PFrGElement r, PFrGElement a, PFrGElement b);
extern "C" void FrG_lnot(PFrGElement r, PFrGElement a);
extern "C" void FrG_toNormal(PFrGElement r, PFrGElement a);
extern "C" void FrG_toLongNormal(PFrGElement r, PFrGElement a);
extern "C" void FrG_toMontgomery(PFrGElement r, PFrGElement a);

extern "C" int FrG_isTrue(PFrGElement pE);
extern "C" int FrG_toInt(PFrGElement pE);

extern "C" void FrG_rawCopy(FrGRawElement pRawResult, const FrGRawElement pRawA);
extern "C" void FrG_rawSwap(FrGRawElement pRawResult, FrGRawElement pRawA);
extern "C" void FrG_rawAdd(FrGRawElement pRawResult, const FrGRawElement pRawA, const FrGRawElement pRawB);
extern "C" void FrG_rawSub(FrGRawElement pRawResult, const FrGRawElement pRawA, const FrGRawElement pRawB);
extern "C" void FrG_rawNeg(FrGRawElement pRawResult, const FrGRawElement pRawA);
extern "C" void FrG_rawMMul(FrGRawElement pRawResult, const FrGRawElement pRawA, const FrGRawElement pRawB);
extern "C" void FrG_rawMSquare(FrGRawElement pRawResult, const FrGRawElement pRawA);
extern "C" void FrG_rawMMul1(FrGRawElement pRawResult, const FrGRawElement pRawA, uint64_t pRawB);
extern "C" void FrG_rawToMontgomery(FrGRawElement pRawResult, const FrGRawElement &pRawA);
extern "C" void FrG_rawFromMontgomery(FrGRawElement pRawResult, const FrGRawElement &pRawA);
extern "C" int FrG_rawIsEq(const FrGRawElement pRawA, const FrGRawElement pRawB);
extern "C" int FrG_rawIsZero(const FrGRawElement pRawB);

extern "C" void FrG_fail();


// Pending functions to convert

void FrG_str2element(PFrGElement pE, char const*s);
void FrG_str2element(PFrGElement pE, char const*s, uint base);
char *FrG_element2str(PFrGElement pE);
void FrG_idiv(PFrGElement r, PFrGElement a, PFrGElement b);
void FrG_mod(PFrGElement r, PFrGElement a, PFrGElement b);
void FrG_inv(PFrGElement r, PFrGElement a);
void FrG_div(PFrGElement r, PFrGElement a, PFrGElement b);
void FrG_pow(PFrGElement r, PFrGElement a, PFrGElement b);

class RawFrG {

public:
    const static int N64 = FrG_N64;
    const static int MaxBits = 64;


    struct Element {
        FrGRawElement v;
    };

private:
    Element fZero;
    Element fOne;
    Element fNegOne;

public:

    RawFrG();
    ~RawFrG();

    const Element &zero() { return fZero; };
    const Element &one() { return fOne; };
    const Element &negOne() { return fNegOne; };
    Element set(int value);
    void set(Element &r, int value);

    void fromString(Element &r, const std::string &n, uint32_t radix = 10);
    std::string toString(const Element &a, uint32_t radix = 10);

    void inline copy(Element &r, const Element &a) { FrG_rawCopy(r.v, a.v); };
    void inline swap(Element &a, Element &b) { FrG_rawSwap(a.v, b.v); };
    void inline add(Element &r, const Element &a, const Element &b) { FrG_rawAdd(r.v, a.v, b.v); };
    void inline sub(Element &r, const Element &a, const Element &b) { FrG_rawSub(r.v, a.v, b.v); };
    void inline mul(Element &r, const Element &a, const Element &b) { FrG_rawMMul(r.v, a.v, b.v); };

    Element inline add(const Element &a, const Element &b) { Element r; FrG_rawAdd(r.v, a.v, b.v); return r;};
    Element inline sub(const Element &a, const Element &b) { Element r; FrG_rawSub(r.v, a.v, b.v); return r;};
    Element inline mul(const Element &a, const Element &b) { Element r; FrG_rawMMul(r.v, a.v, b.v); return r;};

    Element inline neg(const Element &a) { Element r; FrG_rawNeg(r.v, a.v); return r; };
    Element inline square(const Element &a) { Element r; FrG_rawMSquare(r.v, a.v); return r; };

    Element inline add(int a, const Element &b) { return add(set(a), b);};
    Element inline sub(int a, const Element &b) { return sub(set(a), b);};
    Element inline mul(int a, const Element &b) { return mul(set(a), b);};

    Element inline add(const Element &a, int b) { return add(a, set(b));};
    Element inline sub(const Element &a, int b) { return sub(a, set(b));};
    Element inline mul(const Element &a, int b) { return mul(a, set(b));};
    
    void inline mul1(Element &r, const Element &a, uint64_t b) { FrG_rawMMul1(r.v, a.v, b); };
    void inline neg(Element &r, const Element &a) { FrG_rawNeg(r.v, a.v); };
    void inline square(Element &r, const Element &a) { FrG_rawMSquare(r.v, a.v); };
    void inv(Element &r, const Element &a);
    void div(Element &r, const Element &a, const Element &b);
    void exp(Element &r, const Element &base, uint8_t* scalar, unsigned int scalarSize);

    void inline toMontgomery(Element &r, const Element &a) { FrG_rawToMontgomery(r.v, a.v); };
    void inline fromMontgomery(Element &r, const Element &a) { FrG_rawFromMontgomery(r.v, a.v); };
    int inline eq(const Element &a, const Element &b) { return FrG_rawIsEq(a.v, b.v); };
    int inline isZero(const Element &a) { return FrG_rawIsZero(a.v); };

    void toMpz(mpz_t r, const Element &a);
    void fromMpz(Element &a, const mpz_t r);

    int toRprBE(const Element &element, uint8_t *data, int bytes);
    int fromRprBE(Element &element, const uint8_t *data, int bytes);
    
    int bytes ( void ) { return FrG_N64 * 8; };
    
    void fromUI(Element &r, unsigned long int v);

    static RawFrG field;

};


#endif // __FR_H



