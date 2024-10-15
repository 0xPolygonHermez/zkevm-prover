#ifndef __NSECP256R1_H
#define __NSECP256R1_H

#include <stdint.h>
#include <string>
#include <gmp.h>

#define nSecp256r1_N64 4
#define nSecp256r1_SHORT 0x00000000
#define nSecp256r1_LONG 0x80000000
#define nSecp256r1_LONGMONTGOMERY 0xC0000000
typedef uint64_t nSecp256r1RawElement[nSecp256r1_N64];
typedef struct __attribute__((__packed__)) {
    int32_t shortVal;
    uint32_t type;
    nSecp256r1RawElement longVal;
} nSecp256r1Element;
typedef nSecp256r1Element *PnSecp256r1Element;
extern nSecp256r1Element nSecp256r1_q;
extern nSecp256r1Element nSecp256r1_R3;
extern nSecp256r1RawElement nSecp256r1_rawq;
extern nSecp256r1RawElement nSecp256r1_rawR3;

extern "C" void nSecp256r1_copy(PnSecp256r1Element r, PnSecp256r1Element a);
extern "C" void nSecp256r1_copyn(PnSecp256r1Element r, PnSecp256r1Element a, int n);
extern "C" void nSecp256r1_add(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_sub(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_neg(PnSecp256r1Element r, PnSecp256r1Element a);
extern "C" void nSecp256r1_mul(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_square(PnSecp256r1Element r, PnSecp256r1Element a);
extern "C" void nSecp256r1_band(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_bor(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_bxor(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_bnot(PnSecp256r1Element r, PnSecp256r1Element a);
extern "C" void nSecp256r1_shl(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_shr(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_eq(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_neq(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_lt(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_gt(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_leq(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_geq(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_land(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_lor(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
extern "C" void nSecp256r1_lnot(PnSecp256r1Element r, PnSecp256r1Element a);
extern "C" void nSecp256r1_toNormal(PnSecp256r1Element r, PnSecp256r1Element a);
extern "C" void nSecp256r1_toLongNormal(PnSecp256r1Element r, PnSecp256r1Element a);
extern "C" void nSecp256r1_toMontgomery(PnSecp256r1Element r, PnSecp256r1Element a);

extern "C" int nSecp256r1_isTrue(PnSecp256r1Element pE);
extern "C" int nSecp256r1_toInt(PnSecp256r1Element pE);

extern "C" void nSecp256r1_rawCopy(nSecp256r1RawElement pRawResult, const nSecp256r1RawElement pRawA);
extern "C" void nSecp256r1_rawSwap(nSecp256r1RawElement pRawResult, nSecp256r1RawElement pRawA);
extern "C" void nSecp256r1_rawAdd(nSecp256r1RawElement pRawResult, const nSecp256r1RawElement pRawA, const nSecp256r1RawElement pRawB);
extern "C" void nSecp256r1_rawSub(nSecp256r1RawElement pRawResult, const nSecp256r1RawElement pRawA, const nSecp256r1RawElement pRawB);
extern "C" void nSecp256r1_rawNeg(nSecp256r1RawElement pRawResult, const nSecp256r1RawElement pRawA);
extern "C" void nSecp256r1_rawMMul(nSecp256r1RawElement pRawResult, const nSecp256r1RawElement pRawA, const nSecp256r1RawElement pRawB);
extern "C" void nSecp256r1_rawMSquare(nSecp256r1RawElement pRawResult, const nSecp256r1RawElement pRawA);
extern "C" void nSecp256r1_rawMMul1(nSecp256r1RawElement pRawResult, const nSecp256r1RawElement pRawA, uint64_t pRawB);
extern "C" void nSecp256r1_rawToMontgomery(nSecp256r1RawElement pRawResult, const nSecp256r1RawElement &pRawA);
extern "C" void nSecp256r1_rawFromMontgomery(nSecp256r1RawElement pRawResult, const nSecp256r1RawElement &pRawA);
extern "C" int nSecp256r1_rawIsEq(const nSecp256r1RawElement pRawA, const nSecp256r1RawElement pRawB);
extern "C" int nSecp256r1_rawIsZero(const nSecp256r1RawElement pRawB);

extern "C" void nSecp256r1_fail();


// Pending functions to convert

void nSecp256r1_str2element(PnSecp256r1Element pE, char const*s);
char *nSecp256r1_element2str(PnSecp256r1Element pE);
void nSecp256r1_idiv(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
void nSecp256r1_mod(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
void nSecp256r1_inv(PnSecp256r1Element r, PnSecp256r1Element a);
void nSecp256r1_div(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);
void nSecp256r1_pow(PnSecp256r1Element r, PnSecp256r1Element a, PnSecp256r1Element b);

class RawnSecp256r1 {

public:
    const static int N64 = nSecp256r1_N64;
    const static int MaxBits = 256;


    struct Element {
        nSecp256r1RawElement v;
    };

private:
    Element fZero;
    Element fOne;
    Element fNegOne;

public:

    RawnSecp256r1();
    ~RawnSecp256r1();

    const Element &zero() { return fZero; };
    const Element &one() { return fOne; };
    const Element &negOne() { return fNegOne; };
    Element set(int value);
    void set(Element &r, int value);

    void fromString(Element &r, const std::string &n, uint32_t radix = 10);
    std::string toString(const Element &a, uint32_t radix = 10);

    void inline copy(Element &r, const Element &a) { nSecp256r1_rawCopy(r.v, a.v); };
    void inline swap(Element &a, Element &b) { nSecp256r1_rawSwap(a.v, b.v); };
    void inline add(Element &r, const Element &a, const Element &b) { nSecp256r1_rawAdd(r.v, a.v, b.v); };
    void inline sub(Element &r, const Element &a, const Element &b) { nSecp256r1_rawSub(r.v, a.v, b.v); };
    void inline mul(Element &r, const Element &a, const Element &b) { nSecp256r1_rawMMul(r.v, a.v, b.v); };

    Element inline add(const Element &a, const Element &b) { Element r; nSecp256r1_rawAdd(r.v, a.v, b.v); return r;};
    Element inline sub(const Element &a, const Element &b) { Element r; nSecp256r1_rawSub(r.v, a.v, b.v); return r;};
    Element inline mul(const Element &a, const Element &b) { Element r; nSecp256r1_rawMMul(r.v, a.v, b.v); return r;};

    Element inline neg(const Element &a) { Element r; nSecp256r1_rawNeg(r.v, a.v); return r; };
    Element inline square(const Element &a) { Element r; nSecp256r1_rawMSquare(r.v, a.v); return r; };

    Element inline add(int a, const Element &b) { return add(set(a), b);};
    Element inline sub(int a, const Element &b) { return sub(set(a), b);};
    Element inline mul(int a, const Element &b) { return mul(set(a), b);};

    Element inline add(const Element &a, int b) { return add(a, set(b));};
    Element inline sub(const Element &a, int b) { return sub(a, set(b));};
    Element inline mul(const Element &a, int b) { return mul(a, set(b));};
    
    void inline mul1(Element &r, const Element &a, uint64_t b) { nSecp256r1_rawMMul1(r.v, a.v, b); };
    void inline neg(Element &r, const Element &a) { nSecp256r1_rawNeg(r.v, a.v); };
    void inline square(Element &r, const Element &a) { nSecp256r1_rawMSquare(r.v, a.v); };
    void inv(Element &r, const Element &a);
    void div(Element &r, const Element &a, const Element &b);
    void exp(Element &r, const Element &base, uint8_t* scalar, unsigned int scalarSize);

    void inline toMontgomery(Element &r, const Element &a) { nSecp256r1_rawToMontgomery(r.v, a.v); };
    void inline fromMontgomery(Element &r, const Element &a) { nSecp256r1_rawFromMontgomery(r.v, a.v); };
    int inline eq(const Element &a, const Element &b) { return nSecp256r1_rawIsEq(a.v, b.v); };
    int inline isZero(const Element &a) { return nSecp256r1_rawIsZero(a.v); };

    void toMpz(mpz_t r, const Element &a);
    void fromMpz(Element &a, const mpz_t r);

    int toRprBE(const Element &element, uint8_t *data, int bytes);
    int fromRprBE(Element &element, const uint8_t *data, int bytes);
    
    int bytes ( void ) { return nSecp256r1_N64 * 8; };
    
    void fromUI(Element &r, unsigned long int v);

    static RawnSecp256r1 field;

};


#endif // __NSECP256R1_H



