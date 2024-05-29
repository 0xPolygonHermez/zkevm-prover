#ifndef __BLS12_381_384_H
#define __BLS12_381_384_H

#include <stdint.h>
#include <string>
#include <gmp.h>

#define BLS12_381_384_N64 6
#define BLS12_381_384_SHORT 0x00000000
#define BLS12_381_384_LONG 0x80000000
#define BLS12_381_384_LONGMONTGOMERY 0xC0000000
typedef uint64_t BLS12_381_384RawElement[BLS12_381_384_N64];
typedef struct __attribute__((__packed__)) {
    int32_t shortVal;
    uint32_t type;
    BLS12_381_384RawElement longVal;
} BLS12_381_384Element;
typedef BLS12_381_384Element *PBLS12_381_384Element;
extern BLS12_381_384Element BLS12_381_384_q;
extern BLS12_381_384Element BLS12_381_384_R3;
extern BLS12_381_384RawElement BLS12_381_384_rawq;
extern BLS12_381_384RawElement BLS12_381_384_rawR3;

extern "C" void BLS12_381_384_copy(PBLS12_381_384Element r, PBLS12_381_384Element a);
extern "C" void BLS12_381_384_copyn(PBLS12_381_384Element r, PBLS12_381_384Element a, int n);
extern "C" void BLS12_381_384_add(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_sub(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_neg(PBLS12_381_384Element r, PBLS12_381_384Element a);
extern "C" void BLS12_381_384_mul(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_square(PBLS12_381_384Element r, PBLS12_381_384Element a);
extern "C" void BLS12_381_384_band(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_bor(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_bxor(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_bnot(PBLS12_381_384Element r, PBLS12_381_384Element a);
extern "C" void BLS12_381_384_shl(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_shr(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_eq(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_neq(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_lt(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_gt(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_leq(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_geq(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_land(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_lor(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
extern "C" void BLS12_381_384_lnot(PBLS12_381_384Element r, PBLS12_381_384Element a);
extern "C" void BLS12_381_384_toNormal(PBLS12_381_384Element r, PBLS12_381_384Element a);
extern "C" void BLS12_381_384_toLongNormal(PBLS12_381_384Element r, PBLS12_381_384Element a);
extern "C" void BLS12_381_384_toMontgomery(PBLS12_381_384Element r, PBLS12_381_384Element a);

extern "C" int BLS12_381_384_isTrue(PBLS12_381_384Element pE);
extern "C" int BLS12_381_384_toInt(PBLS12_381_384Element pE);

extern "C" void BLS12_381_384_rawCopy(BLS12_381_384RawElement pRawResult, const BLS12_381_384RawElement pRawA);
extern "C" void BLS12_381_384_rawSwap(BLS12_381_384RawElement pRawResult, BLS12_381_384RawElement pRawA);
extern "C" void BLS12_381_384_rawAdd(BLS12_381_384RawElement pRawResult, const BLS12_381_384RawElement pRawA, const BLS12_381_384RawElement pRawB);
extern "C" void BLS12_381_384_rawSub(BLS12_381_384RawElement pRawResult, const BLS12_381_384RawElement pRawA, const BLS12_381_384RawElement pRawB);
extern "C" void BLS12_381_384_rawNeg(BLS12_381_384RawElement pRawResult, const BLS12_381_384RawElement pRawA);
extern "C" void BLS12_381_384_rawMMul(BLS12_381_384RawElement pRawResult, const BLS12_381_384RawElement pRawA, const BLS12_381_384RawElement pRawB);
extern "C" void BLS12_381_384_rawMSquare(BLS12_381_384RawElement pRawResult, const BLS12_381_384RawElement pRawA);
extern "C" void BLS12_381_384_rawMMul1(BLS12_381_384RawElement pRawResult, const BLS12_381_384RawElement pRawA, uint64_t pRawB);
extern "C" void BLS12_381_384_rawToMontgomery(BLS12_381_384RawElement pRawResult, const BLS12_381_384RawElement &pRawA);
extern "C" void BLS12_381_384_rawFromMontgomery(BLS12_381_384RawElement pRawResult, const BLS12_381_384RawElement &pRawA);
extern "C" int BLS12_381_384_rawIsEq(const BLS12_381_384RawElement pRawA, const BLS12_381_384RawElement pRawB);
extern "C" int BLS12_381_384_rawIsZero(const BLS12_381_384RawElement pRawB);

extern "C" void BLS12_381_384_fail();


// Pending functions to convert

void BLS12_381_384_str2element(PBLS12_381_384Element pE, char const*s);
char *BLS12_381_384_element2str(PBLS12_381_384Element pE);
void BLS12_381_384_idiv(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
void BLS12_381_384_mod(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
void BLS12_381_384_inv(PBLS12_381_384Element r, PBLS12_381_384Element a);
void BLS12_381_384_div(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);
void BLS12_381_384_pow(PBLS12_381_384Element r, PBLS12_381_384Element a, PBLS12_381_384Element b);

class RawBLS12_381_384 {

public:
    const static int N64 = BLS12_381_384_N64;
    const static int MaxBits = 381;


    struct Element {
        BLS12_381_384RawElement v;
    };

private:
    Element fZero;
    Element fOne;
    Element fNegOne;

public:

    RawBLS12_381_384();
    ~RawBLS12_381_384();

    const Element &zero() { return fZero; };
    const Element &one() { return fOne; };
    const Element &negOne() { return fNegOne; };
    Element set(int value);
    void set(Element &r, int value);

    void fromString(Element &r, const std::string &n, uint32_t radix = 10);
    std::string toString(const Element &a, uint32_t radix = 10);

    void inline copy(Element &r, const Element &a) { BLS12_381_384_rawCopy(r.v, a.v); };
    void inline swap(Element &a, Element &b) { BLS12_381_384_rawSwap(a.v, b.v); };
    void inline add(Element &r, const Element &a, const Element &b) { BLS12_381_384_rawAdd(r.v, a.v, b.v); };
    void inline sub(Element &r, const Element &a, const Element &b) { BLS12_381_384_rawSub(r.v, a.v, b.v); };
    void inline mul(Element &r, const Element &a, const Element &b) { BLS12_381_384_rawMMul(r.v, a.v, b.v); };

    Element inline add(const Element &a, const Element &b) { Element r; BLS12_381_384_rawAdd(r.v, a.v, b.v); return r;};
    Element inline sub(const Element &a, const Element &b) { Element r; BLS12_381_384_rawSub(r.v, a.v, b.v); return r;};
    Element inline mul(const Element &a, const Element &b) { Element r; BLS12_381_384_rawMMul(r.v, a.v, b.v); return r;};

    Element inline neg(const Element &a) { Element r; BLS12_381_384_rawNeg(r.v, a.v); return r; };
    Element inline square(const Element &a) { Element r; BLS12_381_384_rawMSquare(r.v, a.v); return r; };

    Element inline add(int a, const Element &b) { return add(set(a), b);};
    Element inline sub(int a, const Element &b) { return sub(set(a), b);};
    Element inline mul(int a, const Element &b) { return mul(set(a), b);};

    Element inline add(const Element &a, int b) { return add(a, set(b));};
    Element inline sub(const Element &a, int b) { return sub(a, set(b));};
    Element inline mul(const Element &a, int b) { return mul(a, set(b));};
    
    void inline mul1(Element &r, const Element &a, uint64_t b) { BLS12_381_384_rawMMul1(r.v, a.v, b); };
    void inline neg(Element &r, const Element &a) { BLS12_381_384_rawNeg(r.v, a.v); };
    void inline square(Element &r, const Element &a) { BLS12_381_384_rawMSquare(r.v, a.v); };
    void inv(Element &r, const Element &a);
    void div(Element &r, const Element &a, const Element &b);
    void exp(Element &r, const Element &base, uint8_t* scalar, unsigned int scalarSize);

    void inline toMontgomery(Element &r, const Element &a) { BLS12_381_384_rawToMontgomery(r.v, a.v); };
    void inline fromMontgomery(Element &r, const Element &a) { BLS12_381_384_rawFromMontgomery(r.v, a.v); };
    int inline eq(const Element &a, const Element &b) { return BLS12_381_384_rawIsEq(a.v, b.v); };
    int inline isZero(const Element &a) { return BLS12_381_384_rawIsZero(a.v); };

    void toMpz(mpz_t r, const Element &a);
    void fromMpz(Element &a, const mpz_t r);

    int toRprBE(const Element &element, uint8_t *data, int bytes);
    int fromRprBE(Element &element, const uint8_t *data, int bytes);
    
    int bytes ( void ) { return BLS12_381_384_N64 * 8; };
    
    void fromUI(Element &r, unsigned long int v);

    static RawBLS12_381_384 field;

};


#endif // __BLS12_381_384_H



