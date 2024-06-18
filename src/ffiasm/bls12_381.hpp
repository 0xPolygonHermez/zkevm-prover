#ifndef __BLS12_381_H
#define __BLS12_381_H

#include <stdint.h>
#include <string>
#include <gmp.h>

#define BLS12_381_N64 4
#define BLS12_381_SHORT 0x00000000
#define BLS12_381_LONG 0x80000000
#define BLS12_381_LONGMONTGOMERY 0xC0000000
typedef uint64_t BLS12_381RawElement[BLS12_381_N64];
typedef struct __attribute__((__packed__)) {
    int32_t shortVal;
    uint32_t type;
    BLS12_381RawElement longVal;
} BLS12_381Element;
typedef BLS12_381Element *PBLS12_381Element;
extern BLS12_381Element BLS12_381_q;
extern BLS12_381Element BLS12_381_R3;
extern BLS12_381RawElement BLS12_381_rawq;
extern BLS12_381RawElement BLS12_381_rawR3;

extern "C" void BLS12_381_copy(PBLS12_381Element r, PBLS12_381Element a);
extern "C" void BLS12_381_copyn(PBLS12_381Element r, PBLS12_381Element a, int n);
extern "C" void BLS12_381_add(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_sub(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_neg(PBLS12_381Element r, PBLS12_381Element a);
extern "C" void BLS12_381_mul(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_square(PBLS12_381Element r, PBLS12_381Element a);
extern "C" void BLS12_381_band(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_bor(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_bxor(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_bnot(PBLS12_381Element r, PBLS12_381Element a);
extern "C" void BLS12_381_shl(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_shr(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_eq(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_neq(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_lt(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_gt(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_leq(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_geq(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_land(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_lor(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
extern "C" void BLS12_381_lnot(PBLS12_381Element r, PBLS12_381Element a);
extern "C" void BLS12_381_toNormal(PBLS12_381Element r, PBLS12_381Element a);
extern "C" void BLS12_381_toLongNormal(PBLS12_381Element r, PBLS12_381Element a);
extern "C" void BLS12_381_toMontgomery(PBLS12_381Element r, PBLS12_381Element a);

extern "C" int BLS12_381_isTrue(PBLS12_381Element pE);
extern "C" int BLS12_381_toInt(PBLS12_381Element pE);

extern "C" void BLS12_381_rawCopy(BLS12_381RawElement pRawResult, const BLS12_381RawElement pRawA);
extern "C" void BLS12_381_rawSwap(BLS12_381RawElement pRawResult, BLS12_381RawElement pRawA);
extern "C" void BLS12_381_rawAdd(BLS12_381RawElement pRawResult, const BLS12_381RawElement pRawA, const BLS12_381RawElement pRawB);
extern "C" void BLS12_381_rawSub(BLS12_381RawElement pRawResult, const BLS12_381RawElement pRawA, const BLS12_381RawElement pRawB);
extern "C" void BLS12_381_rawNeg(BLS12_381RawElement pRawResult, const BLS12_381RawElement pRawA);
extern "C" void BLS12_381_rawMMul(BLS12_381RawElement pRawResult, const BLS12_381RawElement pRawA, const BLS12_381RawElement pRawB);
extern "C" void BLS12_381_rawMSquare(BLS12_381RawElement pRawResult, const BLS12_381RawElement pRawA);
extern "C" void BLS12_381_rawMMul1(BLS12_381RawElement pRawResult, const BLS12_381RawElement pRawA, uint64_t pRawB);
extern "C" void BLS12_381_rawToMontgomery(BLS12_381RawElement pRawResult, const BLS12_381RawElement &pRawA);
extern "C" void BLS12_381_rawFromMontgomery(BLS12_381RawElement pRawResult, const BLS12_381RawElement &pRawA);
extern "C" int BLS12_381_rawIsEq(const BLS12_381RawElement pRawA, const BLS12_381RawElement pRawB);
extern "C" int BLS12_381_rawIsZero(const BLS12_381RawElement pRawB);

extern "C" void BLS12_381_fail();


// Pending functions to convert

void BLS12_381_str2element(PBLS12_381Element pE, char const*s);
char *BLS12_381_element2str(PBLS12_381Element pE);
void BLS12_381_idiv(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
void BLS12_381_mod(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
void BLS12_381_inv(PBLS12_381Element r, PBLS12_381Element a);
void BLS12_381_div(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);
void BLS12_381_pow(PBLS12_381Element r, PBLS12_381Element a, PBLS12_381Element b);

class RawBLS12_381 {

public:
    const static int N64 = BLS12_381_N64;
    const static int MaxBits = 255;


    struct Element {
        BLS12_381RawElement v;
    };

private:
    Element fZero;
    Element fOne;
    Element fNegOne;

public:

    RawBLS12_381();
    ~RawBLS12_381();

    const Element &zero() { return fZero; };
    const Element &one() { return fOne; };
    const Element &negOne() { return fNegOne; };
    Element set(int value);
    void set(Element &r, int value);

    void fromString(Element &r, const std::string &n, uint32_t radix = 10);
    std::string toString(const Element &a, uint32_t radix = 10);

    void inline copy(Element &r, const Element &a) { BLS12_381_rawCopy(r.v, a.v); };
    void inline swap(Element &a, Element &b) { BLS12_381_rawSwap(a.v, b.v); };
    void inline add(Element &r, const Element &a, const Element &b) { BLS12_381_rawAdd(r.v, a.v, b.v); };
    void inline sub(Element &r, const Element &a, const Element &b) { BLS12_381_rawSub(r.v, a.v, b.v); };
    void inline mul(Element &r, const Element &a, const Element &b) { BLS12_381_rawMMul(r.v, a.v, b.v); };

    Element inline add(const Element &a, const Element &b) { Element r; BLS12_381_rawAdd(r.v, a.v, b.v); return r;};
    Element inline sub(const Element &a, const Element &b) { Element r; BLS12_381_rawSub(r.v, a.v, b.v); return r;};
    Element inline mul(const Element &a, const Element &b) { Element r; BLS12_381_rawMMul(r.v, a.v, b.v); return r;};

    Element inline neg(const Element &a) { Element r; BLS12_381_rawNeg(r.v, a.v); return r; };
    Element inline square(const Element &a) { Element r; BLS12_381_rawMSquare(r.v, a.v); return r; };

    Element inline add(int a, const Element &b) { return add(set(a), b);};
    Element inline sub(int a, const Element &b) { return sub(set(a), b);};
    Element inline mul(int a, const Element &b) { return mul(set(a), b);};

    Element inline add(const Element &a, int b) { return add(a, set(b));};
    Element inline sub(const Element &a, int b) { return sub(a, set(b));};
    Element inline mul(const Element &a, int b) { return mul(a, set(b));};
    
    void inline mul1(Element &r, const Element &a, uint64_t b) { BLS12_381_rawMMul1(r.v, a.v, b); };
    void inline neg(Element &r, const Element &a) { BLS12_381_rawNeg(r.v, a.v); };
    void inline square(Element &r, const Element &a) { BLS12_381_rawMSquare(r.v, a.v); };
    void inv(Element &r, const Element &a);
    void div(Element &r, const Element &a, const Element &b);
    void exp(Element &r, const Element &base, uint8_t* scalar, unsigned int scalarSize);

    void inline toMontgomery(Element &r, const Element &a) { BLS12_381_rawToMontgomery(r.v, a.v); };
    void inline fromMontgomery(Element &r, const Element &a) { BLS12_381_rawFromMontgomery(r.v, a.v); };
    int inline eq(const Element &a, const Element &b) { return BLS12_381_rawIsEq(a.v, b.v); };
    int inline isZero(const Element &a) { return BLS12_381_rawIsZero(a.v); };

    void toMpz(mpz_t r, const Element &a);
    void fromMpz(Element &a, const mpz_t r);

    int toRprBE(const Element &element, uint8_t *data, int bytes);
    int fromRprBE(Element &element, const uint8_t *data, int bytes);
    
    int bytes ( void ) { return BLS12_381_N64 * 8; };
    
    void fromUI(Element &r, unsigned long int v);

    static RawBLS12_381 field;

};


#endif // __BLS12_381_H



