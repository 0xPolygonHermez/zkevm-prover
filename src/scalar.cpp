#include <iostream>
#include "scalar.hpp"

void fea2scalar (RawFr &fr, mpz_t &scalar, RawFr::Element &fe0, uint64_t fe1, uint64_t fe2, uint64_t fe3)
{
    // Convert field elements to mpz
    mpz_t r0, r1, r2, r3;
    mpz_init_set_ui(r0,0);
    mpz_init_set_ui(r1,fe1);
    mpz_init_set_ui(r2,fe2);
    mpz_init_set_ui(r3,fe3);
    fr.toMpz(r0, fe0);

    // Multiply by the proper power of 2, i.e. shift left
    mpz_t r1_64, r2_128, r3_192;
    mpz_init_set_ui(r1_64,0);
    mpz_init_set_ui(r2_128,0);
    mpz_init_set_ui(r3_192,0);
    mpz_mul_2exp(r1_64, r1, 64U);
    mpz_mul_2exp(r2_128, r2, 128U);
    mpz_mul_2exp(r3_192, r3, 192U);

    // Aggregate in result
    mpz_t result01, result23;
    mpz_init(result01);
    mpz_init(result23);
    mpz_add(result01, r0, r1_64);
    mpz_add(result23, r2_128, r3_192);
    mpz_add(scalar, result01, result23);

    // Free memory
    mpz_clear(r0);
    mpz_clear(r1);
    mpz_clear(r2);
    mpz_clear(r3); 
    mpz_clear(r1_64); 
    mpz_clear(r2_128); 
    mpz_clear(r3_192); 
    mpz_clear(result01); 
    mpz_clear(result23); 
}

void fe2scalar  (RawFr &fr, mpz_class &scalar, RawFr::Element &fe)
{
    mpz_t r;
    mpz_init(r);
    fr.toMpz(r, fe);
    mpz_class s(r);
    scalar = s;
    mpz_clear(r);
}

void fea2scalar (RawFr &fr, mpz_class &scalar, RawFr::Element &fe0, uint64_t fe1, uint64_t fe2, uint64_t fe3)
{
    mpz_t r0;
    mpz_init(r0);
    fr.toMpz(r0, fe0);
    mpz_class s0(r0);
    mpz_class s1(fe1);
    mpz_class s2(fe2);
    mpz_class s3(fe3);
    scalar = s0 + (s1<<64) + (s2<<128) + (s3<<192);
    mpz_clear(r0);
}

void fea2scalar (RawFr &fr, mpz_class &scalar, RawFr::Element &fe0, RawFr::Element fe1, RawFr::Element fe2, RawFr::Element fe3)
{
    mpz_t r0;
    mpz_init(r0);
    fr.toMpz(r0, fe0);
    mpz_t r1;
    mpz_init(r1);
    fr.toMpz(r1, fe1);
    mpz_t r2;
    mpz_init(r2);
    fr.toMpz(r2, fe2);
    mpz_t r3;
    mpz_init(r3);
    fr.toMpz(r3, fe3);
    mpz_class s0(r0);
    mpz_class s1(r1);
    mpz_class s2(r2);
    mpz_class s3(r3);
    scalar = s0 + (s1<<64) + (s2<<128) + (s3<<192);
    mpz_clear(r0);
    mpz_clear(r1);
    mpz_clear(r2);
    mpz_clear(r3);
}

void scalar2fea (RawFr &fr, const mpz_t scalar, RawFr::Element &fe0, RawFr::Element &fe1, RawFr::Element &fe2, RawFr::Element &fe3)
{
    mpz_t aux1;
    mpz_init_set(aux1, scalar);
    mpz_t aux2;
    mpz_init(aux2);
    mpz_t result;
    mpz_init(result);
    mpz_t band;
    mpz_init_set_ui(band, 0xFFFFFFFFFFFFFFFF);

    mpz_and(result, aux1, band);
    fr.fromMpz(fe0, result);

    mpz_div_2exp(aux2, aux1, 64);
    mpz_and(result, aux2, band);
    fr.fromMpz(fe1, result);


    mpz_div_2exp(aux1, aux2, 64);
    mpz_and(result, aux1, band);
    fr.fromMpz(fe2, result);

    mpz_div_2exp(aux2, aux1, 64);
    mpz_and(result, aux2, band);
    fr.fromMpz(fe3, result);

    mpz_clear(aux1);
    mpz_clear(aux2);
    mpz_clear(result);
    mpz_clear(band);
}

void scalar2fe (RawFr &fr, mpz_class &scalar, RawFr::Element &fe)
{
    fr.fromMpz(fe, scalar.get_mpz_t());
}

void scalar2fea (RawFr &fr, mpz_class &scalar, RawFr::Element &fe0, RawFr::Element &fe1, RawFr::Element &fe2, RawFr::Element &fe3)
{
    mpz_class band(0xFFFFFFFFFFFFFFFF);
    mpz_class aux;
    aux = scalar & band;
    scalar2fe(fr, aux, fe0);
    aux = scalar>>64 & band;
    scalar2fe(fr, aux, fe1);
    aux = scalar>>128 & band;
    scalar2fe(fr, aux, fe2);
    aux = scalar>>192 & band;
    scalar2fe(fr, aux, fe3);
}

/*
// Field element array to Big Number
function fea2bn(Fr, arr) {
    let res = Fr.toObject(arr[0]);
    res = Scalar.add(res, Scalar.shl(Fr.toObject(arr[1]), 64));
    res = Scalar.add(res, Scalar.shl(Fr.toObject(arr[2]), 128));
    res = Scalar.add(res, Scalar.shl(Fr.toObject(arr[3]), 192));
    return res;
}
*/
//void bn2bna(RawFr &fr, mpz_t bn, RawFr::Element &result[4])
//{
  //  ;//mfz_
//}
/*
// Big Number to field element array 
function bn2bna(Fr, bn) {
    bn = Scalar.e(bn);
    const r0 = Scalar.band(bn, Scalar.e("0xFFFFFFFFFFFFFFFF"));
    const r1 = Scalar.band(Scalar.shr(bn, 64), Scalar.e("0xFFFFFFFFFFFFFFFF"));
    const r2 = Scalar.band(Scalar.shr(bn, 128), Scalar.e("0xFFFFFFFFFFFFFFFF"));
    const r3 = Scalar.band(Scalar.shr(bn, 192), Scalar.e("0xFFFFFFFFFFFFFFFF"));
    return [Fr.e(r0), Fr.e(r1), Fr.e(r2),Fr.e(r3)];
}
*/

// Field Element to Number
/*
int64_t fe2n (RawFr &fr, RawFr::Element &fe)
{
    int64_t result;
    mpz_t maxInt;
    mpz_init_set_str(maxInt, "0x7FFFFFFF", 16);
    mpz_t minInt;
    mpz_init_set_str(minInt, "0x80000000", 16);
    mpz_t n;
    mpz_init_set_str(n, fr.toString(fe,10).c_str(), 10); // TODO: Refactor not to use strings
    if ( mpz_cmp(n,maxInt) > 0 )
    {
        mpz_t on;
        mpz_init_set_si(on,0);
        mpz_t q;
        mpz_init_set_str(q, Fr_element2str(&Fr_q), 16); // TODO: Refactor not to use strings
        //RawFr::Element prime;
        //fr.fromUI(prime, Fr_q.longVal[0]);
        //fr.toMpz(q, prime);
        mpz_sub(on, q, n);
        if ( mpz_cmp(on, minInt) > 0 )
        {
            result = -mpz_get_ui(on);
        } else {
            cerr << "Error: fe2n() Accessing a no 32bit value" << endl;
            exit(-1);
        }
        mpz_clear(q);
        mpz_clear(on);
    } else {
        result = mpz_get_ui(n);
    }
    mpz_clear(maxInt);
    mpz_clear(minInt);
    mpz_clear(n);
    return result;
}*/
/*
// Field Element to Number
function fe2n(Fr, fe) {
    const maxInt = Scalar.e("0x7FFFFFFF");
    const minInt = Scalar.sub(Fr.p, Scalar.e("0x80000000"));
    const o = Fr.toObject(fe);
    if (Scalar.gt(o, maxInt)) {
        const on = Scalar.sub(Fr.p, o);
        if (Scalar.gt(o, minInt)) {
            return -Scalar.toNumber(on);
        }
        throw new Error(`Accessing a no 32bit value: ${ctx.ln}`);
    } else {
        return Scalar.toNumber(o);
    }
}
*/

// Field Element to Number
int64_t fe2n (RawFr &fr, RawFr::Element &fe)
{
    // Get S32 limits
    mpz_class maxInt(0x7FFFFFFF);
    mpz_class minInt(0x80000000);
    
    // Get o = fe
    mpz_t raw;
    mpz_init(raw);
    fr.toMpz(raw, fe);
    mpz_class o(raw);
    mpz_clear(raw);
    
    // Get the prime number of the finite field
    mpz_class p(Fr_element2str(&Fr_q), 16); // TODO: avoid using strings, but Fr_q is not a Raw::Element

    if (o > maxInt)
    {
        mpz_class on = p - o;
        if (o > minInt) {
            return -on.get_si();
        }
        cerr << "Error: fe2n() accessing a non-32bit value: " << fr.toString(fe,16) << endl;
        exit(-1);
    }
    else {
        return o.get_si();
    }
}

