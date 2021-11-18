#include <iostream>
#include "scalar.hpp"

void fea2bn (Context &ctx, mpz_t &result, RawFr::Element fe0, uint64_t fe1, uint64_t fe2, uint64_t fe3)
{
    // Convert field elements to mpz
    mpz_t r0, r1, r2, r3;
    mpz_init_set_ui(r0,0);
    mpz_init_set_ui(r1,fe1);
    mpz_init_set_ui(r2,fe2);
    mpz_init_set_ui(r3,fe3);
    ctx.pFr->toMpz(r0, fe0);
    //ctx.pFr->toMpz(r1, fe1);
    //ctx.pFr->toMpz(r2, fe2);
    //ctx.pFr->toMpz(r3, fe3);

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
    mpz_add(result, result01, result23);

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
int64_t fe2n (RawFr &fr, RawFr::Element &fe) {
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
}
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

