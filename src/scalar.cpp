#include <iostream>
#include <sstream> 
#include <iomanip>
#include "scalar.hpp"
#include "keccak-tiny.h"
#include "ecrecover/ecrecover.hpp"

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

// Field Element to Number
int64_t fe2n (Context &ctx, RawFr::Element &fe)
{
    // Get S32 limits     
    mpz_class maxInt(0x7FFFFFFF);
    mpz_class minInt;
    minInt = ctx.prime - 0x80000000;

    mpz_class o;
    fe2scalar(ctx.fr, o, fe);

    if (o > maxInt)
    {
        mpz_class on = ctx.prime - o;
        if (o > minInt) {
            return -on.get_si();
        }
        cerr << "Error: fe2n() accessing a non-32bit value: " << ctx.fr.toString(fe,16) << endl;
        exit(-1);
    }
    else {
        return o.get_si();
    }
}

uint64_t fe2u64 (RawFr &fr, RawFr::Element &fe)
{
    mpz_class aux;
    fe2scalar(fr, aux, fe);
    
    if (aux.fits_ulong_p()) return aux.get_ui();

    cerr << "Error: fe2u64() called with non-64B fe: " << fr.toString(fe,16) << endl;
    exit(-1);
}

string RemoveOxIfPresent(string s)
{
    uint64_t position = 0;
    if (s.find("0x") == 0) position = 2;
    return s.substr(position);
}

string PrependZeros (string s, uint64_t n)
{
    if (s.size() > n)
    {
        cerr << "Error: RemovePrependZerosOxIfPresent() called with a string with too large size: " << s.size() << endl;
        exit(-1);
    }
    while (s.size() < n) s = "0" + s;
    return s;
}

string NormalizeToNFormat (string s, uint64_t n)
{
    return PrependZeros(RemoveOxIfPresent(s), n);
}

string NormalizeTo0xNFormat (string s, uint64_t n)
{
    return "0x" + NormalizeToNFormat(s, n);
}

void GetPrimeNumber (RawFr &fr, mpz_class &p) // TODO: Hardcode this value to avoid overhead
{
    fe2scalar(fr, p, fr.negOne());
    p += 1;
}


string keccak256 (uint8_t * pData, uint64_t &dataSize)
{
    std::array<uint8_t,32> hash;
    keccak_256(hash.data(), hash.size(), pData, dataSize);

    // Convert an array of bytes to an hexa string
    std::stringstream s;
    s.fill('0');
    for ( size_t i = 0 ; i < 32 ; i++ )
       s << std::setw(2) << std::hex << hash[i];    // TODO: Can we avoid converting to/from strings?  This is not efficient.
    return "0x" + s.str();
}
  
  