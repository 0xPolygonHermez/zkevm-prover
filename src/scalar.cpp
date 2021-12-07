#include <iostream>
#include <sstream> 
#include <iomanip>
#include "scalar.hpp"
#include "ecrecover/ecrecover.hpp"
#include "XKCP/Keccak.hpp"

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
    mpz_class s0;
    fe2scalar(fr, s0, fe0);
    mpz_class s1(fe1);
    mpz_class s2(fe2);
    mpz_class s3(fe3);
    scalar = s0 + (s1<<64) + (s2<<128) + (s3<<192);
}

void fea2scalar (RawFr &fr, mpz_class &scalar, RawFr::Element &fe0, RawFr::Element fe1, RawFr::Element fe2, RawFr::Element fe3)
{
    mpz_class s0;
    mpz_class s1;
    mpz_class s2;
    mpz_class s3;
    fe2scalar(fr, s0, fe0);
    fe2scalar(fr, s1, fe1);
    fe2scalar(fr, s2, fe2);
    fe2scalar(fr, s3, fe3);
    scalar = s0 + (s1<<64) + (s2<<128) + (s3<<192);
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

void string2fe (RawFr &fr, string s, RawFr::Element &fe)
{
    mpz_class aux(s);
    fr.fromMpz(fe, aux.get_mpz_t());
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


void keccak256(const uint8_t *pInputData, uint64_t inputDataSize, uint8_t *pOutputData, uint64_t outputDataSize)
{
    Keccak(1088, 512, pInputData, inputDataSize, 0x1, pOutputData, outputDataSize);
}

string keccak256 (uint8_t *pInputData, uint64_t inputDataSize)
{
    std::array<uint8_t,32> hash;
    keccak256(pInputData, inputDataSize, hash.data(), hash.size());

    string s;
    ba2string(s, hash.data(), hash.size());
    return "0x" + s;
}


void keccak256 (string &inputString, uint8_t *pOutputData, uint64_t outputDataSize)
{
    string s = RemoveOxIfPresent(inputString);
    uint64_t bufferSize = s.size()/2 + 2;
    uint8_t * pData = (uint8_t *)malloc (bufferSize);
    if (pData == NULL)
    {
        cerr << "ERROR: keccak256(string) failed calling malloc" << endl;
        exit(-1);
    }
    uint64_t dataSize = string2ba(s, pData, dataSize);
    keccak256(pData, dataSize, pOutputData, outputDataSize);
}

string keccak256 (string &inputString)
{
    string s = RemoveOxIfPresent(inputString);
    uint64_t bufferSize = s.size()/2 + 2;
    uint8_t * pData = (uint8_t *)malloc (bufferSize);
    if (pData == NULL)
    {
        cerr << "ERROR: keccak256(string) failed calling malloc" << endl;
        exit(-1);
    }
    uint64_t dataSize = string2ba(s, pData, dataSize);
    string result = keccak256(pData, dataSize);
    free(pData);
    return result;
}
  
uint8_t char2byte (char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    cerr << "Error: char2byte() called with an invalid, non-hex char: " << c << endl;
    exit(-1);
}

char byte2char (uint8_t b)
{
    if (b < 10) return '0' + b;
    if (b < 16) return 'A' + b - 10;
    cerr << "Error: char2byte() called with an invalid byte: " << b << endl;
    exit(-1);  
}

string byte2string(uint8_t b)
{
    char s[3];
    s[0] = byte2char(b>>4);
    s[1] = byte2char(b & 0x0F);
    s[2] = 0;
    string ss(s);
    return ss;
}

uint64_t string2ba (string os, uint8_t *pData, uint64_t &dataSize)
{
    string s = RemoveOxIfPresent(os);

    if (s.size()%2 != 0)
    {
        s = "0" + s;
    }

    uint64_t dsize = s.size()/2;
    if (dsize > dataSize)
    {
        cerr << "Error: string2ba() called with a too short buffer: " << dsize << ">" << dataSize << endl;
        exit(-1);
    }

    const char *p = s.c_str();
    for (int i=0; i<dsize; i++)
    {
        pData[i] = char2byte(p[2*i])*16 + char2byte(p[2*i + 1]);
    }
    return dsize;
}

void ba2string (string &s, const uint8_t *pData, uint64_t dataSize)
{
    s = "";
    for (uint64_t i=0; i<dataSize; i++)
    {
        s.append(1, byte2char(pData[i] >> 4));
        s.append(1, byte2char(pData[i] & 0x0F));
    }
}

void ba2u16(const uint8_t *pData, uint16_t &n)
{
    n = pData[0]*256 + pData[1];
}

void ba2scalar(const uint8_t *pData, mpz_class &n)
{
    n = 0;
    for (uint64_t i=0; i<32; i++)
    {
        n *= 256;
        n += pData[i];
    }
}