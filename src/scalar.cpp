#include <iostream>
#include <sstream> 
#include <iomanip>
#include <vector>
#include "scalar.hpp"
#include "ecrecover/ecrecover.hpp"
#include "XKCP/Keccak-more-compact.hpp"
#include "config.hpp"

void fea2scalar (FiniteField &fr, mpz_class &scalar, const FieldElement (&fea)[8])
{
    for (uint64_t i=0; i<8; i++)
    {
        if (fea[i]>=0x100000000)
        {
            cerr << "fea2scalar() found element i=" << i << " has a too high value=" << fr.toString(fea[i], 16) << endl;
            exit(-1);
        }
    }
    scalar = fea[7];
    scalar = scalar<<32;
    scalar += fea[6];
    scalar = scalar<<32;
    scalar += fea[5];
    scalar = scalar<<32;
    scalar += fea[4];
    scalar = scalar<<32;
    scalar += fea[3];
    scalar = scalar<<32;
    scalar += fea[2];
    scalar = scalar<<32;
    scalar += fea[1];
    scalar = scalar<<32;
    scalar += fea[0];
}

void fea2scalar (FiniteField &fr, mpz_class &scalar, FieldElement &fe0, FieldElement &fe1, FieldElement &fe2, FieldElement &fe3, FieldElement &fe4, FieldElement &fe5, FieldElement &fe6, FieldElement &fe7)
{
    FieldElement fea[8] ={fe0, fe1, fe2, fe3, fe4, fe5, fe6, fe7};
    fea2scalar(fr, scalar, fea);
}

void fea2scalar (FiniteField &fr, mpz_class &scalar, FieldElement &fe0, uint32_t &fe1, uint32_t &fe2, uint32_t &fe3, uint32_t &fe4, uint32_t &fe5, uint32_t &fe6, uint32_t &fe7)
{
    FieldElement fea[8] ={fe0, fe1, fe2, fe3, fe4, fe5, fe6, fe7};
    fea2scalar(fr, scalar, fea);
}

void fea2scalar (FiniteField &fr, mpz_class &scalar, uint32_t &fe0, uint32_t &fe1, uint32_t &fe2, uint32_t &fe3, uint32_t &fe4, uint32_t &fe5, uint32_t &fe6, uint32_t &fe7)
{
    FieldElement fea[8] ={fe0, fe1, fe2, fe3, fe4, fe5, fe6, fe7};
    fea2scalar(fr, scalar, fea);
}

void fea2scalar (FiniteField &fr, mpz_class &scalar, const FieldElement (&fea)[4])
{
    // To be enabled if field elements become bigger than 64 bits
    //zkassert(fe0<0x10000000000000000);
    //zkassert(fe1<0x10000000000000000);
    //zkassert(fe2<0x10000000000000000);
    //zkassert(fe3<0x10000000000000000);
    scalar += fea[3];
    scalar = scalar<<64;
    scalar += fea[2];
    scalar = scalar<<64;
    scalar += fea[1];
    scalar = scalar<<64;
    scalar += fea[0];
}

void fe2scalar  (FiniteField &fr, mpz_class &scalar, const FieldElement &fe)
{
    scalar = fe;
}

void scalar2fe  (FiniteField &fr, const mpz_class &scalar, FieldElement &fe)
{
    if (scalar>0xFFFFFFFFFFFFFFFF || scalar<0)
    {
        cerr << "scalar2fe() found scalar too large:" << scalar.get_str(16) << endl;
        exit(-1);
    }
    fe = scalar.get_ui();
}

void scalar2fea (FiniteField &fr, const mpz_class &scalar, FieldElement &fe0, FieldElement &fe1, FieldElement &fe2, FieldElement &fe3, FieldElement &fe4, FieldElement &fe5, FieldElement &fe6, FieldElement &fe7)
{
    mpz_class band(0xFFFFFFFF);
    mpz_class aux;
    aux = scalar & band;
    fe0 = aux.get_ui();
    aux = scalar>>32 & band;
    fe1 = aux.get_ui();
    aux = scalar>>64 & band;
    fe2 = aux.get_ui();
    aux = scalar>>96 & band;
    fe3 = aux.get_ui();
    aux = scalar>>128 & band;
    fe4 = aux.get_ui();
    aux = scalar>>160 & band;
    fe5 = aux.get_ui();
    aux = scalar>>192 & band;
    fe6 = aux.get_ui();
    aux = scalar>>224 & band;
    fe7 = aux.get_ui();    
}

void scalar2fea (FiniteField &fr, const mpz_class &scalar, FieldElement (&fea)[8])
{
    scalar2fea(fr, scalar, fea[0], fea[1], fea[2], fea[3], fea[4], fea[5], fea[6], fea[7]);
}

void scalar2fea (FiniteField &fr, const mpz_class &scalar, FieldElement (&fea)[4])
{
    mpz_class band(0xFFFFFFFFFFFFFFFF);
    mpz_class aux;
    aux = scalar & band;
    if (aux>=fr.prime())
    {
        cerr << "Error: scalar2fea() found value higher than prime: " << aux.get_str(16) << endl;
        exit(-1);
    }
    fea[0] = aux.get_ui();
    aux = scalar>>64 & band;
    if (aux>=fr.prime())
    {
        cerr << "Error: scalar2fea() found value higher than prime: " << aux.get_str(16) << endl;
        exit(-1);
    }
    fea[1] = aux.get_ui();
    aux = scalar>>128 & band;
    if (aux>=fr.prime())
    {
        cerr << "Error: scalar2fea() found value higher than prime: " << aux.get_str(16) << endl;
        exit(-1);
    }
    fea[2] = aux.get_ui();
    aux = scalar>>192 & band;
    if (aux>=fr.prime())
    {
        cerr << "Error: scalar2fea() found value higher than prime: " << aux.get_str(16) << endl;
        exit(-1);
    }
    fea[3] = aux.get_ui();
}

void scalar2key (FiniteField &fr, mpz_class &s, FieldElement (&key)[4])
{
    mpz_class auxk[4] = {0, 0, 0, 0};
    mpz_class r = s;
    mpz_class one = 1;
    uint64_t i = 0;

    while (r != 0)
    {
        if ((r&1) != 0)
        {
            auxk[i%4] = auxk[i%4] + (one << i/4);
        }
        r = r >> 1;
        i++;
    }
    
    for (uint64_t j=0; j<4; j++) key[j] = auxk[j].get_ui();
}

void string2fe (FiniteField &fr, const string &s, FieldElement &fe)
{
    fr.fromString(fe, Remove0xIfPresent(s), 16);
}

string fea2string (FiniteField &fr, const FieldElement(&fea)[4])
{
    mpz_class auxScalar;
    fea2scalar(fr, auxScalar, fea);
    return auxScalar.get_str(16);
}

// Field Element to Number
int64_t fe2n (FiniteField &fr, const FieldElement &fe)
{
    // Get S32 limits     
    mpz_class maxInt(0x7FFFFFFF);
    mpz_class minInt;
    minInt = fr.prime() - 0x80000000;

    mpz_class o = fe;

    if (o > maxInt)
    {
        mpz_class on = fr.prime() - o;
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

uint64_t fe2u64 (FiniteField &fr, const FieldElement &fe)
{
    return fe;
}

void u82fe (FiniteField &fr, FieldElement &fe, uint8_t n)
{
    fr.fromUI(fe, n);
}

void s82fe (FiniteField &fr, FieldElement &fe, int8_t n)
{
    if (n>=0) fr.fromUI(fe, n);
    else
    {
        fr.fromUI(fe, -n);
        fr.neg(fe, fe);
    }
}

void u162fe (FiniteField &fr, FieldElement &fe, uint16_t n)
{
    fr.fromUI(fe, n);
}

void s162fe (FiniteField &fr, FieldElement &fe, int16_t  n)
{
    if (n>=0) fr.fromUI(fe, n);
    else
    {
        fr.fromUI(fe, -n);
        fr.neg(fe, fe);
    }
}

void u322fe (FiniteField &fr, FieldElement &fe, uint32_t n)
{
    fr.fromUI(fe, n);
}

void s322fe (FiniteField &fr, FieldElement &fe, int32_t  n)
{
    if (n>=0) fr.fromUI(fe, n);
    else
    {
        fr.fromUI(fe, -n);
        fr.neg(fe, fe);
    }
}

void u642fe (FiniteField &fr, FieldElement &fe, uint64_t n)
{
    fr.fromUI(fe, n);
}

void s642fe (FiniteField &fr, FieldElement &fe, int64_t n)
{
    if (n>=0) fr.fromUI(fe, n);
    else
    {
        fr.fromUI(fe, -n);
        fr.neg(fe, fe);
    }
}

string Remove0xIfPresent(const string &s)
{
    uint64_t position = 0;
    if (s.find("0x") == 0) position = 2;
    return s.substr(position);
}

string Add0xIfMissing(string s)
{
    if (s.find("0x") == 0) return s;
    return "0x" + s;
}

string PrependZeros (string s, uint64_t n)
{
    if (s.size() > n)
    {
        cerr << "Error: PrependZeros() called with a string with too large size: " << s.size() << endl;
        exit(-1);
    }
    while (s.size() < n) s = "0" + s;
    return s;
}

string NormalizeToNFormat (string s, uint64_t n)
{
    return PrependZeros(Remove0xIfPresent(s), n);
}

string NormalizeTo0xNFormat (string s, uint64_t n)
{
    return "0x" + NormalizeToNFormat(s, n);
}
void keccak256(const uint8_t *pInputData, uint64_t inputDataSize, uint8_t *pOutputData, uint64_t outputDataSize)
{
    Keccak(1088, 512, pInputData, inputDataSize, 0x1, pOutputData, outputDataSize);
}

string keccak256 (uint8_t *pInputData, uint64_t inputDataSize)
{
    std::array<uint8_t,32> hash = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    keccak256(pInputData, inputDataSize, hash.data(), hash.size());

    string s;
    ba2string(s, hash.data(), hash.size());
    return "0x" + s;
}

void keccak256 (string &inputString, uint8_t *pOutputData, uint64_t outputDataSize)
{
    string s = Remove0xIfPresent(inputString);
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
    string s = Remove0xIfPresent(inputString);
    uint64_t bufferSize = s.size()/2 + 2;
    uint8_t * pData = (uint8_t *)malloc (bufferSize);
    if (pData == NULL)
    {
        cerr << "ERROR: keccak256(string) failed calling malloc" << endl;
        exit(-1);
    }
    uint64_t dataSize = string2ba(s, pData, bufferSize);
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

uint64_t string2ba (const string &os, uint8_t *pData, uint64_t &dataSize)
{
    string s = Remove0xIfPresent(os);

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
    for (uint64_t i=0; i<dsize; i++)
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

void ba2scalar(const uint8_t *pData, uint64_t dataSize, mpz_class &s)
{
    s = 0;
    for (uint64_t i=0; i<dataSize; i++)
    {
        s *= 256;
        s += pData[i];
    }
}

void scalar2ba(uint8_t *pData, uint64_t &dataSize, mpz_class s)
{
    uint64_t i=0;
    for (; i<dataSize; i++)
    {
        // Shift left 1B the byte array content
        for (uint64_t j=i; j>0; j--) pData[j] = pData[j-1];

        // Add the next byte to the byte array
        mpz_class auxScalar = s & 0xFF;
        pData[0] = auxScalar.get_ui();

        // Shift right 1B the scalar content
        s = s >> 8;

        // When we run out of significant bytes, break
        if (s == 0) break;
    }
    if (s!=0)
    {
        cerr << "Error: scalar2ba() run out of buffer of " << dataSize << " bytes" << endl;
        exit(-1);
    }
    dataSize = i+1;
}

void scalar2ba16(uint64_t *pData, uint64_t &dataSize, mpz_class s)
{
    uint64_t i=0;
    for (; i<dataSize; i++)
    {
        // Shift left the 2-byte array content
        for (uint64_t j=i; j>0; j--) pData[j] = pData[j-1];

        // Add the next byte to the byte array
        mpz_class auxScalar = s & (i<(dataSize-1)?0xFFFF:0xFFFFF);
        pData[0] = auxScalar.get_ui();

        // Shift right 1B the scalar content
        s = s >> 16;

        // When we run out of significant bytes, break
        if (s == 0) break;
    }
    if (s!=0)
    {
        cerr << "Error: scalar2ba() run out of buffer of " << dataSize << " bytes" << endl;
        exit(-1);
    }
    dataSize = i+1;
}

void scalar2bits(mpz_class s, vector<uint8_t> &bits)
{
    while (s > 0)
    {
        if ((s&1) == 1)
        {
            bits.push_back(1);
        }
        else
        {
            bits.push_back(0);
        }
        s = s >> 1;
    }
}

void byte2bits(uint8_t byte, uint8_t *pBits)
{
    for (uint64_t i=0; i<8; i++)
    {
        if ((byte&1) == 1)
        {
            pBits[i] = 1;
        }
        else
        {
            pBits[i] = 0;
        }
        byte = byte >> 1;
    }    
}

void bits2byte(const uint8_t *pBits, uint8_t &byte)
{
    byte = 0;
    for (uint64_t i=0; i<8; i++)
    {
        byte = byte << 1;
        if ((pBits[7-i]&0x01) == 1)
        {
            byte |= 1;
        }
    }
}

void sr8to4 ( FiniteField &fr,
              FieldElement a0,
              FieldElement a1,
              FieldElement a2,
              FieldElement a3,
              FieldElement a4,
              FieldElement a5,
              FieldElement a6,
              FieldElement a7,
              FieldElement &r0,
              FieldElement &r1,
              FieldElement &r2,
              FieldElement &r3 )
{
    r0 = fr.add(a0, fr.mul(a1, 0x100000000));
    r1 = fr.add(a2, fr.mul(a3, 0x100000000));
    r2 = fr.add(a4, fr.mul(a5, 0x100000000));
    r3 = fr.add(a6, fr.mul(a7, 0x100000000));
}

void sr4to8 ( FiniteField &fr,
              FieldElement a0,
              FieldElement a1,
              FieldElement a2,
              FieldElement a3,
              FieldElement &r0,
              FieldElement &r1,
              FieldElement &r2,
              FieldElement &r3,
              FieldElement &r4,
              FieldElement &r5,
              FieldElement &r6,
              FieldElement &r7 )
{
    r0 = a0 & 0xFFFFFFFF;
    r1 = a0 >> 32;
    r2 = a1 & 0xFFFFFFFF;
    r3 = a1 >> 32;
    r4 = a2 & 0xFFFFFFFF;
    r5 = a2 >> 32;
    r6 = a3 & 0xFFFFFFFF;
    r7 = a3 >> 32;
}

mpz_class Mask8("FF", 16);
mpz_class Mask256("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
mpz_class twoTo64 ("10000000000000000", 16);
mpz_class twoTo128("100000000000000000000000000000000", 16);
mpz_class twoTo192("1000000000000000000000000000000000000000000000000", 16);
mpz_class twoTo256("10000000000000000000000000000000000000000000000000000000000000000", 16);
mpz_class twoTo255("8000000000000000000000000000000000000000000000000000000000000000", 16);