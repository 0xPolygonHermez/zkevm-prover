#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "scalar.hpp"
#include "XKCP/Keccak-more-compact.hpp"
#include "config.hpp"
#include "utils.hpp"

/* Global scalar variables */

mpz_class Mask4   ("F", 16);
mpz_class Mask8   ("FF", 16);
mpz_class Mask16  ("FFFF", 16);
mpz_class Mask20  ("FFFFF", 16);
mpz_class Mask32  ("FFFFFFFF", 16);
mpz_class Mask64  ("FFFFFFFFFFFFFFFF", 16);
mpz_class Mask256 ("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
mpz_class TwoTo8  ("100", 16);
mpz_class TwoTo16 ("10000", 16);
mpz_class TwoTo18 ("40000", 16);
mpz_class TwoTo32 ("100000000", 16);
mpz_class TwoTo64 ("10000000000000000", 16);
mpz_class TwoTo128("100000000000000000000000000000000", 16);
mpz_class TwoTo192("1000000000000000000000000000000000000000000000000", 16);
mpz_class TwoTo256("10000000000000000000000000000000000000000000000000000000000000000", 16);
mpz_class TwoTo255("8000000000000000000000000000000000000000000000000000000000000000", 16);
mpz_class TwoTo258("40000000000000000000000000000000000000000000000000000000000000000", 16);
mpz_class Zero    ("0", 16);
mpz_class One     ("1", 16);
mpz_class GoldilocksPrime = (uint64_t)GOLDILOCKS_PRIME;

/* Scalar to/from field element conversion */

void fe2scalar (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element &fe)
{
    scalar = fr.toU64(fe);
}

void scalar2fe (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element &fe)
{
    if (scalar>Mask64 || scalar<Zero)
    {
        cerr << "scalar2fe() found scalar out of u64 range:" << scalar.get_str(16) << endl;
        exitProcess();
    }
    fe = fr.fromU64(scalar.get_ui());
}

/* Scalar to/from field element array */

void fea2scalar (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element (&fea)[8])
{
    uint64_t aux;

    // Add field element 7
    aux = fr.toU64(fea[7]);
    if (aux >= 0x100000000)
    {
        cerr << "fea2scalar() found element 7 has a too high value=" << fr.toString(fea[7], 16) << endl;
        exitProcess();
    }
    scalar = aux;
    scalar = scalar<<32;

    // Add field element 6
    aux = fr.toU64(fea[6]);
    if (aux >= 0x100000000)
    {
        cerr << "fea2scalar() found element 6 has a too high value=" << fr.toString(fea[6], 16) << endl;
        exitProcess();
    }
    scalar += aux;
    scalar = scalar<<32;

    // Add field element 5
    aux = fr.toU64(fea[5]);
    if (aux >= 0x100000000)
    {
        cerr << "fea2scalar() found element 5 has a too high value=" << fr.toString(fea[5], 16) << endl;
        exitProcess();
    }
    scalar += aux;
    scalar = scalar<<32;

    // Add field element 4
    aux = fr.toU64(fea[4]);
    if (aux >= 0x100000000)
    {
        cerr << "fea2scalar() found element 4 has a too high value=" << fr.toString(fea[4], 16) << endl;
        exitProcess();
    }
    scalar += aux;
    scalar = scalar<<32;

    // Add field element 3
    aux = fr.toU64(fea[3]);
    if (aux >= 0x100000000)
    {
        cerr << "fea2scalar() found element 3 has a too high value=" << fr.toString(fea[3], 16) << endl;
        exitProcess();
    }
    scalar += aux;
    scalar = scalar<<32;

    // Add field element 2
    aux = fr.toU64(fea[2]);
    if (aux >= 0x100000000)
    {
        cerr << "fea2scalar() found element 2 has a too high value=" << fr.toString(fea[2], 16) << endl;
        exitProcess();
    }
    scalar += aux;
    scalar = scalar<<32;

    // Add field element 1
    aux = fr.toU64(fea[1]);
    if (aux >= 0x100000000)
    {
        cerr << "fea2scalar() found element 1 has a too high value=" << fr.toString(fea[1], 16) << endl;
        exitProcess();
    }
    scalar += aux;
    scalar = scalar<<32;

    // Add field element 0
    aux = fr.toU64(fea[0]);
    if (aux >= 0x100000000)
    {
        cerr << "fea2scalar() found element 0 has a too high value=" << fr.toString(fea[0], 16) << endl;
        exitProcess();
    }
    scalar += aux;
}

void fea2scalar (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element (&fea)[4])
{
    scalar += fr.toU64(fea[3]);
    scalar = scalar<<64;
    scalar += fr.toU64(fea[2]);
    scalar = scalar<<64;
    scalar += fr.toU64(fea[1]);
    scalar = scalar<<64;
    scalar += fr.toU64(fea[0]);
}

void fea2scalar (Goldilocks &fr, mpz_class &scalar, Goldilocks::Element &fe0, Goldilocks::Element &fe1, Goldilocks::Element &fe2, Goldilocks::Element &fe3, Goldilocks::Element &fe4, Goldilocks::Element &fe5, Goldilocks::Element &fe6, Goldilocks::Element &fe7)
{
    Goldilocks::Element fea[8] ={fe0, fe1, fe2, fe3, fe4, fe5, fe6, fe7};
    fea2scalar(fr, scalar, fea);
}

void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element (&fea)[8])
{
    scalar2fea(fr, scalar, fea[0], fea[1], fea[2], fea[3], fea[4], fea[5], fea[6], fea[7]);
}

void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element (&fea)[4])
{
    mpz_class aux;

    aux = scalar & Mask64;
    if (aux >= GoldilocksPrime)
    {
        cerr << "Error: scalar2fea() found value higher than prime: " << aux.get_str(16) << endl;
        exitProcess();
    }
    fea[0] = fr.fromU64(aux.get_ui());

    aux = scalar>>64 & Mask64;
    if (aux >= GoldilocksPrime)
    {
        cerr << "Error: scalar2fea() found value higher than prime: " << aux.get_str(16) << endl;
        exitProcess();
    }
    fea[1] = fr.fromU64(aux.get_ui());

    aux = scalar>>128 & Mask64;
    if (aux >= GoldilocksPrime)
    {
        cerr << "Error: scalar2fea() found value higher than prime: " << aux.get_str(16) << endl;
        exitProcess();
    }
    fea[2] = fr.fromU64(aux.get_ui());

    aux = scalar>>192 & Mask64;
    if (aux >= GoldilocksPrime)
    {
        cerr << "Error: scalar2fea() found value higher than prime: " << aux.get_str(16) << endl;
        exitProcess();
    }
    fea[3] = fr.fromU64(aux.get_ui());
}

void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element &fe0, Goldilocks::Element &fe1, Goldilocks::Element &fe2, Goldilocks::Element &fe3, Goldilocks::Element &fe4, Goldilocks::Element &fe5, Goldilocks::Element &fe6, Goldilocks::Element &fe7)
{
    mpz_class aux;
    aux = scalar & Mask32;
    fe0 = fr.fromU64(aux.get_ui());
    aux = scalar>>32 & Mask32;
    fe1 = fr.fromU64(aux.get_ui());
    aux = scalar>>64 & Mask32;
    fe2 = fr.fromU64(aux.get_ui());
    aux = scalar>>96 & Mask32;
    fe3 = fr.fromU64(aux.get_ui());
    aux = scalar>>128 & Mask32;
    fe4 = fr.fromU64(aux.get_ui());
    aux = scalar>>160 & Mask32;
    fe5 = fr.fromU64(aux.get_ui());
    aux = scalar>>192 & Mask32;
    fe6 = fr.fromU64(aux.get_ui());
    aux = scalar>>224 & Mask32;
    fe7 = fr.fromU64(aux.get_ui());
}

/* Scalar to/from a Sparse Merkle Tree key, interleaving bits */

void scalar2key (Goldilocks &fr, mpz_class &s, Goldilocks::Element (&key)[4])
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

    for (uint64_t j=0; j<4; j++) key[j] = fr.fromU64(auxk[j].get_ui());
}

/* Hexa string to/from field element (array) conversion */

void string2fe (Goldilocks &fr, const string &s, Goldilocks::Element &fe)
{
    fr.fromString(fe, Remove0xIfPresent(s), 16);
}

string fea2string (Goldilocks &fr, const Goldilocks::Element(&fea)[4])
{
    mpz_class auxScalar;
    fea2scalar(fr, auxScalar, fea);
    return auxScalar.get_str(16);
}

string fea2string (Goldilocks &fr, const Goldilocks::Element &fea0, const Goldilocks::Element &fea1, const Goldilocks::Element &fea2, const Goldilocks::Element &fea3)
{
    const Goldilocks::Element fea[4] = {fea0, fea1, fea2, fea3};
    return fea2string(fr, fea);
}

/* Normalized strings */

string Remove0xIfPresent(const string &s)
{
    if ( (s.size() >= 2) && (s.at(1) == 'x') && (s.at(0) == '0') ) return s.substr(2);
    return s;
}

string Add0xIfMissing(string s)
{
    if ( (s.size() >= 2) && (s.at(1) == 'x') && (s.at(0) == '0') ) return s;
    return "0x" + s;
}


// A set of strings with zeros is available in memory for performance reasons
string sZeros[64] = {
    "",
    "0",
    "00",
    "000",
    "0000",
    "00000",
    "000000",
    "0000000",
    "00000000",
    "000000000",
    "0000000000",
    "00000000000",
    "000000000000",
    "0000000000000",
    "00000000000000",
    "000000000000000",
    "0000000000000000",
    "00000000000000000",
    "000000000000000000",
    "0000000000000000000",
    "00000000000000000000",
    "000000000000000000000",
    "0000000000000000000000",
    "00000000000000000000000",
    "000000000000000000000000",
    "0000000000000000000000000",
    "00000000000000000000000000",
    "000000000000000000000000000",
    "0000000000000000000000000000",
    "00000000000000000000000000000",
    "000000000000000000000000000000",
    "0000000000000000000000000000000",
    "00000000000000000000000000000000",
    "000000000000000000000000000000000",
    "0000000000000000000000000000000000",
    "00000000000000000000000000000000000",
    "000000000000000000000000000000000000",
    "0000000000000000000000000000000000000",
    "00000000000000000000000000000000000000",
    "000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000"
};

string PrependZeros (string s, uint64_t n)
{
    // Check that n is not too big
    if (n > 64)
    {
        cerr << "Error: PrependZeros() called with an that is too big n=" << n << endl;
        exitProcess();
    }
    // Check that string size is not too big
    uint64_t stringSize = s.size();
    if ( (stringSize > n) || (stringSize > 64) )
    {
        cerr << "Error: PrependZeros() called with a string with too large s.size=" << stringSize << " n=" << n << endl;
        exitProcess();
    }

    // Prepend zeros if needed
    if (stringSize < n) return sZeros[n-stringSize] + s;

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

string stringToLower (const string &s)
{
    string result = s;
    transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

/* Keccak */

void keccak256(const uint8_t *pInputData, uint64_t inputDataSize, uint8_t *pOutputData, uint64_t outputDataSize)
{
    Keccak(1088, 512, pInputData, inputDataSize, 0x1, pOutputData, outputDataSize);
}

string keccak256 (const uint8_t *pInputData, uint64_t inputDataSize)
{
    std::array<uint8_t,32> hash = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    keccak256(pInputData, inputDataSize, hash.data(), hash.size());

    string s;
    ba2string(s, hash.data(), hash.size());
    return "0x" + s;
}

string keccak256 (const vector<uint8_t> &input)
{
    string baString;
    uint64_t inputSize = input.size();
    for (uint64_t i=0; i<inputSize; i++)
    {
        baString.push_back(input[i]);
    }
    return keccak256((uint8_t *)baString.c_str(), baString.size());
}

string keccak256 (const string &inputString)
{
    string s = Remove0xIfPresent(inputString);
    string baString;
    string2ba(s, baString);
    return keccak256((uint8_t *)baString.c_str(), baString.size());
}

/* Byte to/from char conversion */

uint8_t char2byte (char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    cerr << "Error: char2byte() called with an invalid, non-hex char: " << c << endl;
    exitProcess();
    return 0;
}

char byte2char (uint8_t b)
{
    if (b < 10) return '0' + b;
    if (b < 16) return 'A' + b - 10;
    cerr << "Error: byte2char() called with an invalid byte: " << b << endl;
    exitProcess();
    return 0;
}

string byte2string(uint8_t b)
{
    string result;
    result.push_back(byte2char(b >> 4));
    result.push_back(byte2char(b & 0x0F));
    return result;
}

/* Strint to/from byte array conversion
   s must be even sized, and must not include the leading "0x"
   pData buffer must be big enough to store converted data */

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
        exitProcess();
    }

    const char *p = s.c_str();
    for (uint64_t i=0; i<dsize; i++)
    {
        pData[i] = char2byte(p[2*i])*16 + char2byte(p[2*i + 1]);
    }
    return dsize;
}

void string2ba (const string &textString, string &baString)
{
    baString.clear();

    string s = Remove0xIfPresent(textString);

    if (s.size()%2 != 0)
    {
        s = "0" + s;
    }

    uint64_t dsize = s.size()/2;

    const char *p = s.c_str();
    for (uint64_t i=0; i<dsize; i++)
    {
        uint8_t aux = char2byte(p[2*i])*16 + char2byte(p[2*i + 1]);
        baString.push_back(aux);
    }
}

string string2ba (const string &textString)
{
    string result;
    string2ba(textString, result);
    return result;
}

uint64_t string2bv (const string &os, vector<uint8_t> &vData)
{
    string s = Remove0xIfPresent(os);

    if (s.size()%2 != 0)
    {
        s = "0" + s;
    }

    uint64_t dsize = s.size()/2;
    const char *p = s.c_str();
    for (uint64_t i=0; i<dsize; i++)
    {
        vData.push_back(char2byte(p[2*i])*16 + char2byte(p[2*i + 1]));
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

string ba2string (const uint8_t *pData, uint64_t dataSize)
{
    string result;
    ba2string(result, pData, dataSize);
    return result;
}

void ba2string (const string &baString, string &textString)
{
    ba2string(textString, (const uint8_t *)baString.c_str(), baString.size());
}

string ba2string (const string &baString)
{
    string result;
    ba2string(result, (const uint8_t *)baString.c_str(), baString.size());
    return result;
}

/* Byte array of exactly 2 bytes conversion */

void ba2u16 (const uint8_t *pData, uint16_t &n)
{
    n = pData[0]*256 + pData[1];
}

void ba2scalar (const uint8_t *pData, uint64_t dataSize, mpz_class &s)
{
    s = 0;
    for (uint64_t i=0; i<dataSize; i++)
    {
        s *= TwoTo8;
        s += pData[i];
    }
}

/* Scalar to byte array conversion (up to dataSize bytes) */

void scalar2ba (uint8_t *pData, uint64_t &dataSize, mpz_class s)
{
    uint64_t i=0;
    for (; i<dataSize; i++)
    {
        // Shift left 1B the byte array content
        for (uint64_t j=i; j>0; j--) pData[j] = pData[j-1];

        // Add the next byte to the byte array
        mpz_class auxScalar = s & Mask8;
        pData[0] = auxScalar.get_ui();

        // Shift right 1B the scalar content
        s = s >> 8;

        // When we run out of significant bytes, break
        if (s == Zero) break;
    }
    if (s != Zero)
    {
        cerr << "Error: scalar2ba() run out of buffer of " << dataSize << " bytes" << endl;
        exitProcess();
    }
    dataSize = i+1;
}

void scalar2ba16(uint64_t *pData, uint64_t &dataSize, mpz_class s)
{
    memset(pData, 0, dataSize*sizeof(uint64_t));
    uint64_t i=0;
    for (; i<dataSize; i++)
    {
        // Add the next byte to the byte array
        mpz_class auxScalar = s & ( (i<(dataSize-1)) ? Mask16 : Mask20 );
        pData[i] = auxScalar.get_ui();

        // Shift right 2 bytes the scalar content
        s = s >> 16;

        // When we run out of significant bytes, break
        if (s == Zero) break;
    }
    if (s > Mask4)
    {
        cerr << "Error: scalar2ba16() run out of buffer of " << dataSize << " bytes" << endl;
        exitProcess();
    }
    dataSize = i+1;
}

void scalar2bytes(mpz_class &s, uint8_t (&bytes)[32])
{
    for (uint64_t i=0; i<32; i++)
    {
        mpz_class aux = s & Mask8;
        bytes[i] = aux.get_ui();
        s = s >> 8;
    }
    if (s != Zero)
    {
        cerr << "Error: scalar2bytes() run out of space of 32 bytes" << endl;
        exitProcess();
    }
}

/* Converts a scalar to a vector of bits of the scalar, with value 1 or 0; bits[0] is least significant bit */

void scalar2bits(mpz_class s, vector<uint8_t> &bits)
{
    while (s > Zero)
    {
        if ((s & 1) == One)
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

/* Byte to/from bits array conversion, with value 1 or 0; bits[0] is the least significant bit */

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

/* 8 fe to/from 4 fe conversion */

void sr8to4 ( Goldilocks &fr,
              Goldilocks::Element a0,
              Goldilocks::Element a1,
              Goldilocks::Element a2,
              Goldilocks::Element a3,
              Goldilocks::Element a4,
              Goldilocks::Element a5,
              Goldilocks::Element a6,
              Goldilocks::Element a7,
              Goldilocks::Element &r0,
              Goldilocks::Element &r1,
              Goldilocks::Element &r2,
              Goldilocks::Element &r3 )
{
    r0 = fr.add(a0, fr.mul(a1, fr.fromU64(0x100000000)));
    r1 = fr.add(a2, fr.mul(a3, fr.fromU64(0x100000000)));
    r2 = fr.add(a4, fr.mul(a5, fr.fromU64(0x100000000)));
    r3 = fr.add(a6, fr.mul(a7, fr.fromU64(0x100000000)));
}

void sr4to8 ( Goldilocks &fr,
              Goldilocks::Element a0,
              Goldilocks::Element a1,
              Goldilocks::Element a2,
              Goldilocks::Element a3,
              Goldilocks::Element &r0,
              Goldilocks::Element &r1,
              Goldilocks::Element &r2,
              Goldilocks::Element &r3,
              Goldilocks::Element &r4,
              Goldilocks::Element &r5,
              Goldilocks::Element &r6,
              Goldilocks::Element &r7 )
{
    uint64_t aux;

    aux = fr.toU64(a0);
    r0 = fr.fromU64( aux & 0xFFFFFFFF );
    r1 = fr.fromU64( aux >> 32 );

    aux = fr.toU64(a1);
    r2 = fr.fromU64( aux & 0xFFFFFFFF );
    r3 = fr.fromU64( aux >> 32 );

    aux = fr.toU64(a2);
    r4 = fr.fromU64( aux & 0xFFFFFFFF );
    r5 = fr.fromU64( aux >> 32 );

    aux = fr.toU64(a3);
    r6 = fr.fromU64( aux & 0xFFFFFFFF );
    r7 = fr.fromU64( aux >> 32 );
}

/* Scalar to/from fec conversion */

void fec2scalar (RawFec &fec, const RawFec::Element &fe, mpz_class &s)
{
    s.set_str(fec.toString(fe,16),16);
}
void scalar2fec (RawFec &fec, RawFec::Element &fe, const mpz_class &s)
{
    fec.fromMpz(fe, s.get_mpz_t());
}