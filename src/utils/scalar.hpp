#ifndef SCALAR_HPP
#define SCALAR_HPP

#include <gmpxx.h>
#include <string>
#include <vector>
#include "goldilocks_base_field.hpp"
#include "ffiasm/fec.hpp"
#include "exit_process.hpp"
#include "zklog.hpp"
#include "zkglobals.hpp"

using namespace std;

/* Global scalar variables */
extern mpz_class ScalarMask4;
extern mpz_class ScalarMask8;
extern mpz_class ScalarMask16;
extern mpz_class ScalarMask20;
extern mpz_class ScalarMask32;
extern mpz_class ScalarMask64;
extern mpz_class ScalarMask160;
extern mpz_class ScalarMask256;
extern mpz_class ScalarTwoTo8;
extern mpz_class ScalarTwoTo16;
extern mpz_class ScalarTwoTo18;
extern mpz_class ScalarTwoTo32;
extern mpz_class ScalarTwoTo64;
extern mpz_class ScalarTwoTo128;
extern mpz_class ScalarTwoTo192;
extern mpz_class ScalarTwoTo256;
extern mpz_class ScalarTwoTo255;
extern mpz_class ScalarTwoTo258;
extern mpz_class ScalarZero;
extern mpz_class ScalarOne;
extern mpz_class ScalarGoldilocksPrime;

/* Scalar to/from field element conversion */

inline void fe2scalar  (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element &fe)
{
    scalar = fr.toU64(fe);
}

inline void scalar2fe  (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element &fe)
{
    if ( !scalar.fits_ulong_p() )
    {
        zklog.error("scalar2fe() found scalar out of u64 range:" + scalar.get_str(16));
        exitProcess();
    }
    fe = fr.fromU64(scalar.get_ui());
}

/* Scalar to/from field element array conversion */

inline void fea2scalar (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element (&fea)[4])
{
    scalar = fr.toU64(fea[3]);
    scalar <<= 64;
    scalar += fr.toU64(fea[2]);
    scalar <<= 64;
    scalar += fr.toU64(fea[1]);
    scalar <<= 64;
    scalar += fr.toU64(fea[0]);
}

inline bool fea2scalar (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element &fe0, const Goldilocks::Element &fe1, const Goldilocks::Element &fe2, const Goldilocks::Element &fe3, const Goldilocks::Element &fe4, const Goldilocks::Element &fe5, const Goldilocks::Element &fe6, const Goldilocks::Element &fe7)
{
    // Add field element 7
    uint64_t auxH = fr.toU64(fe7);
    if (auxH >= 0x100000000)
    {
        zklog.error("fea2scalar() found element 7 has a too high value=" + fr.toString(fe7, 16));
        return false;
    }

    // Add field element 6
    uint64_t auxL = fr.toU64(fe6);
    if (auxL >= 0x100000000)
    {
        zklog.error("fea2scalar() found element 6 has a too high value=" + fr.toString(fe6, 16));
        return false;
    }

    scalar = (auxH<<32) + auxL;
    scalar <<= 64;

    // Add field element 5
    auxH = fr.toU64(fe5);
    if (auxH >= 0x100000000)
    {
        zklog.error("fea2scalar() found element 5 has a too high value=" + fr.toString(fe5, 16));
        return false;
    }

    // Add field element 4
    auxL = fr.toU64(fe4);
    if (auxL >= 0x100000000)
    {
        zklog.error("fea2scalar() found element 4 has a too high value=" + fr.toString(fe4, 16));
        return false;
    }
    
    scalar += (auxH<<32) + auxL;
    scalar <<= 64;

    // Add field element 3
    auxH = fr.toU64(fe3);
    if (auxH >= 0x100000000)
    {
        zklog.error("fea2scalar() found element 3 has a too high value=" + fr.toString(fe3, 16));
        return false;
    }

    // Add field element 2
    auxL = fr.toU64(fe2);
    if (auxL >= 0x100000000)
    {
        zklog.error("fea2scalar() found element 2 has a too high value=" + fr.toString(fe2, 16));
        return false;
    }
    
    scalar += (auxH<<32) + auxL;
    scalar <<= 64;

    // Add field element 1
    auxH = fr.toU64(fe1);
    if (auxH >= 0x100000000)
    {
        zklog.error("fea2scalar() found element 1 has a too high value=" + fr.toString(fe1, 16));
        return false;
    }

    // Add field element 0
    auxL = fr.toU64(fe0);
    if (auxL >= 0x100000000)
    {
        zklog.error("fea2scalar() found element 0 has a too high value=" + fr.toString(fe0, 16));
        return false;
    }
    
    scalar += (auxH<<32) + auxL;
    return true;
}

inline bool fea2scalar (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element (&fea)[8])
{
   return fea2scalar(fr, scalar, fea[0], fea[1], fea[2], fea[3], fea[4], fea[5], fea[6], fea[7]);
}

inline void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element (&fea)[4])
{
    mpz_class aux = scalar & ScalarMask64;

    if (aux >= ScalarGoldilocksPrime)
    {
        zklog.error("scalar2fea() found value higher than prime: " + aux.get_str(16));
        exitProcess();
    }

    fea[0] = fr.fromU64(aux.get_ui());
    aux = scalar>>64 & ScalarMask64;

    if (aux >= ScalarGoldilocksPrime)
    {
        zklog.error("scalar2fea() found value higher than prime: " + aux.get_str(16));
        exitProcess();
    }

    fea[1] = fr.fromU64(aux.get_ui());
    aux = scalar>>128 & ScalarMask64;

    if (aux >= ScalarGoldilocksPrime)
    {
        zklog.error("scalar2fea() found value higher than prime: " + aux.get_str(16));
        exitProcess();
    }

    fea[2] = fr.fromU64(aux.get_ui());
    aux = scalar>>192 & ScalarMask64;

    if (aux >= ScalarGoldilocksPrime)
    {
        zklog.error("scalar2fea() found value higher than prime: " + aux.get_str(16));
        exitProcess();
    }

    fea[3] = fr.fromU64(aux.get_ui());
}

inline void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element &fe0, Goldilocks::Element &fe1, Goldilocks::Element &fe2, Goldilocks::Element &fe3, Goldilocks::Element &fe4, Goldilocks::Element &fe5, Goldilocks::Element &fe6, Goldilocks::Element &fe7)
{
    mpz_class aux;
    aux = scalar & ScalarMask32;
    fe0 = fr.fromU64(aux.get_ui());
    aux = scalar>>32 & ScalarMask32;
    fe1 = fr.fromU64(aux.get_ui());
    aux = scalar>>64 & ScalarMask32;
    fe2 = fr.fromU64(aux.get_ui());
    aux = scalar>>96 & ScalarMask32;
    fe3 = fr.fromU64(aux.get_ui());
    aux = scalar>>128 & ScalarMask32;
    fe4 = fr.fromU64(aux.get_ui());
    aux = scalar>>160 & ScalarMask32;
    fe5 = fr.fromU64(aux.get_ui());
    aux = scalar>>192 & ScalarMask32;
    fe6 = fr.fromU64(aux.get_ui());
    aux = scalar>>224 & ScalarMask32;
    fe7 = fr.fromU64(aux.get_ui());
}

inline void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element (&fea)[8])
{
    scalar2fea(fr, scalar, fea[0], fea[1], fea[2], fea[3], fea[4], fea[5], fea[6], fea[7]);
}

/* Scalar to/from a Sparse Merkle Tree key conversion, interleaving bits */
void scalar2key (Goldilocks &fr, mpz_class &s, Goldilocks::Element (&key)[4]);

/* Hexa string to/from field element (array) conversion */
void string2fe  (Goldilocks &fr, const string &s, Goldilocks::Element &fe);
void string2fea (Goldilocks &fr, const string& os, vector<Goldilocks::Element> &fea);
void string2fea (Goldilocks &fr, const string& os, Goldilocks::Element (&fea)[4]);
string fea2string (Goldilocks &fr, const Goldilocks::Element(&fea)[4]);
string fea2string (Goldilocks &fr, const Goldilocks::Element &fea0, const Goldilocks::Element &fea1, const Goldilocks::Element &fea2, const Goldilocks::Element &fea3);
string fea2string (Goldilocks &fr, const Goldilocks::Element &fea0, const Goldilocks::Element &fea1, const Goldilocks::Element &fea2, const Goldilocks::Element &fea3, const Goldilocks::Element &fea4, const Goldilocks::Element &fea5, const Goldilocks::Element &fea6, const Goldilocks::Element &fea7);

/* Normalized strings */
string Remove0xIfPresent      (const string &s);
void   Remove0xIfPresentNoCopy(      string &s);
string Add0xIfMissing         (const string &s);
string PrependZeros           (const string &s, uint64_t n);
void   PrependZerosNoCopy     (      string &s, uint64_t n);
string NormalizeTo0xNFormat   (const string &s, uint64_t n);
string NormalizeToNFormat     (const string &s, uint64_t n);
string stringToLower          (const string &s);

// Check that a char is an hex character
inline bool charIsHex (char c)
{
    if ( (c >= '0') && (c <= '9') ) return true;
    if ( (c >= 'a') && (c <= 'f') ) return true;
    if ( (c >= 'A') && (c <= 'F') ) return true;
    return false;
}

// Check that the string contains only hex characters
bool stringIsHex (const string &s);

// Check that the string contains only 0x + hex characters
bool stringIs0xHex (const string &s);

/* Keccak */
void   keccak256 (const uint8_t *pInputData, uint64_t inputDataSize, uint8_t *pOutputData, uint64_t outputDataSize);
void   keccak256 (const uint8_t *pInputData, uint64_t inputDataSize, uint8_t (&hash)[32]);
void   keccak256 (const uint8_t *pInputData, uint64_t inputDataSize, mpz_class &hash);
string keccak256 (const uint8_t *pInputData, uint64_t inputDataSize);
void   keccak256 (const vector<uint8_t> &input, mpz_class &hash);

/* Byte to/from char conversion */
uint8_t char2byte (char c);
char    byte2char (uint8_t b);
string  byte2string(uint8_t b);

/* Strint to/from byte array conversion
   s must be even sized, and must not include the leading "0x"
   pData buffer must be big enough to store converted data */
uint64_t string2ba (const string &s, uint8_t *pData, uint64_t &dataSize);
void     string2ba (const string &textString, string &baString);
string   string2ba (const string &textString);
void     string2ba (const string os, vector<uint8_t> &data);

void     ba2string (string &s, const uint8_t *pData, uint64_t dataSize);
string   ba2string (const uint8_t *pData, uint64_t dataSize);
void     ba2string (const string &baString, string &textString);
string   ba2string (const string &baString);
void     ba2ba     (const string &baString, vector<uint8_t> (&baVector));
void     ba2ba     (const vector<uint8_t> (&baVector), string &baString);
void     ba2ba     (string &baString, const uint64_t ba);
uint64_t ba2ba     (const string &baString);

/* Byte array of exactly 2 bytes conversion */
void ba2u16(const uint8_t *pData, uint16_t &n);
void ba2u32(const uint8_t *pData, uint32_t &n);
void ba2scalar(const uint8_t *pData, uint64_t dataSize, mpz_class &s);


inline void ba2fea (Goldilocks &fr, const uint8_t * pData, uint64_t len, Goldilocks::Element &fe0, Goldilocks::Element &fe1, Goldilocks::Element &fe2, Goldilocks::Element &fe3, Goldilocks::Element &fe4, Goldilocks::Element &fe5, Goldilocks::Element &fe6, Goldilocks::Element &fe7)
{
    if (len == 1)
    {
        fr.fromU64(fe0, *pData);
        fe1 = fr.zero();
        fe2 = fr.zero();
        fe3 = fr.zero();
        fe4 = fr.zero();
        fe5 = fr.zero();
        fe6 = fr.zero();
        fe7 = fr.zero();
    }
    else
    {
        uint8_t data[32] = {0};
        for (uint64_t i=0; i<len; i++)
        {
            data[32-len+i] = pData[i];
        }
        uint32_t aux;
        ba2u32(data, aux);
        fr.fromU64(fe7, aux);
        ba2u32(data+4, aux);
        fr.fromU64(fe6, aux);
        ba2u32(data+8, aux);
        fr.fromU64(fe5, aux);
        ba2u32(data+12, aux);
        fr.fromU64(fe4, aux);
        ba2u32(data+16, aux);
        fr.fromU64(fe3, aux);
        ba2u32(data+20, aux);
        fr.fromU64(fe2, aux);
        ba2u32(data+24, aux);
        fr.fromU64(fe1, aux);
        ba2u32(data+28, aux);
        fr.fromU64(fe0, aux);
    }
}

/* Scalar to byte array conversion (up to dataSize bytes) */
void scalar2ba(uint8_t *pData, uint64_t &dataSize, mpz_class s);
void scalar2ba16(uint64_t *pData, uint64_t &dataSize, mpz_class s);
string scalar2ba32(const mpz_class &s); // Returns exactly 32 bytes
void scalar2bytes(mpz_class &s, uint8_t (&bytes)[32]);
void scalar2bytesBE(mpz_class &s, uint8_t *pBytes); // pBytes must be a 32-bytes array


/* Scalar to byte array string conversion */
string scalar2ba(const mpz_class &s);

inline void ba2scalar(mpz_class &s, const string &ba)
{
    mpz_import(s.get_mpz_t(), ba.size(), 1, 1, 0, 0, ba.c_str());
}

inline void ba2scalar(mpz_class &s, const uint8_t (&hash)[32])
{
    mpz_import(s.get_mpz_t(), 32, 1, 1, 0, 0, hash);
}

/* Converts a scalar to a vector of bits of the scalar, with value 1 or 0; bits[0] is least significant bit */
void scalar2bits(mpz_class s, vector<uint8_t> &bits);

/* Converts an unsigned 32 to a vector of bits, with value 1 or 0; bits[0] is least significant bit */
void u322bits(uint32_t value, vector<uint8_t> &bits);
uint32_t bits2u32(const vector<uint8_t> &bits);

/* Converts an unsigned 64 to a vector of bits, with value 1 or 0; bits[0] is least significant bit */
void u642bits(uint64_t value, vector<uint8_t> &bits);
uint64_t bits2u64(const vector<uint8_t> &bits);

/* Byte to/from bits array conversion, with value 1 or 0; bits[0] is the least significant bit */
void byte2bits(uint8_t byte, uint8_t *pBits);
void bits2byte(const uint8_t *pBits, uint8_t &byte);

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
              Goldilocks::Element &r3 );
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
              Goldilocks::Element &r7 );

/* Scalar to/from fec conversion */
void fec2scalar(RawFec &fec, const RawFec::Element &fe, mpz_class &s);
void scalar2fec(RawFec &fec, RawFec::Element &fe, const mpz_class &s);

/* Unsigned 64 to an array of bytes.  pOutput must be 8 bytes long */
void u642bytes (uint64_t input, uint8_t * pOutput, bool bBigEndian);

/* Array of bytes to unsigned 32.  pInput must be 4 bytes long */
void bytes2u32 (const uint8_t * pInput, uint32_t &output, bool bBigEndian);

/* Array of bytes to unsigned 64.  pInput must be 8 bytes long */
void bytes2u64 (const uint8_t * pInput, uint64_t &output, bool bBigEndian);

/* unsigned64 to string*/
inline void U64toString(std::string &result, const uint64_t in1, const int radix)
{
    mpz_class aux = in1;
    result = aux.get_str(radix);
}

/* unsigned64 to string*/
inline std::string U64toString( const uint64_t in1, const int radix)
{
    mpz_class aux = in1;
    string result = aux.get_str(radix);
    return result;
}
/* Swap bytes, e.g. little to big endian, and vice-versa */
uint64_t swapBytes64 (uint64_t input);

/* Rotate */
uint32_t inline rotateRight32( uint32_t input, uint64_t bits) { return (input >> bits) | (input << (32-bits)); }
uint32_t inline rotateLeft32( uint32_t input, uint64_t bits) { return (input << bits) | (input >> (32-bits)); }
uint64_t inline rotateRight64( uint64_t input, uint64_t bits) { return (input >> bits) | (input << (64-bits)); }
uint64_t inline rotateLeft64( uint64_t input, uint64_t bits) { return (input << bits) | (input >> (64-bits)); }

bool inline feaIsZero (const Goldilocks::Element (&fea)[4])
{
    return fr.isZero(fea[0]) && fr.isZero(fea[1]) && fr.isZero(fea[2]) && fr.isZero(fea[3]);
}

#endif