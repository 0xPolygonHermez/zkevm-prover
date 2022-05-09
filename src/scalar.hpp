#ifndef SCALAR_HPP
#define SCALAR_HPP

#include <gmpxx.h>
#include <string>
#include "ff/ff.hpp"

using namespace std;

/* Converts a field element into a signed 64b integer */
/* Precondition: p - 2^31 <= fe < 2^31 */
int64_t fe2n (FiniteField &fr, const FieldElement &fe);

/* Converts a field element into an unsigned 64b integer */
/* Precondition: 0 <= fe < 2^64 */
uint64_t fe2u64 (FiniteField &fr, const FieldElement &fe);

/* Converts any polynomial type to a field element */
void u82fe  (FiniteField &fr, FieldElement &fe, uint8_t  n);
void s82fe  (FiniteField &fr, FieldElement &fe, int8_t   n);
void u162fe (FiniteField &fr, FieldElement &fe, uint16_t n);
void s162fe (FiniteField &fr, FieldElement &fe, int16_t  n);
void u322fe (FiniteField &fr, FieldElement &fe, uint32_t n);
void s322fe (FiniteField &fr, FieldElement &fe, int32_t  n);
void u642fe (FiniteField &fr, FieldElement &fe, uint64_t n);
void s642fe (FiniteField &fr, FieldElement &fe, int64_t  n);

/* Using mpz_t as scalar*/
void fea2scalar (FiniteField &fr, mpz_class &scalar, FieldElement &fe0, FieldElement &fe1, FieldElement &fe2, FieldElement &fe3, FieldElement &fe4, FieldElement &fe5, FieldElement &fe6, FieldElement &fe7);
void fea2scalar (FiniteField &fr, mpz_class &scalar, FieldElement &fe0, uint32_t &fe1, uint32_t &fe2, uint32_t &fe3, uint32_t &fe4, uint32_t &fe5, uint32_t &fe6, uint32_t &fe7);
void fea2scalar (FiniteField &fr, mpz_class &scalar, uint32_t &fe0, uint32_t &fe1, uint32_t &fe2, uint32_t &fe3, uint32_t &fe4, uint32_t &fe5, uint32_t &fe6, uint32_t &fe7);
void fea2scalar (FiniteField &fr, mpz_class &scalar, const FieldElement (&fea)[4]);
void fea2scalar (FiniteField &fr, mpz_class &scalar, const FieldElement (&fea)[8]);

/* Using mpz_class as scalar */
void fe2scalar  (FiniteField &fr, mpz_class &scalar, const FieldElement &fe);
void scalar2fe  (FiniteField &fr, const mpz_class &scalar, FieldElement &fe);
void scalar2fea (FiniteField &fr, const mpz_class &scalar, FieldElement &fe0, FieldElement &fe1, FieldElement &fe2, FieldElement &fe3, FieldElement &fe4, FieldElement &fe5, FieldElement &fe6, FieldElement &fe7);
void scalar2fea (FiniteField &fr, const mpz_class &scalar, FieldElement (&fea)[8]);
void scalar2fea (FiniteField &fr, const mpz_class &scalar, FieldElement (&fea)[4]);

/* Convert a scalar to a key, interleaving bits */
void scalar2key (FiniteField &fr, mpz_class &s, FieldElement (&key)[4]);

// Converts an hexa string to a field element
void string2fe  (FiniteField &fr, const string &s, FieldElement &fe);
string fea2string (FiniteField &fr, const FieldElement(&fea)[4]);

/* Normalized strings */
string Remove0xIfPresent      (const string &s);
string Add0xIfMissing         (string s);
string PrependZeros           (string s, uint64_t n);
string NormalizeTo0xNFormat   (string s, uint64_t n);
string NormalizeToNFormat     (string s, uint64_t n);

// Keccak
void   keccak256 (const uint8_t *pInputData, uint64_t inputDataSize, uint8_t *pOutputData, uint64_t outputDataSize);
string keccak256 (uint8_t *pInputData, uint64_t inputDataSize);
void   keccak256 (string &inputString, uint8_t *pOutputData, uint64_t outputDataSize);
string keccak256 (string &inputString);

/* Converts a string to a byte array 
   s must be even sized, and must not include the leading "0x"
   pData buffer must be big enough to store converted data */
uint64_t string2ba (const string &s, uint8_t *pData, uint64_t &dataSize);

void ba2string (string &s, const uint8_t *pData, uint64_t dataSize);

uint8_t char2byte (char c);
char byte2char (uint8_t b);
string byte2string(uint8_t b);

// Converta a byte array of exactly 2 bytes to unsigned int 16 bit
void ba2u16(const uint8_t *pData, uint16_t &n);

// Converts a byte array of dataSize bytes to scalar
void ba2scalar(const uint8_t *pData, uint64_t dataSize, mpz_class &s);

// Converts a scalar to a byte array of up to dataSize bytes
void scalar2ba(uint8_t *pData, uint64_t &dataSize, mpz_class s);
void scalar2ba16(uint64_t *pData, uint64_t &dataSize, mpz_class s);

// Converts a scalar to a vector of bits of the scalar, with value 1 or 0; bits[0] is least significant bit
void scalar2bits(mpz_class s, vector<uint8_t> &bits);

// Converts a byte to an array of bits, with value 1 or 0; bits[0] is the least significant bit
void byte2bits(uint8_t byte, uint8_t *pBits);

// Converts 8 bits to 1 byte
void bits2byte(const uint8_t *pBits, uint8_t &byte);

// Converts 8 fe to 4 fe
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
              FieldElement &r3 );

// Converts 4 fe to 8 fe
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
              FieldElement &r7 );

extern mpz_class Mask8;
extern mpz_class Mask256;
extern mpz_class twoTo64;
extern mpz_class twoTo128;
extern mpz_class twoTo192;
extern mpz_class twoTo256;
extern mpz_class twoTo255;

#endif