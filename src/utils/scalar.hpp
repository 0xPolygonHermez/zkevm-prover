#ifndef SCALAR_HPP
#define SCALAR_HPP

#include <gmpxx.h>
#include <string>
#include "goldilocks/goldilocks_base_field.hpp"
#include "ffiasm/fec.hpp"

using namespace std;

/* Using mpz_t as scalar*/
void fea2scalar (Goldilocks &fr, mpz_class &scalar, Goldilocks::Element &fe0, Goldilocks::Element &fe1, Goldilocks::Element &fe2, Goldilocks::Element &fe3, Goldilocks::Element &fe4, Goldilocks::Element &fe5, Goldilocks::Element &fe6, Goldilocks::Element &fe7);
void fea2scalar (Goldilocks &fr, mpz_class &scalar, Goldilocks::Element &fe0, uint32_t &fe1, uint32_t &fe2, uint32_t &fe3, uint32_t &fe4, uint32_t &fe5, uint32_t &fe6, uint32_t &fe7);
void fea2scalar (Goldilocks &fr, mpz_class &scalar, uint32_t &fe0, uint32_t &fe1, uint32_t &fe2, uint32_t &fe3, uint32_t &fe4, uint32_t &fe5, uint32_t &fe6, uint32_t &fe7);
void fea2scalar (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element (&fea)[4]);
void fea2scalar (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element (&fea)[8]);

/* Using mpz_class as scalar */
void fe2scalar  (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element &fe);
void scalar2fe  (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element &fe);
void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element &fe0, Goldilocks::Element &fe1, Goldilocks::Element &fe2, Goldilocks::Element &fe3, Goldilocks::Element &fe4, Goldilocks::Element &fe5, Goldilocks::Element &fe6, Goldilocks::Element &fe7);
void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element (&fea)[8]);
void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element (&fea)[4]);

/* Convert a scalar to a key, interleaving bits */
void scalar2key (Goldilocks &fr, mpz_class &s, Goldilocks::Element (&key)[4]);

// Converts an hexa string to a field element
void string2fe  (Goldilocks &fr, const string &s, Goldilocks::Element &fe);
string fea2string (Goldilocks &fr, const Goldilocks::Element(&fea)[4]);
string fea2string (Goldilocks &fr, const Goldilocks::Element &fea0, const Goldilocks::Element &fea1, const Goldilocks::Element &fea2, const Goldilocks::Element &fea3);

/* Normalized strings */
string Remove0xIfPresent      (const string &s);
string Add0xIfMissing         (string s);
string PrependZeros           (string s, uint64_t n);
string NormalizeTo0xNFormat   (string s, uint64_t n);
string NormalizeToNFormat     (string s, uint64_t n);
string stringToLower          (const string &s);

// Keccak
void   keccak256 (const uint8_t *pInputData, uint64_t inputDataSize, uint8_t *pOutputData, uint64_t outputDataSize);
string keccak256 (const uint8_t *pInputData, uint64_t inputDataSize);
string keccak256 (const vector<uint8_t> &input);
void   keccak256 (const string &inputString, uint8_t *pOutputData, uint64_t outputDataSize);
string keccak256 (const string &inputString);

/* Converts a string to a byte array
   s must be even sized, and must not include the leading "0x"
   pData buffer must be big enough to store converted data */
uint64_t string2ba (const string &s, uint8_t *pData, uint64_t &dataSize);
void string2ba (const string &textString, string &baString);
string string2ba(const string &textString);

void ba2string (string &s, const uint8_t *pData, uint64_t dataSize);
string ba2string (const uint8_t *pData, uint64_t dataSize);
void ba2string (const string &baString, string &textString);
string ba2string (const string &baString);

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
void scalar2bytes(mpz_class &s, uint8_t (&bytes)[32]);

// Converts a scalar to a vector of bits of the scalar, with value 1 or 0; bits[0] is least significant bit
void scalar2bits(mpz_class s, vector<uint8_t> &bits);

// Converts a byte to an array of bits, with value 1 or 0; bits[0] is the least significant bit
void byte2bits(uint8_t byte, uint8_t *pBits);

// Converts 8 bits to 1 byte
void bits2byte(const uint8_t *pBits, uint8_t &byte);

// Converts 8 fe to 4 fe
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

// Converts 4 fe to 8 fe
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

extern mpz_class Mask8;
extern mpz_class Mask256;
extern mpz_class TwoTo16;
extern mpz_class TwoTo18;
extern mpz_class TwoTo64;
extern mpz_class TwoTo128;
extern mpz_class TwoTo192;
extern mpz_class TwoTo256;
extern mpz_class TwoTo255;
extern mpz_class TwoTo258;
extern mpz_class One;

void fec2scalar(RawFec &fec, const RawFec::Element &fe, mpz_class &s);
void scalar2fec(RawFec &fec, RawFec::Element &fe, const mpz_class &s);

#endif