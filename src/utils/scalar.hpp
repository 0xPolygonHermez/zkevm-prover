#ifndef SCALAR_HPP
#define SCALAR_HPP

#include <gmpxx.h>
#include <string>
#include "goldilocks_base_field.hpp"
#include "ffiasm/fec.hpp"

using namespace std;

/* Global scalar variables */
extern mpz_class Mask4;
extern mpz_class Mask8;
extern mpz_class Mask16;
extern mpz_class Mask20;
extern mpz_class Mask32;
extern mpz_class Mask64;
extern mpz_class Mask256;
extern mpz_class TwoTo8;
extern mpz_class TwoTo16;
extern mpz_class TwoTo18;
extern mpz_class TwoTo32;
extern mpz_class TwoTo64;
extern mpz_class TwoTo128;
extern mpz_class TwoTo192;
extern mpz_class TwoTo256;
extern mpz_class TwoTo255;
extern mpz_class TwoTo258;
extern mpz_class Zero;
extern mpz_class One;
extern mpz_class GoldilocksPrime;

/* Scalar to/from field element conversion */
void fe2scalar  (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element &fe);
void scalar2fe  (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element &fe);

/* Scalar to/from field element array conversion */
void fea2scalar (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element (&fea)[8]);
void fea2scalar (Goldilocks &fr, mpz_class &scalar, const Goldilocks::Element (&fea)[4]);
void fea2scalar (Goldilocks &fr, mpz_class &scalar, Goldilocks::Element &fe0, Goldilocks::Element &fe1, Goldilocks::Element &fe2, Goldilocks::Element &fe3, Goldilocks::Element &fe4, Goldilocks::Element &fe5, Goldilocks::Element &fe6, Goldilocks::Element &fe7);
void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element (&fea)[8]);
void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element (&fea)[4]);
void scalar2fea (Goldilocks &fr, const mpz_class &scalar, Goldilocks::Element &fe0, Goldilocks::Element &fe1, Goldilocks::Element &fe2, Goldilocks::Element &fe3, Goldilocks::Element &fe4, Goldilocks::Element &fe5, Goldilocks::Element &fe6, Goldilocks::Element &fe7);

/* Scalar to/from a Sparse Merkle Tree key conversion, interleaving bits */
void scalar2key (Goldilocks &fr, mpz_class &s, Goldilocks::Element (&key)[4]);

/* Hexa string to/from field element (array) conversion */
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

/* Keccak */
void   keccak256 (const uint8_t *pInputData, uint64_t inputDataSize, uint8_t *pOutputData, uint64_t outputDataSize);
string keccak256 (const uint8_t *pInputData, uint64_t inputDataSize);
string keccak256 (const vector<uint8_t> &input);
string keccak256 (const string &inputString);

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
uint64_t string2bv (const string &os, vector<uint8_t> &vData);
void     ba2string (string &s, const uint8_t *pData, uint64_t dataSize);
string   ba2string (const uint8_t *pData, uint64_t dataSize);
void     ba2string (const string &baString, string &textString);
string   ba2string (const string &baString);

/* Byte array of exactly 2 bytes conversion */
void ba2u16(const uint8_t *pData, uint16_t &n);
void ba2scalar(const uint8_t *pData, uint64_t dataSize, mpz_class &s);

/* Scalar to byte array conversion (up to dataSize bytes) */
void scalar2ba(uint8_t *pData, uint64_t &dataSize, mpz_class s);
void scalar2ba16(uint64_t *pData, uint64_t &dataSize, mpz_class s);
void scalar2bytes(mpz_class &s, uint8_t (&bytes)[32]);

/* Converts a scalar to a vector of bits of the scalar, with value 1 or 0; bits[0] is least significant bit */
void scalar2bits(mpz_class s, vector<uint8_t> &bits);

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

#endif