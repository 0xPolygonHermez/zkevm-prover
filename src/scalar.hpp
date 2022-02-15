#ifndef SCALAR_HPP
#define SCALAR_HPP

#include <gmpxx.h>
#include <string>
#include "ffiasm/fr.hpp"

using namespace std;

/* Converts a field element into a signed 64b integer */
/* Precondition: p - 2^63 <= fe < 2^63 */
int64_t fe2n (RawFr &fr, const mpz_class &prime, const RawFr::Element &fe);

/* Converts a field element into an unsigned 64b integer */
/* Precondition: 0 <= fe < 2^64 */
uint64_t fe2u64 (RawFr &fr, const RawFr::Element &fe);

/* Converts any polynomial type to a field element */
void u82fe  (RawFr &fr, RawFr::Element &fe, uint8_t  n);
void s82fe  (RawFr &fr, RawFr::Element &fe, int8_t   n);
void u162fe (RawFr &fr, RawFr::Element &fe, uint16_t n);
void s162fe (RawFr &fr, RawFr::Element &fe, int16_t  n);
void u322fe (RawFr &fr, RawFr::Element &fe, uint32_t n);
void s322fe (RawFr &fr, RawFr::Element &fe, int32_t  n);
void u642fe (RawFr &fr, RawFr::Element &fe, uint64_t n);
void s642fe (RawFr &fr, RawFr::Element &fe, int64_t  n);

/* Using mpz_t as scalar*/
void fea2scalar (RawFr &fr, mpz_t &scalar, const RawFr::Element &fe0, uint64_t fe1, uint64_t fe2, uint64_t fe3);
void scalar2fea (RawFr &fr, const mpz_t scalar, RawFr::Element &fe0, RawFr::Element &fe1, RawFr::Element &fe2, RawFr::Element &fe3);

/* Using mpz_class as scalar */
void fe2scalar  (RawFr &fr, mpz_class &scalar, const RawFr::Element &fe);
void fea2scalar (RawFr &fr, mpz_class &scalar, const RawFr::Element &fe0, uint64_t fe1, uint64_t fe2, uint64_t fe3);
void fea2scalar (RawFr &fr, mpz_class &scalar, const RawFr::Element &fe0, const RawFr::Element fe1, const RawFr::Element fe2, const RawFr::Element fe3);
void scalar2fe  (RawFr &fr, const mpz_class &scalar, RawFr::Element &fe);
void scalar2fea (RawFr &fr, const mpz_class &scalar, RawFr::Element &fe0, RawFr::Element &fe1, RawFr::Element &fe2, RawFr::Element &fe3);

// Converts an hexa string to a field element
void string2fe  (RawFr &fr, const string &s, RawFr::Element &fe);

/* Normalized strings */
string Remove0xIfPresent      (const string &s);
string Add0xIfMissing         (string s);
string PrependZeros           (string s, uint64_t n);
string NormalizeTo0xNFormat   (string s, uint64_t n);
string NormalizeToNFormat     (string s, uint64_t n);

// Gets the prime number of the finite field
void GetPrimeNumber (RawFr &fr, mpz_class &p);

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

// Converts a scalar to a vector of bits of the scalar, with value 1 or 0; bits[0] is least significant bit
void scalar2bits(mpz_class s, vector<uint8_t> &bits);

// Converts a byte to an array of bits, with value 1 or 0; bits[0] is the least significant bit
void byte2bits(uint8_t byte, uint8_t *pBits);

// Converts 8 bits to 1 byte
void bits2byte(uint8_t *pBits, uint8_t &byte);

#endif