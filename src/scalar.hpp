#ifndef SCALAR_HPP
#define SCALAR_HPP

#include <gmpxx.h>
#include "ffiasm/fr.hpp"
#include "context.hpp"

/* Converts a field element into a signed 64b integer */
/* Precondition: p - 2^63 <= fe < 2^63 */
int64_t fe2n (Context &ctx, RawFr::Element &fe);

/* Converts a field element into an unsigned 64b integer */
/* Precondition: 0 <= fe < 2^64 */
uint64_t fe2u64 (RawFr &fr, RawFr::Element &fe);

/* Using mpz_t as scalar*/
void fea2scalar (RawFr &fr, mpz_t &scalar, RawFr::Element &fe0, uint64_t fe1, uint64_t fe2, uint64_t fe3);
void scalar2fea (RawFr &fr, const mpz_t scalar, RawFr::Element &fe0, RawFr::Element &fe1, RawFr::Element &fe2, RawFr::Element &fe3);

/* Using mpz_class as scalar */
void fe2scalar  (RawFr &fr, mpz_class &scalar, RawFr::Element &fe);
void fea2scalar (RawFr &fr, mpz_class &scalar, RawFr::Element &fe0, uint64_t fe1, uint64_t fe2, uint64_t fe3);
void fea2scalar (RawFr &fr, mpz_class &scalar, RawFr::Element &fe0, RawFr::Element fe1, RawFr::Element fe2, RawFr::Element fe3);
void scalar2fe  (RawFr &fr, mpz_class &scalar, RawFr::Element &fe);
void scalar2fea (RawFr &fr, mpz_class &scalar, RawFr::Element &fe0, RawFr::Element &fe1, RawFr::Element &fe2, RawFr::Element &fe3);

/* Normalized strings */
string RemoveOxIfPresent      (string s);
string PrependZeros           (string s, uint64_t n);
string NormalizeTo0xNFormat   (string s, uint64_t n);
string NormalizeToNFormat     (string s, uint64_t n);

void GetPrimeNumber (RawFr &fr, mpz_class &p);

string keccak256 (uint8_t *pData, uint64_t &dataSize);

// TODO: FE cannot be passed as a const reference because fr.xxx() methods expect non-const arguments

#endif