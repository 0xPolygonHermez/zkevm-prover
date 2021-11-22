#ifndef SCALAR_HPP
#define SCALAR_HPP

#include "ffiasm/fr.hpp"
#include "context.hpp"

int64_t fe2n (RawFr &fr, RawFr::Element &fe);
void fea2scalar (RawFr &fr, mpz_t &scalar, RawFr::Element fe0, uint64_t fe1, uint64_t fe2, uint64_t fe3);
void scalar2fea (RawFr &fr, mpz_t scalar, RawFr::Element &fe0, RawFr::Element &fe1, RawFr::Element &fe2, RawFr::Element &fe3);

#endif