#ifndef ECRECOVER_HPP
#define ECRECOVER_HPP

#include <gmpxx.h>

bool ECRecover (mpz_class &hash, mpz_class &r, mpz_class &s, mpz_class &v);

#endif