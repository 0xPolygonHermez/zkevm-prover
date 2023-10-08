#ifndef ECRECOVER_HPP
#define ECRECOVER_HPP

#include <gmpxx.h>
#include "ffiasm/fec.hpp"
#include "scalar.hpp"

typedef enum
{
    ECR_NO_ERROR = 0,
    ECR_R_IS_ZERO = 1,
    ECR_R_IS_TOO_BIG = 2,
    ECR_S_IS_ZERO = 3,
    ECR_S_IS_TOO_BIG = 4,
    ECR_V_INVALID = 5,
    ECR_NO_SQRT_Y = 6,
    ECR_NO_SQRT_BUT_IT_HAS_SOLUTION = 100
} ECRecoverResult;

ECRecoverResult ECRecover(mpz_class &signature, mpz_class &r, mpz_class &s, mpz_class &v, bool bPrecompiled, mpz_class &address);
int ECRecoverPrecalc(mpz_class &signature, mpz_class &r, mpz_class &s, mpz_class &v, bool bPrecompiled, RawFec::Element* buffer, int nthreads = 16);

// We use that p = 3 mod 4 => r = a^((p+1)/4) is a square root of a
// https://www.rieselprime.de/ziki/Modular_square_root
// n = p+1/4
inline void sqrtF3mod4(mpz_class& r, const mpz_class &a){
    mpz_class auxa = a;
    mpz_class n("0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c");
    mpz_class p("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    mpz_powm(r.get_mpz_t(), a.get_mpz_t(), n.get_mpz_t(), p.get_mpz_t());
    if ((r * r) % p != auxa)
    {
        r = ScalarMask256;
    }
}

#endif