#ifndef ECRECOVER_HPP
#define ECRECOVER_HPP

#include <gmpxx.h>

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

ECRecoverResult ECRecover (mpz_class &signature, mpz_class &r, mpz_class &s, mpz_class &v, bool bPrecompiled, mpz_class &address);

#endif