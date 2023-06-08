#ifndef REGISTERS_C_HPP_fork_5
#define REGISTERS_C_HPP_fork_5

#include <gmpxx.h>

namespace fork_5
{

class RegistersC
{
public:
    mpz_class A, B, C, D, E;
    mpz_class SR;
    mpz_class CNT_KECCAK_F;
    mpz_class HASHPOS;
};

}

#endif