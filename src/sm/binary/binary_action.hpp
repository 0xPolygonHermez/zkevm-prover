#ifndef BINARY_ACTION_HPP
#define BINARY_ACTION_HPP

#include <gmpxx.h>

class BinaryAction
{
public:
    mpz_class a;
    mpz_class b;
    mpz_class c;
    uint64_t opcode;
};

#endif