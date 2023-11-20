#ifndef ARITH_ACTION_HPP
#define ARITH_ACTION_HPP

#include <gmpxx.h>

class ArithAction
{
public:
    mpz_class x1;
    mpz_class y1;
    mpz_class x2;
    mpz_class y2;
    mpz_class x3;
    mpz_class y3;
    uint64_t selEq0;
    uint64_t selEq1;
    uint64_t selEq2;
    uint64_t selEq3;
    uint64_t selEq4;
    uint64_t selEq5;
    uint64_t selEq6;

    ArithAction() : selEq0(0), selEq1(0), selEq2(0), selEq3(0), selEq4(0), selEq5(0), selEq6(0) {};
};

#endif