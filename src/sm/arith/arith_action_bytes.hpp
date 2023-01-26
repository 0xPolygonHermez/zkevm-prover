#ifndef ARITH_ACTION_BYTES_HPP
#define ARITH_ACTION_BYTES_HPP

#include <cstdint>

class ArithActionBytes
{
public:
    // Original input data
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

    // These arrays will contain 16-bit numbers, except the last (15) one, which can be up to 20-bits long
    // For this reason, we use 64-bit numbers, to have room for all possible values
    uint64_t _x1[16];
    uint64_t _y1[16];
    uint64_t _x2[16];
    uint64_t _y2[16];
    uint64_t _x3[16];
    uint64_t _y3[16];
    uint64_t _selEq0[16];
    uint64_t _selEq1[16];
    uint64_t _selEq2[16];
    uint64_t _selEq3[16];
    uint64_t _s[16];
    uint64_t _q0[16];
    uint64_t _q1[16];
    uint64_t _q2[16];
};

#endif