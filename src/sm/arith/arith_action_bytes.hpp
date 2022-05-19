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

    ArithActionBytes()
    {
        selEq0 = 0;
        selEq1 = 0;
        selEq2 = 0;
        selEq3 = 0;
        memset(_x1, 0, sizeof(_x1));
        memset(_y1, 0, sizeof(_y1));
        memset(_x2, 0, sizeof(_x2));
        memset(_y2, 0, sizeof(_y2));
        memset(_x3, 0, sizeof(_x3));
        memset(_x3, 0, sizeof(_y3));
        memset(_selEq0, 0, sizeof(_selEq0));
        memset(_selEq1, 0, sizeof(_selEq1));
        memset(_selEq2, 0, sizeof(_selEq2));
        memset(_selEq3, 0, sizeof(_selEq3));
        memset(_s, 0, sizeof(_s));
        memset(_q0, 0, sizeof(_q0));
        memset(_q1, 0, sizeof(_q1));
        memset(_q2, 0, sizeof(_q2));
    }
};

#endif