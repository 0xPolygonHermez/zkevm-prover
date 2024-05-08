#ifndef ARITH_EQUATION_HPP
#define ARITH_EQUATION_HPP

#include <string>

using namespace std;

const uint64_t ARITH_BASE = 1;
const uint64_t ARITH_ECADD_DIFFERENT = 2;
const uint64_t ARITH_ECADD_SAME = 3;
const uint64_t ARITH_BN254_MULFP2 = 4;
const uint64_t ARITH_BN254_ADDFP2 = 5;
const uint64_t ARITH_BN254_SUBFP2 = 6;
const uint64_t ARITH_MOD = 7;
const uint64_t ARITH_384_MOD = 8;
const uint64_t ARITH_BLS12381_MULFP2 = 9;
const uint64_t ARITH_BLS12381_ADDFP2 = 10;
const uint64_t ARITH_BLS12381_SUBFP2 = 11;
const uint64_t ARITH_256TO384 = 12;

string arith2string (uint64_t arithEquation);

#endif