#ifndef COUNTERS_HPP
#define COUNTERS_HPP

#include <stdint.h>

using namespace std;

class Counters
{
public:
    uint64_t arith;
    uint64_t binary;
    uint64_t memAlign;
    uint64_t keccakF;
    uint64_t poseidonG;
    uint64_t paddingPG;
    uint64_t steps;
    Counters():
        arith(0),
        binary(0),
        memAlign(0),
        keccakF(0),
        poseidonG(0),
        paddingPG(0),
        steps(0) {};
};

#endif