#ifndef COUNTERS_HPP
#define COUNTERS_HPP

#include <stdint.h>

using namespace std;

class Counters
{
public:
    uint64_t ecRecover;
    uint64_t hashPoseidon;
    uint64_t hashKeccak;
    uint64_t arith;
    Counters():
        ecRecover(0),
        hashPoseidon(0),
        hashKeccak(0),
        arith(0)
         {};
};

#endif