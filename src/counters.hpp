#ifndef COUNTERS_HPP
#define COUNTERS_HPP

#include <stdint.h>

using namespace std;

class Counters
{
public:
    uint64_t arith;
    Counters(): arith(0) {};
};

#endif