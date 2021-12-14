#ifndef STARK_STRUCT_HPP
#define STARK_STRUCK_HPP

#include <iostream>

class StarkConfig
{
public:
    uint64_t nBits;
    uint64_t nQueries;
    StarkConfig(uint64_t b, uint64_t q) : nBits(b), nQueries(q) {;};
};

extern StarkConfig starkStruct[2];

#endif