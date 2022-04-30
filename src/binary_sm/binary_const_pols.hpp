#ifndef BINARY_CONST_POLS_HPP
#define BINARY_CONST_POLS_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include "config.hpp"
#include "ff/ff.hpp"
#include "binary_defines.hpp"

class BinaryConstPols
{
private:
    const Config &config;

public:
    uint64_t * P_OPCODE;
    uint64_t * P_A;
    uint64_t * P_B;
    uint64_t * P_CIN;
    uint64_t * P_LAST;
    uint64_t * P_C;
    uint64_t * P_COUT;
    uint64_t * RESET;
    uint64_t * FACTOR[REGISTERS_NUM];

private:
    // Internal attributes
    uint64_t nCommitments;
    uint64_t length;
    uint64_t polSize;
    uint64_t numberOfPols;
    uint64_t totalSize;
    uint64_t * pAddress;

public:
    BinaryConstPols(const Config &config) : config(config)
    {
        pAddress = NULL;
    }

    void alloc (uint64_t len, json &j);
    void dealloc (void);

    uint64_t getPolOrder (json &j, const char * pPolName);
};

#endif

