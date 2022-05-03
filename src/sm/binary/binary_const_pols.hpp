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
    uint8_t * P_OPCODE;
    uint8_t * P_A;
    uint8_t * P_B;
    uint8_t * P_CIN;
    uint8_t * P_LAST;
    uint8_t * P_C;
    uint8_t * P_COUT;
    uint8_t * RESET;
    uint32_t * FACTOR[REGISTERS_NUM];

private:
    // Internal attributes
    uint64_t length;
    uint64_t totalSize;
    uint8_t * pAddress;

public:
    BinaryConstPols(const Config &config) : config(config)
    {
        pAddress = NULL;
    }

    void alloc (uint64_t len, json &j);
    void dealloc (void);
};

#endif

