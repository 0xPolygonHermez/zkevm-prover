#ifndef BINARY_POLS_HPP
#define BINARY_POLS_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include "config.hpp"
#include "ff/ff.hpp"

class BinaryPols
{
private:
    const Config &config;
public:
    uint8_t * freeInA;
    uint8_t * freeInB;
    uint8_t * freeInC;
    uint32_t * a0;
    uint32_t * a1;
    uint32_t * a2;
    uint32_t * a3;
    uint32_t * a4;
    uint32_t * a5;
    uint32_t * a6;
    uint32_t * a7;
    uint32_t * b0;
    uint32_t * b1;
    uint32_t * b2;
    uint32_t * b3;
    uint32_t * b4;
    uint32_t * b5;
    uint32_t * b6;
    uint32_t * b7;
    uint32_t * c0;
    uint32_t * c1;
    uint32_t * c2;
    uint32_t * c3;
    uint32_t * c4;
    uint32_t * c5;
    uint32_t * c6;
    uint32_t * c7;
    uint32_t * c0Temp;
    uint8_t * opcode;
    uint8_t * cIn;
    uint8_t * cOut;
    uint8_t * last;
    uint8_t * useCarry;

private:
    // Internal attributes
    uint64_t length;
    uint64_t totalSize;
    uint8_t * pAddress;

public:
    BinaryPols(const Config &config) : config(config)
    {
        pAddress = NULL;
    }

    void alloc (uint64_t len, json &j);
    void dealloc (void);
};

#endif