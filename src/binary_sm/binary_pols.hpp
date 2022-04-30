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
    uint64_t * freeInA;
    uint64_t * freeInB;
    uint64_t * freeInC;
    uint64_t * a0;
    uint64_t * a1;
    uint64_t * a2;
    uint64_t * a3;
    uint64_t * a4;
    uint64_t * a5;
    uint64_t * a6;
    uint64_t * a7;
    uint64_t * b0;
    uint64_t * b1;
    uint64_t * b2;
    uint64_t * b3;
    uint64_t * b4;
    uint64_t * b5;
    uint64_t * b6;
    uint64_t * b7;
    uint64_t * c0;
    uint64_t * c1;
    uint64_t * c2;
    uint64_t * c3;
    uint64_t * c4;
    uint64_t * c5;
    uint64_t * c6;
    uint64_t * c7;
    uint64_t * c0Temp;
    uint64_t * opcode;
    uint64_t * cIn;
    uint64_t * cOut;
    uint64_t * last;
    uint64_t * useCarry;

private:
    // Internal attributes
    uint64_t nCommitments;
    uint64_t length;
    uint64_t polSize;
    uint64_t numberOfPols;
    uint64_t totalSize;
    uint64_t * pAddress;

public:
    BinaryPols(const Config &config) : config(config)
    {
        pAddress = NULL;
    }

    void alloc (uint64_t len, json &j);
    void dealloc (void);

    uint64_t getPolOrder (json &j, const char * pPolName);
};

#endif