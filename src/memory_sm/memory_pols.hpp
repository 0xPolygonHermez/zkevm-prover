#ifndef MEMORY_POLS_HPP
#define MEMORY_POLS_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include "config.hpp"
#include "ff/ff.hpp"

class MemoryPols
{
public:
    // Committed polynomials
    uint64_t * addr;        // Memmory address accessed by the main state machine
    uint64_t * step;        // Execution step, i.e. evaluation number
    uint64_t * mOp;         // =1 if this is a memory read or memory write operation; =0 otherwise
    uint64_t * mRd;         // =1 if this is a memory read operation; =0 otherwise
    uint64_t * mWr;         // =1 if this is a memory write operation; =0 otherwise
    FieldElement * val[8];  // Value stored in memory: 8 x 32 bits
    uint64_t * lastAccess;  // =1 if this is the last access of this memory address

private:
    // Internal attributes
    uint64_t nCommitments;
    uint64_t firstPolId;
    uint64_t length;
    uint64_t polSize;
    uint64_t numberOfPols;
    uint64_t totalSize;
    uint64_t * pAddress;

public:
    MemoryPols()
    {
        pAddress = NULL;
    }

    void alloc (uint64_t len, json &j);
    void dealloc (void);

    uint64_t getPolOrder (json &j, const char * pPolName);
};

#endif