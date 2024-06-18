#ifndef FORK_INFO_HPP
#define FORK_INFO_HPP

#include <stdint.h>

class ForkInfo
{
public:
    uint64_t id; // Fork ID: 1, 2, ...
    uint64_t parentId; // Fork ID of the main executor that supports this ID, e.g. fork 10 supports both 10 and 11
    uint64_t Nbits; // log2(N)
    uint64_t N; // Number of evaluations N = 2**Nbits, i.e. polynomials degree
    uint64_t N_NoCounters;// Number of evaluations when no counters flag is set
    ForkInfo(
        uint64_t id,
        uint64_t parentId,
        uint64_t Nbits,
        uint64_t N,
        uint64_t N_NoCounters) :
            id(id),
            parentId(parentId),
            Nbits(Nbits),
            N(N),
            N_NoCounters(N_NoCounters) {};
    ForkInfo() : id(0), parentId(0), Nbits(0), N(0), N_NoCounters(0) {};
};

// Returns true on success, false if forkID is not supported
bool getForkInfo (uint64_t forkID, ForkInfo &forkInfo);

// Return N for this fork, or 0 if forkID is not supported
uint64_t getForkN (uint64_t forkID);

#endif