#ifndef FIB_COMMIT_POLS_HPP
#define FIB_COMMIT_POLS_HPP

#include <cstdint>
#include "goldilocks/goldilocks_base_field.hpp"

class FibCommitGeneratedPol
{
private:
    Goldilocks::Element * pData;
public:
    FibCommitGeneratedPol() : pData(NULL) {};
    Goldilocks::Element & operator[](int i) { return pData[i*2]; };
    Goldilocks::Element * operator=(Goldilocks::Element * pAddress) { pData = pAddress; return pData; };
};

class FibonacciCommitPols
{
public:
    FibCommitGeneratedPol l1;
    FibCommitGeneratedPol l2;

    FibonacciCommitPols (void * pAddress)
    {
        l1 = (Goldilocks::Element *)((uint8_t *)pAddress + 0);
        l2 = (Goldilocks::Element *)((uint8_t *)pAddress + 8);
    }

    FibonacciCommitPols (void * pAddress, uint64_t degree)
    {
        l1 = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        l2 = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
    }

    static uint64_t degree (void) { return 1024; }
    static uint64_t size (void) { return 16; }
};

class FibCommitPols
{
public:
    FibonacciCommitPols Fibonacci;

    FibCommitPols (void * pAddress) : Fibonacci(pAddress) {}

    static uint64_t size (void) { return 16384; }
};

#endif // COMMIT_POLS_HPP
