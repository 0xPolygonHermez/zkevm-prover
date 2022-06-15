#ifndef FIB_CONSTANT_POLS_HPP
#define FIB_CONSTANT_POLS_HPP

#include <cstdint>
#include "goldilocks/goldilocks_base_field.hpp"

class FibConstantGeneratedPol
{
private:
    Goldilocks::Element * pData;
public:
    FibConstantGeneratedPol() : pData(NULL) {};
    Goldilocks::Element & operator[](int i) { return pData[i*2]; };
    Goldilocks::Element * operator=(Goldilocks::Element * pAddress) { pData = pAddress; return pData; };
};

class FibonacciConstantPols
{
public:
    FibConstantGeneratedPol L1;
    FibConstantGeneratedPol LLAST;

    FibonacciConstantPols (void * pAddress)
    {
        L1 = (Goldilocks::Element *)((uint8_t *)pAddress + 0);
        LLAST = (Goldilocks::Element *)((uint8_t *)pAddress + 8);
    }

    FibonacciConstantPols (void * pAddress, uint64_t degree)
    {
        L1 = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        LLAST = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
    }

    static uint64_t degree (void) { return 1024; }
    static uint64_t size (void) { return 16; }
};

class FibConstantPols
{
public:
    FibonacciConstantPols Fibonacci;

    FibConstantPols (void * pAddress) : Fibonacci(pAddress) {}

    static uint64_t size (void) { return 16384; }
};

#endif // CONSTANT_POLS_HPP
