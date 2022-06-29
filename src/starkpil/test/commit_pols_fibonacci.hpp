#ifndef FIB_COMMIT_POLS_HPP
#define FIB_COMMIT_POLS_HPP

#include <cstdint>
#include "goldilocks/goldilocks_base_field.hpp"

class FibCommitGeneratedPol
{
private:
    Goldilocks::Element *pData;
    uint64_t length;

public:
    FibCommitGeneratedPol(void *data, uint _length) : pData((Goldilocks::Element *)data), length(_length){};
    Goldilocks::Element &operator[](int i) { return pData[i * length]; };
    Goldilocks::Element *elments() { return pData; };
    FibCommitGeneratedPol &operator=(const FibCommitGeneratedPol &other)
    {
        if (this != &other)
        {
            if (pData != nullptr)
            {
                *pData = *other.pData;
            }
        }
        return *this;
    }
};

class FibonacciCommitPols
{
public:
    FibCommitGeneratedPol l1;
    FibCommitGeneratedPol l2;

    Goldilocks::Element *pData;

    uint64_t degree;
    uint64_t length;

    FibonacciCommitPols(void *pAddress, uint64_t _degree, uint64_t _legnth) : l1(pAddress, _legnth), l2((uint8_t *)pAddress + sizeof(Goldilocks::Element), _legnth), pData((Goldilocks::Element *)pAddress), degree(_degree), length(_legnth) {}
};

class FibCommitPols
{
public:
    FibonacciCommitPols Fibonacci;

    uint64_t degree;
    uint64_t length;

    FibCommitPols(void *pAddress, uint64_t degree, uint64_t length) : Fibonacci(pAddress, degree, length), degree(degree), length(length) {}
    uint64_t size(void) { return degree * length * sizeof(Goldilocks::Element); }
};

#endif // COMMIT_POLS_HPP
