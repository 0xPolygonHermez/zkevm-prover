#ifndef FIBONACCIEXPS
#define FIBONACCIEXPS

#include "goldilocks/goldilocks_base_field.hpp"

class FibonacciExps
{

public:
    Goldilocks::Element *pData;
    uint64_t nExps;
    uint64_t degree;

    FibonacciExps() : pData(nullptr), nExps(0), degree(0) {}
    FibonacciExps(const FibonacciExps &other) : pData(nullptr), nExps(other.nExps), degree(other.degree)
    { // copy constructor
        pData = (Goldilocks::Element *)malloc(other.nExps * other.degree * sizeof(Goldilocks::Element));
        std::memcpy(pData, other.pData, other.nExps * other.degree * sizeof(Goldilocks::Element));
    }
    FibonacciExps &operator=(const FibonacciExps &other)
    { // copy assignment constructor
        // protect against self assignment
        if (this != &other)
        {
            if (pData != nullptr)
            {
                *pData = *other.pData;
            }
            else
            { // p is null - no memory allocated yet
                pData = (Goldilocks::Element *)malloc(other.nExps * other.degree * sizeof(Goldilocks::Element));
            }
        }
        return *this;
    }
    FibonacciExps(uint64_t _degree, uint64_t _nExps) : pData(nullptr), nExps(_nExps), degree(_degree)
    {
        pData = (Goldilocks::Element *)malloc(_nExps * _degree * sizeof(Goldilocks::Element));
    };

    ~FibonacciExps()
    {
        if (pData != nullptr)
            free(pData);
    };

    Goldilocks::Element &operator[](int i) { return pData[i]; };
};

#endif