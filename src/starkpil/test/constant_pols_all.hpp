#ifndef CONSTANT_POLS_ALL_HPP
#define CONSTANT_POLS_ALL_HPP

#include <cstdint>
#include "goldilocks/goldilocks_base_field.hpp"

class ConstantPolAll
{
private:
    Goldilocks::Element *_pAddress;
    uint64_t _degree;
    uint64_t _index;

public:
    ConstantPolAll(Goldilocks::Element *pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index){};
    Goldilocks::Element &operator[](int i) { return _pAddress[i * 9]; };
    Goldilocks::Element *operator=(Goldilocks::Element *pAddress)
    {
        _pAddress = pAddress;
        return _pAddress;
    };

    Goldilocks::Element *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t index(void) { return _index; }
};

class GlobalConstantPols
{
public:
    ConstantPolAll L1;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    GlobalConstantPols(void *pAddress, uint64_t degree) : L1((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
                                                          _pAddress(pAddress),
                                                          _degree(degree){};

    static uint64_t pilDegree(void) { return 1024; }
    static uint64_t pilSize(void) { return 8; }
    static uint64_t numPols(void) { return 1; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 1 * sizeof(Goldilocks::Element); }
};

class FibonacciConstantPols
{
public:
    ConstantPolAll L1;
    ConstantPolAll LLAST;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    FibonacciConstantPols(void *pAddress, uint64_t degree) : L1((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1),
                                                             LLAST((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
                                                             _pAddress(pAddress),
                                                             _degree(degree){};

    static uint64_t pilDegree(void) { return 1024; }
    static uint64_t pilSize(void) { return 16; }
    static uint64_t numPols(void) { return 2; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 2 * sizeof(Goldilocks::Element); }
};

class ConnectionConstantPols
{
public:
    ConstantPolAll S1;
    ConstantPolAll S2;
    ConstantPolAll S3;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    ConnectionConstantPols(void *pAddress, uint64_t degree) : S1((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
                                                              S2((Goldilocks::Element *)((uint8_t *)pAddress + 32), degree, 4),
                                                              S3((Goldilocks::Element *)((uint8_t *)pAddress + 40), degree, 5),
                                                              _pAddress(pAddress),
                                                              _degree(degree){};

    static uint64_t pilDegree(void) { return 1024; }
    static uint64_t pilSize(void) { return 24; }
    static uint64_t numPols(void) { return 3; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 3 * sizeof(Goldilocks::Element); }
};

class PlookupConstantPols
{
public:
    ConstantPolAll SEL;
    ConstantPolAll A;
    ConstantPolAll B;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    PlookupConstantPols(void *pAddress, uint64_t degree) : SEL((Goldilocks::Element *)((uint8_t *)pAddress + 48), degree, 6),
                                                           A((Goldilocks::Element *)((uint8_t *)pAddress + 56), degree, 7),
                                                           B((Goldilocks::Element *)((uint8_t *)pAddress + 64), degree, 8),
                                                           _pAddress(pAddress),
                                                           _degree(degree){};

    static uint64_t pilDegree(void) { return 1024; }
    static uint64_t pilSize(void) { return 24; }
    static uint64_t numPols(void) { return 3; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 3 * sizeof(Goldilocks::Element); }
};

class ConstantPolsAll
{
public:
    GlobalConstantPols Global;
    FibonacciConstantPols Fibonacci;
    ConnectionConstantPols Connection;
    PlookupConstantPols Plookup;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    ConstantPolsAll(void *pAddress, uint64_t degree) : Global(pAddress, degree),
                                                       Fibonacci(pAddress, degree),
                                                       Connection(pAddress, degree),
                                                       Plookup(pAddress, degree),
                                                       _pAddress(pAddress),
                                                       _degree(degree) {}

    static uint64_t pilSize(void) { return 73728; }
    static uint64_t pilDegree(void) { return 1024; }
    static uint64_t numPols(void) { return 9; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 9 * sizeof(Goldilocks::Element); }

    Goldilocks::Element &getElement(uint64_t i, uint64_t j)
    {
        zkassert(i < numPols() && j < degree());
        return ((Goldilocks::Element *)_pAddress)[i + j * numPols()];
    };
};

#endif // CONSTANT_POLS_HPP
