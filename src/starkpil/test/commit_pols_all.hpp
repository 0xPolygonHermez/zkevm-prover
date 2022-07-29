#ifndef COMMIT_POLS_ALL_HPP
#define COMMIT_POLS_ALL_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"

class CommitPolAll
{
private:
    Goldilocks::Element *_pAddress;
    uint64_t _degree;
    uint64_t _index;

public:
    CommitPolAll(Goldilocks::Element *pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index){};
    Goldilocks::Element &operator[](int i) { return _pAddress[i * 15]; };
    Goldilocks::Element *operator=(Goldilocks::Element *pAddress)
    {
        _pAddress = pAddress;
        return _pAddress;
    };

    Goldilocks::Element *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t index(void) { return _index; }
};

class FibonacciCommitPols
{
public:
    CommitPolAll l1;
    CommitPolAll l2;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    FibonacciCommitPols(void *pAddress, uint64_t degree) : l1((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
                                                           l2((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1),
                                                           _pAddress(pAddress),
                                                           _degree(degree){};

    static uint64_t pilDegree(void) { return 1024; }
    static uint64_t pilSize(void) { return 16; }
    static uint64_t numPols(void) { return 2; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 2 * sizeof(Goldilocks::Element); }
};

class ConnectionCommitPols
{
public:
    CommitPolAll a;
    CommitPolAll b;
    CommitPolAll c;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    ConnectionCommitPols(void *pAddress, uint64_t degree) : a((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
                                                            b((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
                                                            c((Goldilocks::Element *)((uint8_t *)pAddress + 32), degree, 4),
                                                            _pAddress(pAddress),
                                                            _degree(degree){};

    static uint64_t pilDegree(void) { return 1024; }
    static uint64_t pilSize(void) { return 24; }
    static uint64_t numPols(void) { return 3; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 3 * sizeof(Goldilocks::Element); }
};

class PermutationCommitPols
{
public:
    CommitPolAll a;
    CommitPolAll b;
    CommitPolAll c;
    CommitPolAll d;
    CommitPolAll selC;
    CommitPolAll selD;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    PermutationCommitPols(void *pAddress, uint64_t degree) : a((Goldilocks::Element *)((uint8_t *)pAddress + 40), degree, 5),
                                                             b((Goldilocks::Element *)((uint8_t *)pAddress + 48), degree, 6),
                                                             c((Goldilocks::Element *)((uint8_t *)pAddress + 56), degree, 7),
                                                             d((Goldilocks::Element *)((uint8_t *)pAddress + 64), degree, 8),
                                                             selC((Goldilocks::Element *)((uint8_t *)pAddress + 72), degree, 9),
                                                             selD((Goldilocks::Element *)((uint8_t *)pAddress + 80), degree, 10),
                                                             _pAddress(pAddress),
                                                             _degree(degree){};

    static uint64_t pilDegree(void) { return 1024; }
    static uint64_t pilSize(void) { return 48; }
    static uint64_t numPols(void) { return 6; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 6 * sizeof(Goldilocks::Element); }
};

class PlookupCommitPols
{
public:
    CommitPolAll sel;
    CommitPolAll a;
    CommitPolAll b;
    CommitPolAll cc;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    PlookupCommitPols(void *pAddress, uint64_t degree) : sel((Goldilocks::Element *)((uint8_t *)pAddress + 88), degree, 11),
                                                         a((Goldilocks::Element *)((uint8_t *)pAddress + 96), degree, 12),
                                                         b((Goldilocks::Element *)((uint8_t *)pAddress + 104), degree, 13),
                                                         cc((Goldilocks::Element *)((uint8_t *)pAddress + 112), degree, 14),
                                                         _pAddress(pAddress),
                                                         _degree(degree){};

    static uint64_t pilDegree(void) { return 1024; }
    static uint64_t pilSize(void) { return 32; }
    static uint64_t numPols(void) { return 4; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 4 * sizeof(Goldilocks::Element); }
};

class CommitPolsAll
{
public:
    FibonacciCommitPols Fibonacci;
    ConnectionCommitPols Connection;
    PermutationCommitPols Permutation;
    PlookupCommitPols Plookup;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    CommitPolsAll(void *pAddress, uint64_t degree) : Fibonacci(pAddress, degree),
                                                     Connection(pAddress, degree),
                                                     Permutation(pAddress, degree),
                                                     Plookup(pAddress, degree),
                                                     _pAddress(pAddress),
                                                     _degree(degree) {}

    static uint64_t pilSize(void) { return 122880; }
    static uint64_t pilDegree(void) { return 1024; }
    static uint64_t numPols(void) { return 15; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 15 * sizeof(Goldilocks::Element); }
};

#endif // COMMIT_POLS_HPP
