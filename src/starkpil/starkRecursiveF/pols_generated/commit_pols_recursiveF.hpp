#ifndef COMMIT_POLS_RECURSIVE_F_HPP
#define COMMIT_POLS_RECURSIVE_F_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"

class CommitPolRecursiveF
{
private:
    Goldilocks::Element * _pAddress;
    uint64_t _degree;
    uint64_t _index;
public:
    CommitPolRecursiveF(Goldilocks::Element * pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index) {};
    Goldilocks::Element & operator[](uint64_t i) { return _pAddress[i*12]; };
    Goldilocks::Element * operator=(Goldilocks::Element * pAddress) { _pAddress = pAddress; return _pAddress; };

    Goldilocks::Element * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t index (void) { return _index; }
};

class CompressorCommitPolsRecursiveF
{
public:
    CommitPolRecursiveF a[12];
private:
    void * _pAddress;
    uint64_t _degree;
public:

    CompressorCommitPolsRecursiveF (void * pAddress, uint64_t degree) :
        a{
            CommitPolRecursiveF((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
            CommitPolRecursiveF((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1),
            CommitPolRecursiveF((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
            CommitPolRecursiveF((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
            CommitPolRecursiveF((Goldilocks::Element *)((uint8_t *)pAddress + 32), degree, 4),
            CommitPolRecursiveF((Goldilocks::Element *)((uint8_t *)pAddress + 40), degree, 5),
            CommitPolRecursiveF((Goldilocks::Element *)((uint8_t *)pAddress + 48), degree, 6),
            CommitPolRecursiveF((Goldilocks::Element *)((uint8_t *)pAddress + 56), degree, 7),
            CommitPolRecursiveF((Goldilocks::Element *)((uint8_t *)pAddress + 64), degree, 8),
            CommitPolRecursiveF((Goldilocks::Element *)((uint8_t *)pAddress + 72), degree, 9),
            CommitPolRecursiveF((Goldilocks::Element *)((uint8_t *)pAddress + 80), degree, 10),
            CommitPolRecursiveF((Goldilocks::Element *)((uint8_t *)pAddress + 88), degree, 11)
        },
        _pAddress(pAddress),
        _degree(degree) {};

    static uint64_t pilDegree (void) { return 4194304; }
    static uint64_t pilSize (void) { return 96; }
    static uint64_t numPols (void) { return 12; }

    void * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t size (void) { return _degree*12*sizeof(Goldilocks::Element); }
};

class CommitPolsRecursiveF
{
public:
    CompressorCommitPolsRecursiveF Compressor;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    CommitPolsRecursiveF (void * pAddress, uint64_t degree) :
        Compressor(pAddress, degree),
        _pAddress(pAddress),
        _degree(degree) {}

    static uint64_t pilSize (void) { return 402653184; }
    static uint64_t pilDegree (void) { return 4194304; }
    static uint64_t numPols (void) { return 12; }

    void * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t size (void) { return _degree*12*sizeof(Goldilocks::Element); }

    Goldilocks::Element &getElement (uint64_t pol, uint64_t evaluation)
    {
        zkassert((pol < numPols()) && (evaluation < degree()));
        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * numPols()];
    }
};

#endif // COMMIT_POLS_HPP
