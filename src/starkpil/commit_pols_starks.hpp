#ifndef COMMIT_POLS_STARKS_HPP
#define COMMIT_POLS_STARKS_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"

#define NUM_COMMIT_POLS 12

class CommitPolStarks
{
private:
    Goldilocks::Element *_pAddress;
    uint64_t _degree;
    uint64_t _index;

public:
    CommitPolStarks(Goldilocks::Element *pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index){};
    Goldilocks::Element &operator[](uint64_t i) { return _pAddress[i * NUM_COMMIT_POLS]; };
    Goldilocks::Element *operator=(Goldilocks::Element *pAddress)
    {
        _pAddress = pAddress;
        return _pAddress;
    };

    Goldilocks::Element *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t index(void) { return _index; }
};

class CompressorCommitPolsStarks
{
public:
    CommitPolStarks a[NUM_COMMIT_POLS];

private:
    void *_pAddress;
    uint64_t _degree;

public:
    CompressorCommitPolsStarks(void *pAddress, uint64_t degree) : a{
                                                                      CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
                                                                      CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1),
                                                                      CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
                                                                      CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
                                                                      CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 32), degree, 4),
                                                                      CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 40), degree, 5),
                                                                      CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 48), degree, 6),
                                                                      CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 56), degree, 7),
                                                                      CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 64), degree, 8),
                                                                      CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 72), degree, 9),
                                                                      CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 80), degree, 10),
                                                                      CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 88), degree, 11)},
                                                                  _pAddress(pAddress), _degree(degree){};

    static uint64_t numPols(void) { return NUM_COMMIT_POLS; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * NUM_COMMIT_POLS * sizeof(Goldilocks::Element); }
};

class CommitPolsStarks
{
public:
    CompressorCommitPolsStarks Compressor;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    CommitPolsStarks(void *pAddress, uint64_t degree) : Compressor(pAddress, degree),
                                                        _pAddress(pAddress),
                                                        _degree(degree) {}

    static uint64_t numPols(void) { return NUM_COMMIT_POLS; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * NUM_COMMIT_POLS * sizeof(Goldilocks::Element); }

    Goldilocks::Element &getElement(uint64_t pol, uint64_t evaluation)
    {
        zkassert((pol < numPols()) && (evaluation < degree()));
        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * numPols()];
    }
};

#endif // COMMIT_POLS_STARKS_HPP
