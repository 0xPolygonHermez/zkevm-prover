#ifndef COMMIT_POLS_STARKS_HPP
#define COMMIT_POLS_STARKS_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"

class CommitPolStarks
{
private:
    Goldilocks::Element *_pAddress;
    uint64_t _degree;
    uint64_t _index;
    uint64_t _nCommitedPols;

public:
    CommitPolStarks() {}

    CommitPolStarks(Goldilocks::Element *pAddress, uint64_t degree, uint64_t index, uint64_t nCommitedPols) : _pAddress(pAddress), _degree(degree), _index(index), _nCommitedPols(nCommitedPols){};
    Goldilocks::Element &operator[](uint64_t i) { return _pAddress[i * _nCommitedPols]; };
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
    CommitPolStarks* a;

private:
    void *_pAddress;
    uint64_t _degree;
    uint64_t _nCommitedPols;

public:
    CompressorCommitPolsStarks(void *pAddress, uint64_t degree, uint64_t nCommitedPols) : _pAddress(pAddress), _degree(degree), _nCommitedPols(nCommitedPols){
        a = new CommitPolStarks[nCommitedPols];

        for (uint64_t i = 0; i < nCommitedPols; i++) {
            a[i] = CommitPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + sizeof(Goldilocks::Element)*i), degree, i, nCommitedPols);
        }
    };
    
    uint64_t numPols(void) { return _nCommitedPols; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * _nCommitedPols * sizeof(Goldilocks::Element); }
    ~CompressorCommitPolsStarks() {
        delete[] a;
    };
};

class CommitPolsStarks
{
public:
    CompressorCommitPolsStarks Compressor;

private:
    void *_pAddress;
    uint64_t _degree;
    uint64_t _nCommitedPols;

public:
    CommitPolsStarks(void *pAddress, uint64_t degree, uint64_t nCommitedPols) : Compressor(pAddress, degree, nCommitedPols),
                                                        _pAddress(pAddress), _degree(degree), _nCommitedPols(nCommitedPols) {}

    uint64_t numPols(void) { return _nCommitedPols; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * _nCommitedPols * sizeof(Goldilocks::Element); }

    Goldilocks::Element &getElement(uint64_t pol, uint64_t evaluation)
    {
        zkassert((pol < numPols()) && (evaluation < degree()));
        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * numPols()];
    }
};

#endif // COMMIT_POLS_STARKS_HPP
