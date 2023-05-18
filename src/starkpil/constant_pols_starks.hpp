#ifndef CONSTANT_POLS_STARKS_HPP
#define CONSTANT_POLS_STARKS_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"

class ConstantPolsStarks
{
private:
    void *_pAddress;
    uint64_t _degree;
    uint64_t _numPols;

public:
    ConstantPolsStarks(void *pAddress, uint64_t degree, uint64_t numPols) : _pAddress(pAddress),_degree(degree), _numPols(numPols){};

    inline uint64_t numPols(void) { return _numPols; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * _numPols * sizeof(Goldilocks::Element); }

    inline Goldilocks::Element &getElement(uint64_t pol, uint64_t evaluation)
    {
        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * _numPols];
    }
};

#endif // CONSTANT_POLS_RECURSIVE_F_HPP
