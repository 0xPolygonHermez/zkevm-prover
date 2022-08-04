#ifndef CONSTANT_POLS_BASIC_HPP
#define CONSTANT_POLS_BASIC_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"

class ConstantPolBasic
{
private:
    Goldilocks::Element * _pAddress;
    uint64_t _degree;
    uint64_t _index;
public:
    ConstantPolBasic(Goldilocks::Element * pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index) {};
    Goldilocks::Element & operator[](int i) { return _pAddress[i*4]; };
    Goldilocks::Element * operator=(Goldilocks::Element * pAddress) { _pAddress = pAddress; return _pAddress; };

    Goldilocks::Element * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t index (void) { return _index; }
};

class GlobalConstantPolsBasic
{
public:
    ConstantPolBasic L1;
    ConstantPolBasic BYTE;
    ConstantPolBasic BYTE2;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    GlobalConstantPolsBasic (void * pAddress, uint64_t degree) :
        L1((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
        BYTE((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1),
        BYTE2((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
        _pAddress(pAddress),
        _degree(degree) {};

    static uint64_t pilDegree (void) { return 262144; }
    static uint64_t pilSize (void) { return 24; }
    static uint64_t numPols (void) { return 3; }

    void * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t size (void) { return _degree*3*sizeof(Goldilocks::Element); }
};

class MainConstantPolsBasic
{
public:
    ConstantPolBasic STEP;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    MainConstantPolsBasic (void * pAddress, uint64_t degree) :
        STEP((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
        _pAddress(pAddress),
        _degree(degree) {};

    static uint64_t pilDegree (void) { return 262144; }
    static uint64_t pilSize (void) { return 8; }
    static uint64_t numPols (void) { return 1; }

    void * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t size (void) { return _degree*1*sizeof(Goldilocks::Element); }
};

class ConstantPolsBasic
{
public:
    GlobalConstantPolsBasic Global;
    MainConstantPolsBasic Main;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    ConstantPolsBasic (void * pAddress, uint64_t degree) :
        Global(pAddress, degree),
        Main(pAddress, degree),
        _pAddress(pAddress),
        _degree(degree) {}

    static uint64_t pilSize (void) { return 8388608; }
    static uint64_t pilDegree (void) { return 262144; }
    static uint64_t numPols (void) { return 4; }

    void * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t size (void) { return _degree*4*sizeof(Goldilocks::Element); }

    Goldilocks::Element &getElement (uint64_t pol, uint64_t evaluation)
    {
        zkassert((pol < numPols()) && (evaluation < degree()));
        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * numPols()];
    }
};

#endif // CONSTANT_POLS_BASIC_HPP
