#ifndef CONSTANT_POLS_BASIC_C12_B_HPP
#define CONSTANT_POLS_BASIC_C12_B_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"

class ConstantPolBasicC12b
{
private:
    Goldilocks::Element * _pAddress;
    uint64_t _degree;
    uint64_t _index;
public:
    ConstantPolBasicC12b(Goldilocks::Element * pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index) {};
    Goldilocks::Element & operator[](int i) { return _pAddress[i*20]; };
    Goldilocks::Element * operator=(Goldilocks::Element * pAddress) { _pAddress = pAddress; return _pAddress; };

    Goldilocks::Element * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t index (void) { return _index; }
};

class GlobalConstantPolBasicC12b
{
public:
    ConstantPolBasicC12b L1;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    GlobalConstantPolBasicC12b (void * pAddress, uint64_t degree) :
        L1((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
        _pAddress(pAddress),
        _degree(degree) {};

    static uint64_t pilDegree (void) { return 262144; }
    static uint64_t pilSize (void) { return 8; }
    static uint64_t numPols (void) { return 1; }

    void * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t size (void) { return _degree*1*sizeof(Goldilocks::Element); }
};

class CompressorConstantPolBasicC12b
{
public:
    ConstantPolBasicC12b S[12];
    ConstantPolBasicC12b Qm;
    ConstantPolBasicC12b Ql;
    ConstantPolBasicC12b Qr;
    ConstantPolBasicC12b Qo;
    ConstantPolBasicC12b Qk;
    ConstantPolBasicC12b QMDS;
    ConstantPolBasicC12b QCMul;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    CompressorConstantPolBasicC12b (void * pAddress, uint64_t degree) :
        S{
            ConstantPolBasicC12b((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1),
            ConstantPolBasicC12b((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
            ConstantPolBasicC12b((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
            ConstantPolBasicC12b((Goldilocks::Element *)((uint8_t *)pAddress + 32), degree, 4),
            ConstantPolBasicC12b((Goldilocks::Element *)((uint8_t *)pAddress + 40), degree, 5),
            ConstantPolBasicC12b((Goldilocks::Element *)((uint8_t *)pAddress + 48), degree, 6),
            ConstantPolBasicC12b((Goldilocks::Element *)((uint8_t *)pAddress + 56), degree, 7),
            ConstantPolBasicC12b((Goldilocks::Element *)((uint8_t *)pAddress + 64), degree, 8),
            ConstantPolBasicC12b((Goldilocks::Element *)((uint8_t *)pAddress + 72), degree, 9),
            ConstantPolBasicC12b((Goldilocks::Element *)((uint8_t *)pAddress + 80), degree, 10),
            ConstantPolBasicC12b((Goldilocks::Element *)((uint8_t *)pAddress + 88), degree, 11),
            ConstantPolBasicC12b((Goldilocks::Element *)((uint8_t *)pAddress + 96), degree, 12)
        },
        Qm((Goldilocks::Element *)((uint8_t *)pAddress + 104), degree, 13),
        Ql((Goldilocks::Element *)((uint8_t *)pAddress + 112), degree, 14),
        Qr((Goldilocks::Element *)((uint8_t *)pAddress + 120), degree, 15),
        Qo((Goldilocks::Element *)((uint8_t *)pAddress + 128), degree, 16),
        Qk((Goldilocks::Element *)((uint8_t *)pAddress + 136), degree, 17),
        QMDS((Goldilocks::Element *)((uint8_t *)pAddress + 144), degree, 18),
        QCMul((Goldilocks::Element *)((uint8_t *)pAddress + 152), degree, 19),
        _pAddress(pAddress),
        _degree(degree) {};

    static uint64_t pilDegree (void) { return 262144; }
    static uint64_t pilSize (void) { return 152; }
    static uint64_t numPols (void) { return 19; }

    void * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t size (void) { return _degree*19*sizeof(Goldilocks::Element); }
};

class ConstantPolsBasicC12b
{
public:
    GlobalConstantPolBasicC12b Global;
    CompressorConstantPolBasicC12b Compressor;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    ConstantPolsBasicC12b (void * pAddress, uint64_t degree) :
        Global(pAddress, degree),
        Compressor(pAddress, degree),
        _pAddress(pAddress),
        _degree(degree) {}

    static uint64_t pilSize (void) { return 41943040; }
    static uint64_t pilDegree (void) { return 262144; }
    static uint64_t numPols (void) { return 20; }

    void * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t size (void) { return _degree*20*sizeof(Goldilocks::Element); }

    Goldilocks::Element &getElement (uint64_t pol, uint64_t evaluation)
    {
        zkassert((pol < numPols()) && (evaluation < degree()));
        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * numPols()];
    }
};

#endif // CONSTANT_POLS_HPP
