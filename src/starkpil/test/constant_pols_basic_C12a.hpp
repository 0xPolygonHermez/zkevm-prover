#ifndef CONSTANT_POLS_BASIC_C12_A_HPP
#define CONSTANT_POLS_BASIC_C12_A_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"

class ConstantPolBasicC12a
{
private:
    Goldilocks::Element *_pAddress;
    uint64_t _degree;
    uint64_t _index;

public:
    ConstantPolBasicC12a(Goldilocks::Element *pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index){};
    Goldilocks::Element &operator[](uint64_t i) { return _pAddress[i * 23]; };
    Goldilocks::Element *operator=(Goldilocks::Element *pAddress)
    {
        _pAddress = pAddress;
        return _pAddress;
    };

    Goldilocks::Element *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t index(void) { return _index; }
};

class GlobalConstantPolsBasicC12a
{
public:
    ConstantPolBasicC12a L1;
    ConstantPolBasicC12a L2;
    ConstantPolBasicC12a L3;
    ConstantPolBasicC12a L4;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    GlobalConstantPolsBasicC12a(void *pAddress, uint64_t degree) : L1((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
                                                                   L2((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1),
                                                                   L3((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
                                                                   L4((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
                                                                   _pAddress(pAddress),
                                                                   _degree(degree){};

    static uint64_t pilDegree(void) { return 131072; }
    static uint64_t pilSize(void) { return 32; }
    static uint64_t numPols(void) { return 4; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 4 * sizeof(Goldilocks::Element); }
};

class CompressorConstantPolsBasicC12
{
public:
    ConstantPolBasicC12a S[12];
    ConstantPolBasicC12a Qm;
    ConstantPolBasicC12a Ql;
    ConstantPolBasicC12a Qr;
    ConstantPolBasicC12a Qo;
    ConstantPolBasicC12a Qk;
    ConstantPolBasicC12a QMDS;
    ConstantPolBasicC12a QCMul;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    CompressorConstantPolsBasicC12(void *pAddress, uint64_t degree) : S{
                                                                          ConstantPolBasicC12a((Goldilocks::Element *)((uint8_t *)pAddress + 32), degree, 4),
                                                                          ConstantPolBasicC12a((Goldilocks::Element *)((uint8_t *)pAddress + 40), degree, 5),
                                                                          ConstantPolBasicC12a((Goldilocks::Element *)((uint8_t *)pAddress + 48), degree, 6),
                                                                          ConstantPolBasicC12a((Goldilocks::Element *)((uint8_t *)pAddress + 56), degree, 7),
                                                                          ConstantPolBasicC12a((Goldilocks::Element *)((uint8_t *)pAddress + 64), degree, 8),
                                                                          ConstantPolBasicC12a((Goldilocks::Element *)((uint8_t *)pAddress + 72), degree, 9),
                                                                          ConstantPolBasicC12a((Goldilocks::Element *)((uint8_t *)pAddress + 80), degree, 10),
                                                                          ConstantPolBasicC12a((Goldilocks::Element *)((uint8_t *)pAddress + 88), degree, 11),
                                                                          ConstantPolBasicC12a((Goldilocks::Element *)((uint8_t *)pAddress + 96), degree, 12),
                                                                          ConstantPolBasicC12a((Goldilocks::Element *)((uint8_t *)pAddress + 104), degree, 13),
                                                                          ConstantPolBasicC12a((Goldilocks::Element *)((uint8_t *)pAddress + 112), degree, 14),
                                                                          ConstantPolBasicC12a((Goldilocks::Element *)((uint8_t *)pAddress + 120), degree, 15)},
                                                                      Qm((Goldilocks::Element *)((uint8_t *)pAddress + 128), degree, 16), Ql((Goldilocks::Element *)((uint8_t *)pAddress + 136), degree, 17), Qr((Goldilocks::Element *)((uint8_t *)pAddress + 144), degree, 18), Qo((Goldilocks::Element *)((uint8_t *)pAddress + 152), degree, 19), Qk((Goldilocks::Element *)((uint8_t *)pAddress + 160), degree, 20), QMDS((Goldilocks::Element *)((uint8_t *)pAddress + 168), degree, 21), QCMul((Goldilocks::Element *)((uint8_t *)pAddress + 176), degree, 22), _pAddress(pAddress), _degree(degree){};

    static uint64_t pilDegree(void) { return 131072; }
    static uint64_t pilSize(void) { return 152; }
    static uint64_t numPols(void) { return 19; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 19 * sizeof(Goldilocks::Element); }
};

class ConstantPolsBasicC12
{
public:
    GlobalConstantPolsBasicC12a Global;
    CompressorConstantPolsBasicC12 Compressor;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    ConstantPolsBasicC12(void *pAddress, uint64_t degree) : Global(pAddress, degree),
                                                            Compressor(pAddress, degree),
                                                            _pAddress(pAddress),
                                                            _degree(degree) {}

    static uint64_t pilSize(void) { return 24117248; }
    static uint64_t pilDegree(void) { return 131072; }
    static uint64_t numPols(void) { return 23; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 23 * sizeof(Goldilocks::Element); }

    Goldilocks::Element &getElement(uint64_t pol, uint64_t evaluation)
    {
        zkassert((pol < numPols()) && (evaluation < degree()));
        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * numPols()];
    }
};

#endif // CONSTANT_POLS_HPP
