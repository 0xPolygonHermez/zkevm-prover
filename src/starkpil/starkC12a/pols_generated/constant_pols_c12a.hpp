#ifndef CONSTANT_POLS_C12_A_HPP
#define CONSTANT_POLS_C12_A_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"
#include "abstractPols.hpp"

class ConstantPolC12a
{
private:
    Goldilocks::Element *_pAddress;
    uint64_t _degree;
    uint64_t _index;

public:
    ConstantPolC12a(Goldilocks::Element *pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index){};
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

class GlobalConstantPolsC12a
{
public:
    ConstantPolC12a L1;
    ConstantPolC12a L2;
    ConstantPolC12a L3;
    ConstantPolC12a L4;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    GlobalConstantPolsC12a(void *pAddress, uint64_t degree) : L1((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
                                                              L2((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1),
                                                              L3((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
                                                              L4((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
                                                              _pAddress(pAddress),
                                                              _degree(degree){};

    static uint64_t pilDegree(void) { return 16777216; }
    static uint64_t pilSize(void) { return 32; }
    static uint64_t numPols(void) { return 4; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 4 * sizeof(Goldilocks::Element); }
};

class CompressorConstantPolsC12a
{
public:
    ConstantPolC12a S[12];
    ConstantPolC12a Qm;
    ConstantPolC12a Ql;
    ConstantPolC12a Qr;
    ConstantPolC12a Qo;
    ConstantPolC12a Qk;
    ConstantPolC12a QMDS;
    ConstantPolC12a QCMul;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    CompressorConstantPolsC12a(void *pAddress, uint64_t degree) : S{
                                                                      ConstantPolC12a((Goldilocks::Element *)((uint8_t *)pAddress + 32), degree, 4),
                                                                      ConstantPolC12a((Goldilocks::Element *)((uint8_t *)pAddress + 40), degree, 5),
                                                                      ConstantPolC12a((Goldilocks::Element *)((uint8_t *)pAddress + 48), degree, 6),
                                                                      ConstantPolC12a((Goldilocks::Element *)((uint8_t *)pAddress + 56), degree, 7),
                                                                      ConstantPolC12a((Goldilocks::Element *)((uint8_t *)pAddress + 64), degree, 8),
                                                                      ConstantPolC12a((Goldilocks::Element *)((uint8_t *)pAddress + 72), degree, 9),
                                                                      ConstantPolC12a((Goldilocks::Element *)((uint8_t *)pAddress + 80), degree, 10),
                                                                      ConstantPolC12a((Goldilocks::Element *)((uint8_t *)pAddress + 88), degree, 11),
                                                                      ConstantPolC12a((Goldilocks::Element *)((uint8_t *)pAddress + 96), degree, 12),
                                                                      ConstantPolC12a((Goldilocks::Element *)((uint8_t *)pAddress + 104), degree, 13),
                                                                      ConstantPolC12a((Goldilocks::Element *)((uint8_t *)pAddress + 112), degree, 14),
                                                                      ConstantPolC12a((Goldilocks::Element *)((uint8_t *)pAddress + 120), degree, 15)},
                                                                  Qm((Goldilocks::Element *)((uint8_t *)pAddress + 128), degree, 16), Ql((Goldilocks::Element *)((uint8_t *)pAddress + 136), degree, 17), Qr((Goldilocks::Element *)((uint8_t *)pAddress + 144), degree, 18), Qo((Goldilocks::Element *)((uint8_t *)pAddress + 152), degree, 19), Qk((Goldilocks::Element *)((uint8_t *)pAddress + 160), degree, 20), QMDS((Goldilocks::Element *)((uint8_t *)pAddress + 168), degree, 21), QCMul((Goldilocks::Element *)((uint8_t *)pAddress + 176), degree, 22), _pAddress(pAddress), _degree(degree){};

    static uint64_t pilDegree(void) { return 16777216; }
    static uint64_t pilSize(void) { return 152; }
    static uint64_t numPols(void) { return 19; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 19 * sizeof(Goldilocks::Element); }
};

class ConstantPolsC12a : public ConstantPolsAbstarct
{
public:
    GlobalConstantPolsC12a Global;
    CompressorConstantPolsC12a Compressor;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    ConstantPolsC12a(void *pAddress, uint64_t degree) : Global(pAddress, degree),
                                                        Compressor(pAddress, degree),
                                                        _pAddress(pAddress),
                                                        _degree(degree) {}

    static uint64_t pilSize(void) { return 3087007744; }
    static uint64_t pilDegree(void) { return 16777216; }
    static uint64_t numPols(void) { return 23; }

    uint64_t pilSizeN(void) { return 3087007744; }
    uint64_t pilDegreeN(void) { return 16777216; }
    uint64_t numPolsN(void) { return 23; }

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
