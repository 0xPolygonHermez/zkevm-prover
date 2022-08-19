#ifndef CONSTANT_POLS_C12_HPP
#define CONSTANT_POLS_C12_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"

class ConstantPolC12
{
private:
    Goldilocks::Element *_pAddress;
    uint64_t _degree;
    uint64_t _index;

public:
    ConstantPolC12(Goldilocks::Element *pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index){};
    Goldilocks::Element &operator[](int i) { return _pAddress[i * 20]; };
    Goldilocks::Element *operator=(Goldilocks::Element *pAddress)
    {
        _pAddress = pAddress;
        return _pAddress;
    };

    Goldilocks::Element *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t index(void) { return _index; }
};

class GlobalConstantPolsC12
{
public:
    ConstantPolC12 L1;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    GlobalConstantPolsC12(void *pAddress, uint64_t degree) : L1((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
                                                          _pAddress(pAddress),
                                                          _degree(degree){};

    static uint64_t pilDegree(void) { return 16777216; }
    static uint64_t pilSize(void) { return 8; }
    static uint64_t numPols(void) { return 1; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 1 * sizeof(Goldilocks::Element); }
};

class CompressorConstantPolsC12
{
public:
    ConstantPolC12 S[12];
    ConstantPolC12 Qm;
    ConstantPolC12 Ql;
    ConstantPolC12 Qr;
    ConstantPolC12 Qo;
    ConstantPolC12 Qk;
    ConstantPolC12 QMDS;
    ConstantPolC12 QCMul;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    CompressorConstantPolsC12(void *pAddress, uint64_t degree) : S{
                                                                  ConstantPolC12((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1),
                                                                  ConstantPolC12((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
                                                                  ConstantPolC12((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
                                                                  ConstantPolC12((Goldilocks::Element *)((uint8_t *)pAddress + 32), degree, 4),
                                                                  ConstantPolC12((Goldilocks::Element *)((uint8_t *)pAddress + 40), degree, 5),
                                                                  ConstantPolC12((Goldilocks::Element *)((uint8_t *)pAddress + 48), degree, 6),
                                                                  ConstantPolC12((Goldilocks::Element *)((uint8_t *)pAddress + 56), degree, 7),
                                                                  ConstantPolC12((Goldilocks::Element *)((uint8_t *)pAddress + 64), degree, 8),
                                                                  ConstantPolC12((Goldilocks::Element *)((uint8_t *)pAddress + 72), degree, 9),
                                                                  ConstantPolC12((Goldilocks::Element *)((uint8_t *)pAddress + 80), degree, 10),
                                                                  ConstantPolC12((Goldilocks::Element *)((uint8_t *)pAddress + 88), degree, 11),
                                                                  ConstantPolC12((Goldilocks::Element *)((uint8_t *)pAddress + 96), degree, 12)},
                                                              Qm((Goldilocks::Element *)((uint8_t *)pAddress + 104), degree, 13), Ql((Goldilocks::Element *)((uint8_t *)pAddress + 112), degree, 14), Qr((Goldilocks::Element *)((uint8_t *)pAddress + 120), degree, 15), Qo((Goldilocks::Element *)((uint8_t *)pAddress + 128), degree, 16), Qk((Goldilocks::Element *)((uint8_t *)pAddress + 136), degree, 17), QMDS((Goldilocks::Element *)((uint8_t *)pAddress + 144), degree, 18), QCMul((Goldilocks::Element *)((uint8_t *)pAddress + 152), degree, 19), _pAddress(pAddress), _degree(degree){};

    static uint64_t pilDegree(void) { return 16777216; }
    static uint64_t pilSize(void) { return 152; }
    static uint64_t numPols(void) { return 19; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 19 * sizeof(Goldilocks::Element); }
};

class ConstantPolsC12
{
public:
    GlobalConstantPolsC12 Global;
    CompressorConstantPolsC12 Compressor;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    ConstantPolsC12(void *pAddress, uint64_t degree) : Global(pAddress, degree),
                                                    Compressor(pAddress, degree),
                                                    _pAddress(pAddress),
                                                    _degree(degree) {}

    static uint64_t pilSize(void) { return 2684354560; }
    static uint64_t pilDegree(void) { return 16777216; }
    static uint64_t numPols(void) { return 20; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * 20 * sizeof(Goldilocks::Element); }

    Goldilocks::Element &getElement(uint64_t pol, uint64_t evaluation)
    {
        zkassert((pol < numPols()) && (evaluation < degree()));
        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * numPols()];
    }
};

#endif // CONSTANT_POLS_HPP
