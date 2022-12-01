#ifndef CONSTANT_POLS_STARKS_HPP
#define CONSTANT_POLS_STARKS_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"

#define NUM_CONSTANT_POLS 23
#define NUM_COMPRESSOR_CONSTANT_POLS_STARKS 19
#define NUM_GLOBAL_CONSTANT 4
#define NUM_S 12

class ConstantPolStarks
{
private:
    Goldilocks::Element *_pAddress;
    uint64_t _degree;
    uint64_t _index;

public:
    ConstantPolStarks(Goldilocks::Element *pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index){};
    Goldilocks::Element &operator[](uint64_t i) { return _pAddress[i * NUM_CONSTANT_POLS]; };
    Goldilocks::Element *operator=(Goldilocks::Element *pAddress)
    {
        _pAddress = pAddress;
        return _pAddress;
    };

    Goldilocks::Element *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t index(void) { return _index; }
};

class GlobalConstantPolsStarks
{
public:
    ConstantPolStarks L1;
    ConstantPolStarks L2;
    ConstantPolStarks L3;
    ConstantPolStarks L4;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    GlobalConstantPolsStarks(void *pAddress, uint64_t degree) : L1((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
                                                                L2((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1),
                                                                L3((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
                                                                L4((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
                                                                _pAddress(pAddress),
                                                                _degree(degree){};

    static uint64_t numPols(void) { return NUM_GLOBAL_CONSTANT; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * NUM_GLOBAL_CONSTANT * sizeof(Goldilocks::Element); }
};

class CompressorConstantPolsStarks
{
public:
    ConstantPolStarks S[NUM_S];
    ConstantPolStarks Qm;
    ConstantPolStarks Ql;
    ConstantPolStarks Qr;
    ConstantPolStarks Qo;
    ConstantPolStarks Qk;
    ConstantPolStarks QMDS;
    ConstantPolStarks QCMul;

private:
    void *_pAddress;
    uint64_t _degree;

public:
    CompressorConstantPolsStarks(void *pAddress, uint64_t degree) : S{
                                                                        ConstantPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 32), degree, 4),
                                                                        ConstantPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 40), degree, 5),
                                                                        ConstantPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 48), degree, 6),
                                                                        ConstantPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 56), degree, 7),
                                                                        ConstantPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 64), degree, 8),
                                                                        ConstantPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 72), degree, 9),
                                                                        ConstantPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 80), degree, 10),
                                                                        ConstantPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 88), degree, 11),
                                                                        ConstantPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 96), degree, 12),
                                                                        ConstantPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 104), degree, 13),
                                                                        ConstantPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 112), degree, 14),
                                                                        ConstantPolStarks((Goldilocks::Element *)((uint8_t *)pAddress + 120), degree, 15)},
                                                                    Qm((Goldilocks::Element *)((uint8_t *)pAddress + 128), degree, 16), Ql((Goldilocks::Element *)((uint8_t *)pAddress + 136), degree, 17), Qr((Goldilocks::Element *)((uint8_t *)pAddress + 144), degree, 18), Qo((Goldilocks::Element *)((uint8_t *)pAddress + 152), degree, 19), Qk((Goldilocks::Element *)((uint8_t *)pAddress + 160), degree, 20), QMDS((Goldilocks::Element *)((uint8_t *)pAddress + 168), degree, 21), QCMul((Goldilocks::Element *)((uint8_t *)pAddress + 176), degree, 22), _pAddress(pAddress), _degree(degree){};

    static uint64_t numPols(void) { return NUM_COMPRESSOR_CONSTANT_POLS_STARKS; }

    void *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t size(void) { return _degree * NUM_COMPRESSOR_CONSTANT_POLS_STARKS * sizeof(Goldilocks::Element); }
};

class ConstantPolsStarks
{
public:
    GlobalConstantPolsStarks Global;
    CompressorConstantPolsStarks Compressor;

private:
    void *_pAddress;
    uint64_t _degree;
    uint64_t _numPols;

public:
    ConstantPolsStarks(void *pAddress, uint64_t degree, uint64_t numPols) : Global(pAddress, degree),
                                                                            Compressor(pAddress, degree),
                                                                            _pAddress(pAddress),
                                                                            _degree(degree),
                                                                            _numPols(numPols){};

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
