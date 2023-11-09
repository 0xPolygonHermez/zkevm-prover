#ifndef CONSTANT_POLS_HPP_fork_7
#define CONSTANT_POLS_HPP_fork_7

#include <cstdint>
#include "goldilocks_base_field.hpp"

namespace fork_7
{

class ConstantPol
{
private:
    Goldilocks::Element * _pAddress;
    uint64_t _degree;
    uint64_t _index;
public:
    ConstantPol(Goldilocks::Element * pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index) {};
    inline Goldilocks::Element & operator[](uint64_t i) { return _pAddress[i*260]; };
    inline Goldilocks::Element * operator=(Goldilocks::Element * pAddress) { _pAddress = pAddress; return _pAddress; };

    inline Goldilocks::Element * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t index (void) { return _index; }
};

class GlobalConstantPols
{
public:
    ConstantPol L1;
    ConstantPol LLAST;
    ConstantPol BYTE;
    ConstantPol BYTE_2A;
    ConstantPol BYTE2;
    ConstantPol CLK32[32];
    ConstantPol BYTE_FACTOR[8];
    ConstantPol STEP;
    ConstantPol STEP32;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    GlobalConstantPols (void * pAddress, uint64_t degree) :
        L1((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
        LLAST((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1),
        BYTE((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
        BYTE_2A((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
        BYTE2((Goldilocks::Element *)((uint8_t *)pAddress + 32), degree, 4),
        CLK32{
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 40), degree, 5),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 48), degree, 6),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 56), degree, 7),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 64), degree, 8),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 72), degree, 9),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 80), degree, 10),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 88), degree, 11),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 96), degree, 12),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 104), degree, 13),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 112), degree, 14),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 120), degree, 15),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 128), degree, 16),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 136), degree, 17),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 144), degree, 18),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 152), degree, 19),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 160), degree, 20),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 168), degree, 21),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 176), degree, 22),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 184), degree, 23),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 192), degree, 24),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 200), degree, 25),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 208), degree, 26),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 216), degree, 27),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 224), degree, 28),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 232), degree, 29),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 240), degree, 30),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 248), degree, 31),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 256), degree, 32),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 264), degree, 33),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 272), degree, 34),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 280), degree, 35),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 288), degree, 36)
        },
        BYTE_FACTOR{
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 296), degree, 37),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 304), degree, 38),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 312), degree, 39),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 320), degree, 40),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 328), degree, 41),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 336), degree, 42),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 344), degree, 43),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 352), degree, 44)
        },
        STEP((Goldilocks::Element *)((uint8_t *)pAddress + 360), degree, 45),
        STEP32((Goldilocks::Element *)((uint8_t *)pAddress + 368), degree, 46),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 376; }
    inline static uint64_t numPols (void) { return 47; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*47*sizeof(Goldilocks::Element); }
};

class RomConstantPols
{
public:
    ConstantPol CONST0;
    ConstantPol CONST1;
    ConstantPol CONST2;
    ConstantPol CONST3;
    ConstantPol CONST4;
    ConstantPol CONST5;
    ConstantPol CONST6;
    ConstantPol CONST7;
    ConstantPol offset;
    ConstantPol inA;
    ConstantPol inB;
    ConstantPol inC;
    ConstantPol inROTL_C;
    ConstantPol inD;
    ConstantPol inE;
    ConstantPol inSR;
    ConstantPol inFREE;
    ConstantPol inCTX;
    ConstantPol inSP;
    ConstantPol inPC;
    ConstantPol inGAS;
    ConstantPol inHASHPOS;
    ConstantPol inSTEP;
    ConstantPol inRR;
    ConstantPol inRCX;
    ConstantPol inCntArith;
    ConstantPol inCntBinary;
    ConstantPol inCntKeccakF;
    ConstantPol inCntMemAlign;
    ConstantPol inCntPaddingPG;
    ConstantPol inCntPoseidonG;
    ConstantPol inCntSha256F;
    ConstantPol incStack;
    ConstantPol binOpcode;
    ConstantPol jmpAddr;
    ConstantPol elseAddr;
    ConstantPol line;
    ConstantPol operations;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    RomConstantPols (void * pAddress, uint64_t degree) :
        CONST0((Goldilocks::Element *)((uint8_t *)pAddress + 376), degree, 47),
        CONST1((Goldilocks::Element *)((uint8_t *)pAddress + 384), degree, 48),
        CONST2((Goldilocks::Element *)((uint8_t *)pAddress + 392), degree, 49),
        CONST3((Goldilocks::Element *)((uint8_t *)pAddress + 400), degree, 50),
        CONST4((Goldilocks::Element *)((uint8_t *)pAddress + 408), degree, 51),
        CONST5((Goldilocks::Element *)((uint8_t *)pAddress + 416), degree, 52),
        CONST6((Goldilocks::Element *)((uint8_t *)pAddress + 424), degree, 53),
        CONST7((Goldilocks::Element *)((uint8_t *)pAddress + 432), degree, 54),
        offset((Goldilocks::Element *)((uint8_t *)pAddress + 440), degree, 55),
        inA((Goldilocks::Element *)((uint8_t *)pAddress + 448), degree, 56),
        inB((Goldilocks::Element *)((uint8_t *)pAddress + 456), degree, 57),
        inC((Goldilocks::Element *)((uint8_t *)pAddress + 464), degree, 58),
        inROTL_C((Goldilocks::Element *)((uint8_t *)pAddress + 472), degree, 59),
        inD((Goldilocks::Element *)((uint8_t *)pAddress + 480), degree, 60),
        inE((Goldilocks::Element *)((uint8_t *)pAddress + 488), degree, 61),
        inSR((Goldilocks::Element *)((uint8_t *)pAddress + 496), degree, 62),
        inFREE((Goldilocks::Element *)((uint8_t *)pAddress + 504), degree, 63),
        inCTX((Goldilocks::Element *)((uint8_t *)pAddress + 512), degree, 64),
        inSP((Goldilocks::Element *)((uint8_t *)pAddress + 520), degree, 65),
        inPC((Goldilocks::Element *)((uint8_t *)pAddress + 528), degree, 66),
        inGAS((Goldilocks::Element *)((uint8_t *)pAddress + 536), degree, 67),
        inHASHPOS((Goldilocks::Element *)((uint8_t *)pAddress + 544), degree, 68),
        inSTEP((Goldilocks::Element *)((uint8_t *)pAddress + 552), degree, 69),
        inRR((Goldilocks::Element *)((uint8_t *)pAddress + 560), degree, 70),
        inRCX((Goldilocks::Element *)((uint8_t *)pAddress + 568), degree, 71),
        inCntArith((Goldilocks::Element *)((uint8_t *)pAddress + 576), degree, 72),
        inCntBinary((Goldilocks::Element *)((uint8_t *)pAddress + 584), degree, 73),
        inCntKeccakF((Goldilocks::Element *)((uint8_t *)pAddress + 592), degree, 74),
        inCntMemAlign((Goldilocks::Element *)((uint8_t *)pAddress + 600), degree, 75),
        inCntPaddingPG((Goldilocks::Element *)((uint8_t *)pAddress + 608), degree, 76),
        inCntPoseidonG((Goldilocks::Element *)((uint8_t *)pAddress + 616), degree, 77),
        inCntSha256F((Goldilocks::Element *)((uint8_t *)pAddress + 624), degree, 78),
        incStack((Goldilocks::Element *)((uint8_t *)pAddress + 632), degree, 79),
        binOpcode((Goldilocks::Element *)((uint8_t *)pAddress + 640), degree, 80),
        jmpAddr((Goldilocks::Element *)((uint8_t *)pAddress + 648), degree, 81),
        elseAddr((Goldilocks::Element *)((uint8_t *)pAddress + 656), degree, 82),
        line((Goldilocks::Element *)((uint8_t *)pAddress + 664), degree, 83),
        operations((Goldilocks::Element *)((uint8_t *)pAddress + 672), degree, 84),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 304; }
    inline static uint64_t numPols (void) { return 38; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*38*sizeof(Goldilocks::Element); }
};

class MemAlignConstantPols
{
public:
    ConstantPol BYTE_C4096;
    ConstantPol FACTOR[8];
    ConstantPol FACTORV[8];
    ConstantPol WR256;
    ConstantPol WR8;
    ConstantPol OFFSET;
    ConstantPol SELM1;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    MemAlignConstantPols (void * pAddress, uint64_t degree) :
        BYTE_C4096((Goldilocks::Element *)((uint8_t *)pAddress + 680), degree, 85),
        FACTOR{
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 688), degree, 86),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 696), degree, 87),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 704), degree, 88),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 712), degree, 89),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 720), degree, 90),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 728), degree, 91),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 736), degree, 92),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 744), degree, 93)
        },
        FACTORV{
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 752), degree, 94),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 760), degree, 95),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 768), degree, 96),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 776), degree, 97),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 784), degree, 98),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 792), degree, 99),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 800), degree, 100),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 808), degree, 101)
        },
        WR256((Goldilocks::Element *)((uint8_t *)pAddress + 816), degree, 102),
        WR8((Goldilocks::Element *)((uint8_t *)pAddress + 824), degree, 103),
        OFFSET((Goldilocks::Element *)((uint8_t *)pAddress + 832), degree, 104),
        SELM1((Goldilocks::Element *)((uint8_t *)pAddress + 840), degree, 105),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 168; }
    inline static uint64_t numPols (void) { return 21; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*21*sizeof(Goldilocks::Element); }
};

class ArithConstantPols
{
public:
    ConstantPol BYTE2_BIT19;
    ConstantPol SEL_BYTE2_BIT19;
    ConstantPol GL_SIGNED_22BITS;
    ConstantPol RANGE_SEL;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    ArithConstantPols (void * pAddress, uint64_t degree) :
        BYTE2_BIT19((Goldilocks::Element *)((uint8_t *)pAddress + 848), degree, 106),
        SEL_BYTE2_BIT19((Goldilocks::Element *)((uint8_t *)pAddress + 856), degree, 107),
        GL_SIGNED_22BITS((Goldilocks::Element *)((uint8_t *)pAddress + 864), degree, 108),
        RANGE_SEL((Goldilocks::Element *)((uint8_t *)pAddress + 872), degree, 109),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 32; }
    inline static uint64_t numPols (void) { return 4; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*4*sizeof(Goldilocks::Element); }
};

class BinaryConstantPols
{
public:
    ConstantPol P_OPCODE;
    ConstantPol P_CIN;
    ConstantPol P_LAST;
    ConstantPol P_USE_CARRY;
    ConstantPol P_C;
    ConstantPol P_COUT;
    ConstantPol FACTOR[8];
private:
    void * _pAddress;
    uint64_t _degree;
public:

    BinaryConstantPols (void * pAddress, uint64_t degree) :
        P_OPCODE((Goldilocks::Element *)((uint8_t *)pAddress + 880), degree, 110),
        P_CIN((Goldilocks::Element *)((uint8_t *)pAddress + 888), degree, 111),
        P_LAST((Goldilocks::Element *)((uint8_t *)pAddress + 896), degree, 112),
        P_USE_CARRY((Goldilocks::Element *)((uint8_t *)pAddress + 904), degree, 113),
        P_C((Goldilocks::Element *)((uint8_t *)pAddress + 912), degree, 114),
        P_COUT((Goldilocks::Element *)((uint8_t *)pAddress + 920), degree, 115),
        FACTOR{
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 928), degree, 116),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 936), degree, 117),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 944), degree, 118),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 952), degree, 119),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 960), degree, 120),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 968), degree, 121),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 976), degree, 122),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 984), degree, 123)
        },
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 112; }
    inline static uint64_t numPols (void) { return 14; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*14*sizeof(Goldilocks::Element); }
};

class PoseidonGConstantPols
{
public:
    ConstantPol LAST;
    ConstantPol LATCH;
    ConstantPol LASTBLOCK;
    ConstantPol PARTIAL;
    ConstantPol C[12];
private:
    void * _pAddress;
    uint64_t _degree;
public:

    PoseidonGConstantPols (void * pAddress, uint64_t degree) :
        LAST((Goldilocks::Element *)((uint8_t *)pAddress + 992), degree, 124),
        LATCH((Goldilocks::Element *)((uint8_t *)pAddress + 1000), degree, 125),
        LASTBLOCK((Goldilocks::Element *)((uint8_t *)pAddress + 1008), degree, 126),
        PARTIAL((Goldilocks::Element *)((uint8_t *)pAddress + 1016), degree, 127),
        C{
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1024), degree, 128),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1032), degree, 129),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1040), degree, 130),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1048), degree, 131),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1056), degree, 132),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1064), degree, 133),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1072), degree, 134),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1080), degree, 135),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1088), degree, 136),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1096), degree, 137),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1104), degree, 138),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1112), degree, 139)
        },
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 128; }
    inline static uint64_t numPols (void) { return 16; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*16*sizeof(Goldilocks::Element); }
};

class PaddingPGConstantPols
{
public:
    ConstantPol F[8];
    ConstantPol lastBlock;
    ConstantPol crValid;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    PaddingPGConstantPols (void * pAddress, uint64_t degree) :
        F{
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1120), degree, 140),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1128), degree, 141),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1136), degree, 142),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1144), degree, 143),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1152), degree, 144),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1160), degree, 145),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1168), degree, 146),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1176), degree, 147)
        },
        lastBlock((Goldilocks::Element *)((uint8_t *)pAddress + 1184), degree, 148),
        crValid((Goldilocks::Element *)((uint8_t *)pAddress + 1192), degree, 149),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 80; }
    inline static uint64_t numPols (void) { return 10; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*10*sizeof(Goldilocks::Element); }
};

class StorageConstantPols
{
public:
    ConstantPol rHash;
    ConstantPol rHashType;
    ConstantPol rLatchGet;
    ConstantPol rLatchSet;
    ConstantPol rClimbRkey;
    ConstantPol rClimbSiblingRkey;
    ConstantPol rClimbSiblingRkeyN;
    ConstantPol rRotateLevel;
    ConstantPol rJmpz;
    ConstantPol rJmp;
    ConstantPol rConst0;
    ConstantPol rConst1;
    ConstantPol rConst2;
    ConstantPol rConst3;
    ConstantPol rAddress;
    ConstantPol rLine;
    ConstantPol rInFree;
    ConstantPol rInNewRoot;
    ConstantPol rInOldRoot;
    ConstantPol rInRkey;
    ConstantPol rInRkeyBit;
    ConstantPol rInSiblingRkey;
    ConstantPol rInSiblingValueHash;
    ConstantPol rInValueLow;
    ConstantPol rInValueHigh;
    ConstantPol rInRotlVh;
    ConstantPol rSetHashLeft;
    ConstantPol rSetHashRight;
    ConstantPol rSetLevel;
    ConstantPol rSetNewRoot;
    ConstantPol rSetOldRoot;
    ConstantPol rSetRkey;
    ConstantPol rSetRkeyBit;
    ConstantPol rSetSiblingRkey;
    ConstantPol rSetSiblingValueHash;
    ConstantPol rSetValueHigh;
    ConstantPol rSetValueLow;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    StorageConstantPols (void * pAddress, uint64_t degree) :
        rHash((Goldilocks::Element *)((uint8_t *)pAddress + 1200), degree, 150),
        rHashType((Goldilocks::Element *)((uint8_t *)pAddress + 1208), degree, 151),
        rLatchGet((Goldilocks::Element *)((uint8_t *)pAddress + 1216), degree, 152),
        rLatchSet((Goldilocks::Element *)((uint8_t *)pAddress + 1224), degree, 153),
        rClimbRkey((Goldilocks::Element *)((uint8_t *)pAddress + 1232), degree, 154),
        rClimbSiblingRkey((Goldilocks::Element *)((uint8_t *)pAddress + 1240), degree, 155),
        rClimbSiblingRkeyN((Goldilocks::Element *)((uint8_t *)pAddress + 1248), degree, 156),
        rRotateLevel((Goldilocks::Element *)((uint8_t *)pAddress + 1256), degree, 157),
        rJmpz((Goldilocks::Element *)((uint8_t *)pAddress + 1264), degree, 158),
        rJmp((Goldilocks::Element *)((uint8_t *)pAddress + 1272), degree, 159),
        rConst0((Goldilocks::Element *)((uint8_t *)pAddress + 1280), degree, 160),
        rConst1((Goldilocks::Element *)((uint8_t *)pAddress + 1288), degree, 161),
        rConst2((Goldilocks::Element *)((uint8_t *)pAddress + 1296), degree, 162),
        rConst3((Goldilocks::Element *)((uint8_t *)pAddress + 1304), degree, 163),
        rAddress((Goldilocks::Element *)((uint8_t *)pAddress + 1312), degree, 164),
        rLine((Goldilocks::Element *)((uint8_t *)pAddress + 1320), degree, 165),
        rInFree((Goldilocks::Element *)((uint8_t *)pAddress + 1328), degree, 166),
        rInNewRoot((Goldilocks::Element *)((uint8_t *)pAddress + 1336), degree, 167),
        rInOldRoot((Goldilocks::Element *)((uint8_t *)pAddress + 1344), degree, 168),
        rInRkey((Goldilocks::Element *)((uint8_t *)pAddress + 1352), degree, 169),
        rInRkeyBit((Goldilocks::Element *)((uint8_t *)pAddress + 1360), degree, 170),
        rInSiblingRkey((Goldilocks::Element *)((uint8_t *)pAddress + 1368), degree, 171),
        rInSiblingValueHash((Goldilocks::Element *)((uint8_t *)pAddress + 1376), degree, 172),
        rInValueLow((Goldilocks::Element *)((uint8_t *)pAddress + 1384), degree, 173),
        rInValueHigh((Goldilocks::Element *)((uint8_t *)pAddress + 1392), degree, 174),
        rInRotlVh((Goldilocks::Element *)((uint8_t *)pAddress + 1400), degree, 175),
        rSetHashLeft((Goldilocks::Element *)((uint8_t *)pAddress + 1408), degree, 176),
        rSetHashRight((Goldilocks::Element *)((uint8_t *)pAddress + 1416), degree, 177),
        rSetLevel((Goldilocks::Element *)((uint8_t *)pAddress + 1424), degree, 178),
        rSetNewRoot((Goldilocks::Element *)((uint8_t *)pAddress + 1432), degree, 179),
        rSetOldRoot((Goldilocks::Element *)((uint8_t *)pAddress + 1440), degree, 180),
        rSetRkey((Goldilocks::Element *)((uint8_t *)pAddress + 1448), degree, 181),
        rSetRkeyBit((Goldilocks::Element *)((uint8_t *)pAddress + 1456), degree, 182),
        rSetSiblingRkey((Goldilocks::Element *)((uint8_t *)pAddress + 1464), degree, 183),
        rSetSiblingValueHash((Goldilocks::Element *)((uint8_t *)pAddress + 1472), degree, 184),
        rSetValueHigh((Goldilocks::Element *)((uint8_t *)pAddress + 1480), degree, 185),
        rSetValueLow((Goldilocks::Element *)((uint8_t *)pAddress + 1488), degree, 186),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 296; }
    inline static uint64_t numPols (void) { return 37; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*37*sizeof(Goldilocks::Element); }
};

class KeccakFConstantPols
{
public:
    ConstantPol ConnA;
    ConstantPol ConnB;
    ConstantPol ConnC;
    ConstantPol GateType;
    ConstantPol kGateType;
    ConstantPol kA;
    ConstantPol kB;
    ConstantPol kC;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    KeccakFConstantPols (void * pAddress, uint64_t degree) :
        ConnA((Goldilocks::Element *)((uint8_t *)pAddress + 1496), degree, 187),
        ConnB((Goldilocks::Element *)((uint8_t *)pAddress + 1504), degree, 188),
        ConnC((Goldilocks::Element *)((uint8_t *)pAddress + 1512), degree, 189),
        GateType((Goldilocks::Element *)((uint8_t *)pAddress + 1520), degree, 190),
        kGateType((Goldilocks::Element *)((uint8_t *)pAddress + 1528), degree, 191),
        kA((Goldilocks::Element *)((uint8_t *)pAddress + 1536), degree, 192),
        kB((Goldilocks::Element *)((uint8_t *)pAddress + 1544), degree, 193),
        kC((Goldilocks::Element *)((uint8_t *)pAddress + 1552), degree, 194),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 64; }
    inline static uint64_t numPols (void) { return 8; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*8*sizeof(Goldilocks::Element); }
};

class Bits2FieldConstantPols
{
public:
    ConstantPol FieldLatch;
    ConstantPol Factor;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    Bits2FieldConstantPols (void * pAddress, uint64_t degree) :
        FieldLatch((Goldilocks::Element *)((uint8_t *)pAddress + 1560), degree, 195),
        Factor((Goldilocks::Element *)((uint8_t *)pAddress + 1568), degree, 196),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 16; }
    inline static uint64_t numPols (void) { return 2; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*2*sizeof(Goldilocks::Element); }
};

class PaddingKKBitConstantPols
{
public:
    ConstantPol r8Id;
    ConstantPol sOutId;
    ConstantPol latchR8;
    ConstantPol Fr8;
    ConstantPol rBitValid;
    ConstantPol latchSOut;
    ConstantPol FSOut0;
    ConstantPol FSOut1;
    ConstantPol FSOut2;
    ConstantPol FSOut3;
    ConstantPol FSOut4;
    ConstantPol FSOut5;
    ConstantPol FSOut6;
    ConstantPol FSOut7;
    ConstantPol ConnSOutBit;
    ConstantPol ConnSInBit;
    ConstantPol ConnBits2FieldBit;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    PaddingKKBitConstantPols (void * pAddress, uint64_t degree) :
        r8Id((Goldilocks::Element *)((uint8_t *)pAddress + 1576), degree, 197),
        sOutId((Goldilocks::Element *)((uint8_t *)pAddress + 1584), degree, 198),
        latchR8((Goldilocks::Element *)((uint8_t *)pAddress + 1592), degree, 199),
        Fr8((Goldilocks::Element *)((uint8_t *)pAddress + 1600), degree, 200),
        rBitValid((Goldilocks::Element *)((uint8_t *)pAddress + 1608), degree, 201),
        latchSOut((Goldilocks::Element *)((uint8_t *)pAddress + 1616), degree, 202),
        FSOut0((Goldilocks::Element *)((uint8_t *)pAddress + 1624), degree, 203),
        FSOut1((Goldilocks::Element *)((uint8_t *)pAddress + 1632), degree, 204),
        FSOut2((Goldilocks::Element *)((uint8_t *)pAddress + 1640), degree, 205),
        FSOut3((Goldilocks::Element *)((uint8_t *)pAddress + 1648), degree, 206),
        FSOut4((Goldilocks::Element *)((uint8_t *)pAddress + 1656), degree, 207),
        FSOut5((Goldilocks::Element *)((uint8_t *)pAddress + 1664), degree, 208),
        FSOut6((Goldilocks::Element *)((uint8_t *)pAddress + 1672), degree, 209),
        FSOut7((Goldilocks::Element *)((uint8_t *)pAddress + 1680), degree, 210),
        ConnSOutBit((Goldilocks::Element *)((uint8_t *)pAddress + 1688), degree, 211),
        ConnSInBit((Goldilocks::Element *)((uint8_t *)pAddress + 1696), degree, 212),
        ConnBits2FieldBit((Goldilocks::Element *)((uint8_t *)pAddress + 1704), degree, 213),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 136; }
    inline static uint64_t numPols (void) { return 17; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*17*sizeof(Goldilocks::Element); }
};

class PaddingKKConstantPols
{
public:
    ConstantPol r8Id;
    ConstantPol lastBlock;
    ConstantPol lastBlockLatch;
    ConstantPol r8valid;
    ConstantPol sOutId;
    ConstantPol forceLastHash;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    PaddingKKConstantPols (void * pAddress, uint64_t degree) :
        r8Id((Goldilocks::Element *)((uint8_t *)pAddress + 1712), degree, 214),
        lastBlock((Goldilocks::Element *)((uint8_t *)pAddress + 1720), degree, 215),
        lastBlockLatch((Goldilocks::Element *)((uint8_t *)pAddress + 1728), degree, 216),
        r8valid((Goldilocks::Element *)((uint8_t *)pAddress + 1736), degree, 217),
        sOutId((Goldilocks::Element *)((uint8_t *)pAddress + 1744), degree, 218),
        forceLastHash((Goldilocks::Element *)((uint8_t *)pAddress + 1752), degree, 219),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 48; }
    inline static uint64_t numPols (void) { return 6; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*6*sizeof(Goldilocks::Element); }
};

class Sha256FConstantPols
{
public:
    ConstantPol kGateType;
    ConstantPol kA;
    ConstantPol kB;
    ConstantPol kC;
    ConstantPol kOut;
    ConstantPol kCarryOut;
    ConstantPol Conn[4];
    ConstantPol GATE_TYPE;
    ConstantPol CARRY_ENABLED;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    Sha256FConstantPols (void * pAddress, uint64_t degree) :
        kGateType((Goldilocks::Element *)((uint8_t *)pAddress + 1760), degree, 220),
        kA((Goldilocks::Element *)((uint8_t *)pAddress + 1768), degree, 221),
        kB((Goldilocks::Element *)((uint8_t *)pAddress + 1776), degree, 222),
        kC((Goldilocks::Element *)((uint8_t *)pAddress + 1784), degree, 223),
        kOut((Goldilocks::Element *)((uint8_t *)pAddress + 1792), degree, 224),
        kCarryOut((Goldilocks::Element *)((uint8_t *)pAddress + 1800), degree, 225),
        Conn{
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1808), degree, 226),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1816), degree, 227),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1824), degree, 228),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1832), degree, 229)
        },
        GATE_TYPE((Goldilocks::Element *)((uint8_t *)pAddress + 1840), degree, 230),
        CARRY_ENABLED((Goldilocks::Element *)((uint8_t *)pAddress + 1848), degree, 231),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 96; }
    inline static uint64_t numPols (void) { return 12; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*12*sizeof(Goldilocks::Element); }
};

class Bits2FieldSha256ConstantPols
{
public:
    ConstantPol FieldLatch;
    ConstantPol Factor;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    Bits2FieldSha256ConstantPols (void * pAddress, uint64_t degree) :
        FieldLatch((Goldilocks::Element *)((uint8_t *)pAddress + 1856), degree, 232),
        Factor((Goldilocks::Element *)((uint8_t *)pAddress + 1864), degree, 233),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 16; }
    inline static uint64_t numPols (void) { return 2; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*2*sizeof(Goldilocks::Element); }
};

class PaddingSha256BitConstantPols
{
public:
    ConstantPol r8Id;
    ConstantPol sOutId;
    ConstantPol latchR8;
    ConstantPol Fr8;
    ConstantPol latchSOut;
    ConstantPol FSOut0;
    ConstantPol FSOut1;
    ConstantPol FSOut2;
    ConstantPol FSOut3;
    ConstantPol FSOut4;
    ConstantPol FSOut5;
    ConstantPol FSOut6;
    ConstantPol FSOut7;
    ConstantPol HIn;
    ConstantPol DoConnect;
    ConstantPol ConnS1;
    ConstantPol ConnS2;
    ConstantPol ConnBits2FieldBit;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    PaddingSha256BitConstantPols (void * pAddress, uint64_t degree) :
        r8Id((Goldilocks::Element *)((uint8_t *)pAddress + 1872), degree, 234),
        sOutId((Goldilocks::Element *)((uint8_t *)pAddress + 1880), degree, 235),
        latchR8((Goldilocks::Element *)((uint8_t *)pAddress + 1888), degree, 236),
        Fr8((Goldilocks::Element *)((uint8_t *)pAddress + 1896), degree, 237),
        latchSOut((Goldilocks::Element *)((uint8_t *)pAddress + 1904), degree, 238),
        FSOut0((Goldilocks::Element *)((uint8_t *)pAddress + 1912), degree, 239),
        FSOut1((Goldilocks::Element *)((uint8_t *)pAddress + 1920), degree, 240),
        FSOut2((Goldilocks::Element *)((uint8_t *)pAddress + 1928), degree, 241),
        FSOut3((Goldilocks::Element *)((uint8_t *)pAddress + 1936), degree, 242),
        FSOut4((Goldilocks::Element *)((uint8_t *)pAddress + 1944), degree, 243),
        FSOut5((Goldilocks::Element *)((uint8_t *)pAddress + 1952), degree, 244),
        FSOut6((Goldilocks::Element *)((uint8_t *)pAddress + 1960), degree, 245),
        FSOut7((Goldilocks::Element *)((uint8_t *)pAddress + 1968), degree, 246),
        HIn((Goldilocks::Element *)((uint8_t *)pAddress + 1976), degree, 247),
        DoConnect((Goldilocks::Element *)((uint8_t *)pAddress + 1984), degree, 248),
        ConnS1((Goldilocks::Element *)((uint8_t *)pAddress + 1992), degree, 249),
        ConnS2((Goldilocks::Element *)((uint8_t *)pAddress + 2000), degree, 250),
        ConnBits2FieldBit((Goldilocks::Element *)((uint8_t *)pAddress + 2008), degree, 251),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 144; }
    inline static uint64_t numPols (void) { return 18; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*18*sizeof(Goldilocks::Element); }
};

class PaddingSha256ConstantPols
{
public:
    ConstantPol r8Id;
    ConstantPol lastBlock;
    ConstantPol lastBlockLatch;
    ConstantPol r8valid;
    ConstantPol PrevLengthSection;
    ConstantPol LengthWeight;
    ConstantPol sOutId;
    ConstantPol forceLastHash;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    PaddingSha256ConstantPols (void * pAddress, uint64_t degree) :
        r8Id((Goldilocks::Element *)((uint8_t *)pAddress + 2016), degree, 252),
        lastBlock((Goldilocks::Element *)((uint8_t *)pAddress + 2024), degree, 253),
        lastBlockLatch((Goldilocks::Element *)((uint8_t *)pAddress + 2032), degree, 254),
        r8valid((Goldilocks::Element *)((uint8_t *)pAddress + 2040), degree, 255),
        PrevLengthSection((Goldilocks::Element *)((uint8_t *)pAddress + 2048), degree, 256),
        LengthWeight((Goldilocks::Element *)((uint8_t *)pAddress + 2056), degree, 257),
        sOutId((Goldilocks::Element *)((uint8_t *)pAddress + 2064), degree, 258),
        forceLastHash((Goldilocks::Element *)((uint8_t *)pAddress + 2072), degree, 259),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t pilSize (void) { return 64; }
    inline static uint64_t numPols (void) { return 8; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*8*sizeof(Goldilocks::Element); }
};

class ConstantPols
{
public:
    GlobalConstantPols Global;
    RomConstantPols Rom;
    MemAlignConstantPols MemAlign;
    ArithConstantPols Arith;
    BinaryConstantPols Binary;
    PoseidonGConstantPols PoseidonG;
    PaddingPGConstantPols PaddingPG;
    StorageConstantPols Storage;
    KeccakFConstantPols KeccakF;
    Bits2FieldConstantPols Bits2Field;
    PaddingKKBitConstantPols PaddingKKBit;
    PaddingKKConstantPols PaddingKK;
    Sha256FConstantPols Sha256F;
    Bits2FieldSha256ConstantPols Bits2FieldSha256;
    PaddingSha256BitConstantPols PaddingSha256Bit;
    PaddingSha256ConstantPols PaddingSha256;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    ConstantPols (void * pAddress, uint64_t degree) :
        Global(pAddress, degree),
        Rom(pAddress, degree),
        MemAlign(pAddress, degree),
        Arith(pAddress, degree),
        Binary(pAddress, degree),
        PoseidonG(pAddress, degree),
        PaddingPG(pAddress, degree),
        Storage(pAddress, degree),
        KeccakF(pAddress, degree),
        Bits2Field(pAddress, degree),
        PaddingKKBit(pAddress, degree),
        PaddingKK(pAddress, degree),
        Sha256F(pAddress, degree),
        Bits2FieldSha256(pAddress, degree),
        PaddingSha256Bit(pAddress, degree),
        PaddingSha256(pAddress, degree),
        _pAddress(pAddress),
        _degree(degree) {}

    inline static uint64_t pilSize (void) { return 17448304640; }
    inline static uint64_t pilDegree (void) { return 8388608; }
    inline static uint64_t numPols (void) { return 260; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*260*sizeof(Goldilocks::Element); }

    inline Goldilocks::Element &getElement (uint64_t pol, uint64_t evaluation)
    {
        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * numPols()];
    }
};

} // namespace

#endif // CONSTANT_POLS_HPP_fork_7

