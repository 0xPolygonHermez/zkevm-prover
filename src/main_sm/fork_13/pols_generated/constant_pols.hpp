#ifndef CONSTANT_POLS_HPP_fork_13
#define CONSTANT_POLS_HPP_fork_13

#include <cstdint>
#include "goldilocks_base_field.hpp"

namespace fork_13
{

class ConstantPol
{
private:
    Goldilocks::Element * _pAddress;
    uint64_t _degree;
    uint64_t _index;
public:
    ConstantPol(Goldilocks::Element * pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index) {};
    inline Goldilocks::Element & operator[](uint64_t i) { return _pAddress[i*235]; };
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
    ConstantPol inFREE0;
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
    ConstantPol arithEq;
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
        inFREE0((Goldilocks::Element *)((uint8_t *)pAddress + 512), degree, 64),
        inCTX((Goldilocks::Element *)((uint8_t *)pAddress + 520), degree, 65),
        inSP((Goldilocks::Element *)((uint8_t *)pAddress + 528), degree, 66),
        inPC((Goldilocks::Element *)((uint8_t *)pAddress + 536), degree, 67),
        inGAS((Goldilocks::Element *)((uint8_t *)pAddress + 544), degree, 68),
        inHASHPOS((Goldilocks::Element *)((uint8_t *)pAddress + 552), degree, 69),
        inSTEP((Goldilocks::Element *)((uint8_t *)pAddress + 560), degree, 70),
        inRR((Goldilocks::Element *)((uint8_t *)pAddress + 568), degree, 71),
        inRCX((Goldilocks::Element *)((uint8_t *)pAddress + 576), degree, 72),
        inCntArith((Goldilocks::Element *)((uint8_t *)pAddress + 584), degree, 73),
        inCntBinary((Goldilocks::Element *)((uint8_t *)pAddress + 592), degree, 74),
        inCntKeccakF((Goldilocks::Element *)((uint8_t *)pAddress + 600), degree, 75),
        inCntMemAlign((Goldilocks::Element *)((uint8_t *)pAddress + 608), degree, 76),
        inCntPaddingPG((Goldilocks::Element *)((uint8_t *)pAddress + 616), degree, 77),
        inCntPoseidonG((Goldilocks::Element *)((uint8_t *)pAddress + 624), degree, 78),
        inCntSha256F((Goldilocks::Element *)((uint8_t *)pAddress + 632), degree, 79),
        incStack((Goldilocks::Element *)((uint8_t *)pAddress + 640), degree, 80),
        binOpcode((Goldilocks::Element *)((uint8_t *)pAddress + 648), degree, 81),
        jmpAddr((Goldilocks::Element *)((uint8_t *)pAddress + 656), degree, 82),
        elseAddr((Goldilocks::Element *)((uint8_t *)pAddress + 664), degree, 83),
        arithEq((Goldilocks::Element *)((uint8_t *)pAddress + 672), degree, 84),
        line((Goldilocks::Element *)((uint8_t *)pAddress + 680), degree, 85),
        operations((Goldilocks::Element *)((uint8_t *)pAddress + 688), degree, 86),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 40; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*40*sizeof(Goldilocks::Element); }
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
        BYTE_C4096((Goldilocks::Element *)((uint8_t *)pAddress + 696), degree, 87),
        FACTOR{
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 704), degree, 88),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 712), degree, 89),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 720), degree, 90),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 728), degree, 91),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 736), degree, 92),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 744), degree, 93),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 752), degree, 94),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 760), degree, 95)
        },
        FACTORV{
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 768), degree, 96),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 776), degree, 97),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 784), degree, 98),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 792), degree, 99),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 800), degree, 100),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 808), degree, 101),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 816), degree, 102),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 824), degree, 103)
        },
        WR256((Goldilocks::Element *)((uint8_t *)pAddress + 832), degree, 104),
        WR8((Goldilocks::Element *)((uint8_t *)pAddress + 840), degree, 105),
        OFFSET((Goldilocks::Element *)((uint8_t *)pAddress + 848), degree, 106),
        SELM1((Goldilocks::Element *)((uint8_t *)pAddress + 856), degree, 107),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 21; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*21*sizeof(Goldilocks::Element); }
};

class ArithConstantPols
{
public:
    ConstantPol BYTE2_BIT21;
    ConstantPol SEL_BYTE2_BIT21;
    ConstantPol GL_SIGNED_22BITS;
    ConstantPol RANGE_SEL;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    ArithConstantPols (void * pAddress, uint64_t degree) :
        BYTE2_BIT21((Goldilocks::Element *)((uint8_t *)pAddress + 864), degree, 108),
        SEL_BYTE2_BIT21((Goldilocks::Element *)((uint8_t *)pAddress + 872), degree, 109),
        GL_SIGNED_22BITS((Goldilocks::Element *)((uint8_t *)pAddress + 880), degree, 110),
        RANGE_SEL((Goldilocks::Element *)((uint8_t *)pAddress + 888), degree, 111),
        _pAddress(pAddress),
        _degree(degree) {};

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
    ConstantPol P_C;
    ConstantPol P_FLAGS;
    ConstantPol FACTOR[8];
private:
    void * _pAddress;
    uint64_t _degree;
public:

    BinaryConstantPols (void * pAddress, uint64_t degree) :
        P_OPCODE((Goldilocks::Element *)((uint8_t *)pAddress + 896), degree, 112),
        P_CIN((Goldilocks::Element *)((uint8_t *)pAddress + 904), degree, 113),
        P_LAST((Goldilocks::Element *)((uint8_t *)pAddress + 912), degree, 114),
        P_C((Goldilocks::Element *)((uint8_t *)pAddress + 920), degree, 115),
        P_FLAGS((Goldilocks::Element *)((uint8_t *)pAddress + 928), degree, 116),
        FACTOR{
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 936), degree, 117),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 944), degree, 118),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 952), degree, 119),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 960), degree, 120),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 968), degree, 121),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 976), degree, 122),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 984), degree, 123),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 992), degree, 124)
        },
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 13; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*13*sizeof(Goldilocks::Element); }
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
        LAST((Goldilocks::Element *)((uint8_t *)pAddress + 1000), degree, 125),
        LATCH((Goldilocks::Element *)((uint8_t *)pAddress + 1008), degree, 126),
        LASTBLOCK((Goldilocks::Element *)((uint8_t *)pAddress + 1016), degree, 127),
        PARTIAL((Goldilocks::Element *)((uint8_t *)pAddress + 1024), degree, 128),
        C{
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
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1112), degree, 139),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1120), degree, 140)
        },
        _pAddress(pAddress),
        _degree(degree) {};

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
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1128), degree, 141),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1136), degree, 142),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1144), degree, 143),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1152), degree, 144),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1160), degree, 145),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1168), degree, 146),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1176), degree, 147),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1184), degree, 148)
        },
        lastBlock((Goldilocks::Element *)((uint8_t *)pAddress + 1192), degree, 149),
        crValid((Goldilocks::Element *)((uint8_t *)pAddress + 1200), degree, 150),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 10; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*10*sizeof(Goldilocks::Element); }
};

class ClimbKeyConstantPols
{
public:
    ConstantPol FACTOR;
    ConstantPol T_CLKEYSEL;
    ConstantPol T_LEVEL;
    ConstantPol T_CHUNK_VALUE;
    ConstantPol T_CARRYLT_IN;
    ConstantPol T_CARRYLT_OUT;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    ClimbKeyConstantPols (void * pAddress, uint64_t degree) :
        FACTOR((Goldilocks::Element *)((uint8_t *)pAddress + 1208), degree, 151),
        T_CLKEYSEL((Goldilocks::Element *)((uint8_t *)pAddress + 1216), degree, 152),
        T_LEVEL((Goldilocks::Element *)((uint8_t *)pAddress + 1224), degree, 153),
        T_CHUNK_VALUE((Goldilocks::Element *)((uint8_t *)pAddress + 1232), degree, 154),
        T_CARRYLT_IN((Goldilocks::Element *)((uint8_t *)pAddress + 1240), degree, 155),
        T_CARRYLT_OUT((Goldilocks::Element *)((uint8_t *)pAddress + 1248), degree, 156),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 6; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*6*sizeof(Goldilocks::Element); }
};

class StorageConstantPols
{
public:
    ConstantPol OPERATION;
    ConstantPol CONST0;
    ConstantPol JMP_ADDRESS;
    ConstantPol LINE;
    ConstantPol IN_SIBLING_RKEY;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    StorageConstantPols (void * pAddress, uint64_t degree) :
        OPERATION((Goldilocks::Element *)((uint8_t *)pAddress + 1256), degree, 157),
        CONST0((Goldilocks::Element *)((uint8_t *)pAddress + 1264), degree, 158),
        JMP_ADDRESS((Goldilocks::Element *)((uint8_t *)pAddress + 1272), degree, 159),
        LINE((Goldilocks::Element *)((uint8_t *)pAddress + 1280), degree, 160),
        IN_SIBLING_RKEY((Goldilocks::Element *)((uint8_t *)pAddress + 1288), degree, 161),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 5; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*5*sizeof(Goldilocks::Element); }
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
        ConnA((Goldilocks::Element *)((uint8_t *)pAddress + 1296), degree, 162),
        ConnB((Goldilocks::Element *)((uint8_t *)pAddress + 1304), degree, 163),
        ConnC((Goldilocks::Element *)((uint8_t *)pAddress + 1312), degree, 164),
        GateType((Goldilocks::Element *)((uint8_t *)pAddress + 1320), degree, 165),
        kGateType((Goldilocks::Element *)((uint8_t *)pAddress + 1328), degree, 166),
        kA((Goldilocks::Element *)((uint8_t *)pAddress + 1336), degree, 167),
        kB((Goldilocks::Element *)((uint8_t *)pAddress + 1344), degree, 168),
        kC((Goldilocks::Element *)((uint8_t *)pAddress + 1352), degree, 169),
        _pAddress(pAddress),
        _degree(degree) {};

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
        FieldLatch((Goldilocks::Element *)((uint8_t *)pAddress + 1360), degree, 170),
        Factor((Goldilocks::Element *)((uint8_t *)pAddress + 1368), degree, 171),
        _pAddress(pAddress),
        _degree(degree) {};

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
        r8Id((Goldilocks::Element *)((uint8_t *)pAddress + 1376), degree, 172),
        sOutId((Goldilocks::Element *)((uint8_t *)pAddress + 1384), degree, 173),
        latchR8((Goldilocks::Element *)((uint8_t *)pAddress + 1392), degree, 174),
        Fr8((Goldilocks::Element *)((uint8_t *)pAddress + 1400), degree, 175),
        rBitValid((Goldilocks::Element *)((uint8_t *)pAddress + 1408), degree, 176),
        latchSOut((Goldilocks::Element *)((uint8_t *)pAddress + 1416), degree, 177),
        FSOut0((Goldilocks::Element *)((uint8_t *)pAddress + 1424), degree, 178),
        FSOut1((Goldilocks::Element *)((uint8_t *)pAddress + 1432), degree, 179),
        FSOut2((Goldilocks::Element *)((uint8_t *)pAddress + 1440), degree, 180),
        FSOut3((Goldilocks::Element *)((uint8_t *)pAddress + 1448), degree, 181),
        FSOut4((Goldilocks::Element *)((uint8_t *)pAddress + 1456), degree, 182),
        FSOut5((Goldilocks::Element *)((uint8_t *)pAddress + 1464), degree, 183),
        FSOut6((Goldilocks::Element *)((uint8_t *)pAddress + 1472), degree, 184),
        FSOut7((Goldilocks::Element *)((uint8_t *)pAddress + 1480), degree, 185),
        ConnSOutBit((Goldilocks::Element *)((uint8_t *)pAddress + 1488), degree, 186),
        ConnSInBit((Goldilocks::Element *)((uint8_t *)pAddress + 1496), degree, 187),
        ConnBits2FieldBit((Goldilocks::Element *)((uint8_t *)pAddress + 1504), degree, 188),
        _pAddress(pAddress),
        _degree(degree) {};

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
        r8Id((Goldilocks::Element *)((uint8_t *)pAddress + 1512), degree, 189),
        lastBlock((Goldilocks::Element *)((uint8_t *)pAddress + 1520), degree, 190),
        lastBlockLatch((Goldilocks::Element *)((uint8_t *)pAddress + 1528), degree, 191),
        r8valid((Goldilocks::Element *)((uint8_t *)pAddress + 1536), degree, 192),
        sOutId((Goldilocks::Element *)((uint8_t *)pAddress + 1544), degree, 193),
        forceLastHash((Goldilocks::Element *)((uint8_t *)pAddress + 1552), degree, 194),
        _pAddress(pAddress),
        _degree(degree) {};

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
        kGateType((Goldilocks::Element *)((uint8_t *)pAddress + 1560), degree, 195),
        kA((Goldilocks::Element *)((uint8_t *)pAddress + 1568), degree, 196),
        kB((Goldilocks::Element *)((uint8_t *)pAddress + 1576), degree, 197),
        kC((Goldilocks::Element *)((uint8_t *)pAddress + 1584), degree, 198),
        kOut((Goldilocks::Element *)((uint8_t *)pAddress + 1592), degree, 199),
        kCarryOut((Goldilocks::Element *)((uint8_t *)pAddress + 1600), degree, 200),
        Conn{
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1608), degree, 201),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1616), degree, 202),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1624), degree, 203),
            ConstantPol((Goldilocks::Element *)((uint8_t *)pAddress + 1632), degree, 204)
        },
        GATE_TYPE((Goldilocks::Element *)((uint8_t *)pAddress + 1640), degree, 205),
        CARRY_ENABLED((Goldilocks::Element *)((uint8_t *)pAddress + 1648), degree, 206),
        _pAddress(pAddress),
        _degree(degree) {};

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
        FieldLatch((Goldilocks::Element *)((uint8_t *)pAddress + 1656), degree, 207),
        Factor((Goldilocks::Element *)((uint8_t *)pAddress + 1664), degree, 208),
        _pAddress(pAddress),
        _degree(degree) {};

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
        r8Id((Goldilocks::Element *)((uint8_t *)pAddress + 1672), degree, 209),
        sOutId((Goldilocks::Element *)((uint8_t *)pAddress + 1680), degree, 210),
        latchR8((Goldilocks::Element *)((uint8_t *)pAddress + 1688), degree, 211),
        Fr8((Goldilocks::Element *)((uint8_t *)pAddress + 1696), degree, 212),
        latchSOut((Goldilocks::Element *)((uint8_t *)pAddress + 1704), degree, 213),
        FSOut0((Goldilocks::Element *)((uint8_t *)pAddress + 1712), degree, 214),
        FSOut1((Goldilocks::Element *)((uint8_t *)pAddress + 1720), degree, 215),
        FSOut2((Goldilocks::Element *)((uint8_t *)pAddress + 1728), degree, 216),
        FSOut3((Goldilocks::Element *)((uint8_t *)pAddress + 1736), degree, 217),
        FSOut4((Goldilocks::Element *)((uint8_t *)pAddress + 1744), degree, 218),
        FSOut5((Goldilocks::Element *)((uint8_t *)pAddress + 1752), degree, 219),
        FSOut6((Goldilocks::Element *)((uint8_t *)pAddress + 1760), degree, 220),
        FSOut7((Goldilocks::Element *)((uint8_t *)pAddress + 1768), degree, 221),
        HIn((Goldilocks::Element *)((uint8_t *)pAddress + 1776), degree, 222),
        DoConnect((Goldilocks::Element *)((uint8_t *)pAddress + 1784), degree, 223),
        ConnS1((Goldilocks::Element *)((uint8_t *)pAddress + 1792), degree, 224),
        ConnS2((Goldilocks::Element *)((uint8_t *)pAddress + 1800), degree, 225),
        ConnBits2FieldBit((Goldilocks::Element *)((uint8_t *)pAddress + 1808), degree, 226),
        _pAddress(pAddress),
        _degree(degree) {};

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
        r8Id((Goldilocks::Element *)((uint8_t *)pAddress + 1816), degree, 227),
        lastBlock((Goldilocks::Element *)((uint8_t *)pAddress + 1824), degree, 228),
        lastBlockLatch((Goldilocks::Element *)((uint8_t *)pAddress + 1832), degree, 229),
        r8valid((Goldilocks::Element *)((uint8_t *)pAddress + 1840), degree, 230),
        PrevLengthSection((Goldilocks::Element *)((uint8_t *)pAddress + 1848), degree, 231),
        LengthWeight((Goldilocks::Element *)((uint8_t *)pAddress + 1856), degree, 232),
        sOutId((Goldilocks::Element *)((uint8_t *)pAddress + 1864), degree, 233),
        forceLastHash((Goldilocks::Element *)((uint8_t *)pAddress + 1872), degree, 234),
        _pAddress(pAddress),
        _degree(degree) {};

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
    ClimbKeyConstantPols ClimbKey;
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
        ClimbKey(pAddress, degree),
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

    inline static uint64_t numPols (void) { return 235; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*235*sizeof(Goldilocks::Element); }

    inline Goldilocks::Element &getElement (uint64_t pol, uint64_t evaluation)
    {
        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * numPols()];
    }
};

inline const char * address2ConstantPolName (uint64_t address)
{
    if ((address >= 0) && (address <= 7)) return "Global.L1";
    if ((address >= 8) && (address <= 15)) return "Global.LLAST";
    if ((address >= 16) && (address <= 23)) return "Global.BYTE";
    if ((address >= 24) && (address <= 31)) return "Global.BYTE_2A";
    if ((address >= 32) && (address <= 39)) return "Global.BYTE2";
    if ((address >= 40) && (address <= 47)) return "Global.CLK32[0]";
    if ((address >= 48) && (address <= 55)) return "Global.CLK32[1]";
    if ((address >= 56) && (address <= 63)) return "Global.CLK32[2]";
    if ((address >= 64) && (address <= 71)) return "Global.CLK32[3]";
    if ((address >= 72) && (address <= 79)) return "Global.CLK32[4]";
    if ((address >= 80) && (address <= 87)) return "Global.CLK32[5]";
    if ((address >= 88) && (address <= 95)) return "Global.CLK32[6]";
    if ((address >= 96) && (address <= 103)) return "Global.CLK32[7]";
    if ((address >= 104) && (address <= 111)) return "Global.CLK32[8]";
    if ((address >= 112) && (address <= 119)) return "Global.CLK32[9]";
    if ((address >= 120) && (address <= 127)) return "Global.CLK32[10]";
    if ((address >= 128) && (address <= 135)) return "Global.CLK32[11]";
    if ((address >= 136) && (address <= 143)) return "Global.CLK32[12]";
    if ((address >= 144) && (address <= 151)) return "Global.CLK32[13]";
    if ((address >= 152) && (address <= 159)) return "Global.CLK32[14]";
    if ((address >= 160) && (address <= 167)) return "Global.CLK32[15]";
    if ((address >= 168) && (address <= 175)) return "Global.CLK32[16]";
    if ((address >= 176) && (address <= 183)) return "Global.CLK32[17]";
    if ((address >= 184) && (address <= 191)) return "Global.CLK32[18]";
    if ((address >= 192) && (address <= 199)) return "Global.CLK32[19]";
    if ((address >= 200) && (address <= 207)) return "Global.CLK32[20]";
    if ((address >= 208) && (address <= 215)) return "Global.CLK32[21]";
    if ((address >= 216) && (address <= 223)) return "Global.CLK32[22]";
    if ((address >= 224) && (address <= 231)) return "Global.CLK32[23]";
    if ((address >= 232) && (address <= 239)) return "Global.CLK32[24]";
    if ((address >= 240) && (address <= 247)) return "Global.CLK32[25]";
    if ((address >= 248) && (address <= 255)) return "Global.CLK32[26]";
    if ((address >= 256) && (address <= 263)) return "Global.CLK32[27]";
    if ((address >= 264) && (address <= 271)) return "Global.CLK32[28]";
    if ((address >= 272) && (address <= 279)) return "Global.CLK32[29]";
    if ((address >= 280) && (address <= 287)) return "Global.CLK32[30]";
    if ((address >= 288) && (address <= 295)) return "Global.CLK32[31]";
    if ((address >= 296) && (address <= 303)) return "Global.BYTE_FACTOR[0]";
    if ((address >= 304) && (address <= 311)) return "Global.BYTE_FACTOR[1]";
    if ((address >= 312) && (address <= 319)) return "Global.BYTE_FACTOR[2]";
    if ((address >= 320) && (address <= 327)) return "Global.BYTE_FACTOR[3]";
    if ((address >= 328) && (address <= 335)) return "Global.BYTE_FACTOR[4]";
    if ((address >= 336) && (address <= 343)) return "Global.BYTE_FACTOR[5]";
    if ((address >= 344) && (address <= 351)) return "Global.BYTE_FACTOR[6]";
    if ((address >= 352) && (address <= 359)) return "Global.BYTE_FACTOR[7]";
    if ((address >= 360) && (address <= 367)) return "Global.STEP";
    if ((address >= 368) && (address <= 375)) return "Global.STEP32";
    if ((address >= 376) && (address <= 383)) return "Rom.CONST0";
    if ((address >= 384) && (address <= 391)) return "Rom.CONST1";
    if ((address >= 392) && (address <= 399)) return "Rom.CONST2";
    if ((address >= 400) && (address <= 407)) return "Rom.CONST3";
    if ((address >= 408) && (address <= 415)) return "Rom.CONST4";
    if ((address >= 416) && (address <= 423)) return "Rom.CONST5";
    if ((address >= 424) && (address <= 431)) return "Rom.CONST6";
    if ((address >= 432) && (address <= 439)) return "Rom.CONST7";
    if ((address >= 440) && (address <= 447)) return "Rom.offset";
    if ((address >= 448) && (address <= 455)) return "Rom.inA";
    if ((address >= 456) && (address <= 463)) return "Rom.inB";
    if ((address >= 464) && (address <= 471)) return "Rom.inC";
    if ((address >= 472) && (address <= 479)) return "Rom.inROTL_C";
    if ((address >= 480) && (address <= 487)) return "Rom.inD";
    if ((address >= 488) && (address <= 495)) return "Rom.inE";
    if ((address >= 496) && (address <= 503)) return "Rom.inSR";
    if ((address >= 504) && (address <= 511)) return "Rom.inFREE";
    if ((address >= 512) && (address <= 519)) return "Rom.inFREE0";
    if ((address >= 520) && (address <= 527)) return "Rom.inCTX";
    if ((address >= 528) && (address <= 535)) return "Rom.inSP";
    if ((address >= 536) && (address <= 543)) return "Rom.inPC";
    if ((address >= 544) && (address <= 551)) return "Rom.inGAS";
    if ((address >= 552) && (address <= 559)) return "Rom.inHASHPOS";
    if ((address >= 560) && (address <= 567)) return "Rom.inSTEP";
    if ((address >= 568) && (address <= 575)) return "Rom.inRR";
    if ((address >= 576) && (address <= 583)) return "Rom.inRCX";
    if ((address >= 584) && (address <= 591)) return "Rom.inCntArith";
    if ((address >= 592) && (address <= 599)) return "Rom.inCntBinary";
    if ((address >= 600) && (address <= 607)) return "Rom.inCntKeccakF";
    if ((address >= 608) && (address <= 615)) return "Rom.inCntMemAlign";
    if ((address >= 616) && (address <= 623)) return "Rom.inCntPaddingPG";
    if ((address >= 624) && (address <= 631)) return "Rom.inCntPoseidonG";
    if ((address >= 632) && (address <= 639)) return "Rom.inCntSha256F";
    if ((address >= 640) && (address <= 647)) return "Rom.incStack";
    if ((address >= 648) && (address <= 655)) return "Rom.binOpcode";
    if ((address >= 656) && (address <= 663)) return "Rom.jmpAddr";
    if ((address >= 664) && (address <= 671)) return "Rom.elseAddr";
    if ((address >= 672) && (address <= 679)) return "Rom.arithEq";
    if ((address >= 680) && (address <= 687)) return "Rom.line";
    if ((address >= 688) && (address <= 695)) return "Rom.operations";
    if ((address >= 696) && (address <= 703)) return "MemAlign.BYTE_C4096";
    if ((address >= 704) && (address <= 711)) return "MemAlign.FACTOR[0]";
    if ((address >= 712) && (address <= 719)) return "MemAlign.FACTOR[1]";
    if ((address >= 720) && (address <= 727)) return "MemAlign.FACTOR[2]";
    if ((address >= 728) && (address <= 735)) return "MemAlign.FACTOR[3]";
    if ((address >= 736) && (address <= 743)) return "MemAlign.FACTOR[4]";
    if ((address >= 744) && (address <= 751)) return "MemAlign.FACTOR[5]";
    if ((address >= 752) && (address <= 759)) return "MemAlign.FACTOR[6]";
    if ((address >= 760) && (address <= 767)) return "MemAlign.FACTOR[7]";
    if ((address >= 768) && (address <= 775)) return "MemAlign.FACTORV[0]";
    if ((address >= 776) && (address <= 783)) return "MemAlign.FACTORV[1]";
    if ((address >= 784) && (address <= 791)) return "MemAlign.FACTORV[2]";
    if ((address >= 792) && (address <= 799)) return "MemAlign.FACTORV[3]";
    if ((address >= 800) && (address <= 807)) return "MemAlign.FACTORV[4]";
    if ((address >= 808) && (address <= 815)) return "MemAlign.FACTORV[5]";
    if ((address >= 816) && (address <= 823)) return "MemAlign.FACTORV[6]";
    if ((address >= 824) && (address <= 831)) return "MemAlign.FACTORV[7]";
    if ((address >= 832) && (address <= 839)) return "MemAlign.WR256";
    if ((address >= 840) && (address <= 847)) return "MemAlign.WR8";
    if ((address >= 848) && (address <= 855)) return "MemAlign.OFFSET";
    if ((address >= 856) && (address <= 863)) return "MemAlign.SELM1";
    if ((address >= 864) && (address <= 871)) return "Arith.BYTE2_BIT21";
    if ((address >= 872) && (address <= 879)) return "Arith.SEL_BYTE2_BIT21";
    if ((address >= 880) && (address <= 887)) return "Arith.GL_SIGNED_22BITS";
    if ((address >= 888) && (address <= 895)) return "Arith.RANGE_SEL";
    if ((address >= 896) && (address <= 903)) return "Binary.P_OPCODE";
    if ((address >= 904) && (address <= 911)) return "Binary.P_CIN";
    if ((address >= 912) && (address <= 919)) return "Binary.P_LAST";
    if ((address >= 920) && (address <= 927)) return "Binary.P_C";
    if ((address >= 928) && (address <= 935)) return "Binary.P_FLAGS";
    if ((address >= 936) && (address <= 943)) return "Binary.FACTOR[0]";
    if ((address >= 944) && (address <= 951)) return "Binary.FACTOR[1]";
    if ((address >= 952) && (address <= 959)) return "Binary.FACTOR[2]";
    if ((address >= 960) && (address <= 967)) return "Binary.FACTOR[3]";
    if ((address >= 968) && (address <= 975)) return "Binary.FACTOR[4]";
    if ((address >= 976) && (address <= 983)) return "Binary.FACTOR[5]";
    if ((address >= 984) && (address <= 991)) return "Binary.FACTOR[6]";
    if ((address >= 992) && (address <= 999)) return "Binary.FACTOR[7]";
    if ((address >= 1000) && (address <= 1007)) return "PoseidonG.LAST";
    if ((address >= 1008) && (address <= 1015)) return "PoseidonG.LATCH";
    if ((address >= 1016) && (address <= 1023)) return "PoseidonG.LASTBLOCK";
    if ((address >= 1024) && (address <= 1031)) return "PoseidonG.PARTIAL";
    if ((address >= 1032) && (address <= 1039)) return "PoseidonG.C[0]";
    if ((address >= 1040) && (address <= 1047)) return "PoseidonG.C[1]";
    if ((address >= 1048) && (address <= 1055)) return "PoseidonG.C[2]";
    if ((address >= 1056) && (address <= 1063)) return "PoseidonG.C[3]";
    if ((address >= 1064) && (address <= 1071)) return "PoseidonG.C[4]";
    if ((address >= 1072) && (address <= 1079)) return "PoseidonG.C[5]";
    if ((address >= 1080) && (address <= 1087)) return "PoseidonG.C[6]";
    if ((address >= 1088) && (address <= 1095)) return "PoseidonG.C[7]";
    if ((address >= 1096) && (address <= 1103)) return "PoseidonG.C[8]";
    if ((address >= 1104) && (address <= 1111)) return "PoseidonG.C[9]";
    if ((address >= 1112) && (address <= 1119)) return "PoseidonG.C[10]";
    if ((address >= 1120) && (address <= 1127)) return "PoseidonG.C[11]";
    if ((address >= 1128) && (address <= 1135)) return "PaddingPG.F[0]";
    if ((address >= 1136) && (address <= 1143)) return "PaddingPG.F[1]";
    if ((address >= 1144) && (address <= 1151)) return "PaddingPG.F[2]";
    if ((address >= 1152) && (address <= 1159)) return "PaddingPG.F[3]";
    if ((address >= 1160) && (address <= 1167)) return "PaddingPG.F[4]";
    if ((address >= 1168) && (address <= 1175)) return "PaddingPG.F[5]";
    if ((address >= 1176) && (address <= 1183)) return "PaddingPG.F[6]";
    if ((address >= 1184) && (address <= 1191)) return "PaddingPG.F[7]";
    if ((address >= 1192) && (address <= 1199)) return "PaddingPG.lastBlock";
    if ((address >= 1200) && (address <= 1207)) return "PaddingPG.crValid";
    if ((address >= 1208) && (address <= 1215)) return "ClimbKey.FACTOR";
    if ((address >= 1216) && (address <= 1223)) return "ClimbKey.T_CLKEYSEL";
    if ((address >= 1224) && (address <= 1231)) return "ClimbKey.T_LEVEL";
    if ((address >= 1232) && (address <= 1239)) return "ClimbKey.T_CHUNK_VALUE";
    if ((address >= 1240) && (address <= 1247)) return "ClimbKey.T_CARRYLT_IN";
    if ((address >= 1248) && (address <= 1255)) return "ClimbKey.T_CARRYLT_OUT";
    if ((address >= 1256) && (address <= 1263)) return "Storage.OPERATION";
    if ((address >= 1264) && (address <= 1271)) return "Storage.CONST0";
    if ((address >= 1272) && (address <= 1279)) return "Storage.JMP_ADDRESS";
    if ((address >= 1280) && (address <= 1287)) return "Storage.LINE";
    if ((address >= 1288) && (address <= 1295)) return "Storage.IN_SIBLING_RKEY";
    if ((address >= 1296) && (address <= 1303)) return "KeccakF.ConnA";
    if ((address >= 1304) && (address <= 1311)) return "KeccakF.ConnB";
    if ((address >= 1312) && (address <= 1319)) return "KeccakF.ConnC";
    if ((address >= 1320) && (address <= 1327)) return "KeccakF.GateType";
    if ((address >= 1328) && (address <= 1335)) return "KeccakF.kGateType";
    if ((address >= 1336) && (address <= 1343)) return "KeccakF.kA";
    if ((address >= 1344) && (address <= 1351)) return "KeccakF.kB";
    if ((address >= 1352) && (address <= 1359)) return "KeccakF.kC";
    if ((address >= 1360) && (address <= 1367)) return "Bits2Field.FieldLatch";
    if ((address >= 1368) && (address <= 1375)) return "Bits2Field.Factor";
    if ((address >= 1376) && (address <= 1383)) return "PaddingKKBit.r8Id";
    if ((address >= 1384) && (address <= 1391)) return "PaddingKKBit.sOutId";
    if ((address >= 1392) && (address <= 1399)) return "PaddingKKBit.latchR8";
    if ((address >= 1400) && (address <= 1407)) return "PaddingKKBit.Fr8";
    if ((address >= 1408) && (address <= 1415)) return "PaddingKKBit.rBitValid";
    if ((address >= 1416) && (address <= 1423)) return "PaddingKKBit.latchSOut";
    if ((address >= 1424) && (address <= 1431)) return "PaddingKKBit.FSOut0";
    if ((address >= 1432) && (address <= 1439)) return "PaddingKKBit.FSOut1";
    if ((address >= 1440) && (address <= 1447)) return "PaddingKKBit.FSOut2";
    if ((address >= 1448) && (address <= 1455)) return "PaddingKKBit.FSOut3";
    if ((address >= 1456) && (address <= 1463)) return "PaddingKKBit.FSOut4";
    if ((address >= 1464) && (address <= 1471)) return "PaddingKKBit.FSOut5";
    if ((address >= 1472) && (address <= 1479)) return "PaddingKKBit.FSOut6";
    if ((address >= 1480) && (address <= 1487)) return "PaddingKKBit.FSOut7";
    if ((address >= 1488) && (address <= 1495)) return "PaddingKKBit.ConnSOutBit";
    if ((address >= 1496) && (address <= 1503)) return "PaddingKKBit.ConnSInBit";
    if ((address >= 1504) && (address <= 1511)) return "PaddingKKBit.ConnBits2FieldBit";
    if ((address >= 1512) && (address <= 1519)) return "PaddingKK.r8Id";
    if ((address >= 1520) && (address <= 1527)) return "PaddingKK.lastBlock";
    if ((address >= 1528) && (address <= 1535)) return "PaddingKK.lastBlockLatch";
    if ((address >= 1536) && (address <= 1543)) return "PaddingKK.r8valid";
    if ((address >= 1544) && (address <= 1551)) return "PaddingKK.sOutId";
    if ((address >= 1552) && (address <= 1559)) return "PaddingKK.forceLastHash";
    if ((address >= 1560) && (address <= 1567)) return "Sha256F.kGateType";
    if ((address >= 1568) && (address <= 1575)) return "Sha256F.kA";
    if ((address >= 1576) && (address <= 1583)) return "Sha256F.kB";
    if ((address >= 1584) && (address <= 1591)) return "Sha256F.kC";
    if ((address >= 1592) && (address <= 1599)) return "Sha256F.kOut";
    if ((address >= 1600) && (address <= 1607)) return "Sha256F.kCarryOut";
    if ((address >= 1608) && (address <= 1615)) return "Sha256F.Conn[0]";
    if ((address >= 1616) && (address <= 1623)) return "Sha256F.Conn[1]";
    if ((address >= 1624) && (address <= 1631)) return "Sha256F.Conn[2]";
    if ((address >= 1632) && (address <= 1639)) return "Sha256F.Conn[3]";
    if ((address >= 1640) && (address <= 1647)) return "Sha256F.GATE_TYPE";
    if ((address >= 1648) && (address <= 1655)) return "Sha256F.CARRY_ENABLED";
    if ((address >= 1656) && (address <= 1663)) return "Bits2FieldSha256.FieldLatch";
    if ((address >= 1664) && (address <= 1671)) return "Bits2FieldSha256.Factor";
    if ((address >= 1672) && (address <= 1679)) return "PaddingSha256Bit.r8Id";
    if ((address >= 1680) && (address <= 1687)) return "PaddingSha256Bit.sOutId";
    if ((address >= 1688) && (address <= 1695)) return "PaddingSha256Bit.latchR8";
    if ((address >= 1696) && (address <= 1703)) return "PaddingSha256Bit.Fr8";
    if ((address >= 1704) && (address <= 1711)) return "PaddingSha256Bit.latchSOut";
    if ((address >= 1712) && (address <= 1719)) return "PaddingSha256Bit.FSOut0";
    if ((address >= 1720) && (address <= 1727)) return "PaddingSha256Bit.FSOut1";
    if ((address >= 1728) && (address <= 1735)) return "PaddingSha256Bit.FSOut2";
    if ((address >= 1736) && (address <= 1743)) return "PaddingSha256Bit.FSOut3";
    if ((address >= 1744) && (address <= 1751)) return "PaddingSha256Bit.FSOut4";
    if ((address >= 1752) && (address <= 1759)) return "PaddingSha256Bit.FSOut5";
    if ((address >= 1760) && (address <= 1767)) return "PaddingSha256Bit.FSOut6";
    if ((address >= 1768) && (address <= 1775)) return "PaddingSha256Bit.FSOut7";
    if ((address >= 1776) && (address <= 1783)) return "PaddingSha256Bit.HIn";
    if ((address >= 1784) && (address <= 1791)) return "PaddingSha256Bit.DoConnect";
    if ((address >= 1792) && (address <= 1799)) return "PaddingSha256Bit.ConnS1";
    if ((address >= 1800) && (address <= 1807)) return "PaddingSha256Bit.ConnS2";
    if ((address >= 1808) && (address <= 1815)) return "PaddingSha256Bit.ConnBits2FieldBit";
    if ((address >= 1816) && (address <= 1823)) return "PaddingSha256.r8Id";
    if ((address >= 1824) && (address <= 1831)) return "PaddingSha256.lastBlock";
    if ((address >= 1832) && (address <= 1839)) return "PaddingSha256.lastBlockLatch";
    if ((address >= 1840) && (address <= 1847)) return "PaddingSha256.r8valid";
    if ((address >= 1848) && (address <= 1855)) return "PaddingSha256.PrevLengthSection";
    if ((address >= 1856) && (address <= 1863)) return "PaddingSha256.LengthWeight";
    if ((address >= 1864) && (address <= 1871)) return "PaddingSha256.sOutId";
    if ((address >= 1872) && (address <= 1879)) return "PaddingSha256.forceLastHash";
    return "ERROR_NOT_FOUND";
}

} // namespace

#endif // CONSTANT_POLS_HPP_fork_13

