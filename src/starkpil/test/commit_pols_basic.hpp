#ifndef COMMIT_POLS_BASIC_HPP
#define COMMIT_POLS_BASIC_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"

class CommitPolBasic
{
private:
    Goldilocks::Element * _pAddress;
    uint64_t _degree;
    uint64_t _index;
public:
    CommitPolBasic(Goldilocks::Element * pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index) {};
    Goldilocks::Element & operator[](int i) { return _pAddress[i*158]; };
    Goldilocks::Element * operator=(Goldilocks::Element * pAddress) { _pAddress = pAddress; return _pAddress; };

    Goldilocks::Element * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t index (void) { return _index; }
};

class MainCommitPolsBasic
{
public:
    CommitPolBasic A7;
    CommitPolBasic A6;
    CommitPolBasic A5;
    CommitPolBasic A4;
    CommitPolBasic A3;
    CommitPolBasic A2;
    CommitPolBasic A1;
    CommitPolBasic A0;
    CommitPolBasic B7;
    CommitPolBasic B6;
    CommitPolBasic B5;
    CommitPolBasic B4;
    CommitPolBasic B3;
    CommitPolBasic B2;
    CommitPolBasic B1;
    CommitPolBasic B0;
    CommitPolBasic C7;
    CommitPolBasic C6;
    CommitPolBasic C5;
    CommitPolBasic C4;
    CommitPolBasic C3;
    CommitPolBasic C2;
    CommitPolBasic C1;
    CommitPolBasic C0;
    CommitPolBasic D7;
    CommitPolBasic D6;
    CommitPolBasic D5;
    CommitPolBasic D4;
    CommitPolBasic D3;
    CommitPolBasic D2;
    CommitPolBasic D1;
    CommitPolBasic D0;
    CommitPolBasic E7;
    CommitPolBasic E6;
    CommitPolBasic E5;
    CommitPolBasic E4;
    CommitPolBasic E3;
    CommitPolBasic E2;
    CommitPolBasic E1;
    CommitPolBasic E0;
    CommitPolBasic SR7;
    CommitPolBasic SR6;
    CommitPolBasic SR5;
    CommitPolBasic SR4;
    CommitPolBasic SR3;
    CommitPolBasic SR2;
    CommitPolBasic SR1;
    CommitPolBasic SR0;
    CommitPolBasic CTX;
    CommitPolBasic SP;
    CommitPolBasic PC;
    CommitPolBasic GAS;
    CommitPolBasic MAXMEM;
    CommitPolBasic zkPC;
    CommitPolBasic RR;
    CommitPolBasic HASHPOS;
    CommitPolBasic CONST7;
    CommitPolBasic CONST6;
    CommitPolBasic CONST5;
    CommitPolBasic CONST4;
    CommitPolBasic CONST3;
    CommitPolBasic CONST2;
    CommitPolBasic CONST1;
    CommitPolBasic CONST0;
    CommitPolBasic FREE7;
    CommitPolBasic FREE6;
    CommitPolBasic FREE5;
    CommitPolBasic FREE4;
    CommitPolBasic FREE3;
    CommitPolBasic FREE2;
    CommitPolBasic FREE1;
    CommitPolBasic FREE0;
    CommitPolBasic inA;
    CommitPolBasic inB;
    CommitPolBasic inC;
    CommitPolBasic inROTL_C;
    CommitPolBasic inD;
    CommitPolBasic inE;
    CommitPolBasic inSR;
    CommitPolBasic inFREE;
    CommitPolBasic inCTX;
    CommitPolBasic inSP;
    CommitPolBasic inPC;
    CommitPolBasic inGAS;
    CommitPolBasic inMAXMEM;
    CommitPolBasic inSTEP;
    CommitPolBasic inRR;
    CommitPolBasic inHASHPOS;
    CommitPolBasic setA;
    CommitPolBasic setB;
    CommitPolBasic setC;
    CommitPolBasic setD;
    CommitPolBasic setE;
    CommitPolBasic setSR;
    CommitPolBasic setCTX;
    CommitPolBasic setSP;
    CommitPolBasic setPC;
    CommitPolBasic setGAS;
    CommitPolBasic setMAXMEM;
    CommitPolBasic JMP;
    CommitPolBasic JMPN;
    CommitPolBasic JMPC;
    CommitPolBasic setRR;
    CommitPolBasic setHASHPOS;
    CommitPolBasic offset;
    CommitPolBasic incStack;
    CommitPolBasic incCode;
    CommitPolBasic isStack;
    CommitPolBasic isCode;
    CommitPolBasic isMem;
    CommitPolBasic ind;
    CommitPolBasic indRR;
    CommitPolBasic useCTX;
    CommitPolBasic carry;
    CommitPolBasic mOp;
    CommitPolBasic mWR;
    CommitPolBasic sWR;
    CommitPolBasic sRD;
    CommitPolBasic arith;
    CommitPolBasic arithEq0;
    CommitPolBasic arithEq1;
    CommitPolBasic arithEq2;
    CommitPolBasic arithEq3;
    CommitPolBasic memAlign;
    CommitPolBasic memAlignWR;
    CommitPolBasic memAlignWR8;
    CommitPolBasic hashK;
    CommitPolBasic hashKLen;
    CommitPolBasic hashKDigest;
    CommitPolBasic hashP;
    CommitPolBasic hashPLen;
    CommitPolBasic hashPDigest;
    CommitPolBasic bin;
    CommitPolBasic binOpcode;
    CommitPolBasic assert_pol;
    CommitPolBasic isNeg;
    CommitPolBasic isMaxMem;
    CommitPolBasic cntArith;
    CommitPolBasic cntBinary;
    CommitPolBasic cntMemAlign;
    CommitPolBasic cntKeccakF;
    CommitPolBasic cntPoseidonG;
    CommitPolBasic cntPaddingPG;
    CommitPolBasic inCntArith;
    CommitPolBasic inCntBinary;
    CommitPolBasic inCntMemAlign;
    CommitPolBasic inCntKeccakF;
    CommitPolBasic inCntPoseidonG;
    CommitPolBasic inCntPaddingPG;
    CommitPolBasic incCounter;
    CommitPolBasic sKeyI[4];
    CommitPolBasic sKey[4];
private:
    void * _pAddress;
    uint64_t _degree;
public:

    MainCommitPolsBasic (void * pAddress, uint64_t degree) :
        A7((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
        A6((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1),
        A5((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
        A4((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
        A3((Goldilocks::Element *)((uint8_t *)pAddress + 32), degree, 4),
        A2((Goldilocks::Element *)((uint8_t *)pAddress + 40), degree, 5),
        A1((Goldilocks::Element *)((uint8_t *)pAddress + 48), degree, 6),
        A0((Goldilocks::Element *)((uint8_t *)pAddress + 56), degree, 7),
        B7((Goldilocks::Element *)((uint8_t *)pAddress + 64), degree, 8),
        B6((Goldilocks::Element *)((uint8_t *)pAddress + 72), degree, 9),
        B5((Goldilocks::Element *)((uint8_t *)pAddress + 80), degree, 10),
        B4((Goldilocks::Element *)((uint8_t *)pAddress + 88), degree, 11),
        B3((Goldilocks::Element *)((uint8_t *)pAddress + 96), degree, 12),
        B2((Goldilocks::Element *)((uint8_t *)pAddress + 104), degree, 13),
        B1((Goldilocks::Element *)((uint8_t *)pAddress + 112), degree, 14),
        B0((Goldilocks::Element *)((uint8_t *)pAddress + 120), degree, 15),
        C7((Goldilocks::Element *)((uint8_t *)pAddress + 128), degree, 16),
        C6((Goldilocks::Element *)((uint8_t *)pAddress + 136), degree, 17),
        C5((Goldilocks::Element *)((uint8_t *)pAddress + 144), degree, 18),
        C4((Goldilocks::Element *)((uint8_t *)pAddress + 152), degree, 19),
        C3((Goldilocks::Element *)((uint8_t *)pAddress + 160), degree, 20),
        C2((Goldilocks::Element *)((uint8_t *)pAddress + 168), degree, 21),
        C1((Goldilocks::Element *)((uint8_t *)pAddress + 176), degree, 22),
        C0((Goldilocks::Element *)((uint8_t *)pAddress + 184), degree, 23),
        D7((Goldilocks::Element *)((uint8_t *)pAddress + 192), degree, 24),
        D6((Goldilocks::Element *)((uint8_t *)pAddress + 200), degree, 25),
        D5((Goldilocks::Element *)((uint8_t *)pAddress + 208), degree, 26),
        D4((Goldilocks::Element *)((uint8_t *)pAddress + 216), degree, 27),
        D3((Goldilocks::Element *)((uint8_t *)pAddress + 224), degree, 28),
        D2((Goldilocks::Element *)((uint8_t *)pAddress + 232), degree, 29),
        D1((Goldilocks::Element *)((uint8_t *)pAddress + 240), degree, 30),
        D0((Goldilocks::Element *)((uint8_t *)pAddress + 248), degree, 31),
        E7((Goldilocks::Element *)((uint8_t *)pAddress + 256), degree, 32),
        E6((Goldilocks::Element *)((uint8_t *)pAddress + 264), degree, 33),
        E5((Goldilocks::Element *)((uint8_t *)pAddress + 272), degree, 34),
        E4((Goldilocks::Element *)((uint8_t *)pAddress + 280), degree, 35),
        E3((Goldilocks::Element *)((uint8_t *)pAddress + 288), degree, 36),
        E2((Goldilocks::Element *)((uint8_t *)pAddress + 296), degree, 37),
        E1((Goldilocks::Element *)((uint8_t *)pAddress + 304), degree, 38),
        E0((Goldilocks::Element *)((uint8_t *)pAddress + 312), degree, 39),
        SR7((Goldilocks::Element *)((uint8_t *)pAddress + 320), degree, 40),
        SR6((Goldilocks::Element *)((uint8_t *)pAddress + 328), degree, 41),
        SR5((Goldilocks::Element *)((uint8_t *)pAddress + 336), degree, 42),
        SR4((Goldilocks::Element *)((uint8_t *)pAddress + 344), degree, 43),
        SR3((Goldilocks::Element *)((uint8_t *)pAddress + 352), degree, 44),
        SR2((Goldilocks::Element *)((uint8_t *)pAddress + 360), degree, 45),
        SR1((Goldilocks::Element *)((uint8_t *)pAddress + 368), degree, 46),
        SR0((Goldilocks::Element *)((uint8_t *)pAddress + 376), degree, 47),
        CTX((Goldilocks::Element *)((uint8_t *)pAddress + 384), degree, 48),
        SP((Goldilocks::Element *)((uint8_t *)pAddress + 392), degree, 49),
        PC((Goldilocks::Element *)((uint8_t *)pAddress + 400), degree, 50),
        GAS((Goldilocks::Element *)((uint8_t *)pAddress + 408), degree, 51),
        MAXMEM((Goldilocks::Element *)((uint8_t *)pAddress + 416), degree, 52),
        zkPC((Goldilocks::Element *)((uint8_t *)pAddress + 424), degree, 53),
        RR((Goldilocks::Element *)((uint8_t *)pAddress + 432), degree, 54),
        HASHPOS((Goldilocks::Element *)((uint8_t *)pAddress + 440), degree, 55),
        CONST7((Goldilocks::Element *)((uint8_t *)pAddress + 448), degree, 56),
        CONST6((Goldilocks::Element *)((uint8_t *)pAddress + 456), degree, 57),
        CONST5((Goldilocks::Element *)((uint8_t *)pAddress + 464), degree, 58),
        CONST4((Goldilocks::Element *)((uint8_t *)pAddress + 472), degree, 59),
        CONST3((Goldilocks::Element *)((uint8_t *)pAddress + 480), degree, 60),
        CONST2((Goldilocks::Element *)((uint8_t *)pAddress + 488), degree, 61),
        CONST1((Goldilocks::Element *)((uint8_t *)pAddress + 496), degree, 62),
        CONST0((Goldilocks::Element *)((uint8_t *)pAddress + 504), degree, 63),
        FREE7((Goldilocks::Element *)((uint8_t *)pAddress + 512), degree, 64),
        FREE6((Goldilocks::Element *)((uint8_t *)pAddress + 520), degree, 65),
        FREE5((Goldilocks::Element *)((uint8_t *)pAddress + 528), degree, 66),
        FREE4((Goldilocks::Element *)((uint8_t *)pAddress + 536), degree, 67),
        FREE3((Goldilocks::Element *)((uint8_t *)pAddress + 544), degree, 68),
        FREE2((Goldilocks::Element *)((uint8_t *)pAddress + 552), degree, 69),
        FREE1((Goldilocks::Element *)((uint8_t *)pAddress + 560), degree, 70),
        FREE0((Goldilocks::Element *)((uint8_t *)pAddress + 568), degree, 71),
        inA((Goldilocks::Element *)((uint8_t *)pAddress + 576), degree, 72),
        inB((Goldilocks::Element *)((uint8_t *)pAddress + 584), degree, 73),
        inC((Goldilocks::Element *)((uint8_t *)pAddress + 592), degree, 74),
        inROTL_C((Goldilocks::Element *)((uint8_t *)pAddress + 600), degree, 75),
        inD((Goldilocks::Element *)((uint8_t *)pAddress + 608), degree, 76),
        inE((Goldilocks::Element *)((uint8_t *)pAddress + 616), degree, 77),
        inSR((Goldilocks::Element *)((uint8_t *)pAddress + 624), degree, 78),
        inFREE((Goldilocks::Element *)((uint8_t *)pAddress + 632), degree, 79),
        inCTX((Goldilocks::Element *)((uint8_t *)pAddress + 640), degree, 80),
        inSP((Goldilocks::Element *)((uint8_t *)pAddress + 648), degree, 81),
        inPC((Goldilocks::Element *)((uint8_t *)pAddress + 656), degree, 82),
        inGAS((Goldilocks::Element *)((uint8_t *)pAddress + 664), degree, 83),
        inMAXMEM((Goldilocks::Element *)((uint8_t *)pAddress + 672), degree, 84),
        inSTEP((Goldilocks::Element *)((uint8_t *)pAddress + 680), degree, 85),
        inRR((Goldilocks::Element *)((uint8_t *)pAddress + 688), degree, 86),
        inHASHPOS((Goldilocks::Element *)((uint8_t *)pAddress + 696), degree, 87),
        setA((Goldilocks::Element *)((uint8_t *)pAddress + 704), degree, 88),
        setB((Goldilocks::Element *)((uint8_t *)pAddress + 712), degree, 89),
        setC((Goldilocks::Element *)((uint8_t *)pAddress + 720), degree, 90),
        setD((Goldilocks::Element *)((uint8_t *)pAddress + 728), degree, 91),
        setE((Goldilocks::Element *)((uint8_t *)pAddress + 736), degree, 92),
        setSR((Goldilocks::Element *)((uint8_t *)pAddress + 744), degree, 93),
        setCTX((Goldilocks::Element *)((uint8_t *)pAddress + 752), degree, 94),
        setSP((Goldilocks::Element *)((uint8_t *)pAddress + 760), degree, 95),
        setPC((Goldilocks::Element *)((uint8_t *)pAddress + 768), degree, 96),
        setGAS((Goldilocks::Element *)((uint8_t *)pAddress + 776), degree, 97),
        setMAXMEM((Goldilocks::Element *)((uint8_t *)pAddress + 784), degree, 98),
        JMP((Goldilocks::Element *)((uint8_t *)pAddress + 792), degree, 99),
        JMPN((Goldilocks::Element *)((uint8_t *)pAddress + 800), degree, 100),
        JMPC((Goldilocks::Element *)((uint8_t *)pAddress + 808), degree, 101),
        setRR((Goldilocks::Element *)((uint8_t *)pAddress + 816), degree, 102),
        setHASHPOS((Goldilocks::Element *)((uint8_t *)pAddress + 824), degree, 103),
        offset((Goldilocks::Element *)((uint8_t *)pAddress + 832), degree, 104),
        incStack((Goldilocks::Element *)((uint8_t *)pAddress + 840), degree, 105),
        incCode((Goldilocks::Element *)((uint8_t *)pAddress + 848), degree, 106),
        isStack((Goldilocks::Element *)((uint8_t *)pAddress + 856), degree, 107),
        isCode((Goldilocks::Element *)((uint8_t *)pAddress + 864), degree, 108),
        isMem((Goldilocks::Element *)((uint8_t *)pAddress + 872), degree, 109),
        ind((Goldilocks::Element *)((uint8_t *)pAddress + 880), degree, 110),
        indRR((Goldilocks::Element *)((uint8_t *)pAddress + 888), degree, 111),
        useCTX((Goldilocks::Element *)((uint8_t *)pAddress + 896), degree, 112),
        carry((Goldilocks::Element *)((uint8_t *)pAddress + 904), degree, 113),
        mOp((Goldilocks::Element *)((uint8_t *)pAddress + 912), degree, 114),
        mWR((Goldilocks::Element *)((uint8_t *)pAddress + 920), degree, 115),
        sWR((Goldilocks::Element *)((uint8_t *)pAddress + 928), degree, 116),
        sRD((Goldilocks::Element *)((uint8_t *)pAddress + 936), degree, 117),
        arith((Goldilocks::Element *)((uint8_t *)pAddress + 944), degree, 118),
        arithEq0((Goldilocks::Element *)((uint8_t *)pAddress + 952), degree, 119),
        arithEq1((Goldilocks::Element *)((uint8_t *)pAddress + 960), degree, 120),
        arithEq2((Goldilocks::Element *)((uint8_t *)pAddress + 968), degree, 121),
        arithEq3((Goldilocks::Element *)((uint8_t *)pAddress + 976), degree, 122),
        memAlign((Goldilocks::Element *)((uint8_t *)pAddress + 984), degree, 123),
        memAlignWR((Goldilocks::Element *)((uint8_t *)pAddress + 992), degree, 124),
        memAlignWR8((Goldilocks::Element *)((uint8_t *)pAddress + 1000), degree, 125),
        hashK((Goldilocks::Element *)((uint8_t *)pAddress + 1008), degree, 126),
        hashKLen((Goldilocks::Element *)((uint8_t *)pAddress + 1016), degree, 127),
        hashKDigest((Goldilocks::Element *)((uint8_t *)pAddress + 1024), degree, 128),
        hashP((Goldilocks::Element *)((uint8_t *)pAddress + 1032), degree, 129),
        hashPLen((Goldilocks::Element *)((uint8_t *)pAddress + 1040), degree, 130),
        hashPDigest((Goldilocks::Element *)((uint8_t *)pAddress + 1048), degree, 131),
        bin((Goldilocks::Element *)((uint8_t *)pAddress + 1056), degree, 132),
        binOpcode((Goldilocks::Element *)((uint8_t *)pAddress + 1064), degree, 133),
        assert_pol((Goldilocks::Element *)((uint8_t *)pAddress + 1072), degree, 134),
        isNeg((Goldilocks::Element *)((uint8_t *)pAddress + 1080), degree, 135),
        isMaxMem((Goldilocks::Element *)((uint8_t *)pAddress + 1088), degree, 136),
        cntArith((Goldilocks::Element *)((uint8_t *)pAddress + 1096), degree, 137),
        cntBinary((Goldilocks::Element *)((uint8_t *)pAddress + 1104), degree, 138),
        cntMemAlign((Goldilocks::Element *)((uint8_t *)pAddress + 1112), degree, 139),
        cntKeccakF((Goldilocks::Element *)((uint8_t *)pAddress + 1120), degree, 140),
        cntPoseidonG((Goldilocks::Element *)((uint8_t *)pAddress + 1128), degree, 141),
        cntPaddingPG((Goldilocks::Element *)((uint8_t *)pAddress + 1136), degree, 142),
        inCntArith((Goldilocks::Element *)((uint8_t *)pAddress + 1144), degree, 143),
        inCntBinary((Goldilocks::Element *)((uint8_t *)pAddress + 1152), degree, 144),
        inCntMemAlign((Goldilocks::Element *)((uint8_t *)pAddress + 1160), degree, 145),
        inCntKeccakF((Goldilocks::Element *)((uint8_t *)pAddress + 1168), degree, 146),
        inCntPoseidonG((Goldilocks::Element *)((uint8_t *)pAddress + 1176), degree, 147),
        inCntPaddingPG((Goldilocks::Element *)((uint8_t *)pAddress + 1184), degree, 148),
        incCounter((Goldilocks::Element *)((uint8_t *)pAddress + 1192), degree, 149),
        sKeyI{
            CommitPolBasic((Goldilocks::Element *)((uint8_t *)pAddress + 1200), degree, 150),
            CommitPolBasic((Goldilocks::Element *)((uint8_t *)pAddress + 1208), degree, 151),
            CommitPolBasic((Goldilocks::Element *)((uint8_t *)pAddress + 1216), degree, 152),
            CommitPolBasic((Goldilocks::Element *)((uint8_t *)pAddress + 1224), degree, 153)
        },
        sKey{
            CommitPolBasic((Goldilocks::Element *)((uint8_t *)pAddress + 1232), degree, 154),
            CommitPolBasic((Goldilocks::Element *)((uint8_t *)pAddress + 1240), degree, 155),
            CommitPolBasic((Goldilocks::Element *)((uint8_t *)pAddress + 1248), degree, 156),
            CommitPolBasic((Goldilocks::Element *)((uint8_t *)pAddress + 1256), degree, 157)
        },
        _pAddress(pAddress),
        _degree(degree) {};

    static uint64_t pilDegree (void) { return 262144; }
    static uint64_t pilSize (void) { return 1264; }
    static uint64_t numPols (void) { return 158; }

    void * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t size (void) { return _degree*158*sizeof(Goldilocks::Element); }
};

class CommitPolsBasic
{
public:
    MainCommitPolsBasic Main;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    CommitPolsBasic (void * pAddress, uint64_t degree) :
        Main(pAddress, degree),
        _pAddress(pAddress),
        _degree(degree) {}

    static uint64_t pilSize (void) { return 331350016; }
    static uint64_t pilDegree (void) { return 262144; }
    static uint64_t numPols (void) { return 158; }

    void * address (void) { return _pAddress; }
    uint64_t degree (void) { return _degree; }
    uint64_t size (void) { return _degree*158*sizeof(Goldilocks::Element); }

    Goldilocks::Element &getElement (uint64_t pol, uint64_t evaluation)
    {
        zkassert((pol < numPols()) && (evaluation < degree()));
        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * numPols()];
    }
};

#endif // COMMIT_POLS_BASIC_HPP
