#ifndef COMMIT_POLS_HPP_fork_12
#define COMMIT_POLS_HPP_fork_12

#include <cstdint>
#include "goldilocks_base_field.hpp"

namespace fork_12
{

class CommitPol
{
private:
    Goldilocks::Element * _pAddress;
    uint64_t _degree;
    uint64_t _index;
public:
    CommitPol(Goldilocks::Element * pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index) {};
    inline Goldilocks::Element & operator[](uint64_t i) { return _pAddress[i*749]; };
    inline Goldilocks::Element * operator=(Goldilocks::Element * pAddress) { _pAddress = pAddress; return _pAddress; };

    inline Goldilocks::Element * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t index (void) { return _index; }
};

class MemAlignCommitPols
{
public:
    CommitPol inM[2];
    CommitPol inV;
    CommitPol wr256;
    CommitPol wr8;
    CommitPol m0[8];
    CommitPol m1[8];
    CommitPol w0[8];
    CommitPol w1[8];
    CommitPol v[8];
    CommitPol selM1;
    CommitPol factorV[8];
    CommitPol offset;
    CommitPol resultRd;
    CommitPol resultWr8;
    CommitPol resultWr256;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    MemAlignCommitPols (void * pAddress, uint64_t degree) :
        inM{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 0), degree, 0),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 8), degree, 1)
        },
        inV((Goldilocks::Element *)((uint8_t *)pAddress + 16), degree, 2),
        wr256((Goldilocks::Element *)((uint8_t *)pAddress + 24), degree, 3),
        wr8((Goldilocks::Element *)((uint8_t *)pAddress + 32), degree, 4),
        m0{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 40), degree, 5),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 48), degree, 6),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 56), degree, 7),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 64), degree, 8),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 72), degree, 9),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 80), degree, 10),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 88), degree, 11),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 96), degree, 12)
        },
        m1{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 104), degree, 13),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 112), degree, 14),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 120), degree, 15),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 128), degree, 16),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 136), degree, 17),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 144), degree, 18),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 152), degree, 19),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 160), degree, 20)
        },
        w0{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 168), degree, 21),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 176), degree, 22),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 184), degree, 23),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 192), degree, 24),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 200), degree, 25),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 208), degree, 26),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 216), degree, 27),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 224), degree, 28)
        },
        w1{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 232), degree, 29),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 240), degree, 30),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 248), degree, 31),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 256), degree, 32),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 264), degree, 33),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 272), degree, 34),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 280), degree, 35),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 288), degree, 36)
        },
        v{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 296), degree, 37),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 304), degree, 38),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 312), degree, 39),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 320), degree, 40),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 328), degree, 41),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 336), degree, 42),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 344), degree, 43),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 352), degree, 44)
        },
        selM1((Goldilocks::Element *)((uint8_t *)pAddress + 360), degree, 45),
        factorV{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 368), degree, 46),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 376), degree, 47),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 384), degree, 48),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 392), degree, 49),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 400), degree, 50),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 408), degree, 51),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 416), degree, 52),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 424), degree, 53)
        },
        offset((Goldilocks::Element *)((uint8_t *)pAddress + 432), degree, 54),
        resultRd((Goldilocks::Element *)((uint8_t *)pAddress + 440), degree, 55),
        resultWr8((Goldilocks::Element *)((uint8_t *)pAddress + 448), degree, 56),
        resultWr256((Goldilocks::Element *)((uint8_t *)pAddress + 456), degree, 57),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 58; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*58*sizeof(Goldilocks::Element); }
};

class ArithCommitPols
{
public:
    CommitPol x1[16];
    CommitPol y1[16];
    CommitPol x2[16];
    CommitPol y2[16];
    CommitPol x3[16];
    CommitPol y3[16];
    CommitPol s[16];
    CommitPol q0[16];
    CommitPol q1[16];
    CommitPol q2[16];
    CommitPol resultEq0;
    CommitPol resultEq1;
    CommitPol resultEq2;
    CommitPol xDeltaChunkInverse;
    CommitPol xAreDifferent;
    CommitPol valueLtPrime;
    CommitPol chunkLtPrime;
    CommitPol selEq[7];
    CommitPol carry[3];
private:
    void * _pAddress;
    uint64_t _degree;
public:

    ArithCommitPols (void * pAddress, uint64_t degree) :
        x1{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 464), degree, 58),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 472), degree, 59),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 480), degree, 60),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 488), degree, 61),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 496), degree, 62),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 504), degree, 63),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 512), degree, 64),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 520), degree, 65),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 528), degree, 66),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 536), degree, 67),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 544), degree, 68),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 552), degree, 69),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 560), degree, 70),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 568), degree, 71),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 576), degree, 72),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 584), degree, 73)
        },
        y1{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 592), degree, 74),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 600), degree, 75),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 608), degree, 76),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 616), degree, 77),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 624), degree, 78),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 632), degree, 79),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 640), degree, 80),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 648), degree, 81),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 656), degree, 82),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 664), degree, 83),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 672), degree, 84),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 680), degree, 85),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 688), degree, 86),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 696), degree, 87),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 704), degree, 88),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 712), degree, 89)
        },
        x2{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 720), degree, 90),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 728), degree, 91),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 736), degree, 92),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 744), degree, 93),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 752), degree, 94),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 760), degree, 95),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 768), degree, 96),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 776), degree, 97),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 784), degree, 98),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 792), degree, 99),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 800), degree, 100),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 808), degree, 101),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 816), degree, 102),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 824), degree, 103),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 832), degree, 104),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 840), degree, 105)
        },
        y2{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 848), degree, 106),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 856), degree, 107),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 864), degree, 108),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 872), degree, 109),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 880), degree, 110),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 888), degree, 111),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 896), degree, 112),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 904), degree, 113),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 912), degree, 114),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 920), degree, 115),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 928), degree, 116),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 936), degree, 117),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 944), degree, 118),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 952), degree, 119),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 960), degree, 120),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 968), degree, 121)
        },
        x3{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 976), degree, 122),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 984), degree, 123),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 992), degree, 124),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1000), degree, 125),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1008), degree, 126),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1016), degree, 127),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1024), degree, 128),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1032), degree, 129),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1040), degree, 130),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1048), degree, 131),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1056), degree, 132),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1064), degree, 133),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1072), degree, 134),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1080), degree, 135),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1088), degree, 136),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1096), degree, 137)
        },
        y3{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1104), degree, 138),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1112), degree, 139),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1120), degree, 140),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1128), degree, 141),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1136), degree, 142),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1144), degree, 143),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1152), degree, 144),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1160), degree, 145),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1168), degree, 146),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1176), degree, 147),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1184), degree, 148),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1192), degree, 149),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1200), degree, 150),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1208), degree, 151),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1216), degree, 152),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1224), degree, 153)
        },
        s{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1232), degree, 154),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1240), degree, 155),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1248), degree, 156),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1256), degree, 157),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1264), degree, 158),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1272), degree, 159),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1280), degree, 160),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1288), degree, 161),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1296), degree, 162),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1304), degree, 163),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1312), degree, 164),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1320), degree, 165),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1328), degree, 166),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1336), degree, 167),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1344), degree, 168),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1352), degree, 169)
        },
        q0{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1360), degree, 170),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1368), degree, 171),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1376), degree, 172),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1384), degree, 173),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1392), degree, 174),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1400), degree, 175),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1408), degree, 176),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1416), degree, 177),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1424), degree, 178),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1432), degree, 179),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1440), degree, 180),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1448), degree, 181),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1456), degree, 182),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1464), degree, 183),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1472), degree, 184),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1480), degree, 185)
        },
        q1{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1488), degree, 186),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1496), degree, 187),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1504), degree, 188),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1512), degree, 189),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1520), degree, 190),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1528), degree, 191),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1536), degree, 192),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1544), degree, 193),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1552), degree, 194),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1560), degree, 195),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1568), degree, 196),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1576), degree, 197),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1584), degree, 198),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1592), degree, 199),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1600), degree, 200),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1608), degree, 201)
        },
        q2{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1616), degree, 202),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1624), degree, 203),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1632), degree, 204),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1640), degree, 205),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1648), degree, 206),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1656), degree, 207),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1664), degree, 208),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1672), degree, 209),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1680), degree, 210),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1688), degree, 211),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1696), degree, 212),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1704), degree, 213),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1712), degree, 214),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1720), degree, 215),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1728), degree, 216),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1736), degree, 217)
        },
        resultEq0((Goldilocks::Element *)((uint8_t *)pAddress + 1744), degree, 218),
        resultEq1((Goldilocks::Element *)((uint8_t *)pAddress + 1752), degree, 219),
        resultEq2((Goldilocks::Element *)((uint8_t *)pAddress + 1760), degree, 220),
        xDeltaChunkInverse((Goldilocks::Element *)((uint8_t *)pAddress + 1768), degree, 221),
        xAreDifferent((Goldilocks::Element *)((uint8_t *)pAddress + 1776), degree, 222),
        valueLtPrime((Goldilocks::Element *)((uint8_t *)pAddress + 1784), degree, 223),
        chunkLtPrime((Goldilocks::Element *)((uint8_t *)pAddress + 1792), degree, 224),
        selEq{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1800), degree, 225),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1808), degree, 226),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1816), degree, 227),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1824), degree, 228),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1832), degree, 229),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1840), degree, 230),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1848), degree, 231)
        },
        carry{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1856), degree, 232),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1864), degree, 233),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1872), degree, 234)
        },
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 177; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*177*sizeof(Goldilocks::Element); }
};

class BinaryCommitPols
{
public:
    CommitPol opcode;
    CommitPol a[8];
    CommitPol b[8];
    CommitPol c[8];
    CommitPol freeInA[2];
    CommitPol freeInB[2];
    CommitPol freeInC[2];
    CommitPol cIn;
    CommitPol cMiddle;
    CommitPol cOut;
    CommitPol lCout;
    CommitPol lOpcode;
    CommitPol previousAreLt4;
    CommitPol usePreviousAreLt4;
    CommitPol reset4;
    CommitPol useCarry;
    CommitPol resultBinOp;
    CommitPol resultValidRange;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    BinaryCommitPols (void * pAddress, uint64_t degree) :
        opcode((Goldilocks::Element *)((uint8_t *)pAddress + 1880), degree, 235),
        a{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1888), degree, 236),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1896), degree, 237),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1904), degree, 238),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1912), degree, 239),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1920), degree, 240),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1928), degree, 241),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1936), degree, 242),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1944), degree, 243)
        },
        b{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1952), degree, 244),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1960), degree, 245),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1968), degree, 246),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1976), degree, 247),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1984), degree, 248),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 1992), degree, 249),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2000), degree, 250),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2008), degree, 251)
        },
        c{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2016), degree, 252),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2024), degree, 253),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2032), degree, 254),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2040), degree, 255),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2048), degree, 256),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2056), degree, 257),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2064), degree, 258),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2072), degree, 259)
        },
        freeInA{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2080), degree, 260),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2088), degree, 261)
        },
        freeInB{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2096), degree, 262),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2104), degree, 263)
        },
        freeInC{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2112), degree, 264),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2120), degree, 265)
        },
        cIn((Goldilocks::Element *)((uint8_t *)pAddress + 2128), degree, 266),
        cMiddle((Goldilocks::Element *)((uint8_t *)pAddress + 2136), degree, 267),
        cOut((Goldilocks::Element *)((uint8_t *)pAddress + 2144), degree, 268),
        lCout((Goldilocks::Element *)((uint8_t *)pAddress + 2152), degree, 269),
        lOpcode((Goldilocks::Element *)((uint8_t *)pAddress + 2160), degree, 270),
        previousAreLt4((Goldilocks::Element *)((uint8_t *)pAddress + 2168), degree, 271),
        usePreviousAreLt4((Goldilocks::Element *)((uint8_t *)pAddress + 2176), degree, 272),
        reset4((Goldilocks::Element *)((uint8_t *)pAddress + 2184), degree, 273),
        useCarry((Goldilocks::Element *)((uint8_t *)pAddress + 2192), degree, 274),
        resultBinOp((Goldilocks::Element *)((uint8_t *)pAddress + 2200), degree, 275),
        resultValidRange((Goldilocks::Element *)((uint8_t *)pAddress + 2208), degree, 276),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 42; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*42*sizeof(Goldilocks::Element); }
};

class PoseidonGCommitPols
{
public:
    CommitPol in0;
    CommitPol in1;
    CommitPol in2;
    CommitPol in3;
    CommitPol in4;
    CommitPol in5;
    CommitPol in6;
    CommitPol in7;
    CommitPol hashType;
    CommitPol cap1;
    CommitPol cap2;
    CommitPol cap3;
    CommitPol hash0;
    CommitPol hash1;
    CommitPol hash2;
    CommitPol hash3;
    CommitPol result1;
    CommitPol result2;
    CommitPol result3;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    PoseidonGCommitPols (void * pAddress, uint64_t degree) :
        in0((Goldilocks::Element *)((uint8_t *)pAddress + 2216), degree, 277),
        in1((Goldilocks::Element *)((uint8_t *)pAddress + 2224), degree, 278),
        in2((Goldilocks::Element *)((uint8_t *)pAddress + 2232), degree, 279),
        in3((Goldilocks::Element *)((uint8_t *)pAddress + 2240), degree, 280),
        in4((Goldilocks::Element *)((uint8_t *)pAddress + 2248), degree, 281),
        in5((Goldilocks::Element *)((uint8_t *)pAddress + 2256), degree, 282),
        in6((Goldilocks::Element *)((uint8_t *)pAddress + 2264), degree, 283),
        in7((Goldilocks::Element *)((uint8_t *)pAddress + 2272), degree, 284),
        hashType((Goldilocks::Element *)((uint8_t *)pAddress + 2280), degree, 285),
        cap1((Goldilocks::Element *)((uint8_t *)pAddress + 2288), degree, 286),
        cap2((Goldilocks::Element *)((uint8_t *)pAddress + 2296), degree, 287),
        cap3((Goldilocks::Element *)((uint8_t *)pAddress + 2304), degree, 288),
        hash0((Goldilocks::Element *)((uint8_t *)pAddress + 2312), degree, 289),
        hash1((Goldilocks::Element *)((uint8_t *)pAddress + 2320), degree, 290),
        hash2((Goldilocks::Element *)((uint8_t *)pAddress + 2328), degree, 291),
        hash3((Goldilocks::Element *)((uint8_t *)pAddress + 2336), degree, 292),
        result1((Goldilocks::Element *)((uint8_t *)pAddress + 2344), degree, 293),
        result2((Goldilocks::Element *)((uint8_t *)pAddress + 2352), degree, 294),
        result3((Goldilocks::Element *)((uint8_t *)pAddress + 2360), degree, 295),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 19; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*19*sizeof(Goldilocks::Element); }
};

class PaddingPGCommitPols
{
public:
    CommitPol acc[8];
    CommitPol freeIn;
    CommitPol addr;
    CommitPol rem;
    CommitPol remInv;
    CommitPol spare;
    CommitPol lastHashLen;
    CommitPol lastHashDigest;
    CommitPol curHash0;
    CommitPol curHash1;
    CommitPol curHash2;
    CommitPol curHash3;
    CommitPol prevHash0;
    CommitPol prevHash1;
    CommitPol prevHash2;
    CommitPol prevHash3;
    CommitPol incCounter;
    CommitPol len;
    CommitPol crOffset;
    CommitPol crLen;
    CommitPol crOffsetInv;
    CommitPol crF0;
    CommitPol crF1;
    CommitPol crF2;
    CommitPol crF3;
    CommitPol crF4;
    CommitPol crF5;
    CommitPol crF6;
    CommitPol crF7;
    CommitPol crV0;
    CommitPol crV1;
    CommitPol crV2;
    CommitPol crV3;
    CommitPol crV4;
    CommitPol crV5;
    CommitPol crV6;
    CommitPol crV7;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    PaddingPGCommitPols (void * pAddress, uint64_t degree) :
        acc{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2368), degree, 296),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2376), degree, 297),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2384), degree, 298),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2392), degree, 299),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2400), degree, 300),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2408), degree, 301),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2416), degree, 302),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 2424), degree, 303)
        },
        freeIn((Goldilocks::Element *)((uint8_t *)pAddress + 2432), degree, 304),
        addr((Goldilocks::Element *)((uint8_t *)pAddress + 2440), degree, 305),
        rem((Goldilocks::Element *)((uint8_t *)pAddress + 2448), degree, 306),
        remInv((Goldilocks::Element *)((uint8_t *)pAddress + 2456), degree, 307),
        spare((Goldilocks::Element *)((uint8_t *)pAddress + 2464), degree, 308),
        lastHashLen((Goldilocks::Element *)((uint8_t *)pAddress + 2472), degree, 309),
        lastHashDigest((Goldilocks::Element *)((uint8_t *)pAddress + 2480), degree, 310),
        curHash0((Goldilocks::Element *)((uint8_t *)pAddress + 2488), degree, 311),
        curHash1((Goldilocks::Element *)((uint8_t *)pAddress + 2496), degree, 312),
        curHash2((Goldilocks::Element *)((uint8_t *)pAddress + 2504), degree, 313),
        curHash3((Goldilocks::Element *)((uint8_t *)pAddress + 2512), degree, 314),
        prevHash0((Goldilocks::Element *)((uint8_t *)pAddress + 2520), degree, 315),
        prevHash1((Goldilocks::Element *)((uint8_t *)pAddress + 2528), degree, 316),
        prevHash2((Goldilocks::Element *)((uint8_t *)pAddress + 2536), degree, 317),
        prevHash3((Goldilocks::Element *)((uint8_t *)pAddress + 2544), degree, 318),
        incCounter((Goldilocks::Element *)((uint8_t *)pAddress + 2552), degree, 319),
        len((Goldilocks::Element *)((uint8_t *)pAddress + 2560), degree, 320),
        crOffset((Goldilocks::Element *)((uint8_t *)pAddress + 2568), degree, 321),
        crLen((Goldilocks::Element *)((uint8_t *)pAddress + 2576), degree, 322),
        crOffsetInv((Goldilocks::Element *)((uint8_t *)pAddress + 2584), degree, 323),
        crF0((Goldilocks::Element *)((uint8_t *)pAddress + 2592), degree, 324),
        crF1((Goldilocks::Element *)((uint8_t *)pAddress + 2600), degree, 325),
        crF2((Goldilocks::Element *)((uint8_t *)pAddress + 2608), degree, 326),
        crF3((Goldilocks::Element *)((uint8_t *)pAddress + 2616), degree, 327),
        crF4((Goldilocks::Element *)((uint8_t *)pAddress + 2624), degree, 328),
        crF5((Goldilocks::Element *)((uint8_t *)pAddress + 2632), degree, 329),
        crF6((Goldilocks::Element *)((uint8_t *)pAddress + 2640), degree, 330),
        crF7((Goldilocks::Element *)((uint8_t *)pAddress + 2648), degree, 331),
        crV0((Goldilocks::Element *)((uint8_t *)pAddress + 2656), degree, 332),
        crV1((Goldilocks::Element *)((uint8_t *)pAddress + 2664), degree, 333),
        crV2((Goldilocks::Element *)((uint8_t *)pAddress + 2672), degree, 334),
        crV3((Goldilocks::Element *)((uint8_t *)pAddress + 2680), degree, 335),
        crV4((Goldilocks::Element *)((uint8_t *)pAddress + 2688), degree, 336),
        crV5((Goldilocks::Element *)((uint8_t *)pAddress + 2696), degree, 337),
        crV6((Goldilocks::Element *)((uint8_t *)pAddress + 2704), degree, 338),
        crV7((Goldilocks::Element *)((uint8_t *)pAddress + 2712), degree, 339),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 44; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*44*sizeof(Goldilocks::Element); }
};

class ClimbKeyCommitPols
{
public:
    CommitPol key0;
    CommitPol key1;
    CommitPol key2;
    CommitPol key3;
    CommitPol level;
    CommitPol keyIn;
    CommitPol keyInChunk;
    CommitPol result;
    CommitPol bit;
    CommitPol keySel0;
    CommitPol keySel1;
    CommitPol keySel2;
    CommitPol keySel3;
    CommitPol carryLt;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    ClimbKeyCommitPols (void * pAddress, uint64_t degree) :
        key0((Goldilocks::Element *)((uint8_t *)pAddress + 2720), degree, 340),
        key1((Goldilocks::Element *)((uint8_t *)pAddress + 2728), degree, 341),
        key2((Goldilocks::Element *)((uint8_t *)pAddress + 2736), degree, 342),
        key3((Goldilocks::Element *)((uint8_t *)pAddress + 2744), degree, 343),
        level((Goldilocks::Element *)((uint8_t *)pAddress + 2752), degree, 344),
        keyIn((Goldilocks::Element *)((uint8_t *)pAddress + 2760), degree, 345),
        keyInChunk((Goldilocks::Element *)((uint8_t *)pAddress + 2768), degree, 346),
        result((Goldilocks::Element *)((uint8_t *)pAddress + 2776), degree, 347),
        bit((Goldilocks::Element *)((uint8_t *)pAddress + 2784), degree, 348),
        keySel0((Goldilocks::Element *)((uint8_t *)pAddress + 2792), degree, 349),
        keySel1((Goldilocks::Element *)((uint8_t *)pAddress + 2800), degree, 350),
        keySel2((Goldilocks::Element *)((uint8_t *)pAddress + 2808), degree, 351),
        keySel3((Goldilocks::Element *)((uint8_t *)pAddress + 2816), degree, 352),
        carryLt((Goldilocks::Element *)((uint8_t *)pAddress + 2824), degree, 353),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 14; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*14*sizeof(Goldilocks::Element); }
};

class StorageCommitPols
{
public:
    CommitPol free0;
    CommitPol free1;
    CommitPol free2;
    CommitPol free3;
    CommitPol hashLeft0;
    CommitPol hashLeft1;
    CommitPol hashLeft2;
    CommitPol hashLeft3;
    CommitPol hashRight0;
    CommitPol hashRight1;
    CommitPol hashRight2;
    CommitPol hashRight3;
    CommitPol oldRoot0;
    CommitPol oldRoot1;
    CommitPol oldRoot2;
    CommitPol oldRoot3;
    CommitPol newRoot0;
    CommitPol newRoot1;
    CommitPol newRoot2;
    CommitPol newRoot3;
    CommitPol valueLow0;
    CommitPol valueLow1;
    CommitPol valueLow2;
    CommitPol valueLow3;
    CommitPol valueHigh0;
    CommitPol valueHigh1;
    CommitPol valueHigh2;
    CommitPol valueHigh3;
    CommitPol siblingValueHash0;
    CommitPol siblingValueHash1;
    CommitPol siblingValueHash2;
    CommitPol siblingValueHash3;
    CommitPol rkey0;
    CommitPol rkey1;
    CommitPol rkey2;
    CommitPol rkey3;
    CommitPol siblingRkey0;
    CommitPol siblingRkey1;
    CommitPol siblingRkey2;
    CommitPol siblingRkey3;
    CommitPol rkeyBit;
    CommitPol level;
    CommitPol pc;
    CommitPol inOldRoot;
    CommitPol inNewRoot;
    CommitPol inValueLow;
    CommitPol inValueHigh;
    CommitPol inSiblingValueHash;
    CommitPol inRkey;
    CommitPol inRkeyBit;
    CommitPol inSiblingRkey;
    CommitPol inFree;
    CommitPol inRotlVh;
    CommitPol inLevel;
    CommitPol setHashLeft;
    CommitPol setHashRight;
    CommitPol setOldRoot;
    CommitPol setNewRoot;
    CommitPol setValueLow;
    CommitPol setValueHigh;
    CommitPol setSiblingValueHash;
    CommitPol setRkey;
    CommitPol setSiblingRkey;
    CommitPol setRkeyBit;
    CommitPol setLevel;
    CommitPol hash;
    CommitPol hashType;
    CommitPol latchSet;
    CommitPol latchGet;
    CommitPol climbRkey;
    CommitPol climbSiblingRkey;
    CommitPol climbBitN;
    CommitPol jmpz;
    CommitPol jmpnz;
    CommitPol jmp;
    CommitPol const0;
    CommitPol jmpAddress;
    CommitPol incCounter;
    CommitPol op0inv;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    StorageCommitPols (void * pAddress, uint64_t degree) :
        free0((Goldilocks::Element *)((uint8_t *)pAddress + 2832), degree, 354),
        free1((Goldilocks::Element *)((uint8_t *)pAddress + 2840), degree, 355),
        free2((Goldilocks::Element *)((uint8_t *)pAddress + 2848), degree, 356),
        free3((Goldilocks::Element *)((uint8_t *)pAddress + 2856), degree, 357),
        hashLeft0((Goldilocks::Element *)((uint8_t *)pAddress + 2864), degree, 358),
        hashLeft1((Goldilocks::Element *)((uint8_t *)pAddress + 2872), degree, 359),
        hashLeft2((Goldilocks::Element *)((uint8_t *)pAddress + 2880), degree, 360),
        hashLeft3((Goldilocks::Element *)((uint8_t *)pAddress + 2888), degree, 361),
        hashRight0((Goldilocks::Element *)((uint8_t *)pAddress + 2896), degree, 362),
        hashRight1((Goldilocks::Element *)((uint8_t *)pAddress + 2904), degree, 363),
        hashRight2((Goldilocks::Element *)((uint8_t *)pAddress + 2912), degree, 364),
        hashRight3((Goldilocks::Element *)((uint8_t *)pAddress + 2920), degree, 365),
        oldRoot0((Goldilocks::Element *)((uint8_t *)pAddress + 2928), degree, 366),
        oldRoot1((Goldilocks::Element *)((uint8_t *)pAddress + 2936), degree, 367),
        oldRoot2((Goldilocks::Element *)((uint8_t *)pAddress + 2944), degree, 368),
        oldRoot3((Goldilocks::Element *)((uint8_t *)pAddress + 2952), degree, 369),
        newRoot0((Goldilocks::Element *)((uint8_t *)pAddress + 2960), degree, 370),
        newRoot1((Goldilocks::Element *)((uint8_t *)pAddress + 2968), degree, 371),
        newRoot2((Goldilocks::Element *)((uint8_t *)pAddress + 2976), degree, 372),
        newRoot3((Goldilocks::Element *)((uint8_t *)pAddress + 2984), degree, 373),
        valueLow0((Goldilocks::Element *)((uint8_t *)pAddress + 2992), degree, 374),
        valueLow1((Goldilocks::Element *)((uint8_t *)pAddress + 3000), degree, 375),
        valueLow2((Goldilocks::Element *)((uint8_t *)pAddress + 3008), degree, 376),
        valueLow3((Goldilocks::Element *)((uint8_t *)pAddress + 3016), degree, 377),
        valueHigh0((Goldilocks::Element *)((uint8_t *)pAddress + 3024), degree, 378),
        valueHigh1((Goldilocks::Element *)((uint8_t *)pAddress + 3032), degree, 379),
        valueHigh2((Goldilocks::Element *)((uint8_t *)pAddress + 3040), degree, 380),
        valueHigh3((Goldilocks::Element *)((uint8_t *)pAddress + 3048), degree, 381),
        siblingValueHash0((Goldilocks::Element *)((uint8_t *)pAddress + 3056), degree, 382),
        siblingValueHash1((Goldilocks::Element *)((uint8_t *)pAddress + 3064), degree, 383),
        siblingValueHash2((Goldilocks::Element *)((uint8_t *)pAddress + 3072), degree, 384),
        siblingValueHash3((Goldilocks::Element *)((uint8_t *)pAddress + 3080), degree, 385),
        rkey0((Goldilocks::Element *)((uint8_t *)pAddress + 3088), degree, 386),
        rkey1((Goldilocks::Element *)((uint8_t *)pAddress + 3096), degree, 387),
        rkey2((Goldilocks::Element *)((uint8_t *)pAddress + 3104), degree, 388),
        rkey3((Goldilocks::Element *)((uint8_t *)pAddress + 3112), degree, 389),
        siblingRkey0((Goldilocks::Element *)((uint8_t *)pAddress + 3120), degree, 390),
        siblingRkey1((Goldilocks::Element *)((uint8_t *)pAddress + 3128), degree, 391),
        siblingRkey2((Goldilocks::Element *)((uint8_t *)pAddress + 3136), degree, 392),
        siblingRkey3((Goldilocks::Element *)((uint8_t *)pAddress + 3144), degree, 393),
        rkeyBit((Goldilocks::Element *)((uint8_t *)pAddress + 3152), degree, 394),
        level((Goldilocks::Element *)((uint8_t *)pAddress + 3160), degree, 395),
        pc((Goldilocks::Element *)((uint8_t *)pAddress + 3168), degree, 396),
        inOldRoot((Goldilocks::Element *)((uint8_t *)pAddress + 3176), degree, 397),
        inNewRoot((Goldilocks::Element *)((uint8_t *)pAddress + 3184), degree, 398),
        inValueLow((Goldilocks::Element *)((uint8_t *)pAddress + 3192), degree, 399),
        inValueHigh((Goldilocks::Element *)((uint8_t *)pAddress + 3200), degree, 400),
        inSiblingValueHash((Goldilocks::Element *)((uint8_t *)pAddress + 3208), degree, 401),
        inRkey((Goldilocks::Element *)((uint8_t *)pAddress + 3216), degree, 402),
        inRkeyBit((Goldilocks::Element *)((uint8_t *)pAddress + 3224), degree, 403),
        inSiblingRkey((Goldilocks::Element *)((uint8_t *)pAddress + 3232), degree, 404),
        inFree((Goldilocks::Element *)((uint8_t *)pAddress + 3240), degree, 405),
        inRotlVh((Goldilocks::Element *)((uint8_t *)pAddress + 3248), degree, 406),
        inLevel((Goldilocks::Element *)((uint8_t *)pAddress + 3256), degree, 407),
        setHashLeft((Goldilocks::Element *)((uint8_t *)pAddress + 3264), degree, 408),
        setHashRight((Goldilocks::Element *)((uint8_t *)pAddress + 3272), degree, 409),
        setOldRoot((Goldilocks::Element *)((uint8_t *)pAddress + 3280), degree, 410),
        setNewRoot((Goldilocks::Element *)((uint8_t *)pAddress + 3288), degree, 411),
        setValueLow((Goldilocks::Element *)((uint8_t *)pAddress + 3296), degree, 412),
        setValueHigh((Goldilocks::Element *)((uint8_t *)pAddress + 3304), degree, 413),
        setSiblingValueHash((Goldilocks::Element *)((uint8_t *)pAddress + 3312), degree, 414),
        setRkey((Goldilocks::Element *)((uint8_t *)pAddress + 3320), degree, 415),
        setSiblingRkey((Goldilocks::Element *)((uint8_t *)pAddress + 3328), degree, 416),
        setRkeyBit((Goldilocks::Element *)((uint8_t *)pAddress + 3336), degree, 417),
        setLevel((Goldilocks::Element *)((uint8_t *)pAddress + 3344), degree, 418),
        hash((Goldilocks::Element *)((uint8_t *)pAddress + 3352), degree, 419),
        hashType((Goldilocks::Element *)((uint8_t *)pAddress + 3360), degree, 420),
        latchSet((Goldilocks::Element *)((uint8_t *)pAddress + 3368), degree, 421),
        latchGet((Goldilocks::Element *)((uint8_t *)pAddress + 3376), degree, 422),
        climbRkey((Goldilocks::Element *)((uint8_t *)pAddress + 3384), degree, 423),
        climbSiblingRkey((Goldilocks::Element *)((uint8_t *)pAddress + 3392), degree, 424),
        climbBitN((Goldilocks::Element *)((uint8_t *)pAddress + 3400), degree, 425),
        jmpz((Goldilocks::Element *)((uint8_t *)pAddress + 3408), degree, 426),
        jmpnz((Goldilocks::Element *)((uint8_t *)pAddress + 3416), degree, 427),
        jmp((Goldilocks::Element *)((uint8_t *)pAddress + 3424), degree, 428),
        const0((Goldilocks::Element *)((uint8_t *)pAddress + 3432), degree, 429),
        jmpAddress((Goldilocks::Element *)((uint8_t *)pAddress + 3440), degree, 430),
        incCounter((Goldilocks::Element *)((uint8_t *)pAddress + 3448), degree, 431),
        op0inv((Goldilocks::Element *)((uint8_t *)pAddress + 3456), degree, 432),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 79; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*79*sizeof(Goldilocks::Element); }
};

class KeccakFCommitPols
{
public:
    CommitPol a[4];
    CommitPol b[4];
    CommitPol c[4];
private:
    void * _pAddress;
    uint64_t _degree;
public:

    KeccakFCommitPols (void * pAddress, uint64_t degree) :
        a{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 3464), degree, 433),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 3472), degree, 434),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 3480), degree, 435),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 3488), degree, 436)
        },
        b{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 3496), degree, 437),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 3504), degree, 438),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 3512), degree, 439),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 3520), degree, 440)
        },
        c{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 3528), degree, 441),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 3536), degree, 442),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 3544), degree, 443),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 3552), degree, 444)
        },
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 12; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*12*sizeof(Goldilocks::Element); }
};

class Bits2FieldCommitPols
{
public:
    CommitPol bit;
    CommitPol field44;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    Bits2FieldCommitPols (void * pAddress, uint64_t degree) :
        bit((Goldilocks::Element *)((uint8_t *)pAddress + 3560), degree, 445),
        field44((Goldilocks::Element *)((uint8_t *)pAddress + 3568), degree, 446),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 2; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*2*sizeof(Goldilocks::Element); }
};

class PaddingKKBitCommitPols
{
public:
    CommitPol rBit;
    CommitPol sOutBit;
    CommitPol r8;
    CommitPol connected;
    CommitPol sOut0;
    CommitPol sOut1;
    CommitPol sOut2;
    CommitPol sOut3;
    CommitPol sOut4;
    CommitPol sOut5;
    CommitPol sOut6;
    CommitPol sOut7;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    PaddingKKBitCommitPols (void * pAddress, uint64_t degree) :
        rBit((Goldilocks::Element *)((uint8_t *)pAddress + 3576), degree, 447),
        sOutBit((Goldilocks::Element *)((uint8_t *)pAddress + 3584), degree, 448),
        r8((Goldilocks::Element *)((uint8_t *)pAddress + 3592), degree, 449),
        connected((Goldilocks::Element *)((uint8_t *)pAddress + 3600), degree, 450),
        sOut0((Goldilocks::Element *)((uint8_t *)pAddress + 3608), degree, 451),
        sOut1((Goldilocks::Element *)((uint8_t *)pAddress + 3616), degree, 452),
        sOut2((Goldilocks::Element *)((uint8_t *)pAddress + 3624), degree, 453),
        sOut3((Goldilocks::Element *)((uint8_t *)pAddress + 3632), degree, 454),
        sOut4((Goldilocks::Element *)((uint8_t *)pAddress + 3640), degree, 455),
        sOut5((Goldilocks::Element *)((uint8_t *)pAddress + 3648), degree, 456),
        sOut6((Goldilocks::Element *)((uint8_t *)pAddress + 3656), degree, 457),
        sOut7((Goldilocks::Element *)((uint8_t *)pAddress + 3664), degree, 458),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 12; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*12*sizeof(Goldilocks::Element); }
};

class PaddingKKCommitPols
{
public:
    CommitPol freeIn;
    CommitPol connected;
    CommitPol addr;
    CommitPol rem;
    CommitPol remInv;
    CommitPol spare;
    CommitPol lastHashLen;
    CommitPol lastHashDigest;
    CommitPol len;
    CommitPol hash0;
    CommitPol hash1;
    CommitPol hash2;
    CommitPol hash3;
    CommitPol hash4;
    CommitPol hash5;
    CommitPol hash6;
    CommitPol hash7;
    CommitPol incCounter;
    CommitPol crOffset;
    CommitPol crLen;
    CommitPol crOffsetInv;
    CommitPol crF0;
    CommitPol crF1;
    CommitPol crF2;
    CommitPol crF3;
    CommitPol crF4;
    CommitPol crF5;
    CommitPol crF6;
    CommitPol crF7;
    CommitPol crV0;
    CommitPol crV1;
    CommitPol crV2;
    CommitPol crV3;
    CommitPol crV4;
    CommitPol crV5;
    CommitPol crV6;
    CommitPol crV7;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    PaddingKKCommitPols (void * pAddress, uint64_t degree) :
        freeIn((Goldilocks::Element *)((uint8_t *)pAddress + 3672), degree, 459),
        connected((Goldilocks::Element *)((uint8_t *)pAddress + 3680), degree, 460),
        addr((Goldilocks::Element *)((uint8_t *)pAddress + 3688), degree, 461),
        rem((Goldilocks::Element *)((uint8_t *)pAddress + 3696), degree, 462),
        remInv((Goldilocks::Element *)((uint8_t *)pAddress + 3704), degree, 463),
        spare((Goldilocks::Element *)((uint8_t *)pAddress + 3712), degree, 464),
        lastHashLen((Goldilocks::Element *)((uint8_t *)pAddress + 3720), degree, 465),
        lastHashDigest((Goldilocks::Element *)((uint8_t *)pAddress + 3728), degree, 466),
        len((Goldilocks::Element *)((uint8_t *)pAddress + 3736), degree, 467),
        hash0((Goldilocks::Element *)((uint8_t *)pAddress + 3744), degree, 468),
        hash1((Goldilocks::Element *)((uint8_t *)pAddress + 3752), degree, 469),
        hash2((Goldilocks::Element *)((uint8_t *)pAddress + 3760), degree, 470),
        hash3((Goldilocks::Element *)((uint8_t *)pAddress + 3768), degree, 471),
        hash4((Goldilocks::Element *)((uint8_t *)pAddress + 3776), degree, 472),
        hash5((Goldilocks::Element *)((uint8_t *)pAddress + 3784), degree, 473),
        hash6((Goldilocks::Element *)((uint8_t *)pAddress + 3792), degree, 474),
        hash7((Goldilocks::Element *)((uint8_t *)pAddress + 3800), degree, 475),
        incCounter((Goldilocks::Element *)((uint8_t *)pAddress + 3808), degree, 476),
        crOffset((Goldilocks::Element *)((uint8_t *)pAddress + 3816), degree, 477),
        crLen((Goldilocks::Element *)((uint8_t *)pAddress + 3824), degree, 478),
        crOffsetInv((Goldilocks::Element *)((uint8_t *)pAddress + 3832), degree, 479),
        crF0((Goldilocks::Element *)((uint8_t *)pAddress + 3840), degree, 480),
        crF1((Goldilocks::Element *)((uint8_t *)pAddress + 3848), degree, 481),
        crF2((Goldilocks::Element *)((uint8_t *)pAddress + 3856), degree, 482),
        crF3((Goldilocks::Element *)((uint8_t *)pAddress + 3864), degree, 483),
        crF4((Goldilocks::Element *)((uint8_t *)pAddress + 3872), degree, 484),
        crF5((Goldilocks::Element *)((uint8_t *)pAddress + 3880), degree, 485),
        crF6((Goldilocks::Element *)((uint8_t *)pAddress + 3888), degree, 486),
        crF7((Goldilocks::Element *)((uint8_t *)pAddress + 3896), degree, 487),
        crV0((Goldilocks::Element *)((uint8_t *)pAddress + 3904), degree, 488),
        crV1((Goldilocks::Element *)((uint8_t *)pAddress + 3912), degree, 489),
        crV2((Goldilocks::Element *)((uint8_t *)pAddress + 3920), degree, 490),
        crV3((Goldilocks::Element *)((uint8_t *)pAddress + 3928), degree, 491),
        crV4((Goldilocks::Element *)((uint8_t *)pAddress + 3936), degree, 492),
        crV5((Goldilocks::Element *)((uint8_t *)pAddress + 3944), degree, 493),
        crV6((Goldilocks::Element *)((uint8_t *)pAddress + 3952), degree, 494),
        crV7((Goldilocks::Element *)((uint8_t *)pAddress + 3960), degree, 495),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 37; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*37*sizeof(Goldilocks::Element); }
};

class MemCommitPols
{
public:
    CommitPol addr;
    CommitPol step;
    CommitPol mOp;
    CommitPol mWr;
    CommitPol val[8];
    CommitPol lastAccess;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    MemCommitPols (void * pAddress, uint64_t degree) :
        addr((Goldilocks::Element *)((uint8_t *)pAddress + 3968), degree, 496),
        step((Goldilocks::Element *)((uint8_t *)pAddress + 3976), degree, 497),
        mOp((Goldilocks::Element *)((uint8_t *)pAddress + 3984), degree, 498),
        mWr((Goldilocks::Element *)((uint8_t *)pAddress + 3992), degree, 499),
        val{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 4000), degree, 500),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 4008), degree, 501),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 4016), degree, 502),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 4024), degree, 503),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 4032), degree, 504),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 4040), degree, 505),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 4048), degree, 506),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 4056), degree, 507)
        },
        lastAccess((Goldilocks::Element *)((uint8_t *)pAddress + 4064), degree, 508),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 13; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*13*sizeof(Goldilocks::Element); }
};

class Sha256FCommitPols
{
public:
    CommitPol input[3];
    CommitPol output;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    Sha256FCommitPols (void * pAddress, uint64_t degree) :
        input{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 4072), degree, 509),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 4080), degree, 510),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 4088), degree, 511)
        },
        output((Goldilocks::Element *)((uint8_t *)pAddress + 4096), degree, 512),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 4; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*4*sizeof(Goldilocks::Element); }
};

class Bits2FieldSha256CommitPols
{
public:
    CommitPol bit;
    CommitPol packField;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    Bits2FieldSha256CommitPols (void * pAddress, uint64_t degree) :
        bit((Goldilocks::Element *)((uint8_t *)pAddress + 4104), degree, 513),
        packField((Goldilocks::Element *)((uint8_t *)pAddress + 4112), degree, 514),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 2; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*2*sizeof(Goldilocks::Element); }
};

class PaddingSha256BitCommitPols
{
public:
    CommitPol s1;
    CommitPol s2;
    CommitPol r8;
    CommitPol connected;
    CommitPol sOut0;
    CommitPol sOut1;
    CommitPol sOut2;
    CommitPol sOut3;
    CommitPol sOut4;
    CommitPol sOut5;
    CommitPol sOut6;
    CommitPol sOut7;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    PaddingSha256BitCommitPols (void * pAddress, uint64_t degree) :
        s1((Goldilocks::Element *)((uint8_t *)pAddress + 4120), degree, 515),
        s2((Goldilocks::Element *)((uint8_t *)pAddress + 4128), degree, 516),
        r8((Goldilocks::Element *)((uint8_t *)pAddress + 4136), degree, 517),
        connected((Goldilocks::Element *)((uint8_t *)pAddress + 4144), degree, 518),
        sOut0((Goldilocks::Element *)((uint8_t *)pAddress + 4152), degree, 519),
        sOut1((Goldilocks::Element *)((uint8_t *)pAddress + 4160), degree, 520),
        sOut2((Goldilocks::Element *)((uint8_t *)pAddress + 4168), degree, 521),
        sOut3((Goldilocks::Element *)((uint8_t *)pAddress + 4176), degree, 522),
        sOut4((Goldilocks::Element *)((uint8_t *)pAddress + 4184), degree, 523),
        sOut5((Goldilocks::Element *)((uint8_t *)pAddress + 4192), degree, 524),
        sOut6((Goldilocks::Element *)((uint8_t *)pAddress + 4200), degree, 525),
        sOut7((Goldilocks::Element *)((uint8_t *)pAddress + 4208), degree, 526),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 12; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*12*sizeof(Goldilocks::Element); }
};

class PaddingSha256CommitPols
{
public:
    CommitPol freeIn;
    CommitPol connected;
    CommitPol addr;
    CommitPol rem;
    CommitPol remInv;
    CommitPol spare;
    CommitPol lengthSection;
    CommitPol accLength;
    CommitPol lastHashLen;
    CommitPol lastHashDigest;
    CommitPol len;
    CommitPol hash0;
    CommitPol hash1;
    CommitPol hash2;
    CommitPol hash3;
    CommitPol hash4;
    CommitPol hash5;
    CommitPol hash6;
    CommitPol hash7;
    CommitPol incCounter;
    CommitPol crOffset;
    CommitPol crLen;
    CommitPol crOffsetInv;
    CommitPol crF0;
    CommitPol crF1;
    CommitPol crF2;
    CommitPol crF3;
    CommitPol crF4;
    CommitPol crF5;
    CommitPol crF6;
    CommitPol crF7;
    CommitPol crV0;
    CommitPol crV1;
    CommitPol crV2;
    CommitPol crV3;
    CommitPol crV4;
    CommitPol crV5;
    CommitPol crV6;
    CommitPol crV7;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    PaddingSha256CommitPols (void * pAddress, uint64_t degree) :
        freeIn((Goldilocks::Element *)((uint8_t *)pAddress + 4216), degree, 527),
        connected((Goldilocks::Element *)((uint8_t *)pAddress + 4224), degree, 528),
        addr((Goldilocks::Element *)((uint8_t *)pAddress + 4232), degree, 529),
        rem((Goldilocks::Element *)((uint8_t *)pAddress + 4240), degree, 530),
        remInv((Goldilocks::Element *)((uint8_t *)pAddress + 4248), degree, 531),
        spare((Goldilocks::Element *)((uint8_t *)pAddress + 4256), degree, 532),
        lengthSection((Goldilocks::Element *)((uint8_t *)pAddress + 4264), degree, 533),
        accLength((Goldilocks::Element *)((uint8_t *)pAddress + 4272), degree, 534),
        lastHashLen((Goldilocks::Element *)((uint8_t *)pAddress + 4280), degree, 535),
        lastHashDigest((Goldilocks::Element *)((uint8_t *)pAddress + 4288), degree, 536),
        len((Goldilocks::Element *)((uint8_t *)pAddress + 4296), degree, 537),
        hash0((Goldilocks::Element *)((uint8_t *)pAddress + 4304), degree, 538),
        hash1((Goldilocks::Element *)((uint8_t *)pAddress + 4312), degree, 539),
        hash2((Goldilocks::Element *)((uint8_t *)pAddress + 4320), degree, 540),
        hash3((Goldilocks::Element *)((uint8_t *)pAddress + 4328), degree, 541),
        hash4((Goldilocks::Element *)((uint8_t *)pAddress + 4336), degree, 542),
        hash5((Goldilocks::Element *)((uint8_t *)pAddress + 4344), degree, 543),
        hash6((Goldilocks::Element *)((uint8_t *)pAddress + 4352), degree, 544),
        hash7((Goldilocks::Element *)((uint8_t *)pAddress + 4360), degree, 545),
        incCounter((Goldilocks::Element *)((uint8_t *)pAddress + 4368), degree, 546),
        crOffset((Goldilocks::Element *)((uint8_t *)pAddress + 4376), degree, 547),
        crLen((Goldilocks::Element *)((uint8_t *)pAddress + 4384), degree, 548),
        crOffsetInv((Goldilocks::Element *)((uint8_t *)pAddress + 4392), degree, 549),
        crF0((Goldilocks::Element *)((uint8_t *)pAddress + 4400), degree, 550),
        crF1((Goldilocks::Element *)((uint8_t *)pAddress + 4408), degree, 551),
        crF2((Goldilocks::Element *)((uint8_t *)pAddress + 4416), degree, 552),
        crF3((Goldilocks::Element *)((uint8_t *)pAddress + 4424), degree, 553),
        crF4((Goldilocks::Element *)((uint8_t *)pAddress + 4432), degree, 554),
        crF5((Goldilocks::Element *)((uint8_t *)pAddress + 4440), degree, 555),
        crF6((Goldilocks::Element *)((uint8_t *)pAddress + 4448), degree, 556),
        crF7((Goldilocks::Element *)((uint8_t *)pAddress + 4456), degree, 557),
        crV0((Goldilocks::Element *)((uint8_t *)pAddress + 4464), degree, 558),
        crV1((Goldilocks::Element *)((uint8_t *)pAddress + 4472), degree, 559),
        crV2((Goldilocks::Element *)((uint8_t *)pAddress + 4480), degree, 560),
        crV3((Goldilocks::Element *)((uint8_t *)pAddress + 4488), degree, 561),
        crV4((Goldilocks::Element *)((uint8_t *)pAddress + 4496), degree, 562),
        crV5((Goldilocks::Element *)((uint8_t *)pAddress + 4504), degree, 563),
        crV6((Goldilocks::Element *)((uint8_t *)pAddress + 4512), degree, 564),
        crV7((Goldilocks::Element *)((uint8_t *)pAddress + 4520), degree, 565),
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 39; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*39*sizeof(Goldilocks::Element); }
};

class MainCommitPols
{
public:
    CommitPol A7;
    CommitPol A6;
    CommitPol A5;
    CommitPol A4;
    CommitPol A3;
    CommitPol A2;
    CommitPol A1;
    CommitPol A0;
    CommitPol B7;
    CommitPol B6;
    CommitPol B5;
    CommitPol B4;
    CommitPol B3;
    CommitPol B2;
    CommitPol B1;
    CommitPol B0;
    CommitPol C7;
    CommitPol C6;
    CommitPol C5;
    CommitPol C4;
    CommitPol C3;
    CommitPol C2;
    CommitPol C1;
    CommitPol C0;
    CommitPol D7;
    CommitPol D6;
    CommitPol D5;
    CommitPol D4;
    CommitPol D3;
    CommitPol D2;
    CommitPol D1;
    CommitPol D0;
    CommitPol E7;
    CommitPol E6;
    CommitPol E5;
    CommitPol E4;
    CommitPol E3;
    CommitPol E2;
    CommitPol E1;
    CommitPol E0;
    CommitPol SR7;
    CommitPol SR6;
    CommitPol SR5;
    CommitPol SR4;
    CommitPol SR3;
    CommitPol SR2;
    CommitPol SR1;
    CommitPol SR0;
    CommitPol CTX;
    CommitPol SP;
    CommitPol PC;
    CommitPol GAS;
    CommitPol zkPC;
    CommitPol RR;
    CommitPol HASHPOS;
    CommitPol RCX;
    CommitPol CONST7;
    CommitPol CONST6;
    CommitPol CONST5;
    CommitPol CONST4;
    CommitPol CONST3;
    CommitPol CONST2;
    CommitPol CONST1;
    CommitPol CONST0;
    CommitPol FREE7;
    CommitPol FREE6;
    CommitPol FREE5;
    CommitPol FREE4;
    CommitPol FREE3;
    CommitPol FREE2;
    CommitPol FREE1;
    CommitPol FREE0;
    CommitPol inA;
    CommitPol inB;
    CommitPol inC;
    CommitPol inROTL_C;
    CommitPol inD;
    CommitPol inE;
    CommitPol inSR;
    CommitPol inFREE;
    CommitPol inFREE0;
    CommitPol inCTX;
    CommitPol inSP;
    CommitPol inPC;
    CommitPol inGAS;
    CommitPol inSTEP;
    CommitPol inRR;
    CommitPol inHASHPOS;
    CommitPol inRCX;
    CommitPol setA;
    CommitPol setB;
    CommitPol setC;
    CommitPol setD;
    CommitPol setE;
    CommitPol setSR;
    CommitPol setCTX;
    CommitPol setSP;
    CommitPol setPC;
    CommitPol setGAS;
    CommitPol setRR;
    CommitPol setHASHPOS;
    CommitPol setRCX;
    CommitPol JMP;
    CommitPol JMPN;
    CommitPol JMPC;
    CommitPol JMPZ;
    CommitPol offset;
    CommitPol incStack;
    CommitPol isStack;
    CommitPol isMem;
    CommitPol ind;
    CommitPol indRR;
    CommitPol useCTX;
    CommitPol carry;
    CommitPol mOp;
    CommitPol mWR;
    CommitPol sWR;
    CommitPol sRD;
    CommitPol arithEq0;
    CommitPol arithEq1;
    CommitPol arithEq2;
    CommitPol arithEq3;
    CommitPol arithEq4;
    CommitPol arithEq5;
    CommitPol memAlignRD;
    CommitPol memAlignWR;
    CommitPol memAlignWR8;
    CommitPol hashK;
    CommitPol hashK1;
    CommitPol hashKLen;
    CommitPol hashKDigest;
    CommitPol hashP;
    CommitPol hashP1;
    CommitPol hashPLen;
    CommitPol hashPDigest;
    CommitPol hashS;
    CommitPol hashS1;
    CommitPol hashSLen;
    CommitPol hashSDigest;
    CommitPol bin;
    CommitPol binOpcode;
    CommitPol assert_pol;
    CommitPol repeat;
    CommitPol call;
    CommitPol return_pol;
    CommitPol isNeg;
    CommitPol cntArith;
    CommitPol cntBinary;
    CommitPol cntMemAlign;
    CommitPol cntKeccakF;
    CommitPol cntSha256F;
    CommitPol cntPoseidonG;
    CommitPol cntPaddingPG;
    CommitPol inCntArith;
    CommitPol inCntBinary;
    CommitPol inCntMemAlign;
    CommitPol inCntKeccakF;
    CommitPol inCntSha256F;
    CommitPol inCntPoseidonG;
    CommitPol inCntPaddingPG;
    CommitPol incCounter;
    CommitPol lJmpnCondValue;
    CommitPol hJmpnCondValueBit[7];
    CommitPol RCXInv;
    CommitPol op0Inv;
    CommitPol jmpAddr;
    CommitPol elseAddr;
    CommitPol useJmpAddr;
    CommitPol useElseAddr;
    CommitPol sKeyI[4];
    CommitPol sKey[4];
private:
    void * _pAddress;
    uint64_t _degree;
public:

    MainCommitPols (void * pAddress, uint64_t degree) :
        A7((Goldilocks::Element *)((uint8_t *)pAddress + 4528), degree, 566),
        A6((Goldilocks::Element *)((uint8_t *)pAddress + 4536), degree, 567),
        A5((Goldilocks::Element *)((uint8_t *)pAddress + 4544), degree, 568),
        A4((Goldilocks::Element *)((uint8_t *)pAddress + 4552), degree, 569),
        A3((Goldilocks::Element *)((uint8_t *)pAddress + 4560), degree, 570),
        A2((Goldilocks::Element *)((uint8_t *)pAddress + 4568), degree, 571),
        A1((Goldilocks::Element *)((uint8_t *)pAddress + 4576), degree, 572),
        A0((Goldilocks::Element *)((uint8_t *)pAddress + 4584), degree, 573),
        B7((Goldilocks::Element *)((uint8_t *)pAddress + 4592), degree, 574),
        B6((Goldilocks::Element *)((uint8_t *)pAddress + 4600), degree, 575),
        B5((Goldilocks::Element *)((uint8_t *)pAddress + 4608), degree, 576),
        B4((Goldilocks::Element *)((uint8_t *)pAddress + 4616), degree, 577),
        B3((Goldilocks::Element *)((uint8_t *)pAddress + 4624), degree, 578),
        B2((Goldilocks::Element *)((uint8_t *)pAddress + 4632), degree, 579),
        B1((Goldilocks::Element *)((uint8_t *)pAddress + 4640), degree, 580),
        B0((Goldilocks::Element *)((uint8_t *)pAddress + 4648), degree, 581),
        C7((Goldilocks::Element *)((uint8_t *)pAddress + 4656), degree, 582),
        C6((Goldilocks::Element *)((uint8_t *)pAddress + 4664), degree, 583),
        C5((Goldilocks::Element *)((uint8_t *)pAddress + 4672), degree, 584),
        C4((Goldilocks::Element *)((uint8_t *)pAddress + 4680), degree, 585),
        C3((Goldilocks::Element *)((uint8_t *)pAddress + 4688), degree, 586),
        C2((Goldilocks::Element *)((uint8_t *)pAddress + 4696), degree, 587),
        C1((Goldilocks::Element *)((uint8_t *)pAddress + 4704), degree, 588),
        C0((Goldilocks::Element *)((uint8_t *)pAddress + 4712), degree, 589),
        D7((Goldilocks::Element *)((uint8_t *)pAddress + 4720), degree, 590),
        D6((Goldilocks::Element *)((uint8_t *)pAddress + 4728), degree, 591),
        D5((Goldilocks::Element *)((uint8_t *)pAddress + 4736), degree, 592),
        D4((Goldilocks::Element *)((uint8_t *)pAddress + 4744), degree, 593),
        D3((Goldilocks::Element *)((uint8_t *)pAddress + 4752), degree, 594),
        D2((Goldilocks::Element *)((uint8_t *)pAddress + 4760), degree, 595),
        D1((Goldilocks::Element *)((uint8_t *)pAddress + 4768), degree, 596),
        D0((Goldilocks::Element *)((uint8_t *)pAddress + 4776), degree, 597),
        E7((Goldilocks::Element *)((uint8_t *)pAddress + 4784), degree, 598),
        E6((Goldilocks::Element *)((uint8_t *)pAddress + 4792), degree, 599),
        E5((Goldilocks::Element *)((uint8_t *)pAddress + 4800), degree, 600),
        E4((Goldilocks::Element *)((uint8_t *)pAddress + 4808), degree, 601),
        E3((Goldilocks::Element *)((uint8_t *)pAddress + 4816), degree, 602),
        E2((Goldilocks::Element *)((uint8_t *)pAddress + 4824), degree, 603),
        E1((Goldilocks::Element *)((uint8_t *)pAddress + 4832), degree, 604),
        E0((Goldilocks::Element *)((uint8_t *)pAddress + 4840), degree, 605),
        SR7((Goldilocks::Element *)((uint8_t *)pAddress + 4848), degree, 606),
        SR6((Goldilocks::Element *)((uint8_t *)pAddress + 4856), degree, 607),
        SR5((Goldilocks::Element *)((uint8_t *)pAddress + 4864), degree, 608),
        SR4((Goldilocks::Element *)((uint8_t *)pAddress + 4872), degree, 609),
        SR3((Goldilocks::Element *)((uint8_t *)pAddress + 4880), degree, 610),
        SR2((Goldilocks::Element *)((uint8_t *)pAddress + 4888), degree, 611),
        SR1((Goldilocks::Element *)((uint8_t *)pAddress + 4896), degree, 612),
        SR0((Goldilocks::Element *)((uint8_t *)pAddress + 4904), degree, 613),
        CTX((Goldilocks::Element *)((uint8_t *)pAddress + 4912), degree, 614),
        SP((Goldilocks::Element *)((uint8_t *)pAddress + 4920), degree, 615),
        PC((Goldilocks::Element *)((uint8_t *)pAddress + 4928), degree, 616),
        GAS((Goldilocks::Element *)((uint8_t *)pAddress + 4936), degree, 617),
        zkPC((Goldilocks::Element *)((uint8_t *)pAddress + 4944), degree, 618),
        RR((Goldilocks::Element *)((uint8_t *)pAddress + 4952), degree, 619),
        HASHPOS((Goldilocks::Element *)((uint8_t *)pAddress + 4960), degree, 620),
        RCX((Goldilocks::Element *)((uint8_t *)pAddress + 4968), degree, 621),
        CONST7((Goldilocks::Element *)((uint8_t *)pAddress + 4976), degree, 622),
        CONST6((Goldilocks::Element *)((uint8_t *)pAddress + 4984), degree, 623),
        CONST5((Goldilocks::Element *)((uint8_t *)pAddress + 4992), degree, 624),
        CONST4((Goldilocks::Element *)((uint8_t *)pAddress + 5000), degree, 625),
        CONST3((Goldilocks::Element *)((uint8_t *)pAddress + 5008), degree, 626),
        CONST2((Goldilocks::Element *)((uint8_t *)pAddress + 5016), degree, 627),
        CONST1((Goldilocks::Element *)((uint8_t *)pAddress + 5024), degree, 628),
        CONST0((Goldilocks::Element *)((uint8_t *)pAddress + 5032), degree, 629),
        FREE7((Goldilocks::Element *)((uint8_t *)pAddress + 5040), degree, 630),
        FREE6((Goldilocks::Element *)((uint8_t *)pAddress + 5048), degree, 631),
        FREE5((Goldilocks::Element *)((uint8_t *)pAddress + 5056), degree, 632),
        FREE4((Goldilocks::Element *)((uint8_t *)pAddress + 5064), degree, 633),
        FREE3((Goldilocks::Element *)((uint8_t *)pAddress + 5072), degree, 634),
        FREE2((Goldilocks::Element *)((uint8_t *)pAddress + 5080), degree, 635),
        FREE1((Goldilocks::Element *)((uint8_t *)pAddress + 5088), degree, 636),
        FREE0((Goldilocks::Element *)((uint8_t *)pAddress + 5096), degree, 637),
        inA((Goldilocks::Element *)((uint8_t *)pAddress + 5104), degree, 638),
        inB((Goldilocks::Element *)((uint8_t *)pAddress + 5112), degree, 639),
        inC((Goldilocks::Element *)((uint8_t *)pAddress + 5120), degree, 640),
        inROTL_C((Goldilocks::Element *)((uint8_t *)pAddress + 5128), degree, 641),
        inD((Goldilocks::Element *)((uint8_t *)pAddress + 5136), degree, 642),
        inE((Goldilocks::Element *)((uint8_t *)pAddress + 5144), degree, 643),
        inSR((Goldilocks::Element *)((uint8_t *)pAddress + 5152), degree, 644),
        inFREE((Goldilocks::Element *)((uint8_t *)pAddress + 5160), degree, 645),
        inFREE0((Goldilocks::Element *)((uint8_t *)pAddress + 5168), degree, 646),
        inCTX((Goldilocks::Element *)((uint8_t *)pAddress + 5176), degree, 647),
        inSP((Goldilocks::Element *)((uint8_t *)pAddress + 5184), degree, 648),
        inPC((Goldilocks::Element *)((uint8_t *)pAddress + 5192), degree, 649),
        inGAS((Goldilocks::Element *)((uint8_t *)pAddress + 5200), degree, 650),
        inSTEP((Goldilocks::Element *)((uint8_t *)pAddress + 5208), degree, 651),
        inRR((Goldilocks::Element *)((uint8_t *)pAddress + 5216), degree, 652),
        inHASHPOS((Goldilocks::Element *)((uint8_t *)pAddress + 5224), degree, 653),
        inRCX((Goldilocks::Element *)((uint8_t *)pAddress + 5232), degree, 654),
        setA((Goldilocks::Element *)((uint8_t *)pAddress + 5240), degree, 655),
        setB((Goldilocks::Element *)((uint8_t *)pAddress + 5248), degree, 656),
        setC((Goldilocks::Element *)((uint8_t *)pAddress + 5256), degree, 657),
        setD((Goldilocks::Element *)((uint8_t *)pAddress + 5264), degree, 658),
        setE((Goldilocks::Element *)((uint8_t *)pAddress + 5272), degree, 659),
        setSR((Goldilocks::Element *)((uint8_t *)pAddress + 5280), degree, 660),
        setCTX((Goldilocks::Element *)((uint8_t *)pAddress + 5288), degree, 661),
        setSP((Goldilocks::Element *)((uint8_t *)pAddress + 5296), degree, 662),
        setPC((Goldilocks::Element *)((uint8_t *)pAddress + 5304), degree, 663),
        setGAS((Goldilocks::Element *)((uint8_t *)pAddress + 5312), degree, 664),
        setRR((Goldilocks::Element *)((uint8_t *)pAddress + 5320), degree, 665),
        setHASHPOS((Goldilocks::Element *)((uint8_t *)pAddress + 5328), degree, 666),
        setRCX((Goldilocks::Element *)((uint8_t *)pAddress + 5336), degree, 667),
        JMP((Goldilocks::Element *)((uint8_t *)pAddress + 5344), degree, 668),
        JMPN((Goldilocks::Element *)((uint8_t *)pAddress + 5352), degree, 669),
        JMPC((Goldilocks::Element *)((uint8_t *)pAddress + 5360), degree, 670),
        JMPZ((Goldilocks::Element *)((uint8_t *)pAddress + 5368), degree, 671),
        offset((Goldilocks::Element *)((uint8_t *)pAddress + 5376), degree, 672),
        incStack((Goldilocks::Element *)((uint8_t *)pAddress + 5384), degree, 673),
        isStack((Goldilocks::Element *)((uint8_t *)pAddress + 5392), degree, 674),
        isMem((Goldilocks::Element *)((uint8_t *)pAddress + 5400), degree, 675),
        ind((Goldilocks::Element *)((uint8_t *)pAddress + 5408), degree, 676),
        indRR((Goldilocks::Element *)((uint8_t *)pAddress + 5416), degree, 677),
        useCTX((Goldilocks::Element *)((uint8_t *)pAddress + 5424), degree, 678),
        carry((Goldilocks::Element *)((uint8_t *)pAddress + 5432), degree, 679),
        mOp((Goldilocks::Element *)((uint8_t *)pAddress + 5440), degree, 680),
        mWR((Goldilocks::Element *)((uint8_t *)pAddress + 5448), degree, 681),
        sWR((Goldilocks::Element *)((uint8_t *)pAddress + 5456), degree, 682),
        sRD((Goldilocks::Element *)((uint8_t *)pAddress + 5464), degree, 683),
        arithEq0((Goldilocks::Element *)((uint8_t *)pAddress + 5472), degree, 684),
        arithEq1((Goldilocks::Element *)((uint8_t *)pAddress + 5480), degree, 685),
        arithEq2((Goldilocks::Element *)((uint8_t *)pAddress + 5488), degree, 686),
        arithEq3((Goldilocks::Element *)((uint8_t *)pAddress + 5496), degree, 687),
        arithEq4((Goldilocks::Element *)((uint8_t *)pAddress + 5504), degree, 688),
        arithEq5((Goldilocks::Element *)((uint8_t *)pAddress + 5512), degree, 689),
        memAlignRD((Goldilocks::Element *)((uint8_t *)pAddress + 5520), degree, 690),
        memAlignWR((Goldilocks::Element *)((uint8_t *)pAddress + 5528), degree, 691),
        memAlignWR8((Goldilocks::Element *)((uint8_t *)pAddress + 5536), degree, 692),
        hashK((Goldilocks::Element *)((uint8_t *)pAddress + 5544), degree, 693),
        hashK1((Goldilocks::Element *)((uint8_t *)pAddress + 5552), degree, 694),
        hashKLen((Goldilocks::Element *)((uint8_t *)pAddress + 5560), degree, 695),
        hashKDigest((Goldilocks::Element *)((uint8_t *)pAddress + 5568), degree, 696),
        hashP((Goldilocks::Element *)((uint8_t *)pAddress + 5576), degree, 697),
        hashP1((Goldilocks::Element *)((uint8_t *)pAddress + 5584), degree, 698),
        hashPLen((Goldilocks::Element *)((uint8_t *)pAddress + 5592), degree, 699),
        hashPDigest((Goldilocks::Element *)((uint8_t *)pAddress + 5600), degree, 700),
        hashS((Goldilocks::Element *)((uint8_t *)pAddress + 5608), degree, 701),
        hashS1((Goldilocks::Element *)((uint8_t *)pAddress + 5616), degree, 702),
        hashSLen((Goldilocks::Element *)((uint8_t *)pAddress + 5624), degree, 703),
        hashSDigest((Goldilocks::Element *)((uint8_t *)pAddress + 5632), degree, 704),
        bin((Goldilocks::Element *)((uint8_t *)pAddress + 5640), degree, 705),
        binOpcode((Goldilocks::Element *)((uint8_t *)pAddress + 5648), degree, 706),
        assert_pol((Goldilocks::Element *)((uint8_t *)pAddress + 5656), degree, 707),
        repeat((Goldilocks::Element *)((uint8_t *)pAddress + 5664), degree, 708),
        call((Goldilocks::Element *)((uint8_t *)pAddress + 5672), degree, 709),
        return_pol((Goldilocks::Element *)((uint8_t *)pAddress + 5680), degree, 710),
        isNeg((Goldilocks::Element *)((uint8_t *)pAddress + 5688), degree, 711),
        cntArith((Goldilocks::Element *)((uint8_t *)pAddress + 5696), degree, 712),
        cntBinary((Goldilocks::Element *)((uint8_t *)pAddress + 5704), degree, 713),
        cntMemAlign((Goldilocks::Element *)((uint8_t *)pAddress + 5712), degree, 714),
        cntKeccakF((Goldilocks::Element *)((uint8_t *)pAddress + 5720), degree, 715),
        cntSha256F((Goldilocks::Element *)((uint8_t *)pAddress + 5728), degree, 716),
        cntPoseidonG((Goldilocks::Element *)((uint8_t *)pAddress + 5736), degree, 717),
        cntPaddingPG((Goldilocks::Element *)((uint8_t *)pAddress + 5744), degree, 718),
        inCntArith((Goldilocks::Element *)((uint8_t *)pAddress + 5752), degree, 719),
        inCntBinary((Goldilocks::Element *)((uint8_t *)pAddress + 5760), degree, 720),
        inCntMemAlign((Goldilocks::Element *)((uint8_t *)pAddress + 5768), degree, 721),
        inCntKeccakF((Goldilocks::Element *)((uint8_t *)pAddress + 5776), degree, 722),
        inCntSha256F((Goldilocks::Element *)((uint8_t *)pAddress + 5784), degree, 723),
        inCntPoseidonG((Goldilocks::Element *)((uint8_t *)pAddress + 5792), degree, 724),
        inCntPaddingPG((Goldilocks::Element *)((uint8_t *)pAddress + 5800), degree, 725),
        incCounter((Goldilocks::Element *)((uint8_t *)pAddress + 5808), degree, 726),
        lJmpnCondValue((Goldilocks::Element *)((uint8_t *)pAddress + 5816), degree, 727),
        hJmpnCondValueBit{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5824), degree, 728),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5832), degree, 729),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5840), degree, 730),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5848), degree, 731),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5856), degree, 732),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5864), degree, 733),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5872), degree, 734)
        },
        RCXInv((Goldilocks::Element *)((uint8_t *)pAddress + 5880), degree, 735),
        op0Inv((Goldilocks::Element *)((uint8_t *)pAddress + 5888), degree, 736),
        jmpAddr((Goldilocks::Element *)((uint8_t *)pAddress + 5896), degree, 737),
        elseAddr((Goldilocks::Element *)((uint8_t *)pAddress + 5904), degree, 738),
        useJmpAddr((Goldilocks::Element *)((uint8_t *)pAddress + 5912), degree, 739),
        useElseAddr((Goldilocks::Element *)((uint8_t *)pAddress + 5920), degree, 740),
        sKeyI{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5928), degree, 741),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5936), degree, 742),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5944), degree, 743),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5952), degree, 744)
        },
        sKey{
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5960), degree, 745),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5968), degree, 746),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5976), degree, 747),
            CommitPol((Goldilocks::Element *)((uint8_t *)pAddress + 5984), degree, 748)
        },
        _pAddress(pAddress),
        _degree(degree) {};

    inline static uint64_t numPols (void) { return 183; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*183*sizeof(Goldilocks::Element); }
};

class CommitPols
{
public:
    MemAlignCommitPols MemAlign;
    ArithCommitPols Arith;
    BinaryCommitPols Binary;
    PoseidonGCommitPols PoseidonG;
    PaddingPGCommitPols PaddingPG;
    ClimbKeyCommitPols ClimbKey;
    StorageCommitPols Storage;
    KeccakFCommitPols KeccakF;
    Bits2FieldCommitPols Bits2Field;
    PaddingKKBitCommitPols PaddingKKBit;
    PaddingKKCommitPols PaddingKK;
    MemCommitPols Mem;
    Sha256FCommitPols Sha256F;
    Bits2FieldSha256CommitPols Bits2FieldSha256;
    PaddingSha256BitCommitPols PaddingSha256Bit;
    PaddingSha256CommitPols PaddingSha256;
    MainCommitPols Main;
private:
    void * _pAddress;
    uint64_t _degree;
public:

    CommitPols (void * pAddress, uint64_t degree) :
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
        Mem(pAddress, degree),
        Sha256F(pAddress, degree),
        Bits2FieldSha256(pAddress, degree),
        PaddingSha256Bit(pAddress, degree),
        PaddingSha256(pAddress, degree),
        Main(pAddress, degree),
        _pAddress(pAddress),
        _degree(degree) {}

    inline static uint64_t numPols (void) { return 749; }

    inline void * address (void) { return _pAddress; }
    inline uint64_t degree (void) { return _degree; }
    inline uint64_t size (void) { return _degree*749*sizeof(Goldilocks::Element); }

    inline Goldilocks::Element &getElement (uint64_t pol, uint64_t evaluation)
    {
        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * numPols()];
    }
};

inline const char * address2CommitPolName (uint64_t address)
{
    if ((address >= 0) && (address <= 7)) return "MemAlign.inM[0]";
    if ((address >= 8) && (address <= 15)) return "MemAlign.inM[1]";
    if ((address >= 16) && (address <= 23)) return "MemAlign.inV";
    if ((address >= 24) && (address <= 31)) return "MemAlign.wr256";
    if ((address >= 32) && (address <= 39)) return "MemAlign.wr8";
    if ((address >= 40) && (address <= 47)) return "MemAlign.m0[0]";
    if ((address >= 48) && (address <= 55)) return "MemAlign.m0[1]";
    if ((address >= 56) && (address <= 63)) return "MemAlign.m0[2]";
    if ((address >= 64) && (address <= 71)) return "MemAlign.m0[3]";
    if ((address >= 72) && (address <= 79)) return "MemAlign.m0[4]";
    if ((address >= 80) && (address <= 87)) return "MemAlign.m0[5]";
    if ((address >= 88) && (address <= 95)) return "MemAlign.m0[6]";
    if ((address >= 96) && (address <= 103)) return "MemAlign.m0[7]";
    if ((address >= 104) && (address <= 111)) return "MemAlign.m1[0]";
    if ((address >= 112) && (address <= 119)) return "MemAlign.m1[1]";
    if ((address >= 120) && (address <= 127)) return "MemAlign.m1[2]";
    if ((address >= 128) && (address <= 135)) return "MemAlign.m1[3]";
    if ((address >= 136) && (address <= 143)) return "MemAlign.m1[4]";
    if ((address >= 144) && (address <= 151)) return "MemAlign.m1[5]";
    if ((address >= 152) && (address <= 159)) return "MemAlign.m1[6]";
    if ((address >= 160) && (address <= 167)) return "MemAlign.m1[7]";
    if ((address >= 168) && (address <= 175)) return "MemAlign.w0[0]";
    if ((address >= 176) && (address <= 183)) return "MemAlign.w0[1]";
    if ((address >= 184) && (address <= 191)) return "MemAlign.w0[2]";
    if ((address >= 192) && (address <= 199)) return "MemAlign.w0[3]";
    if ((address >= 200) && (address <= 207)) return "MemAlign.w0[4]";
    if ((address >= 208) && (address <= 215)) return "MemAlign.w0[5]";
    if ((address >= 216) && (address <= 223)) return "MemAlign.w0[6]";
    if ((address >= 224) && (address <= 231)) return "MemAlign.w0[7]";
    if ((address >= 232) && (address <= 239)) return "MemAlign.w1[0]";
    if ((address >= 240) && (address <= 247)) return "MemAlign.w1[1]";
    if ((address >= 248) && (address <= 255)) return "MemAlign.w1[2]";
    if ((address >= 256) && (address <= 263)) return "MemAlign.w1[3]";
    if ((address >= 264) && (address <= 271)) return "MemAlign.w1[4]";
    if ((address >= 272) && (address <= 279)) return "MemAlign.w1[5]";
    if ((address >= 280) && (address <= 287)) return "MemAlign.w1[6]";
    if ((address >= 288) && (address <= 295)) return "MemAlign.w1[7]";
    if ((address >= 296) && (address <= 303)) return "MemAlign.v[0]";
    if ((address >= 304) && (address <= 311)) return "MemAlign.v[1]";
    if ((address >= 312) && (address <= 319)) return "MemAlign.v[2]";
    if ((address >= 320) && (address <= 327)) return "MemAlign.v[3]";
    if ((address >= 328) && (address <= 335)) return "MemAlign.v[4]";
    if ((address >= 336) && (address <= 343)) return "MemAlign.v[5]";
    if ((address >= 344) && (address <= 351)) return "MemAlign.v[6]";
    if ((address >= 352) && (address <= 359)) return "MemAlign.v[7]";
    if ((address >= 360) && (address <= 367)) return "MemAlign.selM1";
    if ((address >= 368) && (address <= 375)) return "MemAlign.factorV[0]";
    if ((address >= 376) && (address <= 383)) return "MemAlign.factorV[1]";
    if ((address >= 384) && (address <= 391)) return "MemAlign.factorV[2]";
    if ((address >= 392) && (address <= 399)) return "MemAlign.factorV[3]";
    if ((address >= 400) && (address <= 407)) return "MemAlign.factorV[4]";
    if ((address >= 408) && (address <= 415)) return "MemAlign.factorV[5]";
    if ((address >= 416) && (address <= 423)) return "MemAlign.factorV[6]";
    if ((address >= 424) && (address <= 431)) return "MemAlign.factorV[7]";
    if ((address >= 432) && (address <= 439)) return "MemAlign.offset";
    if ((address >= 440) && (address <= 447)) return "MemAlign.resultRd";
    if ((address >= 448) && (address <= 455)) return "MemAlign.resultWr8";
    if ((address >= 456) && (address <= 463)) return "MemAlign.resultWr256";
    if ((address >= 464) && (address <= 471)) return "Arith.x1[0]";
    if ((address >= 472) && (address <= 479)) return "Arith.x1[1]";
    if ((address >= 480) && (address <= 487)) return "Arith.x1[2]";
    if ((address >= 488) && (address <= 495)) return "Arith.x1[3]";
    if ((address >= 496) && (address <= 503)) return "Arith.x1[4]";
    if ((address >= 504) && (address <= 511)) return "Arith.x1[5]";
    if ((address >= 512) && (address <= 519)) return "Arith.x1[6]";
    if ((address >= 520) && (address <= 527)) return "Arith.x1[7]";
    if ((address >= 528) && (address <= 535)) return "Arith.x1[8]";
    if ((address >= 536) && (address <= 543)) return "Arith.x1[9]";
    if ((address >= 544) && (address <= 551)) return "Arith.x1[10]";
    if ((address >= 552) && (address <= 559)) return "Arith.x1[11]";
    if ((address >= 560) && (address <= 567)) return "Arith.x1[12]";
    if ((address >= 568) && (address <= 575)) return "Arith.x1[13]";
    if ((address >= 576) && (address <= 583)) return "Arith.x1[14]";
    if ((address >= 584) && (address <= 591)) return "Arith.x1[15]";
    if ((address >= 592) && (address <= 599)) return "Arith.y1[0]";
    if ((address >= 600) && (address <= 607)) return "Arith.y1[1]";
    if ((address >= 608) && (address <= 615)) return "Arith.y1[2]";
    if ((address >= 616) && (address <= 623)) return "Arith.y1[3]";
    if ((address >= 624) && (address <= 631)) return "Arith.y1[4]";
    if ((address >= 632) && (address <= 639)) return "Arith.y1[5]";
    if ((address >= 640) && (address <= 647)) return "Arith.y1[6]";
    if ((address >= 648) && (address <= 655)) return "Arith.y1[7]";
    if ((address >= 656) && (address <= 663)) return "Arith.y1[8]";
    if ((address >= 664) && (address <= 671)) return "Arith.y1[9]";
    if ((address >= 672) && (address <= 679)) return "Arith.y1[10]";
    if ((address >= 680) && (address <= 687)) return "Arith.y1[11]";
    if ((address >= 688) && (address <= 695)) return "Arith.y1[12]";
    if ((address >= 696) && (address <= 703)) return "Arith.y1[13]";
    if ((address >= 704) && (address <= 711)) return "Arith.y1[14]";
    if ((address >= 712) && (address <= 719)) return "Arith.y1[15]";
    if ((address >= 720) && (address <= 727)) return "Arith.x2[0]";
    if ((address >= 728) && (address <= 735)) return "Arith.x2[1]";
    if ((address >= 736) && (address <= 743)) return "Arith.x2[2]";
    if ((address >= 744) && (address <= 751)) return "Arith.x2[3]";
    if ((address >= 752) && (address <= 759)) return "Arith.x2[4]";
    if ((address >= 760) && (address <= 767)) return "Arith.x2[5]";
    if ((address >= 768) && (address <= 775)) return "Arith.x2[6]";
    if ((address >= 776) && (address <= 783)) return "Arith.x2[7]";
    if ((address >= 784) && (address <= 791)) return "Arith.x2[8]";
    if ((address >= 792) && (address <= 799)) return "Arith.x2[9]";
    if ((address >= 800) && (address <= 807)) return "Arith.x2[10]";
    if ((address >= 808) && (address <= 815)) return "Arith.x2[11]";
    if ((address >= 816) && (address <= 823)) return "Arith.x2[12]";
    if ((address >= 824) && (address <= 831)) return "Arith.x2[13]";
    if ((address >= 832) && (address <= 839)) return "Arith.x2[14]";
    if ((address >= 840) && (address <= 847)) return "Arith.x2[15]";
    if ((address >= 848) && (address <= 855)) return "Arith.y2[0]";
    if ((address >= 856) && (address <= 863)) return "Arith.y2[1]";
    if ((address >= 864) && (address <= 871)) return "Arith.y2[2]";
    if ((address >= 872) && (address <= 879)) return "Arith.y2[3]";
    if ((address >= 880) && (address <= 887)) return "Arith.y2[4]";
    if ((address >= 888) && (address <= 895)) return "Arith.y2[5]";
    if ((address >= 896) && (address <= 903)) return "Arith.y2[6]";
    if ((address >= 904) && (address <= 911)) return "Arith.y2[7]";
    if ((address >= 912) && (address <= 919)) return "Arith.y2[8]";
    if ((address >= 920) && (address <= 927)) return "Arith.y2[9]";
    if ((address >= 928) && (address <= 935)) return "Arith.y2[10]";
    if ((address >= 936) && (address <= 943)) return "Arith.y2[11]";
    if ((address >= 944) && (address <= 951)) return "Arith.y2[12]";
    if ((address >= 952) && (address <= 959)) return "Arith.y2[13]";
    if ((address >= 960) && (address <= 967)) return "Arith.y2[14]";
    if ((address >= 968) && (address <= 975)) return "Arith.y2[15]";
    if ((address >= 976) && (address <= 983)) return "Arith.x3[0]";
    if ((address >= 984) && (address <= 991)) return "Arith.x3[1]";
    if ((address >= 992) && (address <= 999)) return "Arith.x3[2]";
    if ((address >= 1000) && (address <= 1007)) return "Arith.x3[3]";
    if ((address >= 1008) && (address <= 1015)) return "Arith.x3[4]";
    if ((address >= 1016) && (address <= 1023)) return "Arith.x3[5]";
    if ((address >= 1024) && (address <= 1031)) return "Arith.x3[6]";
    if ((address >= 1032) && (address <= 1039)) return "Arith.x3[7]";
    if ((address >= 1040) && (address <= 1047)) return "Arith.x3[8]";
    if ((address >= 1048) && (address <= 1055)) return "Arith.x3[9]";
    if ((address >= 1056) && (address <= 1063)) return "Arith.x3[10]";
    if ((address >= 1064) && (address <= 1071)) return "Arith.x3[11]";
    if ((address >= 1072) && (address <= 1079)) return "Arith.x3[12]";
    if ((address >= 1080) && (address <= 1087)) return "Arith.x3[13]";
    if ((address >= 1088) && (address <= 1095)) return "Arith.x3[14]";
    if ((address >= 1096) && (address <= 1103)) return "Arith.x3[15]";
    if ((address >= 1104) && (address <= 1111)) return "Arith.y3[0]";
    if ((address >= 1112) && (address <= 1119)) return "Arith.y3[1]";
    if ((address >= 1120) && (address <= 1127)) return "Arith.y3[2]";
    if ((address >= 1128) && (address <= 1135)) return "Arith.y3[3]";
    if ((address >= 1136) && (address <= 1143)) return "Arith.y3[4]";
    if ((address >= 1144) && (address <= 1151)) return "Arith.y3[5]";
    if ((address >= 1152) && (address <= 1159)) return "Arith.y3[6]";
    if ((address >= 1160) && (address <= 1167)) return "Arith.y3[7]";
    if ((address >= 1168) && (address <= 1175)) return "Arith.y3[8]";
    if ((address >= 1176) && (address <= 1183)) return "Arith.y3[9]";
    if ((address >= 1184) && (address <= 1191)) return "Arith.y3[10]";
    if ((address >= 1192) && (address <= 1199)) return "Arith.y3[11]";
    if ((address >= 1200) && (address <= 1207)) return "Arith.y3[12]";
    if ((address >= 1208) && (address <= 1215)) return "Arith.y3[13]";
    if ((address >= 1216) && (address <= 1223)) return "Arith.y3[14]";
    if ((address >= 1224) && (address <= 1231)) return "Arith.y3[15]";
    if ((address >= 1232) && (address <= 1239)) return "Arith.s[0]";
    if ((address >= 1240) && (address <= 1247)) return "Arith.s[1]";
    if ((address >= 1248) && (address <= 1255)) return "Arith.s[2]";
    if ((address >= 1256) && (address <= 1263)) return "Arith.s[3]";
    if ((address >= 1264) && (address <= 1271)) return "Arith.s[4]";
    if ((address >= 1272) && (address <= 1279)) return "Arith.s[5]";
    if ((address >= 1280) && (address <= 1287)) return "Arith.s[6]";
    if ((address >= 1288) && (address <= 1295)) return "Arith.s[7]";
    if ((address >= 1296) && (address <= 1303)) return "Arith.s[8]";
    if ((address >= 1304) && (address <= 1311)) return "Arith.s[9]";
    if ((address >= 1312) && (address <= 1319)) return "Arith.s[10]";
    if ((address >= 1320) && (address <= 1327)) return "Arith.s[11]";
    if ((address >= 1328) && (address <= 1335)) return "Arith.s[12]";
    if ((address >= 1336) && (address <= 1343)) return "Arith.s[13]";
    if ((address >= 1344) && (address <= 1351)) return "Arith.s[14]";
    if ((address >= 1352) && (address <= 1359)) return "Arith.s[15]";
    if ((address >= 1360) && (address <= 1367)) return "Arith.q0[0]";
    if ((address >= 1368) && (address <= 1375)) return "Arith.q0[1]";
    if ((address >= 1376) && (address <= 1383)) return "Arith.q0[2]";
    if ((address >= 1384) && (address <= 1391)) return "Arith.q0[3]";
    if ((address >= 1392) && (address <= 1399)) return "Arith.q0[4]";
    if ((address >= 1400) && (address <= 1407)) return "Arith.q0[5]";
    if ((address >= 1408) && (address <= 1415)) return "Arith.q0[6]";
    if ((address >= 1416) && (address <= 1423)) return "Arith.q0[7]";
    if ((address >= 1424) && (address <= 1431)) return "Arith.q0[8]";
    if ((address >= 1432) && (address <= 1439)) return "Arith.q0[9]";
    if ((address >= 1440) && (address <= 1447)) return "Arith.q0[10]";
    if ((address >= 1448) && (address <= 1455)) return "Arith.q0[11]";
    if ((address >= 1456) && (address <= 1463)) return "Arith.q0[12]";
    if ((address >= 1464) && (address <= 1471)) return "Arith.q0[13]";
    if ((address >= 1472) && (address <= 1479)) return "Arith.q0[14]";
    if ((address >= 1480) && (address <= 1487)) return "Arith.q0[15]";
    if ((address >= 1488) && (address <= 1495)) return "Arith.q1[0]";
    if ((address >= 1496) && (address <= 1503)) return "Arith.q1[1]";
    if ((address >= 1504) && (address <= 1511)) return "Arith.q1[2]";
    if ((address >= 1512) && (address <= 1519)) return "Arith.q1[3]";
    if ((address >= 1520) && (address <= 1527)) return "Arith.q1[4]";
    if ((address >= 1528) && (address <= 1535)) return "Arith.q1[5]";
    if ((address >= 1536) && (address <= 1543)) return "Arith.q1[6]";
    if ((address >= 1544) && (address <= 1551)) return "Arith.q1[7]";
    if ((address >= 1552) && (address <= 1559)) return "Arith.q1[8]";
    if ((address >= 1560) && (address <= 1567)) return "Arith.q1[9]";
    if ((address >= 1568) && (address <= 1575)) return "Arith.q1[10]";
    if ((address >= 1576) && (address <= 1583)) return "Arith.q1[11]";
    if ((address >= 1584) && (address <= 1591)) return "Arith.q1[12]";
    if ((address >= 1592) && (address <= 1599)) return "Arith.q1[13]";
    if ((address >= 1600) && (address <= 1607)) return "Arith.q1[14]";
    if ((address >= 1608) && (address <= 1615)) return "Arith.q1[15]";
    if ((address >= 1616) && (address <= 1623)) return "Arith.q2[0]";
    if ((address >= 1624) && (address <= 1631)) return "Arith.q2[1]";
    if ((address >= 1632) && (address <= 1639)) return "Arith.q2[2]";
    if ((address >= 1640) && (address <= 1647)) return "Arith.q2[3]";
    if ((address >= 1648) && (address <= 1655)) return "Arith.q2[4]";
    if ((address >= 1656) && (address <= 1663)) return "Arith.q2[5]";
    if ((address >= 1664) && (address <= 1671)) return "Arith.q2[6]";
    if ((address >= 1672) && (address <= 1679)) return "Arith.q2[7]";
    if ((address >= 1680) && (address <= 1687)) return "Arith.q2[8]";
    if ((address >= 1688) && (address <= 1695)) return "Arith.q2[9]";
    if ((address >= 1696) && (address <= 1703)) return "Arith.q2[10]";
    if ((address >= 1704) && (address <= 1711)) return "Arith.q2[11]";
    if ((address >= 1712) && (address <= 1719)) return "Arith.q2[12]";
    if ((address >= 1720) && (address <= 1727)) return "Arith.q2[13]";
    if ((address >= 1728) && (address <= 1735)) return "Arith.q2[14]";
    if ((address >= 1736) && (address <= 1743)) return "Arith.q2[15]";
    if ((address >= 1744) && (address <= 1751)) return "Arith.resultEq0";
    if ((address >= 1752) && (address <= 1759)) return "Arith.resultEq1";
    if ((address >= 1760) && (address <= 1767)) return "Arith.resultEq2";
    if ((address >= 1768) && (address <= 1775)) return "Arith.xDeltaChunkInverse";
    if ((address >= 1776) && (address <= 1783)) return "Arith.xAreDifferent";
    if ((address >= 1784) && (address <= 1791)) return "Arith.valueLtPrime";
    if ((address >= 1792) && (address <= 1799)) return "Arith.chunkLtPrime";
    if ((address >= 1800) && (address <= 1807)) return "Arith.selEq[0]";
    if ((address >= 1808) && (address <= 1815)) return "Arith.selEq[1]";
    if ((address >= 1816) && (address <= 1823)) return "Arith.selEq[2]";
    if ((address >= 1824) && (address <= 1831)) return "Arith.selEq[3]";
    if ((address >= 1832) && (address <= 1839)) return "Arith.selEq[4]";
    if ((address >= 1840) && (address <= 1847)) return "Arith.selEq[5]";
    if ((address >= 1848) && (address <= 1855)) return "Arith.selEq[6]";
    if ((address >= 1856) && (address <= 1863)) return "Arith.carry[0]";
    if ((address >= 1864) && (address <= 1871)) return "Arith.carry[1]";
    if ((address >= 1872) && (address <= 1879)) return "Arith.carry[2]";
    if ((address >= 1880) && (address <= 1887)) return "Binary.opcode";
    if ((address >= 1888) && (address <= 1895)) return "Binary.a[0]";
    if ((address >= 1896) && (address <= 1903)) return "Binary.a[1]";
    if ((address >= 1904) && (address <= 1911)) return "Binary.a[2]";
    if ((address >= 1912) && (address <= 1919)) return "Binary.a[3]";
    if ((address >= 1920) && (address <= 1927)) return "Binary.a[4]";
    if ((address >= 1928) && (address <= 1935)) return "Binary.a[5]";
    if ((address >= 1936) && (address <= 1943)) return "Binary.a[6]";
    if ((address >= 1944) && (address <= 1951)) return "Binary.a[7]";
    if ((address >= 1952) && (address <= 1959)) return "Binary.b[0]";
    if ((address >= 1960) && (address <= 1967)) return "Binary.b[1]";
    if ((address >= 1968) && (address <= 1975)) return "Binary.b[2]";
    if ((address >= 1976) && (address <= 1983)) return "Binary.b[3]";
    if ((address >= 1984) && (address <= 1991)) return "Binary.b[4]";
    if ((address >= 1992) && (address <= 1999)) return "Binary.b[5]";
    if ((address >= 2000) && (address <= 2007)) return "Binary.b[6]";
    if ((address >= 2008) && (address <= 2015)) return "Binary.b[7]";
    if ((address >= 2016) && (address <= 2023)) return "Binary.c[0]";
    if ((address >= 2024) && (address <= 2031)) return "Binary.c[1]";
    if ((address >= 2032) && (address <= 2039)) return "Binary.c[2]";
    if ((address >= 2040) && (address <= 2047)) return "Binary.c[3]";
    if ((address >= 2048) && (address <= 2055)) return "Binary.c[4]";
    if ((address >= 2056) && (address <= 2063)) return "Binary.c[5]";
    if ((address >= 2064) && (address <= 2071)) return "Binary.c[6]";
    if ((address >= 2072) && (address <= 2079)) return "Binary.c[7]";
    if ((address >= 2080) && (address <= 2087)) return "Binary.freeInA[0]";
    if ((address >= 2088) && (address <= 2095)) return "Binary.freeInA[1]";
    if ((address >= 2096) && (address <= 2103)) return "Binary.freeInB[0]";
    if ((address >= 2104) && (address <= 2111)) return "Binary.freeInB[1]";
    if ((address >= 2112) && (address <= 2119)) return "Binary.freeInC[0]";
    if ((address >= 2120) && (address <= 2127)) return "Binary.freeInC[1]";
    if ((address >= 2128) && (address <= 2135)) return "Binary.cIn";
    if ((address >= 2136) && (address <= 2143)) return "Binary.cMiddle";
    if ((address >= 2144) && (address <= 2151)) return "Binary.cOut";
    if ((address >= 2152) && (address <= 2159)) return "Binary.lCout";
    if ((address >= 2160) && (address <= 2167)) return "Binary.lOpcode";
    if ((address >= 2168) && (address <= 2175)) return "Binary.previousAreLt4";
    if ((address >= 2176) && (address <= 2183)) return "Binary.usePreviousAreLt4";
    if ((address >= 2184) && (address <= 2191)) return "Binary.reset4";
    if ((address >= 2192) && (address <= 2199)) return "Binary.useCarry";
    if ((address >= 2200) && (address <= 2207)) return "Binary.resultBinOp";
    if ((address >= 2208) && (address <= 2215)) return "Binary.resultValidRange";
    if ((address >= 2216) && (address <= 2223)) return "PoseidonG.in0";
    if ((address >= 2224) && (address <= 2231)) return "PoseidonG.in1";
    if ((address >= 2232) && (address <= 2239)) return "PoseidonG.in2";
    if ((address >= 2240) && (address <= 2247)) return "PoseidonG.in3";
    if ((address >= 2248) && (address <= 2255)) return "PoseidonG.in4";
    if ((address >= 2256) && (address <= 2263)) return "PoseidonG.in5";
    if ((address >= 2264) && (address <= 2271)) return "PoseidonG.in6";
    if ((address >= 2272) && (address <= 2279)) return "PoseidonG.in7";
    if ((address >= 2280) && (address <= 2287)) return "PoseidonG.hashType";
    if ((address >= 2288) && (address <= 2295)) return "PoseidonG.cap1";
    if ((address >= 2296) && (address <= 2303)) return "PoseidonG.cap2";
    if ((address >= 2304) && (address <= 2311)) return "PoseidonG.cap3";
    if ((address >= 2312) && (address <= 2319)) return "PoseidonG.hash0";
    if ((address >= 2320) && (address <= 2327)) return "PoseidonG.hash1";
    if ((address >= 2328) && (address <= 2335)) return "PoseidonG.hash2";
    if ((address >= 2336) && (address <= 2343)) return "PoseidonG.hash3";
    if ((address >= 2344) && (address <= 2351)) return "PoseidonG.result1";
    if ((address >= 2352) && (address <= 2359)) return "PoseidonG.result2";
    if ((address >= 2360) && (address <= 2367)) return "PoseidonG.result3";
    if ((address >= 2368) && (address <= 2375)) return "PaddingPG.acc[0]";
    if ((address >= 2376) && (address <= 2383)) return "PaddingPG.acc[1]";
    if ((address >= 2384) && (address <= 2391)) return "PaddingPG.acc[2]";
    if ((address >= 2392) && (address <= 2399)) return "PaddingPG.acc[3]";
    if ((address >= 2400) && (address <= 2407)) return "PaddingPG.acc[4]";
    if ((address >= 2408) && (address <= 2415)) return "PaddingPG.acc[5]";
    if ((address >= 2416) && (address <= 2423)) return "PaddingPG.acc[6]";
    if ((address >= 2424) && (address <= 2431)) return "PaddingPG.acc[7]";
    if ((address >= 2432) && (address <= 2439)) return "PaddingPG.freeIn";
    if ((address >= 2440) && (address <= 2447)) return "PaddingPG.addr";
    if ((address >= 2448) && (address <= 2455)) return "PaddingPG.rem";
    if ((address >= 2456) && (address <= 2463)) return "PaddingPG.remInv";
    if ((address >= 2464) && (address <= 2471)) return "PaddingPG.spare";
    if ((address >= 2472) && (address <= 2479)) return "PaddingPG.lastHashLen";
    if ((address >= 2480) && (address <= 2487)) return "PaddingPG.lastHashDigest";
    if ((address >= 2488) && (address <= 2495)) return "PaddingPG.curHash0";
    if ((address >= 2496) && (address <= 2503)) return "PaddingPG.curHash1";
    if ((address >= 2504) && (address <= 2511)) return "PaddingPG.curHash2";
    if ((address >= 2512) && (address <= 2519)) return "PaddingPG.curHash3";
    if ((address >= 2520) && (address <= 2527)) return "PaddingPG.prevHash0";
    if ((address >= 2528) && (address <= 2535)) return "PaddingPG.prevHash1";
    if ((address >= 2536) && (address <= 2543)) return "PaddingPG.prevHash2";
    if ((address >= 2544) && (address <= 2551)) return "PaddingPG.prevHash3";
    if ((address >= 2552) && (address <= 2559)) return "PaddingPG.incCounter";
    if ((address >= 2560) && (address <= 2567)) return "PaddingPG.len";
    if ((address >= 2568) && (address <= 2575)) return "PaddingPG.crOffset";
    if ((address >= 2576) && (address <= 2583)) return "PaddingPG.crLen";
    if ((address >= 2584) && (address <= 2591)) return "PaddingPG.crOffsetInv";
    if ((address >= 2592) && (address <= 2599)) return "PaddingPG.crF0";
    if ((address >= 2600) && (address <= 2607)) return "PaddingPG.crF1";
    if ((address >= 2608) && (address <= 2615)) return "PaddingPG.crF2";
    if ((address >= 2616) && (address <= 2623)) return "PaddingPG.crF3";
    if ((address >= 2624) && (address <= 2631)) return "PaddingPG.crF4";
    if ((address >= 2632) && (address <= 2639)) return "PaddingPG.crF5";
    if ((address >= 2640) && (address <= 2647)) return "PaddingPG.crF6";
    if ((address >= 2648) && (address <= 2655)) return "PaddingPG.crF7";
    if ((address >= 2656) && (address <= 2663)) return "PaddingPG.crV0";
    if ((address >= 2664) && (address <= 2671)) return "PaddingPG.crV1";
    if ((address >= 2672) && (address <= 2679)) return "PaddingPG.crV2";
    if ((address >= 2680) && (address <= 2687)) return "PaddingPG.crV3";
    if ((address >= 2688) && (address <= 2695)) return "PaddingPG.crV4";
    if ((address >= 2696) && (address <= 2703)) return "PaddingPG.crV5";
    if ((address >= 2704) && (address <= 2711)) return "PaddingPG.crV6";
    if ((address >= 2712) && (address <= 2719)) return "PaddingPG.crV7";
    if ((address >= 2720) && (address <= 2727)) return "ClimbKey.key0";
    if ((address >= 2728) && (address <= 2735)) return "ClimbKey.key1";
    if ((address >= 2736) && (address <= 2743)) return "ClimbKey.key2";
    if ((address >= 2744) && (address <= 2751)) return "ClimbKey.key3";
    if ((address >= 2752) && (address <= 2759)) return "ClimbKey.level";
    if ((address >= 2760) && (address <= 2767)) return "ClimbKey.keyIn";
    if ((address >= 2768) && (address <= 2775)) return "ClimbKey.keyInChunk";
    if ((address >= 2776) && (address <= 2783)) return "ClimbKey.result";
    if ((address >= 2784) && (address <= 2791)) return "ClimbKey.bit";
    if ((address >= 2792) && (address <= 2799)) return "ClimbKey.keySel0";
    if ((address >= 2800) && (address <= 2807)) return "ClimbKey.keySel1";
    if ((address >= 2808) && (address <= 2815)) return "ClimbKey.keySel2";
    if ((address >= 2816) && (address <= 2823)) return "ClimbKey.keySel3";
    if ((address >= 2824) && (address <= 2831)) return "ClimbKey.carryLt";
    if ((address >= 2832) && (address <= 2839)) return "Storage.free0";
    if ((address >= 2840) && (address <= 2847)) return "Storage.free1";
    if ((address >= 2848) && (address <= 2855)) return "Storage.free2";
    if ((address >= 2856) && (address <= 2863)) return "Storage.free3";
    if ((address >= 2864) && (address <= 2871)) return "Storage.hashLeft0";
    if ((address >= 2872) && (address <= 2879)) return "Storage.hashLeft1";
    if ((address >= 2880) && (address <= 2887)) return "Storage.hashLeft2";
    if ((address >= 2888) && (address <= 2895)) return "Storage.hashLeft3";
    if ((address >= 2896) && (address <= 2903)) return "Storage.hashRight0";
    if ((address >= 2904) && (address <= 2911)) return "Storage.hashRight1";
    if ((address >= 2912) && (address <= 2919)) return "Storage.hashRight2";
    if ((address >= 2920) && (address <= 2927)) return "Storage.hashRight3";
    if ((address >= 2928) && (address <= 2935)) return "Storage.oldRoot0";
    if ((address >= 2936) && (address <= 2943)) return "Storage.oldRoot1";
    if ((address >= 2944) && (address <= 2951)) return "Storage.oldRoot2";
    if ((address >= 2952) && (address <= 2959)) return "Storage.oldRoot3";
    if ((address >= 2960) && (address <= 2967)) return "Storage.newRoot0";
    if ((address >= 2968) && (address <= 2975)) return "Storage.newRoot1";
    if ((address >= 2976) && (address <= 2983)) return "Storage.newRoot2";
    if ((address >= 2984) && (address <= 2991)) return "Storage.newRoot3";
    if ((address >= 2992) && (address <= 2999)) return "Storage.valueLow0";
    if ((address >= 3000) && (address <= 3007)) return "Storage.valueLow1";
    if ((address >= 3008) && (address <= 3015)) return "Storage.valueLow2";
    if ((address >= 3016) && (address <= 3023)) return "Storage.valueLow3";
    if ((address >= 3024) && (address <= 3031)) return "Storage.valueHigh0";
    if ((address >= 3032) && (address <= 3039)) return "Storage.valueHigh1";
    if ((address >= 3040) && (address <= 3047)) return "Storage.valueHigh2";
    if ((address >= 3048) && (address <= 3055)) return "Storage.valueHigh3";
    if ((address >= 3056) && (address <= 3063)) return "Storage.siblingValueHash0";
    if ((address >= 3064) && (address <= 3071)) return "Storage.siblingValueHash1";
    if ((address >= 3072) && (address <= 3079)) return "Storage.siblingValueHash2";
    if ((address >= 3080) && (address <= 3087)) return "Storage.siblingValueHash3";
    if ((address >= 3088) && (address <= 3095)) return "Storage.rkey0";
    if ((address >= 3096) && (address <= 3103)) return "Storage.rkey1";
    if ((address >= 3104) && (address <= 3111)) return "Storage.rkey2";
    if ((address >= 3112) && (address <= 3119)) return "Storage.rkey3";
    if ((address >= 3120) && (address <= 3127)) return "Storage.siblingRkey0";
    if ((address >= 3128) && (address <= 3135)) return "Storage.siblingRkey1";
    if ((address >= 3136) && (address <= 3143)) return "Storage.siblingRkey2";
    if ((address >= 3144) && (address <= 3151)) return "Storage.siblingRkey3";
    if ((address >= 3152) && (address <= 3159)) return "Storage.rkeyBit";
    if ((address >= 3160) && (address <= 3167)) return "Storage.level";
    if ((address >= 3168) && (address <= 3175)) return "Storage.pc";
    if ((address >= 3176) && (address <= 3183)) return "Storage.inOldRoot";
    if ((address >= 3184) && (address <= 3191)) return "Storage.inNewRoot";
    if ((address >= 3192) && (address <= 3199)) return "Storage.inValueLow";
    if ((address >= 3200) && (address <= 3207)) return "Storage.inValueHigh";
    if ((address >= 3208) && (address <= 3215)) return "Storage.inSiblingValueHash";
    if ((address >= 3216) && (address <= 3223)) return "Storage.inRkey";
    if ((address >= 3224) && (address <= 3231)) return "Storage.inRkeyBit";
    if ((address >= 3232) && (address <= 3239)) return "Storage.inSiblingRkey";
    if ((address >= 3240) && (address <= 3247)) return "Storage.inFree";
    if ((address >= 3248) && (address <= 3255)) return "Storage.inRotlVh";
    if ((address >= 3256) && (address <= 3263)) return "Storage.inLevel";
    if ((address >= 3264) && (address <= 3271)) return "Storage.setHashLeft";
    if ((address >= 3272) && (address <= 3279)) return "Storage.setHashRight";
    if ((address >= 3280) && (address <= 3287)) return "Storage.setOldRoot";
    if ((address >= 3288) && (address <= 3295)) return "Storage.setNewRoot";
    if ((address >= 3296) && (address <= 3303)) return "Storage.setValueLow";
    if ((address >= 3304) && (address <= 3311)) return "Storage.setValueHigh";
    if ((address >= 3312) && (address <= 3319)) return "Storage.setSiblingValueHash";
    if ((address >= 3320) && (address <= 3327)) return "Storage.setRkey";
    if ((address >= 3328) && (address <= 3335)) return "Storage.setSiblingRkey";
    if ((address >= 3336) && (address <= 3343)) return "Storage.setRkeyBit";
    if ((address >= 3344) && (address <= 3351)) return "Storage.setLevel";
    if ((address >= 3352) && (address <= 3359)) return "Storage.hash";
    if ((address >= 3360) && (address <= 3367)) return "Storage.hashType";
    if ((address >= 3368) && (address <= 3375)) return "Storage.latchSet";
    if ((address >= 3376) && (address <= 3383)) return "Storage.latchGet";
    if ((address >= 3384) && (address <= 3391)) return "Storage.climbRkey";
    if ((address >= 3392) && (address <= 3399)) return "Storage.climbSiblingRkey";
    if ((address >= 3400) && (address <= 3407)) return "Storage.climbBitN";
    if ((address >= 3408) && (address <= 3415)) return "Storage.jmpz";
    if ((address >= 3416) && (address <= 3423)) return "Storage.jmpnz";
    if ((address >= 3424) && (address <= 3431)) return "Storage.jmp";
    if ((address >= 3432) && (address <= 3439)) return "Storage.const0";
    if ((address >= 3440) && (address <= 3447)) return "Storage.jmpAddress";
    if ((address >= 3448) && (address <= 3455)) return "Storage.incCounter";
    if ((address >= 3456) && (address <= 3463)) return "Storage.op0inv";
    if ((address >= 3464) && (address <= 3471)) return "KeccakF.a[0]";
    if ((address >= 3472) && (address <= 3479)) return "KeccakF.a[1]";
    if ((address >= 3480) && (address <= 3487)) return "KeccakF.a[2]";
    if ((address >= 3488) && (address <= 3495)) return "KeccakF.a[3]";
    if ((address >= 3496) && (address <= 3503)) return "KeccakF.b[0]";
    if ((address >= 3504) && (address <= 3511)) return "KeccakF.b[1]";
    if ((address >= 3512) && (address <= 3519)) return "KeccakF.b[2]";
    if ((address >= 3520) && (address <= 3527)) return "KeccakF.b[3]";
    if ((address >= 3528) && (address <= 3535)) return "KeccakF.c[0]";
    if ((address >= 3536) && (address <= 3543)) return "KeccakF.c[1]";
    if ((address >= 3544) && (address <= 3551)) return "KeccakF.c[2]";
    if ((address >= 3552) && (address <= 3559)) return "KeccakF.c[3]";
    if ((address >= 3560) && (address <= 3567)) return "Bits2Field.bit";
    if ((address >= 3568) && (address <= 3575)) return "Bits2Field.field44";
    if ((address >= 3576) && (address <= 3583)) return "PaddingKKBit.rBit";
    if ((address >= 3584) && (address <= 3591)) return "PaddingKKBit.sOutBit";
    if ((address >= 3592) && (address <= 3599)) return "PaddingKKBit.r8";
    if ((address >= 3600) && (address <= 3607)) return "PaddingKKBit.connected";
    if ((address >= 3608) && (address <= 3615)) return "PaddingKKBit.sOut0";
    if ((address >= 3616) && (address <= 3623)) return "PaddingKKBit.sOut1";
    if ((address >= 3624) && (address <= 3631)) return "PaddingKKBit.sOut2";
    if ((address >= 3632) && (address <= 3639)) return "PaddingKKBit.sOut3";
    if ((address >= 3640) && (address <= 3647)) return "PaddingKKBit.sOut4";
    if ((address >= 3648) && (address <= 3655)) return "PaddingKKBit.sOut5";
    if ((address >= 3656) && (address <= 3663)) return "PaddingKKBit.sOut6";
    if ((address >= 3664) && (address <= 3671)) return "PaddingKKBit.sOut7";
    if ((address >= 3672) && (address <= 3679)) return "PaddingKK.freeIn";
    if ((address >= 3680) && (address <= 3687)) return "PaddingKK.connected";
    if ((address >= 3688) && (address <= 3695)) return "PaddingKK.addr";
    if ((address >= 3696) && (address <= 3703)) return "PaddingKK.rem";
    if ((address >= 3704) && (address <= 3711)) return "PaddingKK.remInv";
    if ((address >= 3712) && (address <= 3719)) return "PaddingKK.spare";
    if ((address >= 3720) && (address <= 3727)) return "PaddingKK.lastHashLen";
    if ((address >= 3728) && (address <= 3735)) return "PaddingKK.lastHashDigest";
    if ((address >= 3736) && (address <= 3743)) return "PaddingKK.len";
    if ((address >= 3744) && (address <= 3751)) return "PaddingKK.hash0";
    if ((address >= 3752) && (address <= 3759)) return "PaddingKK.hash1";
    if ((address >= 3760) && (address <= 3767)) return "PaddingKK.hash2";
    if ((address >= 3768) && (address <= 3775)) return "PaddingKK.hash3";
    if ((address >= 3776) && (address <= 3783)) return "PaddingKK.hash4";
    if ((address >= 3784) && (address <= 3791)) return "PaddingKK.hash5";
    if ((address >= 3792) && (address <= 3799)) return "PaddingKK.hash6";
    if ((address >= 3800) && (address <= 3807)) return "PaddingKK.hash7";
    if ((address >= 3808) && (address <= 3815)) return "PaddingKK.incCounter";
    if ((address >= 3816) && (address <= 3823)) return "PaddingKK.crOffset";
    if ((address >= 3824) && (address <= 3831)) return "PaddingKK.crLen";
    if ((address >= 3832) && (address <= 3839)) return "PaddingKK.crOffsetInv";
    if ((address >= 3840) && (address <= 3847)) return "PaddingKK.crF0";
    if ((address >= 3848) && (address <= 3855)) return "PaddingKK.crF1";
    if ((address >= 3856) && (address <= 3863)) return "PaddingKK.crF2";
    if ((address >= 3864) && (address <= 3871)) return "PaddingKK.crF3";
    if ((address >= 3872) && (address <= 3879)) return "PaddingKK.crF4";
    if ((address >= 3880) && (address <= 3887)) return "PaddingKK.crF5";
    if ((address >= 3888) && (address <= 3895)) return "PaddingKK.crF6";
    if ((address >= 3896) && (address <= 3903)) return "PaddingKK.crF7";
    if ((address >= 3904) && (address <= 3911)) return "PaddingKK.crV0";
    if ((address >= 3912) && (address <= 3919)) return "PaddingKK.crV1";
    if ((address >= 3920) && (address <= 3927)) return "PaddingKK.crV2";
    if ((address >= 3928) && (address <= 3935)) return "PaddingKK.crV3";
    if ((address >= 3936) && (address <= 3943)) return "PaddingKK.crV4";
    if ((address >= 3944) && (address <= 3951)) return "PaddingKK.crV5";
    if ((address >= 3952) && (address <= 3959)) return "PaddingKK.crV6";
    if ((address >= 3960) && (address <= 3967)) return "PaddingKK.crV7";
    if ((address >= 3968) && (address <= 3975)) return "Mem.addr";
    if ((address >= 3976) && (address <= 3983)) return "Mem.step";
    if ((address >= 3984) && (address <= 3991)) return "Mem.mOp";
    if ((address >= 3992) && (address <= 3999)) return "Mem.mWr";
    if ((address >= 4000) && (address <= 4007)) return "Mem.val[0]";
    if ((address >= 4008) && (address <= 4015)) return "Mem.val[1]";
    if ((address >= 4016) && (address <= 4023)) return "Mem.val[2]";
    if ((address >= 4024) && (address <= 4031)) return "Mem.val[3]";
    if ((address >= 4032) && (address <= 4039)) return "Mem.val[4]";
    if ((address >= 4040) && (address <= 4047)) return "Mem.val[5]";
    if ((address >= 4048) && (address <= 4055)) return "Mem.val[6]";
    if ((address >= 4056) && (address <= 4063)) return "Mem.val[7]";
    if ((address >= 4064) && (address <= 4071)) return "Mem.lastAccess";
    if ((address >= 4072) && (address <= 4079)) return "Sha256F.input[0]";
    if ((address >= 4080) && (address <= 4087)) return "Sha256F.input[1]";
    if ((address >= 4088) && (address <= 4095)) return "Sha256F.input[2]";
    if ((address >= 4096) && (address <= 4103)) return "Sha256F.output";
    if ((address >= 4104) && (address <= 4111)) return "Bits2FieldSha256.bit";
    if ((address >= 4112) && (address <= 4119)) return "Bits2FieldSha256.packField";
    if ((address >= 4120) && (address <= 4127)) return "PaddingSha256Bit.s1";
    if ((address >= 4128) && (address <= 4135)) return "PaddingSha256Bit.s2";
    if ((address >= 4136) && (address <= 4143)) return "PaddingSha256Bit.r8";
    if ((address >= 4144) && (address <= 4151)) return "PaddingSha256Bit.connected";
    if ((address >= 4152) && (address <= 4159)) return "PaddingSha256Bit.sOut0";
    if ((address >= 4160) && (address <= 4167)) return "PaddingSha256Bit.sOut1";
    if ((address >= 4168) && (address <= 4175)) return "PaddingSha256Bit.sOut2";
    if ((address >= 4176) && (address <= 4183)) return "PaddingSha256Bit.sOut3";
    if ((address >= 4184) && (address <= 4191)) return "PaddingSha256Bit.sOut4";
    if ((address >= 4192) && (address <= 4199)) return "PaddingSha256Bit.sOut5";
    if ((address >= 4200) && (address <= 4207)) return "PaddingSha256Bit.sOut6";
    if ((address >= 4208) && (address <= 4215)) return "PaddingSha256Bit.sOut7";
    if ((address >= 4216) && (address <= 4223)) return "PaddingSha256.freeIn";
    if ((address >= 4224) && (address <= 4231)) return "PaddingSha256.connected";
    if ((address >= 4232) && (address <= 4239)) return "PaddingSha256.addr";
    if ((address >= 4240) && (address <= 4247)) return "PaddingSha256.rem";
    if ((address >= 4248) && (address <= 4255)) return "PaddingSha256.remInv";
    if ((address >= 4256) && (address <= 4263)) return "PaddingSha256.spare";
    if ((address >= 4264) && (address <= 4271)) return "PaddingSha256.lengthSection";
    if ((address >= 4272) && (address <= 4279)) return "PaddingSha256.accLength";
    if ((address >= 4280) && (address <= 4287)) return "PaddingSha256.lastHashLen";
    if ((address >= 4288) && (address <= 4295)) return "PaddingSha256.lastHashDigest";
    if ((address >= 4296) && (address <= 4303)) return "PaddingSha256.len";
    if ((address >= 4304) && (address <= 4311)) return "PaddingSha256.hash0";
    if ((address >= 4312) && (address <= 4319)) return "PaddingSha256.hash1";
    if ((address >= 4320) && (address <= 4327)) return "PaddingSha256.hash2";
    if ((address >= 4328) && (address <= 4335)) return "PaddingSha256.hash3";
    if ((address >= 4336) && (address <= 4343)) return "PaddingSha256.hash4";
    if ((address >= 4344) && (address <= 4351)) return "PaddingSha256.hash5";
    if ((address >= 4352) && (address <= 4359)) return "PaddingSha256.hash6";
    if ((address >= 4360) && (address <= 4367)) return "PaddingSha256.hash7";
    if ((address >= 4368) && (address <= 4375)) return "PaddingSha256.incCounter";
    if ((address >= 4376) && (address <= 4383)) return "PaddingSha256.crOffset";
    if ((address >= 4384) && (address <= 4391)) return "PaddingSha256.crLen";
    if ((address >= 4392) && (address <= 4399)) return "PaddingSha256.crOffsetInv";
    if ((address >= 4400) && (address <= 4407)) return "PaddingSha256.crF0";
    if ((address >= 4408) && (address <= 4415)) return "PaddingSha256.crF1";
    if ((address >= 4416) && (address <= 4423)) return "PaddingSha256.crF2";
    if ((address >= 4424) && (address <= 4431)) return "PaddingSha256.crF3";
    if ((address >= 4432) && (address <= 4439)) return "PaddingSha256.crF4";
    if ((address >= 4440) && (address <= 4447)) return "PaddingSha256.crF5";
    if ((address >= 4448) && (address <= 4455)) return "PaddingSha256.crF6";
    if ((address >= 4456) && (address <= 4463)) return "PaddingSha256.crF7";
    if ((address >= 4464) && (address <= 4471)) return "PaddingSha256.crV0";
    if ((address >= 4472) && (address <= 4479)) return "PaddingSha256.crV1";
    if ((address >= 4480) && (address <= 4487)) return "PaddingSha256.crV2";
    if ((address >= 4488) && (address <= 4495)) return "PaddingSha256.crV3";
    if ((address >= 4496) && (address <= 4503)) return "PaddingSha256.crV4";
    if ((address >= 4504) && (address <= 4511)) return "PaddingSha256.crV5";
    if ((address >= 4512) && (address <= 4519)) return "PaddingSha256.crV6";
    if ((address >= 4520) && (address <= 4527)) return "PaddingSha256.crV7";
    if ((address >= 4528) && (address <= 4535)) return "Main.A7";
    if ((address >= 4536) && (address <= 4543)) return "Main.A6";
    if ((address >= 4544) && (address <= 4551)) return "Main.A5";
    if ((address >= 4552) && (address <= 4559)) return "Main.A4";
    if ((address >= 4560) && (address <= 4567)) return "Main.A3";
    if ((address >= 4568) && (address <= 4575)) return "Main.A2";
    if ((address >= 4576) && (address <= 4583)) return "Main.A1";
    if ((address >= 4584) && (address <= 4591)) return "Main.A0";
    if ((address >= 4592) && (address <= 4599)) return "Main.B7";
    if ((address >= 4600) && (address <= 4607)) return "Main.B6";
    if ((address >= 4608) && (address <= 4615)) return "Main.B5";
    if ((address >= 4616) && (address <= 4623)) return "Main.B4";
    if ((address >= 4624) && (address <= 4631)) return "Main.B3";
    if ((address >= 4632) && (address <= 4639)) return "Main.B2";
    if ((address >= 4640) && (address <= 4647)) return "Main.B1";
    if ((address >= 4648) && (address <= 4655)) return "Main.B0";
    if ((address >= 4656) && (address <= 4663)) return "Main.C7";
    if ((address >= 4664) && (address <= 4671)) return "Main.C6";
    if ((address >= 4672) && (address <= 4679)) return "Main.C5";
    if ((address >= 4680) && (address <= 4687)) return "Main.C4";
    if ((address >= 4688) && (address <= 4695)) return "Main.C3";
    if ((address >= 4696) && (address <= 4703)) return "Main.C2";
    if ((address >= 4704) && (address <= 4711)) return "Main.C1";
    if ((address >= 4712) && (address <= 4719)) return "Main.C0";
    if ((address >= 4720) && (address <= 4727)) return "Main.D7";
    if ((address >= 4728) && (address <= 4735)) return "Main.D6";
    if ((address >= 4736) && (address <= 4743)) return "Main.D5";
    if ((address >= 4744) && (address <= 4751)) return "Main.D4";
    if ((address >= 4752) && (address <= 4759)) return "Main.D3";
    if ((address >= 4760) && (address <= 4767)) return "Main.D2";
    if ((address >= 4768) && (address <= 4775)) return "Main.D1";
    if ((address >= 4776) && (address <= 4783)) return "Main.D0";
    if ((address >= 4784) && (address <= 4791)) return "Main.E7";
    if ((address >= 4792) && (address <= 4799)) return "Main.E6";
    if ((address >= 4800) && (address <= 4807)) return "Main.E5";
    if ((address >= 4808) && (address <= 4815)) return "Main.E4";
    if ((address >= 4816) && (address <= 4823)) return "Main.E3";
    if ((address >= 4824) && (address <= 4831)) return "Main.E2";
    if ((address >= 4832) && (address <= 4839)) return "Main.E1";
    if ((address >= 4840) && (address <= 4847)) return "Main.E0";
    if ((address >= 4848) && (address <= 4855)) return "Main.SR7";
    if ((address >= 4856) && (address <= 4863)) return "Main.SR6";
    if ((address >= 4864) && (address <= 4871)) return "Main.SR5";
    if ((address >= 4872) && (address <= 4879)) return "Main.SR4";
    if ((address >= 4880) && (address <= 4887)) return "Main.SR3";
    if ((address >= 4888) && (address <= 4895)) return "Main.SR2";
    if ((address >= 4896) && (address <= 4903)) return "Main.SR1";
    if ((address >= 4904) && (address <= 4911)) return "Main.SR0";
    if ((address >= 4912) && (address <= 4919)) return "Main.CTX";
    if ((address >= 4920) && (address <= 4927)) return "Main.SP";
    if ((address >= 4928) && (address <= 4935)) return "Main.PC";
    if ((address >= 4936) && (address <= 4943)) return "Main.GAS";
    if ((address >= 4944) && (address <= 4951)) return "Main.zkPC";
    if ((address >= 4952) && (address <= 4959)) return "Main.RR";
    if ((address >= 4960) && (address <= 4967)) return "Main.HASHPOS";
    if ((address >= 4968) && (address <= 4975)) return "Main.RCX";
    if ((address >= 4976) && (address <= 4983)) return "Main.CONST7";
    if ((address >= 4984) && (address <= 4991)) return "Main.CONST6";
    if ((address >= 4992) && (address <= 4999)) return "Main.CONST5";
    if ((address >= 5000) && (address <= 5007)) return "Main.CONST4";
    if ((address >= 5008) && (address <= 5015)) return "Main.CONST3";
    if ((address >= 5016) && (address <= 5023)) return "Main.CONST2";
    if ((address >= 5024) && (address <= 5031)) return "Main.CONST1";
    if ((address >= 5032) && (address <= 5039)) return "Main.CONST0";
    if ((address >= 5040) && (address <= 5047)) return "Main.FREE7";
    if ((address >= 5048) && (address <= 5055)) return "Main.FREE6";
    if ((address >= 5056) && (address <= 5063)) return "Main.FREE5";
    if ((address >= 5064) && (address <= 5071)) return "Main.FREE4";
    if ((address >= 5072) && (address <= 5079)) return "Main.FREE3";
    if ((address >= 5080) && (address <= 5087)) return "Main.FREE2";
    if ((address >= 5088) && (address <= 5095)) return "Main.FREE1";
    if ((address >= 5096) && (address <= 5103)) return "Main.FREE0";
    if ((address >= 5104) && (address <= 5111)) return "Main.inA";
    if ((address >= 5112) && (address <= 5119)) return "Main.inB";
    if ((address >= 5120) && (address <= 5127)) return "Main.inC";
    if ((address >= 5128) && (address <= 5135)) return "Main.inROTL_C";
    if ((address >= 5136) && (address <= 5143)) return "Main.inD";
    if ((address >= 5144) && (address <= 5151)) return "Main.inE";
    if ((address >= 5152) && (address <= 5159)) return "Main.inSR";
    if ((address >= 5160) && (address <= 5167)) return "Main.inFREE";
    if ((address >= 5168) && (address <= 5175)) return "Main.inFREE0";
    if ((address >= 5176) && (address <= 5183)) return "Main.inCTX";
    if ((address >= 5184) && (address <= 5191)) return "Main.inSP";
    if ((address >= 5192) && (address <= 5199)) return "Main.inPC";
    if ((address >= 5200) && (address <= 5207)) return "Main.inGAS";
    if ((address >= 5208) && (address <= 5215)) return "Main.inSTEP";
    if ((address >= 5216) && (address <= 5223)) return "Main.inRR";
    if ((address >= 5224) && (address <= 5231)) return "Main.inHASHPOS";
    if ((address >= 5232) && (address <= 5239)) return "Main.inRCX";
    if ((address >= 5240) && (address <= 5247)) return "Main.setA";
    if ((address >= 5248) && (address <= 5255)) return "Main.setB";
    if ((address >= 5256) && (address <= 5263)) return "Main.setC";
    if ((address >= 5264) && (address <= 5271)) return "Main.setD";
    if ((address >= 5272) && (address <= 5279)) return "Main.setE";
    if ((address >= 5280) && (address <= 5287)) return "Main.setSR";
    if ((address >= 5288) && (address <= 5295)) return "Main.setCTX";
    if ((address >= 5296) && (address <= 5303)) return "Main.setSP";
    if ((address >= 5304) && (address <= 5311)) return "Main.setPC";
    if ((address >= 5312) && (address <= 5319)) return "Main.setGAS";
    if ((address >= 5320) && (address <= 5327)) return "Main.setRR";
    if ((address >= 5328) && (address <= 5335)) return "Main.setHASHPOS";
    if ((address >= 5336) && (address <= 5343)) return "Main.setRCX";
    if ((address >= 5344) && (address <= 5351)) return "Main.JMP";
    if ((address >= 5352) && (address <= 5359)) return "Main.JMPN";
    if ((address >= 5360) && (address <= 5367)) return "Main.JMPC";
    if ((address >= 5368) && (address <= 5375)) return "Main.JMPZ";
    if ((address >= 5376) && (address <= 5383)) return "Main.offset";
    if ((address >= 5384) && (address <= 5391)) return "Main.incStack";
    if ((address >= 5392) && (address <= 5399)) return "Main.isStack";
    if ((address >= 5400) && (address <= 5407)) return "Main.isMem";
    if ((address >= 5408) && (address <= 5415)) return "Main.ind";
    if ((address >= 5416) && (address <= 5423)) return "Main.indRR";
    if ((address >= 5424) && (address <= 5431)) return "Main.useCTX";
    if ((address >= 5432) && (address <= 5439)) return "Main.carry";
    if ((address >= 5440) && (address <= 5447)) return "Main.mOp";
    if ((address >= 5448) && (address <= 5455)) return "Main.mWR";
    if ((address >= 5456) && (address <= 5463)) return "Main.sWR";
    if ((address >= 5464) && (address <= 5471)) return "Main.sRD";
    if ((address >= 5472) && (address <= 5479)) return "Main.arithEq0";
    if ((address >= 5480) && (address <= 5487)) return "Main.arithEq1";
    if ((address >= 5488) && (address <= 5495)) return "Main.arithEq2";
    if ((address >= 5496) && (address <= 5503)) return "Main.arithEq3";
    if ((address >= 5504) && (address <= 5511)) return "Main.arithEq4";
    if ((address >= 5512) && (address <= 5519)) return "Main.arithEq5";
    if ((address >= 5520) && (address <= 5527)) return "Main.memAlignRD";
    if ((address >= 5528) && (address <= 5535)) return "Main.memAlignWR";
    if ((address >= 5536) && (address <= 5543)) return "Main.memAlignWR8";
    if ((address >= 5544) && (address <= 5551)) return "Main.hashK";
    if ((address >= 5552) && (address <= 5559)) return "Main.hashK1";
    if ((address >= 5560) && (address <= 5567)) return "Main.hashKLen";
    if ((address >= 5568) && (address <= 5575)) return "Main.hashKDigest";
    if ((address >= 5576) && (address <= 5583)) return "Main.hashP";
    if ((address >= 5584) && (address <= 5591)) return "Main.hashP1";
    if ((address >= 5592) && (address <= 5599)) return "Main.hashPLen";
    if ((address >= 5600) && (address <= 5607)) return "Main.hashPDigest";
    if ((address >= 5608) && (address <= 5615)) return "Main.hashS";
    if ((address >= 5616) && (address <= 5623)) return "Main.hashS1";
    if ((address >= 5624) && (address <= 5631)) return "Main.hashSLen";
    if ((address >= 5632) && (address <= 5639)) return "Main.hashSDigest";
    if ((address >= 5640) && (address <= 5647)) return "Main.bin";
    if ((address >= 5648) && (address <= 5655)) return "Main.binOpcode";
    if ((address >= 5656) && (address <= 5663)) return "Main.assert_pol";
    if ((address >= 5664) && (address <= 5671)) return "Main.repeat";
    if ((address >= 5672) && (address <= 5679)) return "Main.call";
    if ((address >= 5680) && (address <= 5687)) return "Main.return_pol";
    if ((address >= 5688) && (address <= 5695)) return "Main.isNeg";
    if ((address >= 5696) && (address <= 5703)) return "Main.cntArith";
    if ((address >= 5704) && (address <= 5711)) return "Main.cntBinary";
    if ((address >= 5712) && (address <= 5719)) return "Main.cntMemAlign";
    if ((address >= 5720) && (address <= 5727)) return "Main.cntKeccakF";
    if ((address >= 5728) && (address <= 5735)) return "Main.cntSha256F";
    if ((address >= 5736) && (address <= 5743)) return "Main.cntPoseidonG";
    if ((address >= 5744) && (address <= 5751)) return "Main.cntPaddingPG";
    if ((address >= 5752) && (address <= 5759)) return "Main.inCntArith";
    if ((address >= 5760) && (address <= 5767)) return "Main.inCntBinary";
    if ((address >= 5768) && (address <= 5775)) return "Main.inCntMemAlign";
    if ((address >= 5776) && (address <= 5783)) return "Main.inCntKeccakF";
    if ((address >= 5784) && (address <= 5791)) return "Main.inCntSha256F";
    if ((address >= 5792) && (address <= 5799)) return "Main.inCntPoseidonG";
    if ((address >= 5800) && (address <= 5807)) return "Main.inCntPaddingPG";
    if ((address >= 5808) && (address <= 5815)) return "Main.incCounter";
    if ((address >= 5816) && (address <= 5823)) return "Main.lJmpnCondValue";
    if ((address >= 5824) && (address <= 5831)) return "Main.hJmpnCondValueBit[0]";
    if ((address >= 5832) && (address <= 5839)) return "Main.hJmpnCondValueBit[1]";
    if ((address >= 5840) && (address <= 5847)) return "Main.hJmpnCondValueBit[2]";
    if ((address >= 5848) && (address <= 5855)) return "Main.hJmpnCondValueBit[3]";
    if ((address >= 5856) && (address <= 5863)) return "Main.hJmpnCondValueBit[4]";
    if ((address >= 5864) && (address <= 5871)) return "Main.hJmpnCondValueBit[5]";
    if ((address >= 5872) && (address <= 5879)) return "Main.hJmpnCondValueBit[6]";
    if ((address >= 5880) && (address <= 5887)) return "Main.RCXInv";
    if ((address >= 5888) && (address <= 5895)) return "Main.op0Inv";
    if ((address >= 5896) && (address <= 5903)) return "Main.jmpAddr";
    if ((address >= 5904) && (address <= 5911)) return "Main.elseAddr";
    if ((address >= 5912) && (address <= 5919)) return "Main.useJmpAddr";
    if ((address >= 5920) && (address <= 5927)) return "Main.useElseAddr";
    if ((address >= 5928) && (address <= 5935)) return "Main.sKeyI[0]";
    if ((address >= 5936) && (address <= 5943)) return "Main.sKeyI[1]";
    if ((address >= 5944) && (address <= 5951)) return "Main.sKeyI[2]";
    if ((address >= 5952) && (address <= 5959)) return "Main.sKeyI[3]";
    if ((address >= 5960) && (address <= 5967)) return "Main.sKey[0]";
    if ((address >= 5968) && (address <= 5975)) return "Main.sKey[1]";
    if ((address >= 5976) && (address <= 5983)) return "Main.sKey[2]";
    if ((address >= 5984) && (address <= 5991)) return "Main.sKey[3]";
    return "ERROR_NOT_FOUND";
}

} // namespace

#endif // COMMIT_POLS_HPP_fork_12

