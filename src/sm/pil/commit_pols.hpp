#ifndef COMMIT_POLS_HPP
#define COMMIT_POLS_HPP

#include <cstdint>
#include "ff/ff.hpp"

class Byte4CommitPols
{
public:
    uint16_t * freeIN;
    uint32_t * out;

    Byte4CommitPols (void * pAddress)
    {
        freeIN = (uint16_t *)((uint8_t *)pAddress + 0);
        out = (uint32_t *)((uint8_t *)pAddress + 4194304);
    }

    Byte4CommitPols (void * pAddress, uint64_t degree)
    {
        freeIN = (uint16_t *)((uint8_t *)pAddress + 0*degree);
        out = (uint32_t *)((uint8_t *)pAddress + 2*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 6; }
};

class MemAlignCommitPols
{
public:
    uint8_t * inM;
    uint8_t * inV;
    uint8_t * wr;
    uint32_t * m0[8];
    uint32_t * m1[8];
    uint32_t * w0[8];
    uint32_t * w1[8];
    uint32_t * v[8];
    uint8_t * offset;
    uint8_t * selW;
    FieldElement * factorV[8];
    FieldElement * latchOffset;
    FieldElement * latchWr;

    MemAlignCommitPols (void * pAddress)
    {
        inM = (uint8_t *)((uint8_t *)pAddress + 12582912);
        inV = (uint8_t *)((uint8_t *)pAddress + 14680064);
        wr = (uint8_t *)((uint8_t *)pAddress + 16777216);
        m0[0] = (uint32_t *)((uint8_t *)pAddress + 18874368);
        m0[1] = (uint32_t *)((uint8_t *)pAddress + 27262976);
        m0[2] = (uint32_t *)((uint8_t *)pAddress + 35651584);
        m0[3] = (uint32_t *)((uint8_t *)pAddress + 44040192);
        m0[4] = (uint32_t *)((uint8_t *)pAddress + 52428800);
        m0[5] = (uint32_t *)((uint8_t *)pAddress + 60817408);
        m0[6] = (uint32_t *)((uint8_t *)pAddress + 69206016);
        m0[7] = (uint32_t *)((uint8_t *)pAddress + 77594624);
        m1[0] = (uint32_t *)((uint8_t *)pAddress + 85983232);
        m1[1] = (uint32_t *)((uint8_t *)pAddress + 94371840);
        m1[2] = (uint32_t *)((uint8_t *)pAddress + 102760448);
        m1[3] = (uint32_t *)((uint8_t *)pAddress + 111149056);
        m1[4] = (uint32_t *)((uint8_t *)pAddress + 119537664);
        m1[5] = (uint32_t *)((uint8_t *)pAddress + 127926272);
        m1[6] = (uint32_t *)((uint8_t *)pAddress + 136314880);
        m1[7] = (uint32_t *)((uint8_t *)pAddress + 144703488);
        w0[0] = (uint32_t *)((uint8_t *)pAddress + 153092096);
        w0[1] = (uint32_t *)((uint8_t *)pAddress + 161480704);
        w0[2] = (uint32_t *)((uint8_t *)pAddress + 169869312);
        w0[3] = (uint32_t *)((uint8_t *)pAddress + 178257920);
        w0[4] = (uint32_t *)((uint8_t *)pAddress + 186646528);
        w0[5] = (uint32_t *)((uint8_t *)pAddress + 195035136);
        w0[6] = (uint32_t *)((uint8_t *)pAddress + 203423744);
        w0[7] = (uint32_t *)((uint8_t *)pAddress + 211812352);
        w1[0] = (uint32_t *)((uint8_t *)pAddress + 220200960);
        w1[1] = (uint32_t *)((uint8_t *)pAddress + 228589568);
        w1[2] = (uint32_t *)((uint8_t *)pAddress + 236978176);
        w1[3] = (uint32_t *)((uint8_t *)pAddress + 245366784);
        w1[4] = (uint32_t *)((uint8_t *)pAddress + 253755392);
        w1[5] = (uint32_t *)((uint8_t *)pAddress + 262144000);
        w1[6] = (uint32_t *)((uint8_t *)pAddress + 270532608);
        w1[7] = (uint32_t *)((uint8_t *)pAddress + 278921216);
        v[0] = (uint32_t *)((uint8_t *)pAddress + 287309824);
        v[1] = (uint32_t *)((uint8_t *)pAddress + 295698432);
        v[2] = (uint32_t *)((uint8_t *)pAddress + 304087040);
        v[3] = (uint32_t *)((uint8_t *)pAddress + 312475648);
        v[4] = (uint32_t *)((uint8_t *)pAddress + 320864256);
        v[5] = (uint32_t *)((uint8_t *)pAddress + 329252864);
        v[6] = (uint32_t *)((uint8_t *)pAddress + 337641472);
        v[7] = (uint32_t *)((uint8_t *)pAddress + 346030080);
        offset = (uint8_t *)((uint8_t *)pAddress + 354418688);
        selW = (uint8_t *)((uint8_t *)pAddress + 356515840);
        factorV[0] = (FieldElement *)((uint8_t *)pAddress + 358612992);
        factorV[1] = (FieldElement *)((uint8_t *)pAddress + 375390208);
        factorV[2] = (FieldElement *)((uint8_t *)pAddress + 392167424);
        factorV[3] = (FieldElement *)((uint8_t *)pAddress + 408944640);
        factorV[4] = (FieldElement *)((uint8_t *)pAddress + 425721856);
        factorV[5] = (FieldElement *)((uint8_t *)pAddress + 442499072);
        factorV[6] = (FieldElement *)((uint8_t *)pAddress + 459276288);
        factorV[7] = (FieldElement *)((uint8_t *)pAddress + 476053504);
        latchOffset = (FieldElement *)((uint8_t *)pAddress + 492830720);
        latchWr = (FieldElement *)((uint8_t *)pAddress + 509607936);
    }

    MemAlignCommitPols (void * pAddress, uint64_t degree)
    {
        inM = (uint8_t *)((uint8_t *)pAddress + 0*degree);
        inV = (uint8_t *)((uint8_t *)pAddress + 1*degree);
        wr = (uint8_t *)((uint8_t *)pAddress + 2*degree);
        m0[0] = (uint32_t *)((uint8_t *)pAddress + 3*degree);
        m0[1] = (uint32_t *)((uint8_t *)pAddress + 7*degree);
        m0[2] = (uint32_t *)((uint8_t *)pAddress + 11*degree);
        m0[3] = (uint32_t *)((uint8_t *)pAddress + 15*degree);
        m0[4] = (uint32_t *)((uint8_t *)pAddress + 19*degree);
        m0[5] = (uint32_t *)((uint8_t *)pAddress + 23*degree);
        m0[6] = (uint32_t *)((uint8_t *)pAddress + 27*degree);
        m0[7] = (uint32_t *)((uint8_t *)pAddress + 31*degree);
        m1[0] = (uint32_t *)((uint8_t *)pAddress + 35*degree);
        m1[1] = (uint32_t *)((uint8_t *)pAddress + 39*degree);
        m1[2] = (uint32_t *)((uint8_t *)pAddress + 43*degree);
        m1[3] = (uint32_t *)((uint8_t *)pAddress + 47*degree);
        m1[4] = (uint32_t *)((uint8_t *)pAddress + 51*degree);
        m1[5] = (uint32_t *)((uint8_t *)pAddress + 55*degree);
        m1[6] = (uint32_t *)((uint8_t *)pAddress + 59*degree);
        m1[7] = (uint32_t *)((uint8_t *)pAddress + 63*degree);
        w0[0] = (uint32_t *)((uint8_t *)pAddress + 67*degree);
        w0[1] = (uint32_t *)((uint8_t *)pAddress + 71*degree);
        w0[2] = (uint32_t *)((uint8_t *)pAddress + 75*degree);
        w0[3] = (uint32_t *)((uint8_t *)pAddress + 79*degree);
        w0[4] = (uint32_t *)((uint8_t *)pAddress + 83*degree);
        w0[5] = (uint32_t *)((uint8_t *)pAddress + 87*degree);
        w0[6] = (uint32_t *)((uint8_t *)pAddress + 91*degree);
        w0[7] = (uint32_t *)((uint8_t *)pAddress + 95*degree);
        w1[0] = (uint32_t *)((uint8_t *)pAddress + 99*degree);
        w1[1] = (uint32_t *)((uint8_t *)pAddress + 103*degree);
        w1[2] = (uint32_t *)((uint8_t *)pAddress + 107*degree);
        w1[3] = (uint32_t *)((uint8_t *)pAddress + 111*degree);
        w1[4] = (uint32_t *)((uint8_t *)pAddress + 115*degree);
        w1[5] = (uint32_t *)((uint8_t *)pAddress + 119*degree);
        w1[6] = (uint32_t *)((uint8_t *)pAddress + 123*degree);
        w1[7] = (uint32_t *)((uint8_t *)pAddress + 127*degree);
        v[0] = (uint32_t *)((uint8_t *)pAddress + 131*degree);
        v[1] = (uint32_t *)((uint8_t *)pAddress + 135*degree);
        v[2] = (uint32_t *)((uint8_t *)pAddress + 139*degree);
        v[3] = (uint32_t *)((uint8_t *)pAddress + 143*degree);
        v[4] = (uint32_t *)((uint8_t *)pAddress + 147*degree);
        v[5] = (uint32_t *)((uint8_t *)pAddress + 151*degree);
        v[6] = (uint32_t *)((uint8_t *)pAddress + 155*degree);
        v[7] = (uint32_t *)((uint8_t *)pAddress + 159*degree);
        offset = (uint8_t *)((uint8_t *)pAddress + 163*degree);
        selW = (uint8_t *)((uint8_t *)pAddress + 164*degree);
        factorV[0] = (FieldElement *)((uint8_t *)pAddress + 165*degree);
        factorV[1] = (FieldElement *)((uint8_t *)pAddress + 173*degree);
        factorV[2] = (FieldElement *)((uint8_t *)pAddress + 181*degree);
        factorV[3] = (FieldElement *)((uint8_t *)pAddress + 189*degree);
        factorV[4] = (FieldElement *)((uint8_t *)pAddress + 197*degree);
        factorV[5] = (FieldElement *)((uint8_t *)pAddress + 205*degree);
        factorV[6] = (FieldElement *)((uint8_t *)pAddress + 213*degree);
        factorV[7] = (FieldElement *)((uint8_t *)pAddress + 221*degree);
        latchOffset = (FieldElement *)((uint8_t *)pAddress + 229*degree);
        latchWr = (FieldElement *)((uint8_t *)pAddress + 237*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 245; }
};

class ArithCommitPols
{
public:
    FieldElement * x1[16];
    FieldElement * y1[16];
    FieldElement * x2[16];
    FieldElement * y2[16];
    FieldElement * x3[16];
    FieldElement * y3[16];
    FieldElement * s[16];
    FieldElement * q0[16];
    FieldElement * q1[16];
    FieldElement * q2[16];
    FieldElement * selEq[4];
    FieldElement * carryL[3];
    FieldElement * carryH[3];

    ArithCommitPols (void * pAddress)
    {
        x1[0] = (FieldElement *)((uint8_t *)pAddress + 526385152);
        x1[1] = (FieldElement *)((uint8_t *)pAddress + 543162368);
        x1[2] = (FieldElement *)((uint8_t *)pAddress + 559939584);
        x1[3] = (FieldElement *)((uint8_t *)pAddress + 576716800);
        x1[4] = (FieldElement *)((uint8_t *)pAddress + 593494016);
        x1[5] = (FieldElement *)((uint8_t *)pAddress + 610271232);
        x1[6] = (FieldElement *)((uint8_t *)pAddress + 627048448);
        x1[7] = (FieldElement *)((uint8_t *)pAddress + 643825664);
        x1[8] = (FieldElement *)((uint8_t *)pAddress + 660602880);
        x1[9] = (FieldElement *)((uint8_t *)pAddress + 677380096);
        x1[10] = (FieldElement *)((uint8_t *)pAddress + 694157312);
        x1[11] = (FieldElement *)((uint8_t *)pAddress + 710934528);
        x1[12] = (FieldElement *)((uint8_t *)pAddress + 727711744);
        x1[13] = (FieldElement *)((uint8_t *)pAddress + 744488960);
        x1[14] = (FieldElement *)((uint8_t *)pAddress + 761266176);
        x1[15] = (FieldElement *)((uint8_t *)pAddress + 778043392);
        y1[0] = (FieldElement *)((uint8_t *)pAddress + 794820608);
        y1[1] = (FieldElement *)((uint8_t *)pAddress + 811597824);
        y1[2] = (FieldElement *)((uint8_t *)pAddress + 828375040);
        y1[3] = (FieldElement *)((uint8_t *)pAddress + 845152256);
        y1[4] = (FieldElement *)((uint8_t *)pAddress + 861929472);
        y1[5] = (FieldElement *)((uint8_t *)pAddress + 878706688);
        y1[6] = (FieldElement *)((uint8_t *)pAddress + 895483904);
        y1[7] = (FieldElement *)((uint8_t *)pAddress + 912261120);
        y1[8] = (FieldElement *)((uint8_t *)pAddress + 929038336);
        y1[9] = (FieldElement *)((uint8_t *)pAddress + 945815552);
        y1[10] = (FieldElement *)((uint8_t *)pAddress + 962592768);
        y1[11] = (FieldElement *)((uint8_t *)pAddress + 979369984);
        y1[12] = (FieldElement *)((uint8_t *)pAddress + 996147200);
        y1[13] = (FieldElement *)((uint8_t *)pAddress + 1012924416);
        y1[14] = (FieldElement *)((uint8_t *)pAddress + 1029701632);
        y1[15] = (FieldElement *)((uint8_t *)pAddress + 1046478848);
        x2[0] = (FieldElement *)((uint8_t *)pAddress + 1063256064);
        x2[1] = (FieldElement *)((uint8_t *)pAddress + 1080033280);
        x2[2] = (FieldElement *)((uint8_t *)pAddress + 1096810496);
        x2[3] = (FieldElement *)((uint8_t *)pAddress + 1113587712);
        x2[4] = (FieldElement *)((uint8_t *)pAddress + 1130364928);
        x2[5] = (FieldElement *)((uint8_t *)pAddress + 1147142144);
        x2[6] = (FieldElement *)((uint8_t *)pAddress + 1163919360);
        x2[7] = (FieldElement *)((uint8_t *)pAddress + 1180696576);
        x2[8] = (FieldElement *)((uint8_t *)pAddress + 1197473792);
        x2[9] = (FieldElement *)((uint8_t *)pAddress + 1214251008);
        x2[10] = (FieldElement *)((uint8_t *)pAddress + 1231028224);
        x2[11] = (FieldElement *)((uint8_t *)pAddress + 1247805440);
        x2[12] = (FieldElement *)((uint8_t *)pAddress + 1264582656);
        x2[13] = (FieldElement *)((uint8_t *)pAddress + 1281359872);
        x2[14] = (FieldElement *)((uint8_t *)pAddress + 1298137088);
        x2[15] = (FieldElement *)((uint8_t *)pAddress + 1314914304);
        y2[0] = (FieldElement *)((uint8_t *)pAddress + 1331691520);
        y2[1] = (FieldElement *)((uint8_t *)pAddress + 1348468736);
        y2[2] = (FieldElement *)((uint8_t *)pAddress + 1365245952);
        y2[3] = (FieldElement *)((uint8_t *)pAddress + 1382023168);
        y2[4] = (FieldElement *)((uint8_t *)pAddress + 1398800384);
        y2[5] = (FieldElement *)((uint8_t *)pAddress + 1415577600);
        y2[6] = (FieldElement *)((uint8_t *)pAddress + 1432354816);
        y2[7] = (FieldElement *)((uint8_t *)pAddress + 1449132032);
        y2[8] = (FieldElement *)((uint8_t *)pAddress + 1465909248);
        y2[9] = (FieldElement *)((uint8_t *)pAddress + 1482686464);
        y2[10] = (FieldElement *)((uint8_t *)pAddress + 1499463680);
        y2[11] = (FieldElement *)((uint8_t *)pAddress + 1516240896);
        y2[12] = (FieldElement *)((uint8_t *)pAddress + 1533018112);
        y2[13] = (FieldElement *)((uint8_t *)pAddress + 1549795328);
        y2[14] = (FieldElement *)((uint8_t *)pAddress + 1566572544);
        y2[15] = (FieldElement *)((uint8_t *)pAddress + 1583349760);
        x3[0] = (FieldElement *)((uint8_t *)pAddress + 1600126976);
        x3[1] = (FieldElement *)((uint8_t *)pAddress + 1616904192);
        x3[2] = (FieldElement *)((uint8_t *)pAddress + 1633681408);
        x3[3] = (FieldElement *)((uint8_t *)pAddress + 1650458624);
        x3[4] = (FieldElement *)((uint8_t *)pAddress + 1667235840);
        x3[5] = (FieldElement *)((uint8_t *)pAddress + 1684013056);
        x3[6] = (FieldElement *)((uint8_t *)pAddress + 1700790272);
        x3[7] = (FieldElement *)((uint8_t *)pAddress + 1717567488);
        x3[8] = (FieldElement *)((uint8_t *)pAddress + 1734344704);
        x3[9] = (FieldElement *)((uint8_t *)pAddress + 1751121920);
        x3[10] = (FieldElement *)((uint8_t *)pAddress + 1767899136);
        x3[11] = (FieldElement *)((uint8_t *)pAddress + 1784676352);
        x3[12] = (FieldElement *)((uint8_t *)pAddress + 1801453568);
        x3[13] = (FieldElement *)((uint8_t *)pAddress + 1818230784);
        x3[14] = (FieldElement *)((uint8_t *)pAddress + 1835008000);
        x3[15] = (FieldElement *)((uint8_t *)pAddress + 1851785216);
        y3[0] = (FieldElement *)((uint8_t *)pAddress + 1868562432);
        y3[1] = (FieldElement *)((uint8_t *)pAddress + 1885339648);
        y3[2] = (FieldElement *)((uint8_t *)pAddress + 1902116864);
        y3[3] = (FieldElement *)((uint8_t *)pAddress + 1918894080);
        y3[4] = (FieldElement *)((uint8_t *)pAddress + 1935671296);
        y3[5] = (FieldElement *)((uint8_t *)pAddress + 1952448512);
        y3[6] = (FieldElement *)((uint8_t *)pAddress + 1969225728);
        y3[7] = (FieldElement *)((uint8_t *)pAddress + 1986002944);
        y3[8] = (FieldElement *)((uint8_t *)pAddress + 2002780160);
        y3[9] = (FieldElement *)((uint8_t *)pAddress + 2019557376);
        y3[10] = (FieldElement *)((uint8_t *)pAddress + 2036334592);
        y3[11] = (FieldElement *)((uint8_t *)pAddress + 2053111808);
        y3[12] = (FieldElement *)((uint8_t *)pAddress + 2069889024);
        y3[13] = (FieldElement *)((uint8_t *)pAddress + 2086666240);
        y3[14] = (FieldElement *)((uint8_t *)pAddress + 2103443456);
        y3[15] = (FieldElement *)((uint8_t *)pAddress + 2120220672);
        s[0] = (FieldElement *)((uint8_t *)pAddress + 2136997888);
        s[1] = (FieldElement *)((uint8_t *)pAddress + 2153775104);
        s[2] = (FieldElement *)((uint8_t *)pAddress + 2170552320);
        s[3] = (FieldElement *)((uint8_t *)pAddress + 2187329536);
        s[4] = (FieldElement *)((uint8_t *)pAddress + 2204106752);
        s[5] = (FieldElement *)((uint8_t *)pAddress + 2220883968);
        s[6] = (FieldElement *)((uint8_t *)pAddress + 2237661184);
        s[7] = (FieldElement *)((uint8_t *)pAddress + 2254438400);
        s[8] = (FieldElement *)((uint8_t *)pAddress + 2271215616);
        s[9] = (FieldElement *)((uint8_t *)pAddress + 2287992832);
        s[10] = (FieldElement *)((uint8_t *)pAddress + 2304770048);
        s[11] = (FieldElement *)((uint8_t *)pAddress + 2321547264);
        s[12] = (FieldElement *)((uint8_t *)pAddress + 2338324480);
        s[13] = (FieldElement *)((uint8_t *)pAddress + 2355101696);
        s[14] = (FieldElement *)((uint8_t *)pAddress + 2371878912);
        s[15] = (FieldElement *)((uint8_t *)pAddress + 2388656128);
        q0[0] = (FieldElement *)((uint8_t *)pAddress + 2405433344);
        q0[1] = (FieldElement *)((uint8_t *)pAddress + 2422210560);
        q0[2] = (FieldElement *)((uint8_t *)pAddress + 2438987776);
        q0[3] = (FieldElement *)((uint8_t *)pAddress + 2455764992);
        q0[4] = (FieldElement *)((uint8_t *)pAddress + 2472542208);
        q0[5] = (FieldElement *)((uint8_t *)pAddress + 2489319424);
        q0[6] = (FieldElement *)((uint8_t *)pAddress + 2506096640);
        q0[7] = (FieldElement *)((uint8_t *)pAddress + 2522873856);
        q0[8] = (FieldElement *)((uint8_t *)pAddress + 2539651072);
        q0[9] = (FieldElement *)((uint8_t *)pAddress + 2556428288);
        q0[10] = (FieldElement *)((uint8_t *)pAddress + 2573205504);
        q0[11] = (FieldElement *)((uint8_t *)pAddress + 2589982720);
        q0[12] = (FieldElement *)((uint8_t *)pAddress + 2606759936);
        q0[13] = (FieldElement *)((uint8_t *)pAddress + 2623537152);
        q0[14] = (FieldElement *)((uint8_t *)pAddress + 2640314368);
        q0[15] = (FieldElement *)((uint8_t *)pAddress + 2657091584);
        q1[0] = (FieldElement *)((uint8_t *)pAddress + 2673868800);
        q1[1] = (FieldElement *)((uint8_t *)pAddress + 2690646016);
        q1[2] = (FieldElement *)((uint8_t *)pAddress + 2707423232);
        q1[3] = (FieldElement *)((uint8_t *)pAddress + 2724200448);
        q1[4] = (FieldElement *)((uint8_t *)pAddress + 2740977664);
        q1[5] = (FieldElement *)((uint8_t *)pAddress + 2757754880);
        q1[6] = (FieldElement *)((uint8_t *)pAddress + 2774532096);
        q1[7] = (FieldElement *)((uint8_t *)pAddress + 2791309312);
        q1[8] = (FieldElement *)((uint8_t *)pAddress + 2808086528);
        q1[9] = (FieldElement *)((uint8_t *)pAddress + 2824863744);
        q1[10] = (FieldElement *)((uint8_t *)pAddress + 2841640960);
        q1[11] = (FieldElement *)((uint8_t *)pAddress + 2858418176);
        q1[12] = (FieldElement *)((uint8_t *)pAddress + 2875195392);
        q1[13] = (FieldElement *)((uint8_t *)pAddress + 2891972608);
        q1[14] = (FieldElement *)((uint8_t *)pAddress + 2908749824);
        q1[15] = (FieldElement *)((uint8_t *)pAddress + 2925527040);
        q2[0] = (FieldElement *)((uint8_t *)pAddress + 2942304256);
        q2[1] = (FieldElement *)((uint8_t *)pAddress + 2959081472);
        q2[2] = (FieldElement *)((uint8_t *)pAddress + 2975858688);
        q2[3] = (FieldElement *)((uint8_t *)pAddress + 2992635904);
        q2[4] = (FieldElement *)((uint8_t *)pAddress + 3009413120);
        q2[5] = (FieldElement *)((uint8_t *)pAddress + 3026190336);
        q2[6] = (FieldElement *)((uint8_t *)pAddress + 3042967552);
        q2[7] = (FieldElement *)((uint8_t *)pAddress + 3059744768);
        q2[8] = (FieldElement *)((uint8_t *)pAddress + 3076521984);
        q2[9] = (FieldElement *)((uint8_t *)pAddress + 3093299200);
        q2[10] = (FieldElement *)((uint8_t *)pAddress + 3110076416);
        q2[11] = (FieldElement *)((uint8_t *)pAddress + 3126853632);
        q2[12] = (FieldElement *)((uint8_t *)pAddress + 3143630848);
        q2[13] = (FieldElement *)((uint8_t *)pAddress + 3160408064);
        q2[14] = (FieldElement *)((uint8_t *)pAddress + 3177185280);
        q2[15] = (FieldElement *)((uint8_t *)pAddress + 3193962496);
        selEq[0] = (FieldElement *)((uint8_t *)pAddress + 3210739712);
        selEq[1] = (FieldElement *)((uint8_t *)pAddress + 3227516928);
        selEq[2] = (FieldElement *)((uint8_t *)pAddress + 3244294144);
        selEq[3] = (FieldElement *)((uint8_t *)pAddress + 3261071360);
        carryL[0] = (FieldElement *)((uint8_t *)pAddress + 3277848576);
        carryL[1] = (FieldElement *)((uint8_t *)pAddress + 3294625792);
        carryL[2] = (FieldElement *)((uint8_t *)pAddress + 3311403008);
        carryH[0] = (FieldElement *)((uint8_t *)pAddress + 3328180224);
        carryH[1] = (FieldElement *)((uint8_t *)pAddress + 3344957440);
        carryH[2] = (FieldElement *)((uint8_t *)pAddress + 3361734656);
    }

    ArithCommitPols (void * pAddress, uint64_t degree)
    {
        x1[0] = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        x1[1] = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        x1[2] = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        x1[3] = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        x1[4] = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        x1[5] = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        x1[6] = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        x1[7] = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        x1[8] = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        x1[9] = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        x1[10] = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        x1[11] = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        x1[12] = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        x1[13] = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        x1[14] = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        x1[15] = (FieldElement *)((uint8_t *)pAddress + 120*degree);
        y1[0] = (FieldElement *)((uint8_t *)pAddress + 128*degree);
        y1[1] = (FieldElement *)((uint8_t *)pAddress + 136*degree);
        y1[2] = (FieldElement *)((uint8_t *)pAddress + 144*degree);
        y1[3] = (FieldElement *)((uint8_t *)pAddress + 152*degree);
        y1[4] = (FieldElement *)((uint8_t *)pAddress + 160*degree);
        y1[5] = (FieldElement *)((uint8_t *)pAddress + 168*degree);
        y1[6] = (FieldElement *)((uint8_t *)pAddress + 176*degree);
        y1[7] = (FieldElement *)((uint8_t *)pAddress + 184*degree);
        y1[8] = (FieldElement *)((uint8_t *)pAddress + 192*degree);
        y1[9] = (FieldElement *)((uint8_t *)pAddress + 200*degree);
        y1[10] = (FieldElement *)((uint8_t *)pAddress + 208*degree);
        y1[11] = (FieldElement *)((uint8_t *)pAddress + 216*degree);
        y1[12] = (FieldElement *)((uint8_t *)pAddress + 224*degree);
        y1[13] = (FieldElement *)((uint8_t *)pAddress + 232*degree);
        y1[14] = (FieldElement *)((uint8_t *)pAddress + 240*degree);
        y1[15] = (FieldElement *)((uint8_t *)pAddress + 248*degree);
        x2[0] = (FieldElement *)((uint8_t *)pAddress + 256*degree);
        x2[1] = (FieldElement *)((uint8_t *)pAddress + 264*degree);
        x2[2] = (FieldElement *)((uint8_t *)pAddress + 272*degree);
        x2[3] = (FieldElement *)((uint8_t *)pAddress + 280*degree);
        x2[4] = (FieldElement *)((uint8_t *)pAddress + 288*degree);
        x2[5] = (FieldElement *)((uint8_t *)pAddress + 296*degree);
        x2[6] = (FieldElement *)((uint8_t *)pAddress + 304*degree);
        x2[7] = (FieldElement *)((uint8_t *)pAddress + 312*degree);
        x2[8] = (FieldElement *)((uint8_t *)pAddress + 320*degree);
        x2[9] = (FieldElement *)((uint8_t *)pAddress + 328*degree);
        x2[10] = (FieldElement *)((uint8_t *)pAddress + 336*degree);
        x2[11] = (FieldElement *)((uint8_t *)pAddress + 344*degree);
        x2[12] = (FieldElement *)((uint8_t *)pAddress + 352*degree);
        x2[13] = (FieldElement *)((uint8_t *)pAddress + 360*degree);
        x2[14] = (FieldElement *)((uint8_t *)pAddress + 368*degree);
        x2[15] = (FieldElement *)((uint8_t *)pAddress + 376*degree);
        y2[0] = (FieldElement *)((uint8_t *)pAddress + 384*degree);
        y2[1] = (FieldElement *)((uint8_t *)pAddress + 392*degree);
        y2[2] = (FieldElement *)((uint8_t *)pAddress + 400*degree);
        y2[3] = (FieldElement *)((uint8_t *)pAddress + 408*degree);
        y2[4] = (FieldElement *)((uint8_t *)pAddress + 416*degree);
        y2[5] = (FieldElement *)((uint8_t *)pAddress + 424*degree);
        y2[6] = (FieldElement *)((uint8_t *)pAddress + 432*degree);
        y2[7] = (FieldElement *)((uint8_t *)pAddress + 440*degree);
        y2[8] = (FieldElement *)((uint8_t *)pAddress + 448*degree);
        y2[9] = (FieldElement *)((uint8_t *)pAddress + 456*degree);
        y2[10] = (FieldElement *)((uint8_t *)pAddress + 464*degree);
        y2[11] = (FieldElement *)((uint8_t *)pAddress + 472*degree);
        y2[12] = (FieldElement *)((uint8_t *)pAddress + 480*degree);
        y2[13] = (FieldElement *)((uint8_t *)pAddress + 488*degree);
        y2[14] = (FieldElement *)((uint8_t *)pAddress + 496*degree);
        y2[15] = (FieldElement *)((uint8_t *)pAddress + 504*degree);
        x3[0] = (FieldElement *)((uint8_t *)pAddress + 512*degree);
        x3[1] = (FieldElement *)((uint8_t *)pAddress + 520*degree);
        x3[2] = (FieldElement *)((uint8_t *)pAddress + 528*degree);
        x3[3] = (FieldElement *)((uint8_t *)pAddress + 536*degree);
        x3[4] = (FieldElement *)((uint8_t *)pAddress + 544*degree);
        x3[5] = (FieldElement *)((uint8_t *)pAddress + 552*degree);
        x3[6] = (FieldElement *)((uint8_t *)pAddress + 560*degree);
        x3[7] = (FieldElement *)((uint8_t *)pAddress + 568*degree);
        x3[8] = (FieldElement *)((uint8_t *)pAddress + 576*degree);
        x3[9] = (FieldElement *)((uint8_t *)pAddress + 584*degree);
        x3[10] = (FieldElement *)((uint8_t *)pAddress + 592*degree);
        x3[11] = (FieldElement *)((uint8_t *)pAddress + 600*degree);
        x3[12] = (FieldElement *)((uint8_t *)pAddress + 608*degree);
        x3[13] = (FieldElement *)((uint8_t *)pAddress + 616*degree);
        x3[14] = (FieldElement *)((uint8_t *)pAddress + 624*degree);
        x3[15] = (FieldElement *)((uint8_t *)pAddress + 632*degree);
        y3[0] = (FieldElement *)((uint8_t *)pAddress + 640*degree);
        y3[1] = (FieldElement *)((uint8_t *)pAddress + 648*degree);
        y3[2] = (FieldElement *)((uint8_t *)pAddress + 656*degree);
        y3[3] = (FieldElement *)((uint8_t *)pAddress + 664*degree);
        y3[4] = (FieldElement *)((uint8_t *)pAddress + 672*degree);
        y3[5] = (FieldElement *)((uint8_t *)pAddress + 680*degree);
        y3[6] = (FieldElement *)((uint8_t *)pAddress + 688*degree);
        y3[7] = (FieldElement *)((uint8_t *)pAddress + 696*degree);
        y3[8] = (FieldElement *)((uint8_t *)pAddress + 704*degree);
        y3[9] = (FieldElement *)((uint8_t *)pAddress + 712*degree);
        y3[10] = (FieldElement *)((uint8_t *)pAddress + 720*degree);
        y3[11] = (FieldElement *)((uint8_t *)pAddress + 728*degree);
        y3[12] = (FieldElement *)((uint8_t *)pAddress + 736*degree);
        y3[13] = (FieldElement *)((uint8_t *)pAddress + 744*degree);
        y3[14] = (FieldElement *)((uint8_t *)pAddress + 752*degree);
        y3[15] = (FieldElement *)((uint8_t *)pAddress + 760*degree);
        s[0] = (FieldElement *)((uint8_t *)pAddress + 768*degree);
        s[1] = (FieldElement *)((uint8_t *)pAddress + 776*degree);
        s[2] = (FieldElement *)((uint8_t *)pAddress + 784*degree);
        s[3] = (FieldElement *)((uint8_t *)pAddress + 792*degree);
        s[4] = (FieldElement *)((uint8_t *)pAddress + 800*degree);
        s[5] = (FieldElement *)((uint8_t *)pAddress + 808*degree);
        s[6] = (FieldElement *)((uint8_t *)pAddress + 816*degree);
        s[7] = (FieldElement *)((uint8_t *)pAddress + 824*degree);
        s[8] = (FieldElement *)((uint8_t *)pAddress + 832*degree);
        s[9] = (FieldElement *)((uint8_t *)pAddress + 840*degree);
        s[10] = (FieldElement *)((uint8_t *)pAddress + 848*degree);
        s[11] = (FieldElement *)((uint8_t *)pAddress + 856*degree);
        s[12] = (FieldElement *)((uint8_t *)pAddress + 864*degree);
        s[13] = (FieldElement *)((uint8_t *)pAddress + 872*degree);
        s[14] = (FieldElement *)((uint8_t *)pAddress + 880*degree);
        s[15] = (FieldElement *)((uint8_t *)pAddress + 888*degree);
        q0[0] = (FieldElement *)((uint8_t *)pAddress + 896*degree);
        q0[1] = (FieldElement *)((uint8_t *)pAddress + 904*degree);
        q0[2] = (FieldElement *)((uint8_t *)pAddress + 912*degree);
        q0[3] = (FieldElement *)((uint8_t *)pAddress + 920*degree);
        q0[4] = (FieldElement *)((uint8_t *)pAddress + 928*degree);
        q0[5] = (FieldElement *)((uint8_t *)pAddress + 936*degree);
        q0[6] = (FieldElement *)((uint8_t *)pAddress + 944*degree);
        q0[7] = (FieldElement *)((uint8_t *)pAddress + 952*degree);
        q0[8] = (FieldElement *)((uint8_t *)pAddress + 960*degree);
        q0[9] = (FieldElement *)((uint8_t *)pAddress + 968*degree);
        q0[10] = (FieldElement *)((uint8_t *)pAddress + 976*degree);
        q0[11] = (FieldElement *)((uint8_t *)pAddress + 984*degree);
        q0[12] = (FieldElement *)((uint8_t *)pAddress + 992*degree);
        q0[13] = (FieldElement *)((uint8_t *)pAddress + 1000*degree);
        q0[14] = (FieldElement *)((uint8_t *)pAddress + 1008*degree);
        q0[15] = (FieldElement *)((uint8_t *)pAddress + 1016*degree);
        q1[0] = (FieldElement *)((uint8_t *)pAddress + 1024*degree);
        q1[1] = (FieldElement *)((uint8_t *)pAddress + 1032*degree);
        q1[2] = (FieldElement *)((uint8_t *)pAddress + 1040*degree);
        q1[3] = (FieldElement *)((uint8_t *)pAddress + 1048*degree);
        q1[4] = (FieldElement *)((uint8_t *)pAddress + 1056*degree);
        q1[5] = (FieldElement *)((uint8_t *)pAddress + 1064*degree);
        q1[6] = (FieldElement *)((uint8_t *)pAddress + 1072*degree);
        q1[7] = (FieldElement *)((uint8_t *)pAddress + 1080*degree);
        q1[8] = (FieldElement *)((uint8_t *)pAddress + 1088*degree);
        q1[9] = (FieldElement *)((uint8_t *)pAddress + 1096*degree);
        q1[10] = (FieldElement *)((uint8_t *)pAddress + 1104*degree);
        q1[11] = (FieldElement *)((uint8_t *)pAddress + 1112*degree);
        q1[12] = (FieldElement *)((uint8_t *)pAddress + 1120*degree);
        q1[13] = (FieldElement *)((uint8_t *)pAddress + 1128*degree);
        q1[14] = (FieldElement *)((uint8_t *)pAddress + 1136*degree);
        q1[15] = (FieldElement *)((uint8_t *)pAddress + 1144*degree);
        q2[0] = (FieldElement *)((uint8_t *)pAddress + 1152*degree);
        q2[1] = (FieldElement *)((uint8_t *)pAddress + 1160*degree);
        q2[2] = (FieldElement *)((uint8_t *)pAddress + 1168*degree);
        q2[3] = (FieldElement *)((uint8_t *)pAddress + 1176*degree);
        q2[4] = (FieldElement *)((uint8_t *)pAddress + 1184*degree);
        q2[5] = (FieldElement *)((uint8_t *)pAddress + 1192*degree);
        q2[6] = (FieldElement *)((uint8_t *)pAddress + 1200*degree);
        q2[7] = (FieldElement *)((uint8_t *)pAddress + 1208*degree);
        q2[8] = (FieldElement *)((uint8_t *)pAddress + 1216*degree);
        q2[9] = (FieldElement *)((uint8_t *)pAddress + 1224*degree);
        q2[10] = (FieldElement *)((uint8_t *)pAddress + 1232*degree);
        q2[11] = (FieldElement *)((uint8_t *)pAddress + 1240*degree);
        q2[12] = (FieldElement *)((uint8_t *)pAddress + 1248*degree);
        q2[13] = (FieldElement *)((uint8_t *)pAddress + 1256*degree);
        q2[14] = (FieldElement *)((uint8_t *)pAddress + 1264*degree);
        q2[15] = (FieldElement *)((uint8_t *)pAddress + 1272*degree);
        selEq[0] = (FieldElement *)((uint8_t *)pAddress + 1280*degree);
        selEq[1] = (FieldElement *)((uint8_t *)pAddress + 1288*degree);
        selEq[2] = (FieldElement *)((uint8_t *)pAddress + 1296*degree);
        selEq[3] = (FieldElement *)((uint8_t *)pAddress + 1304*degree);
        carryL[0] = (FieldElement *)((uint8_t *)pAddress + 1312*degree);
        carryL[1] = (FieldElement *)((uint8_t *)pAddress + 1320*degree);
        carryL[2] = (FieldElement *)((uint8_t *)pAddress + 1328*degree);
        carryH[0] = (FieldElement *)((uint8_t *)pAddress + 1336*degree);
        carryH[1] = (FieldElement *)((uint8_t *)pAddress + 1344*degree);
        carryH[2] = (FieldElement *)((uint8_t *)pAddress + 1352*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 1360; }
};

class BinaryCommitPols
{
public:
    uint8_t * freeInA;
    uint8_t * freeInB;
    uint8_t * freeInC;
    uint32_t * a0;
    uint32_t * a1;
    uint32_t * a2;
    uint32_t * a3;
    uint32_t * a4;
    uint32_t * a5;
    uint32_t * a6;
    uint32_t * a7;
    uint32_t * b0;
    uint32_t * b1;
    uint32_t * b2;
    uint32_t * b3;
    uint32_t * b4;
    uint32_t * b5;
    uint32_t * b6;
    uint32_t * b7;
    uint32_t * c0;
    uint32_t * c1;
    uint32_t * c2;
    uint32_t * c3;
    uint32_t * c4;
    uint32_t * c5;
    uint32_t * c6;
    uint32_t * c7;
    uint8_t * opcode;
    uint8_t * cIn;
    uint8_t * cOut;
    uint8_t * lCout;
    uint8_t * lOpcode;
    uint8_t * last;
    uint8_t * useCarry;

    BinaryCommitPols (void * pAddress)
    {
        freeInA = (uint8_t *)((uint8_t *)pAddress + 3378511872);
        freeInB = (uint8_t *)((uint8_t *)pAddress + 3380609024);
        freeInC = (uint8_t *)((uint8_t *)pAddress + 3382706176);
        a0 = (uint32_t *)((uint8_t *)pAddress + 3384803328);
        a1 = (uint32_t *)((uint8_t *)pAddress + 3393191936);
        a2 = (uint32_t *)((uint8_t *)pAddress + 3401580544);
        a3 = (uint32_t *)((uint8_t *)pAddress + 3409969152);
        a4 = (uint32_t *)((uint8_t *)pAddress + 3418357760);
        a5 = (uint32_t *)((uint8_t *)pAddress + 3426746368);
        a6 = (uint32_t *)((uint8_t *)pAddress + 3435134976);
        a7 = (uint32_t *)((uint8_t *)pAddress + 3443523584);
        b0 = (uint32_t *)((uint8_t *)pAddress + 3451912192);
        b1 = (uint32_t *)((uint8_t *)pAddress + 3460300800);
        b2 = (uint32_t *)((uint8_t *)pAddress + 3468689408);
        b3 = (uint32_t *)((uint8_t *)pAddress + 3477078016);
        b4 = (uint32_t *)((uint8_t *)pAddress + 3485466624);
        b5 = (uint32_t *)((uint8_t *)pAddress + 3493855232);
        b6 = (uint32_t *)((uint8_t *)pAddress + 3502243840);
        b7 = (uint32_t *)((uint8_t *)pAddress + 3510632448);
        c0 = (uint32_t *)((uint8_t *)pAddress + 3519021056);
        c1 = (uint32_t *)((uint8_t *)pAddress + 3527409664);
        c2 = (uint32_t *)((uint8_t *)pAddress + 3535798272);
        c3 = (uint32_t *)((uint8_t *)pAddress + 3544186880);
        c4 = (uint32_t *)((uint8_t *)pAddress + 3552575488);
        c5 = (uint32_t *)((uint8_t *)pAddress + 3560964096);
        c6 = (uint32_t *)((uint8_t *)pAddress + 3569352704);
        c7 = (uint32_t *)((uint8_t *)pAddress + 3577741312);
        opcode = (uint8_t *)((uint8_t *)pAddress + 3586129920);
        cIn = (uint8_t *)((uint8_t *)pAddress + 3588227072);
        cOut = (uint8_t *)((uint8_t *)pAddress + 3590324224);
        lCout = (uint8_t *)((uint8_t *)pAddress + 3592421376);
        lOpcode = (uint8_t *)((uint8_t *)pAddress + 3594518528);
        last = (uint8_t *)((uint8_t *)pAddress + 3596615680);
        useCarry = (uint8_t *)((uint8_t *)pAddress + 3598712832);
    }

    BinaryCommitPols (void * pAddress, uint64_t degree)
    {
        freeInA = (uint8_t *)((uint8_t *)pAddress + 0*degree);
        freeInB = (uint8_t *)((uint8_t *)pAddress + 1*degree);
        freeInC = (uint8_t *)((uint8_t *)pAddress + 2*degree);
        a0 = (uint32_t *)((uint8_t *)pAddress + 3*degree);
        a1 = (uint32_t *)((uint8_t *)pAddress + 7*degree);
        a2 = (uint32_t *)((uint8_t *)pAddress + 11*degree);
        a3 = (uint32_t *)((uint8_t *)pAddress + 15*degree);
        a4 = (uint32_t *)((uint8_t *)pAddress + 19*degree);
        a5 = (uint32_t *)((uint8_t *)pAddress + 23*degree);
        a6 = (uint32_t *)((uint8_t *)pAddress + 27*degree);
        a7 = (uint32_t *)((uint8_t *)pAddress + 31*degree);
        b0 = (uint32_t *)((uint8_t *)pAddress + 35*degree);
        b1 = (uint32_t *)((uint8_t *)pAddress + 39*degree);
        b2 = (uint32_t *)((uint8_t *)pAddress + 43*degree);
        b3 = (uint32_t *)((uint8_t *)pAddress + 47*degree);
        b4 = (uint32_t *)((uint8_t *)pAddress + 51*degree);
        b5 = (uint32_t *)((uint8_t *)pAddress + 55*degree);
        b6 = (uint32_t *)((uint8_t *)pAddress + 59*degree);
        b7 = (uint32_t *)((uint8_t *)pAddress + 63*degree);
        c0 = (uint32_t *)((uint8_t *)pAddress + 67*degree);
        c1 = (uint32_t *)((uint8_t *)pAddress + 71*degree);
        c2 = (uint32_t *)((uint8_t *)pAddress + 75*degree);
        c3 = (uint32_t *)((uint8_t *)pAddress + 79*degree);
        c4 = (uint32_t *)((uint8_t *)pAddress + 83*degree);
        c5 = (uint32_t *)((uint8_t *)pAddress + 87*degree);
        c6 = (uint32_t *)((uint8_t *)pAddress + 91*degree);
        c7 = (uint32_t *)((uint8_t *)pAddress + 95*degree);
        opcode = (uint8_t *)((uint8_t *)pAddress + 99*degree);
        cIn = (uint8_t *)((uint8_t *)pAddress + 100*degree);
        cOut = (uint8_t *)((uint8_t *)pAddress + 101*degree);
        lCout = (uint8_t *)((uint8_t *)pAddress + 102*degree);
        lOpcode = (uint8_t *)((uint8_t *)pAddress + 103*degree);
        last = (uint8_t *)((uint8_t *)pAddress + 104*degree);
        useCarry = (uint8_t *)((uint8_t *)pAddress + 105*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 106; }
};

class PoseidonGCommitPols
{
public:
    FieldElement * in0;
    FieldElement * in1;
    FieldElement * in2;
    FieldElement * in3;
    FieldElement * in4;
    FieldElement * in5;
    FieldElement * in6;
    FieldElement * in7;
    FieldElement * hashType;
    FieldElement * cap1;
    FieldElement * cap2;
    FieldElement * cap3;
    FieldElement * hash0;
    FieldElement * hash1;
    FieldElement * hash2;
    FieldElement * hash3;

    PoseidonGCommitPols (void * pAddress)
    {
        in0 = (FieldElement *)((uint8_t *)pAddress + 3600809984);
        in1 = (FieldElement *)((uint8_t *)pAddress + 3617587200);
        in2 = (FieldElement *)((uint8_t *)pAddress + 3634364416);
        in3 = (FieldElement *)((uint8_t *)pAddress + 3651141632);
        in4 = (FieldElement *)((uint8_t *)pAddress + 3667918848);
        in5 = (FieldElement *)((uint8_t *)pAddress + 3684696064);
        in6 = (FieldElement *)((uint8_t *)pAddress + 3701473280);
        in7 = (FieldElement *)((uint8_t *)pAddress + 3718250496);
        hashType = (FieldElement *)((uint8_t *)pAddress + 3735027712);
        cap1 = (FieldElement *)((uint8_t *)pAddress + 3751804928);
        cap2 = (FieldElement *)((uint8_t *)pAddress + 3768582144);
        cap3 = (FieldElement *)((uint8_t *)pAddress + 3785359360);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 3802136576);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 3818913792);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 3835691008);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 3852468224);
    }

    PoseidonGCommitPols (void * pAddress, uint64_t degree)
    {
        in0 = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        in1 = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        in2 = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        in3 = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        in4 = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        in5 = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        in6 = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        in7 = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        hashType = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        cap1 = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        cap2 = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        cap3 = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 120*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 128; }
};

class PaddingPGCommitPols
{
public:
    FieldElement * acc[8];
    FieldElement * freeIn;
    FieldElement * addr;
    FieldElement * rem;
    FieldElement * remInv;
    FieldElement * spare;
    FieldElement * firstHash;
    FieldElement * curHash0;
    FieldElement * curHash1;
    FieldElement * curHash2;
    FieldElement * curHash3;
    FieldElement * prevHash0;
    FieldElement * prevHash1;
    FieldElement * prevHash2;
    FieldElement * prevHash3;
    FieldElement * len;
    FieldElement * crOffset;
    FieldElement * crLen;
    FieldElement * crOffsetInv;
    FieldElement * crF0;
    FieldElement * crF1;
    FieldElement * crF2;
    FieldElement * crF3;
    FieldElement * crF4;
    FieldElement * crF5;
    FieldElement * crF6;
    FieldElement * crF7;
    FieldElement * crV0;
    FieldElement * crV1;
    FieldElement * crV2;
    FieldElement * crV3;
    FieldElement * crV4;
    FieldElement * crV5;
    FieldElement * crV6;
    FieldElement * crV7;

    PaddingPGCommitPols (void * pAddress)
    {
        acc[0] = (FieldElement *)((uint8_t *)pAddress + 3869245440);
        acc[1] = (FieldElement *)((uint8_t *)pAddress + 3886022656);
        acc[2] = (FieldElement *)((uint8_t *)pAddress + 3902799872);
        acc[3] = (FieldElement *)((uint8_t *)pAddress + 3919577088);
        acc[4] = (FieldElement *)((uint8_t *)pAddress + 3936354304);
        acc[5] = (FieldElement *)((uint8_t *)pAddress + 3953131520);
        acc[6] = (FieldElement *)((uint8_t *)pAddress + 3969908736);
        acc[7] = (FieldElement *)((uint8_t *)pAddress + 3986685952);
        freeIn = (FieldElement *)((uint8_t *)pAddress + 4003463168);
        addr = (FieldElement *)((uint8_t *)pAddress + 4020240384);
        rem = (FieldElement *)((uint8_t *)pAddress + 4037017600);
        remInv = (FieldElement *)((uint8_t *)pAddress + 4053794816);
        spare = (FieldElement *)((uint8_t *)pAddress + 4070572032);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 4087349248);
        curHash0 = (FieldElement *)((uint8_t *)pAddress + 4104126464);
        curHash1 = (FieldElement *)((uint8_t *)pAddress + 4120903680);
        curHash2 = (FieldElement *)((uint8_t *)pAddress + 4137680896);
        curHash3 = (FieldElement *)((uint8_t *)pAddress + 4154458112);
        prevHash0 = (FieldElement *)((uint8_t *)pAddress + 4171235328);
        prevHash1 = (FieldElement *)((uint8_t *)pAddress + 4188012544);
        prevHash2 = (FieldElement *)((uint8_t *)pAddress + 4204789760);
        prevHash3 = (FieldElement *)((uint8_t *)pAddress + 4221566976);
        len = (FieldElement *)((uint8_t *)pAddress + 4238344192);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 4255121408);
        crLen = (FieldElement *)((uint8_t *)pAddress + 4271898624);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 4288675840);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 4305453056);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 4322230272);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 4339007488);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 4355784704);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 4372561920);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 4389339136);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 4406116352);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 4422893568);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 4439670784);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 4456448000);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 4473225216);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 4490002432);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 4506779648);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 4523556864);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 4540334080);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 4557111296);
    }

    PaddingPGCommitPols (void * pAddress, uint64_t degree)
    {
        acc[0] = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        acc[1] = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        acc[2] = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        acc[3] = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        acc[4] = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        acc[5] = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        acc[6] = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        acc[7] = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        freeIn = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        addr = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        rem = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        remInv = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        spare = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        curHash0 = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        curHash1 = (FieldElement *)((uint8_t *)pAddress + 120*degree);
        curHash2 = (FieldElement *)((uint8_t *)pAddress + 128*degree);
        curHash3 = (FieldElement *)((uint8_t *)pAddress + 136*degree);
        prevHash0 = (FieldElement *)((uint8_t *)pAddress + 144*degree);
        prevHash1 = (FieldElement *)((uint8_t *)pAddress + 152*degree);
        prevHash2 = (FieldElement *)((uint8_t *)pAddress + 160*degree);
        prevHash3 = (FieldElement *)((uint8_t *)pAddress + 168*degree);
        len = (FieldElement *)((uint8_t *)pAddress + 176*degree);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 184*degree);
        crLen = (FieldElement *)((uint8_t *)pAddress + 192*degree);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 200*degree);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 208*degree);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 216*degree);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 224*degree);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 232*degree);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 240*degree);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 248*degree);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 256*degree);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 264*degree);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 272*degree);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 280*degree);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 288*degree);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 296*degree);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 304*degree);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 312*degree);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 320*degree);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 328*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 336; }
};

class StorageCommitPols
{
public:
    uint64_t * free0;
    uint64_t * free1;
    uint64_t * free2;
    uint64_t * free3;
    uint64_t * hashLeft0;
    uint64_t * hashLeft1;
    uint64_t * hashLeft2;
    uint64_t * hashLeft3;
    uint64_t * hashRight0;
    uint64_t * hashRight1;
    uint64_t * hashRight2;
    uint64_t * hashRight3;
    uint64_t * oldRoot0;
    uint64_t * oldRoot1;
    uint64_t * oldRoot2;
    uint64_t * oldRoot3;
    uint64_t * newRoot0;
    uint64_t * newRoot1;
    uint64_t * newRoot2;
    uint64_t * newRoot3;
    uint64_t * valueLow0;
    uint64_t * valueLow1;
    uint64_t * valueLow2;
    uint64_t * valueLow3;
    uint64_t * valueHigh0;
    uint64_t * valueHigh1;
    uint64_t * valueHigh2;
    uint64_t * valueHigh3;
    uint64_t * siblingValueHash0;
    uint64_t * siblingValueHash1;
    uint64_t * siblingValueHash2;
    uint64_t * siblingValueHash3;
    uint64_t * rkey0;
    uint64_t * rkey1;
    uint64_t * rkey2;
    uint64_t * rkey3;
    uint64_t * siblingRkey0;
    uint64_t * siblingRkey1;
    uint64_t * siblingRkey2;
    uint64_t * siblingRkey3;
    uint64_t * rkeyBit;
    uint64_t * level0;
    uint64_t * level1;
    uint64_t * level2;
    uint64_t * level3;
    uint64_t * pc;
    uint8_t * selOldRoot;
    uint8_t * selNewRoot;
    uint8_t * selValueLow;
    uint8_t * selValueHigh;
    uint8_t * selSiblingValueHash;
    uint8_t * selRkey;
    uint8_t * selRkeyBit;
    uint8_t * selSiblingRkey;
    uint8_t * selFree;
    uint8_t * setHashLeft;
    uint8_t * setHashRight;
    uint8_t * setOldRoot;
    uint8_t * setNewRoot;
    uint8_t * setValueLow;
    uint8_t * setValueHigh;
    uint8_t * setSiblingValueHash;
    uint8_t * setRkey;
    uint8_t * setSiblingRkey;
    uint8_t * setRkeyBit;
    uint8_t * setLevel;
    uint8_t * iHash;
    uint8_t * iHashType;
    uint8_t * iLatchSet;
    uint8_t * iLatchGet;
    uint8_t * iClimbRkey;
    uint8_t * iClimbSiblingRkey;
    uint8_t * iClimbSiblingRkeyN;
    uint8_t * iRotateLevel;
    uint8_t * iJmpz;
    uint8_t * iJmp;
    uint64_t * iConst0;
    uint64_t * iConst1;
    uint64_t * iConst2;
    uint64_t * iConst3;
    uint64_t * iAddress;
    FieldElement * op0inv;

    StorageCommitPols (void * pAddress)
    {
        free0 = (uint64_t *)((uint8_t *)pAddress + 4573888512);
        free1 = (uint64_t *)((uint8_t *)pAddress + 4590665728);
        free2 = (uint64_t *)((uint8_t *)pAddress + 4607442944);
        free3 = (uint64_t *)((uint8_t *)pAddress + 4624220160);
        hashLeft0 = (uint64_t *)((uint8_t *)pAddress + 4640997376);
        hashLeft1 = (uint64_t *)((uint8_t *)pAddress + 4657774592);
        hashLeft2 = (uint64_t *)((uint8_t *)pAddress + 4674551808);
        hashLeft3 = (uint64_t *)((uint8_t *)pAddress + 4691329024);
        hashRight0 = (uint64_t *)((uint8_t *)pAddress + 4708106240);
        hashRight1 = (uint64_t *)((uint8_t *)pAddress + 4724883456);
        hashRight2 = (uint64_t *)((uint8_t *)pAddress + 4741660672);
        hashRight3 = (uint64_t *)((uint8_t *)pAddress + 4758437888);
        oldRoot0 = (uint64_t *)((uint8_t *)pAddress + 4775215104);
        oldRoot1 = (uint64_t *)((uint8_t *)pAddress + 4791992320);
        oldRoot2 = (uint64_t *)((uint8_t *)pAddress + 4808769536);
        oldRoot3 = (uint64_t *)((uint8_t *)pAddress + 4825546752);
        newRoot0 = (uint64_t *)((uint8_t *)pAddress + 4842323968);
        newRoot1 = (uint64_t *)((uint8_t *)pAddress + 4859101184);
        newRoot2 = (uint64_t *)((uint8_t *)pAddress + 4875878400);
        newRoot3 = (uint64_t *)((uint8_t *)pAddress + 4892655616);
        valueLow0 = (uint64_t *)((uint8_t *)pAddress + 4909432832);
        valueLow1 = (uint64_t *)((uint8_t *)pAddress + 4926210048);
        valueLow2 = (uint64_t *)((uint8_t *)pAddress + 4942987264);
        valueLow3 = (uint64_t *)((uint8_t *)pAddress + 4959764480);
        valueHigh0 = (uint64_t *)((uint8_t *)pAddress + 4976541696);
        valueHigh1 = (uint64_t *)((uint8_t *)pAddress + 4993318912);
        valueHigh2 = (uint64_t *)((uint8_t *)pAddress + 5010096128);
        valueHigh3 = (uint64_t *)((uint8_t *)pAddress + 5026873344);
        siblingValueHash0 = (uint64_t *)((uint8_t *)pAddress + 5043650560);
        siblingValueHash1 = (uint64_t *)((uint8_t *)pAddress + 5060427776);
        siblingValueHash2 = (uint64_t *)((uint8_t *)pAddress + 5077204992);
        siblingValueHash3 = (uint64_t *)((uint8_t *)pAddress + 5093982208);
        rkey0 = (uint64_t *)((uint8_t *)pAddress + 5110759424);
        rkey1 = (uint64_t *)((uint8_t *)pAddress + 5127536640);
        rkey2 = (uint64_t *)((uint8_t *)pAddress + 5144313856);
        rkey3 = (uint64_t *)((uint8_t *)pAddress + 5161091072);
        siblingRkey0 = (uint64_t *)((uint8_t *)pAddress + 5177868288);
        siblingRkey1 = (uint64_t *)((uint8_t *)pAddress + 5194645504);
        siblingRkey2 = (uint64_t *)((uint8_t *)pAddress + 5211422720);
        siblingRkey3 = (uint64_t *)((uint8_t *)pAddress + 5228199936);
        rkeyBit = (uint64_t *)((uint8_t *)pAddress + 5244977152);
        level0 = (uint64_t *)((uint8_t *)pAddress + 5261754368);
        level1 = (uint64_t *)((uint8_t *)pAddress + 5278531584);
        level2 = (uint64_t *)((uint8_t *)pAddress + 5295308800);
        level3 = (uint64_t *)((uint8_t *)pAddress + 5312086016);
        pc = (uint64_t *)((uint8_t *)pAddress + 5328863232);
        selOldRoot = (uint8_t *)((uint8_t *)pAddress + 5345640448);
        selNewRoot = (uint8_t *)((uint8_t *)pAddress + 5347737600);
        selValueLow = (uint8_t *)((uint8_t *)pAddress + 5349834752);
        selValueHigh = (uint8_t *)((uint8_t *)pAddress + 5351931904);
        selSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 5354029056);
        selRkey = (uint8_t *)((uint8_t *)pAddress + 5356126208);
        selRkeyBit = (uint8_t *)((uint8_t *)pAddress + 5358223360);
        selSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 5360320512);
        selFree = (uint8_t *)((uint8_t *)pAddress + 5362417664);
        setHashLeft = (uint8_t *)((uint8_t *)pAddress + 5364514816);
        setHashRight = (uint8_t *)((uint8_t *)pAddress + 5366611968);
        setOldRoot = (uint8_t *)((uint8_t *)pAddress + 5368709120);
        setNewRoot = (uint8_t *)((uint8_t *)pAddress + 5370806272);
        setValueLow = (uint8_t *)((uint8_t *)pAddress + 5372903424);
        setValueHigh = (uint8_t *)((uint8_t *)pAddress + 5375000576);
        setSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 5377097728);
        setRkey = (uint8_t *)((uint8_t *)pAddress + 5379194880);
        setSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 5381292032);
        setRkeyBit = (uint8_t *)((uint8_t *)pAddress + 5383389184);
        setLevel = (uint8_t *)((uint8_t *)pAddress + 5385486336);
        iHash = (uint8_t *)((uint8_t *)pAddress + 5387583488);
        iHashType = (uint8_t *)((uint8_t *)pAddress + 5389680640);
        iLatchSet = (uint8_t *)((uint8_t *)pAddress + 5391777792);
        iLatchGet = (uint8_t *)((uint8_t *)pAddress + 5393874944);
        iClimbRkey = (uint8_t *)((uint8_t *)pAddress + 5395972096);
        iClimbSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 5398069248);
        iClimbSiblingRkeyN = (uint8_t *)((uint8_t *)pAddress + 5400166400);
        iRotateLevel = (uint8_t *)((uint8_t *)pAddress + 5402263552);
        iJmpz = (uint8_t *)((uint8_t *)pAddress + 5404360704);
        iJmp = (uint8_t *)((uint8_t *)pAddress + 5406457856);
        iConst0 = (uint64_t *)((uint8_t *)pAddress + 5408555008);
        iConst1 = (uint64_t *)((uint8_t *)pAddress + 5425332224);
        iConst2 = (uint64_t *)((uint8_t *)pAddress + 5442109440);
        iConst3 = (uint64_t *)((uint8_t *)pAddress + 5458886656);
        iAddress = (uint64_t *)((uint8_t *)pAddress + 5475663872);
        op0inv = (FieldElement *)((uint8_t *)pAddress + 5492441088);
    }

    StorageCommitPols (void * pAddress, uint64_t degree)
    {
        free0 = (uint64_t *)((uint8_t *)pAddress + 0*degree);
        free1 = (uint64_t *)((uint8_t *)pAddress + 8*degree);
        free2 = (uint64_t *)((uint8_t *)pAddress + 16*degree);
        free3 = (uint64_t *)((uint8_t *)pAddress + 24*degree);
        hashLeft0 = (uint64_t *)((uint8_t *)pAddress + 32*degree);
        hashLeft1 = (uint64_t *)((uint8_t *)pAddress + 40*degree);
        hashLeft2 = (uint64_t *)((uint8_t *)pAddress + 48*degree);
        hashLeft3 = (uint64_t *)((uint8_t *)pAddress + 56*degree);
        hashRight0 = (uint64_t *)((uint8_t *)pAddress + 64*degree);
        hashRight1 = (uint64_t *)((uint8_t *)pAddress + 72*degree);
        hashRight2 = (uint64_t *)((uint8_t *)pAddress + 80*degree);
        hashRight3 = (uint64_t *)((uint8_t *)pAddress + 88*degree);
        oldRoot0 = (uint64_t *)((uint8_t *)pAddress + 96*degree);
        oldRoot1 = (uint64_t *)((uint8_t *)pAddress + 104*degree);
        oldRoot2 = (uint64_t *)((uint8_t *)pAddress + 112*degree);
        oldRoot3 = (uint64_t *)((uint8_t *)pAddress + 120*degree);
        newRoot0 = (uint64_t *)((uint8_t *)pAddress + 128*degree);
        newRoot1 = (uint64_t *)((uint8_t *)pAddress + 136*degree);
        newRoot2 = (uint64_t *)((uint8_t *)pAddress + 144*degree);
        newRoot3 = (uint64_t *)((uint8_t *)pAddress + 152*degree);
        valueLow0 = (uint64_t *)((uint8_t *)pAddress + 160*degree);
        valueLow1 = (uint64_t *)((uint8_t *)pAddress + 168*degree);
        valueLow2 = (uint64_t *)((uint8_t *)pAddress + 176*degree);
        valueLow3 = (uint64_t *)((uint8_t *)pAddress + 184*degree);
        valueHigh0 = (uint64_t *)((uint8_t *)pAddress + 192*degree);
        valueHigh1 = (uint64_t *)((uint8_t *)pAddress + 200*degree);
        valueHigh2 = (uint64_t *)((uint8_t *)pAddress + 208*degree);
        valueHigh3 = (uint64_t *)((uint8_t *)pAddress + 216*degree);
        siblingValueHash0 = (uint64_t *)((uint8_t *)pAddress + 224*degree);
        siblingValueHash1 = (uint64_t *)((uint8_t *)pAddress + 232*degree);
        siblingValueHash2 = (uint64_t *)((uint8_t *)pAddress + 240*degree);
        siblingValueHash3 = (uint64_t *)((uint8_t *)pAddress + 248*degree);
        rkey0 = (uint64_t *)((uint8_t *)pAddress + 256*degree);
        rkey1 = (uint64_t *)((uint8_t *)pAddress + 264*degree);
        rkey2 = (uint64_t *)((uint8_t *)pAddress + 272*degree);
        rkey3 = (uint64_t *)((uint8_t *)pAddress + 280*degree);
        siblingRkey0 = (uint64_t *)((uint8_t *)pAddress + 288*degree);
        siblingRkey1 = (uint64_t *)((uint8_t *)pAddress + 296*degree);
        siblingRkey2 = (uint64_t *)((uint8_t *)pAddress + 304*degree);
        siblingRkey3 = (uint64_t *)((uint8_t *)pAddress + 312*degree);
        rkeyBit = (uint64_t *)((uint8_t *)pAddress + 320*degree);
        level0 = (uint64_t *)((uint8_t *)pAddress + 328*degree);
        level1 = (uint64_t *)((uint8_t *)pAddress + 336*degree);
        level2 = (uint64_t *)((uint8_t *)pAddress + 344*degree);
        level3 = (uint64_t *)((uint8_t *)pAddress + 352*degree);
        pc = (uint64_t *)((uint8_t *)pAddress + 360*degree);
        selOldRoot = (uint8_t *)((uint8_t *)pAddress + 368*degree);
        selNewRoot = (uint8_t *)((uint8_t *)pAddress + 369*degree);
        selValueLow = (uint8_t *)((uint8_t *)pAddress + 370*degree);
        selValueHigh = (uint8_t *)((uint8_t *)pAddress + 371*degree);
        selSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 372*degree);
        selRkey = (uint8_t *)((uint8_t *)pAddress + 373*degree);
        selRkeyBit = (uint8_t *)((uint8_t *)pAddress + 374*degree);
        selSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 375*degree);
        selFree = (uint8_t *)((uint8_t *)pAddress + 376*degree);
        setHashLeft = (uint8_t *)((uint8_t *)pAddress + 377*degree);
        setHashRight = (uint8_t *)((uint8_t *)pAddress + 378*degree);
        setOldRoot = (uint8_t *)((uint8_t *)pAddress + 379*degree);
        setNewRoot = (uint8_t *)((uint8_t *)pAddress + 380*degree);
        setValueLow = (uint8_t *)((uint8_t *)pAddress + 381*degree);
        setValueHigh = (uint8_t *)((uint8_t *)pAddress + 382*degree);
        setSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 383*degree);
        setRkey = (uint8_t *)((uint8_t *)pAddress + 384*degree);
        setSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 385*degree);
        setRkeyBit = (uint8_t *)((uint8_t *)pAddress + 386*degree);
        setLevel = (uint8_t *)((uint8_t *)pAddress + 387*degree);
        iHash = (uint8_t *)((uint8_t *)pAddress + 388*degree);
        iHashType = (uint8_t *)((uint8_t *)pAddress + 389*degree);
        iLatchSet = (uint8_t *)((uint8_t *)pAddress + 390*degree);
        iLatchGet = (uint8_t *)((uint8_t *)pAddress + 391*degree);
        iClimbRkey = (uint8_t *)((uint8_t *)pAddress + 392*degree);
        iClimbSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 393*degree);
        iClimbSiblingRkeyN = (uint8_t *)((uint8_t *)pAddress + 394*degree);
        iRotateLevel = (uint8_t *)((uint8_t *)pAddress + 395*degree);
        iJmpz = (uint8_t *)((uint8_t *)pAddress + 396*degree);
        iJmp = (uint8_t *)((uint8_t *)pAddress + 397*degree);
        iConst0 = (uint64_t *)((uint8_t *)pAddress + 398*degree);
        iConst1 = (uint64_t *)((uint8_t *)pAddress + 406*degree);
        iConst2 = (uint64_t *)((uint8_t *)pAddress + 414*degree);
        iConst3 = (uint64_t *)((uint8_t *)pAddress + 422*degree);
        iAddress = (uint64_t *)((uint8_t *)pAddress + 430*degree);
        op0inv = (FieldElement *)((uint8_t *)pAddress + 438*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 446; }
};

class NormGate9CommitPols
{
public:
    FieldElement * freeA;
    FieldElement * freeB;
    FieldElement * gateType;
    FieldElement * freeANorm;
    FieldElement * freeBNorm;
    FieldElement * freeCNorm;
    FieldElement * a;
    FieldElement * b;
    FieldElement * c;

    NormGate9CommitPols (void * pAddress)
    {
        freeA = (FieldElement *)((uint8_t *)pAddress + 5509218304);
        freeB = (FieldElement *)((uint8_t *)pAddress + 5525995520);
        gateType = (FieldElement *)((uint8_t *)pAddress + 5542772736);
        freeANorm = (FieldElement *)((uint8_t *)pAddress + 5559549952);
        freeBNorm = (FieldElement *)((uint8_t *)pAddress + 5576327168);
        freeCNorm = (FieldElement *)((uint8_t *)pAddress + 5593104384);
        a = (FieldElement *)((uint8_t *)pAddress + 5609881600);
        b = (FieldElement *)((uint8_t *)pAddress + 5626658816);
        c = (FieldElement *)((uint8_t *)pAddress + 5643436032);
    }

    NormGate9CommitPols (void * pAddress, uint64_t degree)
    {
        freeA = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        freeB = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        gateType = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        freeANorm = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        freeBNorm = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        freeCNorm = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        a = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        b = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        c = (FieldElement *)((uint8_t *)pAddress + 64*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 72; }
};

class KeccakFCommitPols
{
public:
    FieldElement * a;
    FieldElement * b;
    FieldElement * c;

    KeccakFCommitPols (void * pAddress)
    {
        a = (FieldElement *)((uint8_t *)pAddress + 5660213248);
        b = (FieldElement *)((uint8_t *)pAddress + 5676990464);
        c = (FieldElement *)((uint8_t *)pAddress + 5693767680);
    }

    KeccakFCommitPols (void * pAddress, uint64_t degree)
    {
        a = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        b = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        c = (FieldElement *)((uint8_t *)pAddress + 16*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 24; }
};

class Nine2OneCommitPols
{
public:
    FieldElement * bit;
    FieldElement * field9;

    Nine2OneCommitPols (void * pAddress)
    {
        bit = (FieldElement *)((uint8_t *)pAddress + 5710544896);
        field9 = (FieldElement *)((uint8_t *)pAddress + 5727322112);
    }

    Nine2OneCommitPols (void * pAddress, uint64_t degree)
    {
        bit = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        field9 = (FieldElement *)((uint8_t *)pAddress + 8*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 16; }
};

class PaddingKKBitCommitPols
{
public:
    FieldElement * rBit;
    FieldElement * sOutBit;
    FieldElement * r8;
    FieldElement * connected;
    FieldElement * sOut0;
    FieldElement * sOut1;
    FieldElement * sOut2;
    FieldElement * sOut3;
    FieldElement * sOut4;
    FieldElement * sOut5;
    FieldElement * sOut6;
    FieldElement * sOut7;

    PaddingKKBitCommitPols (void * pAddress)
    {
        rBit = (FieldElement *)((uint8_t *)pAddress + 5744099328);
        sOutBit = (FieldElement *)((uint8_t *)pAddress + 5760876544);
        r8 = (FieldElement *)((uint8_t *)pAddress + 5777653760);
        connected = (FieldElement *)((uint8_t *)pAddress + 5794430976);
        sOut0 = (FieldElement *)((uint8_t *)pAddress + 5811208192);
        sOut1 = (FieldElement *)((uint8_t *)pAddress + 5827985408);
        sOut2 = (FieldElement *)((uint8_t *)pAddress + 5844762624);
        sOut3 = (FieldElement *)((uint8_t *)pAddress + 5861539840);
        sOut4 = (FieldElement *)((uint8_t *)pAddress + 5878317056);
        sOut5 = (FieldElement *)((uint8_t *)pAddress + 5895094272);
        sOut6 = (FieldElement *)((uint8_t *)pAddress + 5911871488);
        sOut7 = (FieldElement *)((uint8_t *)pAddress + 5928648704);
    }

    PaddingKKBitCommitPols (void * pAddress, uint64_t degree)
    {
        rBit = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        sOutBit = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        r8 = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        connected = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        sOut0 = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        sOut1 = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        sOut2 = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        sOut3 = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        sOut4 = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        sOut5 = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        sOut6 = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        sOut7 = (FieldElement *)((uint8_t *)pAddress + 88*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 96; }
};

class PaddingKKCommitPols
{
public:
    FieldElement * freeIn;
    FieldElement * connected;
    FieldElement * addr;
    FieldElement * rem;
    FieldElement * remInv;
    FieldElement * spare;
    FieldElement * firstHash;
    FieldElement * len;
    FieldElement * hash0;
    FieldElement * hash1;
    FieldElement * hash2;
    FieldElement * hash3;
    FieldElement * hash4;
    FieldElement * hash5;
    FieldElement * hash6;
    FieldElement * hash7;
    FieldElement * crOffset;
    FieldElement * crLen;
    FieldElement * crOffsetInv;
    FieldElement * crF0;
    FieldElement * crF1;
    FieldElement * crF2;
    FieldElement * crF3;
    FieldElement * crF4;
    FieldElement * crF5;
    FieldElement * crF6;
    FieldElement * crF7;
    FieldElement * crV0;
    FieldElement * crV1;
    FieldElement * crV2;
    FieldElement * crV3;
    FieldElement * crV4;
    FieldElement * crV5;
    FieldElement * crV6;
    FieldElement * crV7;

    PaddingKKCommitPols (void * pAddress)
    {
        freeIn = (FieldElement *)((uint8_t *)pAddress + 5945425920);
        connected = (FieldElement *)((uint8_t *)pAddress + 5962203136);
        addr = (FieldElement *)((uint8_t *)pAddress + 5978980352);
        rem = (FieldElement *)((uint8_t *)pAddress + 5995757568);
        remInv = (FieldElement *)((uint8_t *)pAddress + 6012534784);
        spare = (FieldElement *)((uint8_t *)pAddress + 6029312000);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 6046089216);
        len = (FieldElement *)((uint8_t *)pAddress + 6062866432);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 6079643648);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 6096420864);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 6113198080);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 6129975296);
        hash4 = (FieldElement *)((uint8_t *)pAddress + 6146752512);
        hash5 = (FieldElement *)((uint8_t *)pAddress + 6163529728);
        hash6 = (FieldElement *)((uint8_t *)pAddress + 6180306944);
        hash7 = (FieldElement *)((uint8_t *)pAddress + 6197084160);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 6213861376);
        crLen = (FieldElement *)((uint8_t *)pAddress + 6230638592);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 6247415808);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 6264193024);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 6280970240);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 6297747456);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 6314524672);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 6331301888);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 6348079104);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 6364856320);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 6381633536);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 6398410752);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 6415187968);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 6431965184);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 6448742400);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 6465519616);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 6482296832);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 6499074048);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 6515851264);
    }

    PaddingKKCommitPols (void * pAddress, uint64_t degree)
    {
        freeIn = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        connected = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        addr = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        rem = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        remInv = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        spare = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        len = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        hash4 = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        hash5 = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        hash6 = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        hash7 = (FieldElement *)((uint8_t *)pAddress + 120*degree);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 128*degree);
        crLen = (FieldElement *)((uint8_t *)pAddress + 136*degree);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 144*degree);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 152*degree);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 160*degree);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 168*degree);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 176*degree);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 184*degree);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 192*degree);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 200*degree);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 208*degree);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 216*degree);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 224*degree);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 232*degree);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 240*degree);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 248*degree);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 256*degree);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 264*degree);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 272*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 280; }
};

class MemCommitPols
{
public:
    FieldElement * addr;
    FieldElement * step;
    FieldElement * mOp;
    FieldElement * mWr;
    FieldElement * val[8];
    FieldElement * lastAccess;

    MemCommitPols (void * pAddress)
    {
        addr = (FieldElement *)((uint8_t *)pAddress + 6532628480);
        step = (FieldElement *)((uint8_t *)pAddress + 6549405696);
        mOp = (FieldElement *)((uint8_t *)pAddress + 6566182912);
        mWr = (FieldElement *)((uint8_t *)pAddress + 6582960128);
        val[0] = (FieldElement *)((uint8_t *)pAddress + 6599737344);
        val[1] = (FieldElement *)((uint8_t *)pAddress + 6616514560);
        val[2] = (FieldElement *)((uint8_t *)pAddress + 6633291776);
        val[3] = (FieldElement *)((uint8_t *)pAddress + 6650068992);
        val[4] = (FieldElement *)((uint8_t *)pAddress + 6666846208);
        val[5] = (FieldElement *)((uint8_t *)pAddress + 6683623424);
        val[6] = (FieldElement *)((uint8_t *)pAddress + 6700400640);
        val[7] = (FieldElement *)((uint8_t *)pAddress + 6717177856);
        lastAccess = (FieldElement *)((uint8_t *)pAddress + 6733955072);
    }

    MemCommitPols (void * pAddress, uint64_t degree)
    {
        addr = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        step = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        mOp = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        mWr = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        val[0] = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        val[1] = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        val[2] = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        val[3] = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        val[4] = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        val[5] = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        val[6] = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        val[7] = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        lastAccess = (FieldElement *)((uint8_t *)pAddress + 96*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 104; }
};

class MainCommitPols
{
public:
    uint32_t * A7;
    uint32_t * A6;
    uint32_t * A5;
    uint32_t * A4;
    uint32_t * A3;
    uint32_t * A2;
    uint32_t * A1;
    FieldElement * A0;
    uint32_t * B7;
    uint32_t * B6;
    uint32_t * B5;
    uint32_t * B4;
    uint32_t * B3;
    uint32_t * B2;
    uint32_t * B1;
    FieldElement * B0;
    uint32_t * C7;
    uint32_t * C6;
    uint32_t * C5;
    uint32_t * C4;
    uint32_t * C3;
    uint32_t * C2;
    uint32_t * C1;
    FieldElement * C0;
    uint32_t * D7;
    uint32_t * D6;
    uint32_t * D5;
    uint32_t * D4;
    uint32_t * D3;
    uint32_t * D2;
    uint32_t * D1;
    FieldElement * D0;
    uint32_t * E7;
    uint32_t * E6;
    uint32_t * E5;
    uint32_t * E4;
    uint32_t * E3;
    uint32_t * E2;
    uint32_t * E1;
    FieldElement * E0;
    uint32_t * SR7;
    uint32_t * SR6;
    uint32_t * SR5;
    uint32_t * SR4;
    uint32_t * SR3;
    uint32_t * SR2;
    uint32_t * SR1;
    uint32_t * SR0;
    uint32_t * CTX;
    uint16_t * SP;
    uint32_t * PC;
    uint64_t * GAS;
    uint32_t * MAXMEM;
    uint32_t * zkPC;
    uint32_t * RR;
    uint32_t * HASHPOS;
    FieldElement * CONST7;
    FieldElement * CONST6;
    FieldElement * CONST5;
    FieldElement * CONST4;
    FieldElement * CONST3;
    FieldElement * CONST2;
    FieldElement * CONST1;
    FieldElement * CONST0;
    FieldElement * FREE7;
    FieldElement * FREE6;
    FieldElement * FREE5;
    FieldElement * FREE4;
    FieldElement * FREE3;
    FieldElement * FREE2;
    FieldElement * FREE1;
    FieldElement * FREE0;
    FieldElement * inA;
    FieldElement * inB;
    FieldElement * inC;
    FieldElement * inD;
    FieldElement * inE;
    FieldElement * inSR;
    FieldElement * inFREE;
    FieldElement * inCTX;
    FieldElement * inSP;
    FieldElement * inPC;
    FieldElement * inGAS;
    FieldElement * inMAXMEM;
    FieldElement * inSTEP;
    FieldElement * inRR;
    FieldElement * inHASHPOS;
    uint8_t * setA;
    uint8_t * setB;
    uint8_t * setC;
    uint8_t * setD;
    uint8_t * setE;
    uint8_t * setSR;
    uint8_t * setCTX;
    uint8_t * setSP;
    uint8_t * setPC;
    uint8_t * setGAS;
    uint8_t * setMAXMEM;
    uint8_t * JMP;
    uint8_t * JMPN;
    uint8_t * JMPC;
    uint8_t * setRR;
    uint8_t * setHASHPOS;
    uint32_t * offset;
    int32_t * incStack;
    int32_t * incCode;
    uint8_t * isStack;
    uint8_t * isCode;
    uint8_t * isMem;
    uint8_t * ind;
    uint8_t * indRR;
    uint8_t * useCTX;
    uint8_t * carry;
    uint8_t * mOp;
    uint8_t * mWR;
    uint8_t * sWR;
    uint8_t * sRD;
    uint8_t * arith;
    uint8_t * arithEq0;
    uint8_t * arithEq1;
    uint8_t * arithEq2;
    uint8_t * arithEq3;
    uint8_t * memAlign;
    uint8_t * memAlignWR;
    uint8_t * hashK;
    uint8_t * hashKLen;
    uint8_t * hashKDigest;
    uint8_t * hashP;
    uint8_t * hashPLen;
    uint8_t * hashPDigest;
    uint8_t * bin;
    uint8_t * binOpcode;
    uint8_t * assert;
    uint8_t * opcodeRomMap;
    uint8_t * isNeg;
    uint8_t * isMaxMem;
    FieldElement * sKeyI[4];
    FieldElement * sKey[4];

    MainCommitPols (void * pAddress)
    {
        A7 = (uint32_t *)((uint8_t *)pAddress + 6750732288);
        A6 = (uint32_t *)((uint8_t *)pAddress + 6759120896);
        A5 = (uint32_t *)((uint8_t *)pAddress + 6767509504);
        A4 = (uint32_t *)((uint8_t *)pAddress + 6775898112);
        A3 = (uint32_t *)((uint8_t *)pAddress + 6784286720);
        A2 = (uint32_t *)((uint8_t *)pAddress + 6792675328);
        A1 = (uint32_t *)((uint8_t *)pAddress + 6801063936);
        A0 = (FieldElement *)((uint8_t *)pAddress + 6809452544);
        B7 = (uint32_t *)((uint8_t *)pAddress + 6826229760);
        B6 = (uint32_t *)((uint8_t *)pAddress + 6834618368);
        B5 = (uint32_t *)((uint8_t *)pAddress + 6843006976);
        B4 = (uint32_t *)((uint8_t *)pAddress + 6851395584);
        B3 = (uint32_t *)((uint8_t *)pAddress + 6859784192);
        B2 = (uint32_t *)((uint8_t *)pAddress + 6868172800);
        B1 = (uint32_t *)((uint8_t *)pAddress + 6876561408);
        B0 = (FieldElement *)((uint8_t *)pAddress + 6884950016);
        C7 = (uint32_t *)((uint8_t *)pAddress + 6901727232);
        C6 = (uint32_t *)((uint8_t *)pAddress + 6910115840);
        C5 = (uint32_t *)((uint8_t *)pAddress + 6918504448);
        C4 = (uint32_t *)((uint8_t *)pAddress + 6926893056);
        C3 = (uint32_t *)((uint8_t *)pAddress + 6935281664);
        C2 = (uint32_t *)((uint8_t *)pAddress + 6943670272);
        C1 = (uint32_t *)((uint8_t *)pAddress + 6952058880);
        C0 = (FieldElement *)((uint8_t *)pAddress + 6960447488);
        D7 = (uint32_t *)((uint8_t *)pAddress + 6977224704);
        D6 = (uint32_t *)((uint8_t *)pAddress + 6985613312);
        D5 = (uint32_t *)((uint8_t *)pAddress + 6994001920);
        D4 = (uint32_t *)((uint8_t *)pAddress + 7002390528);
        D3 = (uint32_t *)((uint8_t *)pAddress + 7010779136);
        D2 = (uint32_t *)((uint8_t *)pAddress + 7019167744);
        D1 = (uint32_t *)((uint8_t *)pAddress + 7027556352);
        D0 = (FieldElement *)((uint8_t *)pAddress + 7035944960);
        E7 = (uint32_t *)((uint8_t *)pAddress + 7052722176);
        E6 = (uint32_t *)((uint8_t *)pAddress + 7061110784);
        E5 = (uint32_t *)((uint8_t *)pAddress + 7069499392);
        E4 = (uint32_t *)((uint8_t *)pAddress + 7077888000);
        E3 = (uint32_t *)((uint8_t *)pAddress + 7086276608);
        E2 = (uint32_t *)((uint8_t *)pAddress + 7094665216);
        E1 = (uint32_t *)((uint8_t *)pAddress + 7103053824);
        E0 = (FieldElement *)((uint8_t *)pAddress + 7111442432);
        SR7 = (uint32_t *)((uint8_t *)pAddress + 7128219648);
        SR6 = (uint32_t *)((uint8_t *)pAddress + 7136608256);
        SR5 = (uint32_t *)((uint8_t *)pAddress + 7144996864);
        SR4 = (uint32_t *)((uint8_t *)pAddress + 7153385472);
        SR3 = (uint32_t *)((uint8_t *)pAddress + 7161774080);
        SR2 = (uint32_t *)((uint8_t *)pAddress + 7170162688);
        SR1 = (uint32_t *)((uint8_t *)pAddress + 7178551296);
        SR0 = (uint32_t *)((uint8_t *)pAddress + 7186939904);
        CTX = (uint32_t *)((uint8_t *)pAddress + 7195328512);
        SP = (uint16_t *)((uint8_t *)pAddress + 7203717120);
        PC = (uint32_t *)((uint8_t *)pAddress + 7207911424);
        GAS = (uint64_t *)((uint8_t *)pAddress + 7216300032);
        MAXMEM = (uint32_t *)((uint8_t *)pAddress + 7233077248);
        zkPC = (uint32_t *)((uint8_t *)pAddress + 7241465856);
        RR = (uint32_t *)((uint8_t *)pAddress + 7249854464);
        HASHPOS = (uint32_t *)((uint8_t *)pAddress + 7258243072);
        CONST7 = (FieldElement *)((uint8_t *)pAddress + 7266631680);
        CONST6 = (FieldElement *)((uint8_t *)pAddress + 7283408896);
        CONST5 = (FieldElement *)((uint8_t *)pAddress + 7300186112);
        CONST4 = (FieldElement *)((uint8_t *)pAddress + 7316963328);
        CONST3 = (FieldElement *)((uint8_t *)pAddress + 7333740544);
        CONST2 = (FieldElement *)((uint8_t *)pAddress + 7350517760);
        CONST1 = (FieldElement *)((uint8_t *)pAddress + 7367294976);
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 7384072192);
        FREE7 = (FieldElement *)((uint8_t *)pAddress + 7400849408);
        FREE6 = (FieldElement *)((uint8_t *)pAddress + 7417626624);
        FREE5 = (FieldElement *)((uint8_t *)pAddress + 7434403840);
        FREE4 = (FieldElement *)((uint8_t *)pAddress + 7451181056);
        FREE3 = (FieldElement *)((uint8_t *)pAddress + 7467958272);
        FREE2 = (FieldElement *)((uint8_t *)pAddress + 7484735488);
        FREE1 = (FieldElement *)((uint8_t *)pAddress + 7501512704);
        FREE0 = (FieldElement *)((uint8_t *)pAddress + 7518289920);
        inA = (FieldElement *)((uint8_t *)pAddress + 7535067136);
        inB = (FieldElement *)((uint8_t *)pAddress + 7551844352);
        inC = (FieldElement *)((uint8_t *)pAddress + 7568621568);
        inD = (FieldElement *)((uint8_t *)pAddress + 7585398784);
        inE = (FieldElement *)((uint8_t *)pAddress + 7602176000);
        inSR = (FieldElement *)((uint8_t *)pAddress + 7618953216);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 7635730432);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 7652507648);
        inSP = (FieldElement *)((uint8_t *)pAddress + 7669284864);
        inPC = (FieldElement *)((uint8_t *)pAddress + 7686062080);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 7702839296);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 7719616512);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 7736393728);
        inRR = (FieldElement *)((uint8_t *)pAddress + 7753170944);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 7769948160);
        setA = (uint8_t *)((uint8_t *)pAddress + 7786725376);
        setB = (uint8_t *)((uint8_t *)pAddress + 7788822528);
        setC = (uint8_t *)((uint8_t *)pAddress + 7790919680);
        setD = (uint8_t *)((uint8_t *)pAddress + 7793016832);
        setE = (uint8_t *)((uint8_t *)pAddress + 7795113984);
        setSR = (uint8_t *)((uint8_t *)pAddress + 7797211136);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 7799308288);
        setSP = (uint8_t *)((uint8_t *)pAddress + 7801405440);
        setPC = (uint8_t *)((uint8_t *)pAddress + 7803502592);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 7805599744);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 7807696896);
        JMP = (uint8_t *)((uint8_t *)pAddress + 7809794048);
        JMPN = (uint8_t *)((uint8_t *)pAddress + 7811891200);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 7813988352);
        setRR = (uint8_t *)((uint8_t *)pAddress + 7816085504);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 7818182656);
        offset = (uint32_t *)((uint8_t *)pAddress + 7820279808);
        incStack = (int32_t *)((uint8_t *)pAddress + 7828668416);
        incCode = (int32_t *)((uint8_t *)pAddress + 7837057024);
        isStack = (uint8_t *)((uint8_t *)pAddress + 7845445632);
        isCode = (uint8_t *)((uint8_t *)pAddress + 7847542784);
        isMem = (uint8_t *)((uint8_t *)pAddress + 7849639936);
        ind = (uint8_t *)((uint8_t *)pAddress + 7851737088);
        indRR = (uint8_t *)((uint8_t *)pAddress + 7853834240);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 7855931392);
        carry = (uint8_t *)((uint8_t *)pAddress + 7858028544);
        mOp = (uint8_t *)((uint8_t *)pAddress + 7860125696);
        mWR = (uint8_t *)((uint8_t *)pAddress + 7862222848);
        sWR = (uint8_t *)((uint8_t *)pAddress + 7864320000);
        sRD = (uint8_t *)((uint8_t *)pAddress + 7866417152);
        arith = (uint8_t *)((uint8_t *)pAddress + 7868514304);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 7870611456);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 7872708608);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 7874805760);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 7876902912);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 7879000064);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 7881097216);
        hashK = (uint8_t *)((uint8_t *)pAddress + 7883194368);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 7885291520);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 7887388672);
        hashP = (uint8_t *)((uint8_t *)pAddress + 7889485824);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 7891582976);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 7893680128);
        bin = (uint8_t *)((uint8_t *)pAddress + 7895777280);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 7897874432);
        assert = (uint8_t *)((uint8_t *)pAddress + 7899971584);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 7902068736);
        isNeg = (uint8_t *)((uint8_t *)pAddress + 7904165888);
        isMaxMem = (uint8_t *)((uint8_t *)pAddress + 7906263040);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 7908360192);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 7925137408);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 7941914624);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 7958691840);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 7975469056);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 7992246272);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 8009023488);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 8025800704);
    }

    MainCommitPols (void * pAddress, uint64_t degree)
    {
        A7 = (uint32_t *)((uint8_t *)pAddress + 0*degree);
        A6 = (uint32_t *)((uint8_t *)pAddress + 4*degree);
        A5 = (uint32_t *)((uint8_t *)pAddress + 8*degree);
        A4 = (uint32_t *)((uint8_t *)pAddress + 12*degree);
        A3 = (uint32_t *)((uint8_t *)pAddress + 16*degree);
        A2 = (uint32_t *)((uint8_t *)pAddress + 20*degree);
        A1 = (uint32_t *)((uint8_t *)pAddress + 24*degree);
        A0 = (FieldElement *)((uint8_t *)pAddress + 28*degree);
        B7 = (uint32_t *)((uint8_t *)pAddress + 36*degree);
        B6 = (uint32_t *)((uint8_t *)pAddress + 40*degree);
        B5 = (uint32_t *)((uint8_t *)pAddress + 44*degree);
        B4 = (uint32_t *)((uint8_t *)pAddress + 48*degree);
        B3 = (uint32_t *)((uint8_t *)pAddress + 52*degree);
        B2 = (uint32_t *)((uint8_t *)pAddress + 56*degree);
        B1 = (uint32_t *)((uint8_t *)pAddress + 60*degree);
        B0 = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        C7 = (uint32_t *)((uint8_t *)pAddress + 72*degree);
        C6 = (uint32_t *)((uint8_t *)pAddress + 76*degree);
        C5 = (uint32_t *)((uint8_t *)pAddress + 80*degree);
        C4 = (uint32_t *)((uint8_t *)pAddress + 84*degree);
        C3 = (uint32_t *)((uint8_t *)pAddress + 88*degree);
        C2 = (uint32_t *)((uint8_t *)pAddress + 92*degree);
        C1 = (uint32_t *)((uint8_t *)pAddress + 96*degree);
        C0 = (FieldElement *)((uint8_t *)pAddress + 100*degree);
        D7 = (uint32_t *)((uint8_t *)pAddress + 108*degree);
        D6 = (uint32_t *)((uint8_t *)pAddress + 112*degree);
        D5 = (uint32_t *)((uint8_t *)pAddress + 116*degree);
        D4 = (uint32_t *)((uint8_t *)pAddress + 120*degree);
        D3 = (uint32_t *)((uint8_t *)pAddress + 124*degree);
        D2 = (uint32_t *)((uint8_t *)pAddress + 128*degree);
        D1 = (uint32_t *)((uint8_t *)pAddress + 132*degree);
        D0 = (FieldElement *)((uint8_t *)pAddress + 136*degree);
        E7 = (uint32_t *)((uint8_t *)pAddress + 144*degree);
        E6 = (uint32_t *)((uint8_t *)pAddress + 148*degree);
        E5 = (uint32_t *)((uint8_t *)pAddress + 152*degree);
        E4 = (uint32_t *)((uint8_t *)pAddress + 156*degree);
        E3 = (uint32_t *)((uint8_t *)pAddress + 160*degree);
        E2 = (uint32_t *)((uint8_t *)pAddress + 164*degree);
        E1 = (uint32_t *)((uint8_t *)pAddress + 168*degree);
        E0 = (FieldElement *)((uint8_t *)pAddress + 172*degree);
        SR7 = (uint32_t *)((uint8_t *)pAddress + 180*degree);
        SR6 = (uint32_t *)((uint8_t *)pAddress + 184*degree);
        SR5 = (uint32_t *)((uint8_t *)pAddress + 188*degree);
        SR4 = (uint32_t *)((uint8_t *)pAddress + 192*degree);
        SR3 = (uint32_t *)((uint8_t *)pAddress + 196*degree);
        SR2 = (uint32_t *)((uint8_t *)pAddress + 200*degree);
        SR1 = (uint32_t *)((uint8_t *)pAddress + 204*degree);
        SR0 = (uint32_t *)((uint8_t *)pAddress + 208*degree);
        CTX = (uint32_t *)((uint8_t *)pAddress + 212*degree);
        SP = (uint16_t *)((uint8_t *)pAddress + 216*degree);
        PC = (uint32_t *)((uint8_t *)pAddress + 218*degree);
        GAS = (uint64_t *)((uint8_t *)pAddress + 222*degree);
        MAXMEM = (uint32_t *)((uint8_t *)pAddress + 230*degree);
        zkPC = (uint32_t *)((uint8_t *)pAddress + 234*degree);
        RR = (uint32_t *)((uint8_t *)pAddress + 238*degree);
        HASHPOS = (uint32_t *)((uint8_t *)pAddress + 242*degree);
        CONST7 = (FieldElement *)((uint8_t *)pAddress + 246*degree);
        CONST6 = (FieldElement *)((uint8_t *)pAddress + 254*degree);
        CONST5 = (FieldElement *)((uint8_t *)pAddress + 262*degree);
        CONST4 = (FieldElement *)((uint8_t *)pAddress + 270*degree);
        CONST3 = (FieldElement *)((uint8_t *)pAddress + 278*degree);
        CONST2 = (FieldElement *)((uint8_t *)pAddress + 286*degree);
        CONST1 = (FieldElement *)((uint8_t *)pAddress + 294*degree);
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 302*degree);
        FREE7 = (FieldElement *)((uint8_t *)pAddress + 310*degree);
        FREE6 = (FieldElement *)((uint8_t *)pAddress + 318*degree);
        FREE5 = (FieldElement *)((uint8_t *)pAddress + 326*degree);
        FREE4 = (FieldElement *)((uint8_t *)pAddress + 334*degree);
        FREE3 = (FieldElement *)((uint8_t *)pAddress + 342*degree);
        FREE2 = (FieldElement *)((uint8_t *)pAddress + 350*degree);
        FREE1 = (FieldElement *)((uint8_t *)pAddress + 358*degree);
        FREE0 = (FieldElement *)((uint8_t *)pAddress + 366*degree);
        inA = (FieldElement *)((uint8_t *)pAddress + 374*degree);
        inB = (FieldElement *)((uint8_t *)pAddress + 382*degree);
        inC = (FieldElement *)((uint8_t *)pAddress + 390*degree);
        inD = (FieldElement *)((uint8_t *)pAddress + 398*degree);
        inE = (FieldElement *)((uint8_t *)pAddress + 406*degree);
        inSR = (FieldElement *)((uint8_t *)pAddress + 414*degree);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 422*degree);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 430*degree);
        inSP = (FieldElement *)((uint8_t *)pAddress + 438*degree);
        inPC = (FieldElement *)((uint8_t *)pAddress + 446*degree);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 454*degree);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 462*degree);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 470*degree);
        inRR = (FieldElement *)((uint8_t *)pAddress + 478*degree);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 486*degree);
        setA = (uint8_t *)((uint8_t *)pAddress + 494*degree);
        setB = (uint8_t *)((uint8_t *)pAddress + 495*degree);
        setC = (uint8_t *)((uint8_t *)pAddress + 496*degree);
        setD = (uint8_t *)((uint8_t *)pAddress + 497*degree);
        setE = (uint8_t *)((uint8_t *)pAddress + 498*degree);
        setSR = (uint8_t *)((uint8_t *)pAddress + 499*degree);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 500*degree);
        setSP = (uint8_t *)((uint8_t *)pAddress + 501*degree);
        setPC = (uint8_t *)((uint8_t *)pAddress + 502*degree);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 503*degree);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 504*degree);
        JMP = (uint8_t *)((uint8_t *)pAddress + 505*degree);
        JMPN = (uint8_t *)((uint8_t *)pAddress + 506*degree);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 507*degree);
        setRR = (uint8_t *)((uint8_t *)pAddress + 508*degree);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 509*degree);
        offset = (uint32_t *)((uint8_t *)pAddress + 510*degree);
        incStack = (int32_t *)((uint8_t *)pAddress + 514*degree);
        incCode = (int32_t *)((uint8_t *)pAddress + 518*degree);
        isStack = (uint8_t *)((uint8_t *)pAddress + 522*degree);
        isCode = (uint8_t *)((uint8_t *)pAddress + 523*degree);
        isMem = (uint8_t *)((uint8_t *)pAddress + 524*degree);
        ind = (uint8_t *)((uint8_t *)pAddress + 525*degree);
        indRR = (uint8_t *)((uint8_t *)pAddress + 526*degree);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 527*degree);
        carry = (uint8_t *)((uint8_t *)pAddress + 528*degree);
        mOp = (uint8_t *)((uint8_t *)pAddress + 529*degree);
        mWR = (uint8_t *)((uint8_t *)pAddress + 530*degree);
        sWR = (uint8_t *)((uint8_t *)pAddress + 531*degree);
        sRD = (uint8_t *)((uint8_t *)pAddress + 532*degree);
        arith = (uint8_t *)((uint8_t *)pAddress + 533*degree);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 534*degree);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 535*degree);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 536*degree);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 537*degree);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 538*degree);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 539*degree);
        hashK = (uint8_t *)((uint8_t *)pAddress + 540*degree);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 541*degree);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 542*degree);
        hashP = (uint8_t *)((uint8_t *)pAddress + 543*degree);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 544*degree);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 545*degree);
        bin = (uint8_t *)((uint8_t *)pAddress + 546*degree);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 547*degree);
        assert = (uint8_t *)((uint8_t *)pAddress + 548*degree);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 549*degree);
        isNeg = (uint8_t *)((uint8_t *)pAddress + 550*degree);
        isMaxMem = (uint8_t *)((uint8_t *)pAddress + 551*degree);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 552*degree);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 560*degree);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 568*degree);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 576*degree);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 584*degree);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 592*degree);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 600*degree);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 608*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 616; }
};

class CommitPols
{
public:
    Byte4CommitPols Byte4;
    MemAlignCommitPols MemAlign;
    ArithCommitPols Arith;
    BinaryCommitPols Binary;
    PoseidonGCommitPols PoseidonG;
    PaddingPGCommitPols PaddingPG;
    StorageCommitPols Storage;
    NormGate9CommitPols NormGate9;
    KeccakFCommitPols KeccakF;
    Nine2OneCommitPols Nine2One;
    PaddingKKBitCommitPols PaddingKKBit;
    PaddingKKCommitPols PaddingKK;
    MemCommitPols Mem;
    MainCommitPols Main;

    CommitPols (void * pAddress) : Byte4(pAddress), MemAlign(pAddress), Arith(pAddress), Binary(pAddress), PoseidonG(pAddress), PaddingPG(pAddress), Storage(pAddress), NormGate9(pAddress), KeccakF(pAddress), Nine2One(pAddress), PaddingKKBit(pAddress), PaddingKK(pAddress), Mem(pAddress), Main(pAddress) {}

    static uint64_t size (void) { return 8042577920; }
};

#endif // COMMIT_POLS_HPP
