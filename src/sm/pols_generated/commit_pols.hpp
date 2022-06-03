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
    FieldElement * inM[2];
    FieldElement * inV;
    FieldElement * wr256;
    FieldElement * wr8;
    FieldElement * m0[8];
    FieldElement * m1[8];
    FieldElement * w0[8];
    FieldElement * w1[8];
    FieldElement * v[8];
    FieldElement * selM1;
    FieldElement * factorV[8];
    FieldElement * offset;

    MemAlignCommitPols (void * pAddress)
    {
        inM[0] = (FieldElement *)((uint8_t *)pAddress + 12582912);
        inM[1] = (FieldElement *)((uint8_t *)pAddress + 29360128);
        inV = (FieldElement *)((uint8_t *)pAddress + 46137344);
        wr256 = (FieldElement *)((uint8_t *)pAddress + 62914560);
        wr8 = (FieldElement *)((uint8_t *)pAddress + 79691776);
        m0[0] = (FieldElement *)((uint8_t *)pAddress + 96468992);
        m0[1] = (FieldElement *)((uint8_t *)pAddress + 113246208);
        m0[2] = (FieldElement *)((uint8_t *)pAddress + 130023424);
        m0[3] = (FieldElement *)((uint8_t *)pAddress + 146800640);
        m0[4] = (FieldElement *)((uint8_t *)pAddress + 163577856);
        m0[5] = (FieldElement *)((uint8_t *)pAddress + 180355072);
        m0[6] = (FieldElement *)((uint8_t *)pAddress + 197132288);
        m0[7] = (FieldElement *)((uint8_t *)pAddress + 213909504);
        m1[0] = (FieldElement *)((uint8_t *)pAddress + 230686720);
        m1[1] = (FieldElement *)((uint8_t *)pAddress + 247463936);
        m1[2] = (FieldElement *)((uint8_t *)pAddress + 264241152);
        m1[3] = (FieldElement *)((uint8_t *)pAddress + 281018368);
        m1[4] = (FieldElement *)((uint8_t *)pAddress + 297795584);
        m1[5] = (FieldElement *)((uint8_t *)pAddress + 314572800);
        m1[6] = (FieldElement *)((uint8_t *)pAddress + 331350016);
        m1[7] = (FieldElement *)((uint8_t *)pAddress + 348127232);
        w0[0] = (FieldElement *)((uint8_t *)pAddress + 364904448);
        w0[1] = (FieldElement *)((uint8_t *)pAddress + 381681664);
        w0[2] = (FieldElement *)((uint8_t *)pAddress + 398458880);
        w0[3] = (FieldElement *)((uint8_t *)pAddress + 415236096);
        w0[4] = (FieldElement *)((uint8_t *)pAddress + 432013312);
        w0[5] = (FieldElement *)((uint8_t *)pAddress + 448790528);
        w0[6] = (FieldElement *)((uint8_t *)pAddress + 465567744);
        w0[7] = (FieldElement *)((uint8_t *)pAddress + 482344960);
        w1[0] = (FieldElement *)((uint8_t *)pAddress + 499122176);
        w1[1] = (FieldElement *)((uint8_t *)pAddress + 515899392);
        w1[2] = (FieldElement *)((uint8_t *)pAddress + 532676608);
        w1[3] = (FieldElement *)((uint8_t *)pAddress + 549453824);
        w1[4] = (FieldElement *)((uint8_t *)pAddress + 566231040);
        w1[5] = (FieldElement *)((uint8_t *)pAddress + 583008256);
        w1[6] = (FieldElement *)((uint8_t *)pAddress + 599785472);
        w1[7] = (FieldElement *)((uint8_t *)pAddress + 616562688);
        v[0] = (FieldElement *)((uint8_t *)pAddress + 633339904);
        v[1] = (FieldElement *)((uint8_t *)pAddress + 650117120);
        v[2] = (FieldElement *)((uint8_t *)pAddress + 666894336);
        v[3] = (FieldElement *)((uint8_t *)pAddress + 683671552);
        v[4] = (FieldElement *)((uint8_t *)pAddress + 700448768);
        v[5] = (FieldElement *)((uint8_t *)pAddress + 717225984);
        v[6] = (FieldElement *)((uint8_t *)pAddress + 734003200);
        v[7] = (FieldElement *)((uint8_t *)pAddress + 750780416);
        selM1 = (FieldElement *)((uint8_t *)pAddress + 767557632);
        factorV[0] = (FieldElement *)((uint8_t *)pAddress + 784334848);
        factorV[1] = (FieldElement *)((uint8_t *)pAddress + 801112064);
        factorV[2] = (FieldElement *)((uint8_t *)pAddress + 817889280);
        factorV[3] = (FieldElement *)((uint8_t *)pAddress + 834666496);
        factorV[4] = (FieldElement *)((uint8_t *)pAddress + 851443712);
        factorV[5] = (FieldElement *)((uint8_t *)pAddress + 868220928);
        factorV[6] = (FieldElement *)((uint8_t *)pAddress + 884998144);
        factorV[7] = (FieldElement *)((uint8_t *)pAddress + 901775360);
        offset = (FieldElement *)((uint8_t *)pAddress + 918552576);
    }

    MemAlignCommitPols (void * pAddress, uint64_t degree)
    {
        inM[0] = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        inM[1] = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        inV = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        wr256 = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        wr8 = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        m0[0] = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        m0[1] = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        m0[2] = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        m0[3] = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        m0[4] = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        m0[5] = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        m0[6] = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        m0[7] = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        m1[0] = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        m1[1] = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        m1[2] = (FieldElement *)((uint8_t *)pAddress + 120*degree);
        m1[3] = (FieldElement *)((uint8_t *)pAddress + 128*degree);
        m1[4] = (FieldElement *)((uint8_t *)pAddress + 136*degree);
        m1[5] = (FieldElement *)((uint8_t *)pAddress + 144*degree);
        m1[6] = (FieldElement *)((uint8_t *)pAddress + 152*degree);
        m1[7] = (FieldElement *)((uint8_t *)pAddress + 160*degree);
        w0[0] = (FieldElement *)((uint8_t *)pAddress + 168*degree);
        w0[1] = (FieldElement *)((uint8_t *)pAddress + 176*degree);
        w0[2] = (FieldElement *)((uint8_t *)pAddress + 184*degree);
        w0[3] = (FieldElement *)((uint8_t *)pAddress + 192*degree);
        w0[4] = (FieldElement *)((uint8_t *)pAddress + 200*degree);
        w0[5] = (FieldElement *)((uint8_t *)pAddress + 208*degree);
        w0[6] = (FieldElement *)((uint8_t *)pAddress + 216*degree);
        w0[7] = (FieldElement *)((uint8_t *)pAddress + 224*degree);
        w1[0] = (FieldElement *)((uint8_t *)pAddress + 232*degree);
        w1[1] = (FieldElement *)((uint8_t *)pAddress + 240*degree);
        w1[2] = (FieldElement *)((uint8_t *)pAddress + 248*degree);
        w1[3] = (FieldElement *)((uint8_t *)pAddress + 256*degree);
        w1[4] = (FieldElement *)((uint8_t *)pAddress + 264*degree);
        w1[5] = (FieldElement *)((uint8_t *)pAddress + 272*degree);
        w1[6] = (FieldElement *)((uint8_t *)pAddress + 280*degree);
        w1[7] = (FieldElement *)((uint8_t *)pAddress + 288*degree);
        v[0] = (FieldElement *)((uint8_t *)pAddress + 296*degree);
        v[1] = (FieldElement *)((uint8_t *)pAddress + 304*degree);
        v[2] = (FieldElement *)((uint8_t *)pAddress + 312*degree);
        v[3] = (FieldElement *)((uint8_t *)pAddress + 320*degree);
        v[4] = (FieldElement *)((uint8_t *)pAddress + 328*degree);
        v[5] = (FieldElement *)((uint8_t *)pAddress + 336*degree);
        v[6] = (FieldElement *)((uint8_t *)pAddress + 344*degree);
        v[7] = (FieldElement *)((uint8_t *)pAddress + 352*degree);
        selM1 = (FieldElement *)((uint8_t *)pAddress + 360*degree);
        factorV[0] = (FieldElement *)((uint8_t *)pAddress + 368*degree);
        factorV[1] = (FieldElement *)((uint8_t *)pAddress + 376*degree);
        factorV[2] = (FieldElement *)((uint8_t *)pAddress + 384*degree);
        factorV[3] = (FieldElement *)((uint8_t *)pAddress + 392*degree);
        factorV[4] = (FieldElement *)((uint8_t *)pAddress + 400*degree);
        factorV[5] = (FieldElement *)((uint8_t *)pAddress + 408*degree);
        factorV[6] = (FieldElement *)((uint8_t *)pAddress + 416*degree);
        factorV[7] = (FieldElement *)((uint8_t *)pAddress + 424*degree);
        offset = (FieldElement *)((uint8_t *)pAddress + 432*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 440; }
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
        x1[0] = (FieldElement *)((uint8_t *)pAddress + 935329792);
        x1[1] = (FieldElement *)((uint8_t *)pAddress + 952107008);
        x1[2] = (FieldElement *)((uint8_t *)pAddress + 968884224);
        x1[3] = (FieldElement *)((uint8_t *)pAddress + 985661440);
        x1[4] = (FieldElement *)((uint8_t *)pAddress + 1002438656);
        x1[5] = (FieldElement *)((uint8_t *)pAddress + 1019215872);
        x1[6] = (FieldElement *)((uint8_t *)pAddress + 1035993088);
        x1[7] = (FieldElement *)((uint8_t *)pAddress + 1052770304);
        x1[8] = (FieldElement *)((uint8_t *)pAddress + 1069547520);
        x1[9] = (FieldElement *)((uint8_t *)pAddress + 1086324736);
        x1[10] = (FieldElement *)((uint8_t *)pAddress + 1103101952);
        x1[11] = (FieldElement *)((uint8_t *)pAddress + 1119879168);
        x1[12] = (FieldElement *)((uint8_t *)pAddress + 1136656384);
        x1[13] = (FieldElement *)((uint8_t *)pAddress + 1153433600);
        x1[14] = (FieldElement *)((uint8_t *)pAddress + 1170210816);
        x1[15] = (FieldElement *)((uint8_t *)pAddress + 1186988032);
        y1[0] = (FieldElement *)((uint8_t *)pAddress + 1203765248);
        y1[1] = (FieldElement *)((uint8_t *)pAddress + 1220542464);
        y1[2] = (FieldElement *)((uint8_t *)pAddress + 1237319680);
        y1[3] = (FieldElement *)((uint8_t *)pAddress + 1254096896);
        y1[4] = (FieldElement *)((uint8_t *)pAddress + 1270874112);
        y1[5] = (FieldElement *)((uint8_t *)pAddress + 1287651328);
        y1[6] = (FieldElement *)((uint8_t *)pAddress + 1304428544);
        y1[7] = (FieldElement *)((uint8_t *)pAddress + 1321205760);
        y1[8] = (FieldElement *)((uint8_t *)pAddress + 1337982976);
        y1[9] = (FieldElement *)((uint8_t *)pAddress + 1354760192);
        y1[10] = (FieldElement *)((uint8_t *)pAddress + 1371537408);
        y1[11] = (FieldElement *)((uint8_t *)pAddress + 1388314624);
        y1[12] = (FieldElement *)((uint8_t *)pAddress + 1405091840);
        y1[13] = (FieldElement *)((uint8_t *)pAddress + 1421869056);
        y1[14] = (FieldElement *)((uint8_t *)pAddress + 1438646272);
        y1[15] = (FieldElement *)((uint8_t *)pAddress + 1455423488);
        x2[0] = (FieldElement *)((uint8_t *)pAddress + 1472200704);
        x2[1] = (FieldElement *)((uint8_t *)pAddress + 1488977920);
        x2[2] = (FieldElement *)((uint8_t *)pAddress + 1505755136);
        x2[3] = (FieldElement *)((uint8_t *)pAddress + 1522532352);
        x2[4] = (FieldElement *)((uint8_t *)pAddress + 1539309568);
        x2[5] = (FieldElement *)((uint8_t *)pAddress + 1556086784);
        x2[6] = (FieldElement *)((uint8_t *)pAddress + 1572864000);
        x2[7] = (FieldElement *)((uint8_t *)pAddress + 1589641216);
        x2[8] = (FieldElement *)((uint8_t *)pAddress + 1606418432);
        x2[9] = (FieldElement *)((uint8_t *)pAddress + 1623195648);
        x2[10] = (FieldElement *)((uint8_t *)pAddress + 1639972864);
        x2[11] = (FieldElement *)((uint8_t *)pAddress + 1656750080);
        x2[12] = (FieldElement *)((uint8_t *)pAddress + 1673527296);
        x2[13] = (FieldElement *)((uint8_t *)pAddress + 1690304512);
        x2[14] = (FieldElement *)((uint8_t *)pAddress + 1707081728);
        x2[15] = (FieldElement *)((uint8_t *)pAddress + 1723858944);
        y2[0] = (FieldElement *)((uint8_t *)pAddress + 1740636160);
        y2[1] = (FieldElement *)((uint8_t *)pAddress + 1757413376);
        y2[2] = (FieldElement *)((uint8_t *)pAddress + 1774190592);
        y2[3] = (FieldElement *)((uint8_t *)pAddress + 1790967808);
        y2[4] = (FieldElement *)((uint8_t *)pAddress + 1807745024);
        y2[5] = (FieldElement *)((uint8_t *)pAddress + 1824522240);
        y2[6] = (FieldElement *)((uint8_t *)pAddress + 1841299456);
        y2[7] = (FieldElement *)((uint8_t *)pAddress + 1858076672);
        y2[8] = (FieldElement *)((uint8_t *)pAddress + 1874853888);
        y2[9] = (FieldElement *)((uint8_t *)pAddress + 1891631104);
        y2[10] = (FieldElement *)((uint8_t *)pAddress + 1908408320);
        y2[11] = (FieldElement *)((uint8_t *)pAddress + 1925185536);
        y2[12] = (FieldElement *)((uint8_t *)pAddress + 1941962752);
        y2[13] = (FieldElement *)((uint8_t *)pAddress + 1958739968);
        y2[14] = (FieldElement *)((uint8_t *)pAddress + 1975517184);
        y2[15] = (FieldElement *)((uint8_t *)pAddress + 1992294400);
        x3[0] = (FieldElement *)((uint8_t *)pAddress + 2009071616);
        x3[1] = (FieldElement *)((uint8_t *)pAddress + 2025848832);
        x3[2] = (FieldElement *)((uint8_t *)pAddress + 2042626048);
        x3[3] = (FieldElement *)((uint8_t *)pAddress + 2059403264);
        x3[4] = (FieldElement *)((uint8_t *)pAddress + 2076180480);
        x3[5] = (FieldElement *)((uint8_t *)pAddress + 2092957696);
        x3[6] = (FieldElement *)((uint8_t *)pAddress + 2109734912);
        x3[7] = (FieldElement *)((uint8_t *)pAddress + 2126512128);
        x3[8] = (FieldElement *)((uint8_t *)pAddress + 2143289344);
        x3[9] = (FieldElement *)((uint8_t *)pAddress + 2160066560);
        x3[10] = (FieldElement *)((uint8_t *)pAddress + 2176843776);
        x3[11] = (FieldElement *)((uint8_t *)pAddress + 2193620992);
        x3[12] = (FieldElement *)((uint8_t *)pAddress + 2210398208);
        x3[13] = (FieldElement *)((uint8_t *)pAddress + 2227175424);
        x3[14] = (FieldElement *)((uint8_t *)pAddress + 2243952640);
        x3[15] = (FieldElement *)((uint8_t *)pAddress + 2260729856);
        y3[0] = (FieldElement *)((uint8_t *)pAddress + 2277507072);
        y3[1] = (FieldElement *)((uint8_t *)pAddress + 2294284288);
        y3[2] = (FieldElement *)((uint8_t *)pAddress + 2311061504);
        y3[3] = (FieldElement *)((uint8_t *)pAddress + 2327838720);
        y3[4] = (FieldElement *)((uint8_t *)pAddress + 2344615936);
        y3[5] = (FieldElement *)((uint8_t *)pAddress + 2361393152);
        y3[6] = (FieldElement *)((uint8_t *)pAddress + 2378170368);
        y3[7] = (FieldElement *)((uint8_t *)pAddress + 2394947584);
        y3[8] = (FieldElement *)((uint8_t *)pAddress + 2411724800);
        y3[9] = (FieldElement *)((uint8_t *)pAddress + 2428502016);
        y3[10] = (FieldElement *)((uint8_t *)pAddress + 2445279232);
        y3[11] = (FieldElement *)((uint8_t *)pAddress + 2462056448);
        y3[12] = (FieldElement *)((uint8_t *)pAddress + 2478833664);
        y3[13] = (FieldElement *)((uint8_t *)pAddress + 2495610880);
        y3[14] = (FieldElement *)((uint8_t *)pAddress + 2512388096);
        y3[15] = (FieldElement *)((uint8_t *)pAddress + 2529165312);
        s[0] = (FieldElement *)((uint8_t *)pAddress + 2545942528);
        s[1] = (FieldElement *)((uint8_t *)pAddress + 2562719744);
        s[2] = (FieldElement *)((uint8_t *)pAddress + 2579496960);
        s[3] = (FieldElement *)((uint8_t *)pAddress + 2596274176);
        s[4] = (FieldElement *)((uint8_t *)pAddress + 2613051392);
        s[5] = (FieldElement *)((uint8_t *)pAddress + 2629828608);
        s[6] = (FieldElement *)((uint8_t *)pAddress + 2646605824);
        s[7] = (FieldElement *)((uint8_t *)pAddress + 2663383040);
        s[8] = (FieldElement *)((uint8_t *)pAddress + 2680160256);
        s[9] = (FieldElement *)((uint8_t *)pAddress + 2696937472);
        s[10] = (FieldElement *)((uint8_t *)pAddress + 2713714688);
        s[11] = (FieldElement *)((uint8_t *)pAddress + 2730491904);
        s[12] = (FieldElement *)((uint8_t *)pAddress + 2747269120);
        s[13] = (FieldElement *)((uint8_t *)pAddress + 2764046336);
        s[14] = (FieldElement *)((uint8_t *)pAddress + 2780823552);
        s[15] = (FieldElement *)((uint8_t *)pAddress + 2797600768);
        q0[0] = (FieldElement *)((uint8_t *)pAddress + 2814377984);
        q0[1] = (FieldElement *)((uint8_t *)pAddress + 2831155200);
        q0[2] = (FieldElement *)((uint8_t *)pAddress + 2847932416);
        q0[3] = (FieldElement *)((uint8_t *)pAddress + 2864709632);
        q0[4] = (FieldElement *)((uint8_t *)pAddress + 2881486848);
        q0[5] = (FieldElement *)((uint8_t *)pAddress + 2898264064);
        q0[6] = (FieldElement *)((uint8_t *)pAddress + 2915041280);
        q0[7] = (FieldElement *)((uint8_t *)pAddress + 2931818496);
        q0[8] = (FieldElement *)((uint8_t *)pAddress + 2948595712);
        q0[9] = (FieldElement *)((uint8_t *)pAddress + 2965372928);
        q0[10] = (FieldElement *)((uint8_t *)pAddress + 2982150144);
        q0[11] = (FieldElement *)((uint8_t *)pAddress + 2998927360);
        q0[12] = (FieldElement *)((uint8_t *)pAddress + 3015704576);
        q0[13] = (FieldElement *)((uint8_t *)pAddress + 3032481792);
        q0[14] = (FieldElement *)((uint8_t *)pAddress + 3049259008);
        q0[15] = (FieldElement *)((uint8_t *)pAddress + 3066036224);
        q1[0] = (FieldElement *)((uint8_t *)pAddress + 3082813440);
        q1[1] = (FieldElement *)((uint8_t *)pAddress + 3099590656);
        q1[2] = (FieldElement *)((uint8_t *)pAddress + 3116367872);
        q1[3] = (FieldElement *)((uint8_t *)pAddress + 3133145088);
        q1[4] = (FieldElement *)((uint8_t *)pAddress + 3149922304);
        q1[5] = (FieldElement *)((uint8_t *)pAddress + 3166699520);
        q1[6] = (FieldElement *)((uint8_t *)pAddress + 3183476736);
        q1[7] = (FieldElement *)((uint8_t *)pAddress + 3200253952);
        q1[8] = (FieldElement *)((uint8_t *)pAddress + 3217031168);
        q1[9] = (FieldElement *)((uint8_t *)pAddress + 3233808384);
        q1[10] = (FieldElement *)((uint8_t *)pAddress + 3250585600);
        q1[11] = (FieldElement *)((uint8_t *)pAddress + 3267362816);
        q1[12] = (FieldElement *)((uint8_t *)pAddress + 3284140032);
        q1[13] = (FieldElement *)((uint8_t *)pAddress + 3300917248);
        q1[14] = (FieldElement *)((uint8_t *)pAddress + 3317694464);
        q1[15] = (FieldElement *)((uint8_t *)pAddress + 3334471680);
        q2[0] = (FieldElement *)((uint8_t *)pAddress + 3351248896);
        q2[1] = (FieldElement *)((uint8_t *)pAddress + 3368026112);
        q2[2] = (FieldElement *)((uint8_t *)pAddress + 3384803328);
        q2[3] = (FieldElement *)((uint8_t *)pAddress + 3401580544);
        q2[4] = (FieldElement *)((uint8_t *)pAddress + 3418357760);
        q2[5] = (FieldElement *)((uint8_t *)pAddress + 3435134976);
        q2[6] = (FieldElement *)((uint8_t *)pAddress + 3451912192);
        q2[7] = (FieldElement *)((uint8_t *)pAddress + 3468689408);
        q2[8] = (FieldElement *)((uint8_t *)pAddress + 3485466624);
        q2[9] = (FieldElement *)((uint8_t *)pAddress + 3502243840);
        q2[10] = (FieldElement *)((uint8_t *)pAddress + 3519021056);
        q2[11] = (FieldElement *)((uint8_t *)pAddress + 3535798272);
        q2[12] = (FieldElement *)((uint8_t *)pAddress + 3552575488);
        q2[13] = (FieldElement *)((uint8_t *)pAddress + 3569352704);
        q2[14] = (FieldElement *)((uint8_t *)pAddress + 3586129920);
        q2[15] = (FieldElement *)((uint8_t *)pAddress + 3602907136);
        selEq[0] = (FieldElement *)((uint8_t *)pAddress + 3619684352);
        selEq[1] = (FieldElement *)((uint8_t *)pAddress + 3636461568);
        selEq[2] = (FieldElement *)((uint8_t *)pAddress + 3653238784);
        selEq[3] = (FieldElement *)((uint8_t *)pAddress + 3670016000);
        carryL[0] = (FieldElement *)((uint8_t *)pAddress + 3686793216);
        carryL[1] = (FieldElement *)((uint8_t *)pAddress + 3703570432);
        carryL[2] = (FieldElement *)((uint8_t *)pAddress + 3720347648);
        carryH[0] = (FieldElement *)((uint8_t *)pAddress + 3737124864);
        carryH[1] = (FieldElement *)((uint8_t *)pAddress + 3753902080);
        carryH[2] = (FieldElement *)((uint8_t *)pAddress + 3770679296);
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
        freeInA = (uint8_t *)((uint8_t *)pAddress + 3787456512);
        freeInB = (uint8_t *)((uint8_t *)pAddress + 3789553664);
        freeInC = (uint8_t *)((uint8_t *)pAddress + 3791650816);
        a0 = (uint32_t *)((uint8_t *)pAddress + 3793747968);
        a1 = (uint32_t *)((uint8_t *)pAddress + 3802136576);
        a2 = (uint32_t *)((uint8_t *)pAddress + 3810525184);
        a3 = (uint32_t *)((uint8_t *)pAddress + 3818913792);
        a4 = (uint32_t *)((uint8_t *)pAddress + 3827302400);
        a5 = (uint32_t *)((uint8_t *)pAddress + 3835691008);
        a6 = (uint32_t *)((uint8_t *)pAddress + 3844079616);
        a7 = (uint32_t *)((uint8_t *)pAddress + 3852468224);
        b0 = (uint32_t *)((uint8_t *)pAddress + 3860856832);
        b1 = (uint32_t *)((uint8_t *)pAddress + 3869245440);
        b2 = (uint32_t *)((uint8_t *)pAddress + 3877634048);
        b3 = (uint32_t *)((uint8_t *)pAddress + 3886022656);
        b4 = (uint32_t *)((uint8_t *)pAddress + 3894411264);
        b5 = (uint32_t *)((uint8_t *)pAddress + 3902799872);
        b6 = (uint32_t *)((uint8_t *)pAddress + 3911188480);
        b7 = (uint32_t *)((uint8_t *)pAddress + 3919577088);
        c0 = (uint32_t *)((uint8_t *)pAddress + 3927965696);
        c1 = (uint32_t *)((uint8_t *)pAddress + 3936354304);
        c2 = (uint32_t *)((uint8_t *)pAddress + 3944742912);
        c3 = (uint32_t *)((uint8_t *)pAddress + 3953131520);
        c4 = (uint32_t *)((uint8_t *)pAddress + 3961520128);
        c5 = (uint32_t *)((uint8_t *)pAddress + 3969908736);
        c6 = (uint32_t *)((uint8_t *)pAddress + 3978297344);
        c7 = (uint32_t *)((uint8_t *)pAddress + 3986685952);
        opcode = (uint8_t *)((uint8_t *)pAddress + 3995074560);
        cIn = (uint8_t *)((uint8_t *)pAddress + 3997171712);
        cOut = (uint8_t *)((uint8_t *)pAddress + 3999268864);
        lCout = (uint8_t *)((uint8_t *)pAddress + 4001366016);
        lOpcode = (uint8_t *)((uint8_t *)pAddress + 4003463168);
        last = (uint8_t *)((uint8_t *)pAddress + 4005560320);
        useCarry = (uint8_t *)((uint8_t *)pAddress + 4007657472);
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
        in0 = (FieldElement *)((uint8_t *)pAddress + 4009754624);
        in1 = (FieldElement *)((uint8_t *)pAddress + 4026531840);
        in2 = (FieldElement *)((uint8_t *)pAddress + 4043309056);
        in3 = (FieldElement *)((uint8_t *)pAddress + 4060086272);
        in4 = (FieldElement *)((uint8_t *)pAddress + 4076863488);
        in5 = (FieldElement *)((uint8_t *)pAddress + 4093640704);
        in6 = (FieldElement *)((uint8_t *)pAddress + 4110417920);
        in7 = (FieldElement *)((uint8_t *)pAddress + 4127195136);
        hashType = (FieldElement *)((uint8_t *)pAddress + 4143972352);
        cap1 = (FieldElement *)((uint8_t *)pAddress + 4160749568);
        cap2 = (FieldElement *)((uint8_t *)pAddress + 4177526784);
        cap3 = (FieldElement *)((uint8_t *)pAddress + 4194304000);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 4211081216);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 4227858432);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 4244635648);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 4261412864);
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
        acc[0] = (FieldElement *)((uint8_t *)pAddress + 4278190080);
        acc[1] = (FieldElement *)((uint8_t *)pAddress + 4294967296);
        acc[2] = (FieldElement *)((uint8_t *)pAddress + 4311744512);
        acc[3] = (FieldElement *)((uint8_t *)pAddress + 4328521728);
        acc[4] = (FieldElement *)((uint8_t *)pAddress + 4345298944);
        acc[5] = (FieldElement *)((uint8_t *)pAddress + 4362076160);
        acc[6] = (FieldElement *)((uint8_t *)pAddress + 4378853376);
        acc[7] = (FieldElement *)((uint8_t *)pAddress + 4395630592);
        freeIn = (FieldElement *)((uint8_t *)pAddress + 4412407808);
        addr = (FieldElement *)((uint8_t *)pAddress + 4429185024);
        rem = (FieldElement *)((uint8_t *)pAddress + 4445962240);
        remInv = (FieldElement *)((uint8_t *)pAddress + 4462739456);
        spare = (FieldElement *)((uint8_t *)pAddress + 4479516672);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 4496293888);
        curHash0 = (FieldElement *)((uint8_t *)pAddress + 4513071104);
        curHash1 = (FieldElement *)((uint8_t *)pAddress + 4529848320);
        curHash2 = (FieldElement *)((uint8_t *)pAddress + 4546625536);
        curHash3 = (FieldElement *)((uint8_t *)pAddress + 4563402752);
        prevHash0 = (FieldElement *)((uint8_t *)pAddress + 4580179968);
        prevHash1 = (FieldElement *)((uint8_t *)pAddress + 4596957184);
        prevHash2 = (FieldElement *)((uint8_t *)pAddress + 4613734400);
        prevHash3 = (FieldElement *)((uint8_t *)pAddress + 4630511616);
        len = (FieldElement *)((uint8_t *)pAddress + 4647288832);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 4664066048);
        crLen = (FieldElement *)((uint8_t *)pAddress + 4680843264);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 4697620480);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 4714397696);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 4731174912);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 4747952128);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 4764729344);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 4781506560);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 4798283776);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 4815060992);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 4831838208);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 4848615424);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 4865392640);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 4882169856);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 4898947072);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 4915724288);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 4932501504);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 4949278720);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 4966055936);
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
        free0 = (uint64_t *)((uint8_t *)pAddress + 4982833152);
        free1 = (uint64_t *)((uint8_t *)pAddress + 4999610368);
        free2 = (uint64_t *)((uint8_t *)pAddress + 5016387584);
        free3 = (uint64_t *)((uint8_t *)pAddress + 5033164800);
        hashLeft0 = (uint64_t *)((uint8_t *)pAddress + 5049942016);
        hashLeft1 = (uint64_t *)((uint8_t *)pAddress + 5066719232);
        hashLeft2 = (uint64_t *)((uint8_t *)pAddress + 5083496448);
        hashLeft3 = (uint64_t *)((uint8_t *)pAddress + 5100273664);
        hashRight0 = (uint64_t *)((uint8_t *)pAddress + 5117050880);
        hashRight1 = (uint64_t *)((uint8_t *)pAddress + 5133828096);
        hashRight2 = (uint64_t *)((uint8_t *)pAddress + 5150605312);
        hashRight3 = (uint64_t *)((uint8_t *)pAddress + 5167382528);
        oldRoot0 = (uint64_t *)((uint8_t *)pAddress + 5184159744);
        oldRoot1 = (uint64_t *)((uint8_t *)pAddress + 5200936960);
        oldRoot2 = (uint64_t *)((uint8_t *)pAddress + 5217714176);
        oldRoot3 = (uint64_t *)((uint8_t *)pAddress + 5234491392);
        newRoot0 = (uint64_t *)((uint8_t *)pAddress + 5251268608);
        newRoot1 = (uint64_t *)((uint8_t *)pAddress + 5268045824);
        newRoot2 = (uint64_t *)((uint8_t *)pAddress + 5284823040);
        newRoot3 = (uint64_t *)((uint8_t *)pAddress + 5301600256);
        valueLow0 = (uint64_t *)((uint8_t *)pAddress + 5318377472);
        valueLow1 = (uint64_t *)((uint8_t *)pAddress + 5335154688);
        valueLow2 = (uint64_t *)((uint8_t *)pAddress + 5351931904);
        valueLow3 = (uint64_t *)((uint8_t *)pAddress + 5368709120);
        valueHigh0 = (uint64_t *)((uint8_t *)pAddress + 5385486336);
        valueHigh1 = (uint64_t *)((uint8_t *)pAddress + 5402263552);
        valueHigh2 = (uint64_t *)((uint8_t *)pAddress + 5419040768);
        valueHigh3 = (uint64_t *)((uint8_t *)pAddress + 5435817984);
        siblingValueHash0 = (uint64_t *)((uint8_t *)pAddress + 5452595200);
        siblingValueHash1 = (uint64_t *)((uint8_t *)pAddress + 5469372416);
        siblingValueHash2 = (uint64_t *)((uint8_t *)pAddress + 5486149632);
        siblingValueHash3 = (uint64_t *)((uint8_t *)pAddress + 5502926848);
        rkey0 = (uint64_t *)((uint8_t *)pAddress + 5519704064);
        rkey1 = (uint64_t *)((uint8_t *)pAddress + 5536481280);
        rkey2 = (uint64_t *)((uint8_t *)pAddress + 5553258496);
        rkey3 = (uint64_t *)((uint8_t *)pAddress + 5570035712);
        siblingRkey0 = (uint64_t *)((uint8_t *)pAddress + 5586812928);
        siblingRkey1 = (uint64_t *)((uint8_t *)pAddress + 5603590144);
        siblingRkey2 = (uint64_t *)((uint8_t *)pAddress + 5620367360);
        siblingRkey3 = (uint64_t *)((uint8_t *)pAddress + 5637144576);
        rkeyBit = (uint64_t *)((uint8_t *)pAddress + 5653921792);
        level0 = (uint64_t *)((uint8_t *)pAddress + 5670699008);
        level1 = (uint64_t *)((uint8_t *)pAddress + 5687476224);
        level2 = (uint64_t *)((uint8_t *)pAddress + 5704253440);
        level3 = (uint64_t *)((uint8_t *)pAddress + 5721030656);
        pc = (uint64_t *)((uint8_t *)pAddress + 5737807872);
        selOldRoot = (uint8_t *)((uint8_t *)pAddress + 5754585088);
        selNewRoot = (uint8_t *)((uint8_t *)pAddress + 5756682240);
        selValueLow = (uint8_t *)((uint8_t *)pAddress + 5758779392);
        selValueHigh = (uint8_t *)((uint8_t *)pAddress + 5760876544);
        selSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 5762973696);
        selRkey = (uint8_t *)((uint8_t *)pAddress + 5765070848);
        selRkeyBit = (uint8_t *)((uint8_t *)pAddress + 5767168000);
        selSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 5769265152);
        selFree = (uint8_t *)((uint8_t *)pAddress + 5771362304);
        setHashLeft = (uint8_t *)((uint8_t *)pAddress + 5773459456);
        setHashRight = (uint8_t *)((uint8_t *)pAddress + 5775556608);
        setOldRoot = (uint8_t *)((uint8_t *)pAddress + 5777653760);
        setNewRoot = (uint8_t *)((uint8_t *)pAddress + 5779750912);
        setValueLow = (uint8_t *)((uint8_t *)pAddress + 5781848064);
        setValueHigh = (uint8_t *)((uint8_t *)pAddress + 5783945216);
        setSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 5786042368);
        setRkey = (uint8_t *)((uint8_t *)pAddress + 5788139520);
        setSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 5790236672);
        setRkeyBit = (uint8_t *)((uint8_t *)pAddress + 5792333824);
        setLevel = (uint8_t *)((uint8_t *)pAddress + 5794430976);
        iHash = (uint8_t *)((uint8_t *)pAddress + 5796528128);
        iHashType = (uint8_t *)((uint8_t *)pAddress + 5798625280);
        iLatchSet = (uint8_t *)((uint8_t *)pAddress + 5800722432);
        iLatchGet = (uint8_t *)((uint8_t *)pAddress + 5802819584);
        iClimbRkey = (uint8_t *)((uint8_t *)pAddress + 5804916736);
        iClimbSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 5807013888);
        iClimbSiblingRkeyN = (uint8_t *)((uint8_t *)pAddress + 5809111040);
        iRotateLevel = (uint8_t *)((uint8_t *)pAddress + 5811208192);
        iJmpz = (uint8_t *)((uint8_t *)pAddress + 5813305344);
        iJmp = (uint8_t *)((uint8_t *)pAddress + 5815402496);
        iConst0 = (uint64_t *)((uint8_t *)pAddress + 5817499648);
        iConst1 = (uint64_t *)((uint8_t *)pAddress + 5834276864);
        iConst2 = (uint64_t *)((uint8_t *)pAddress + 5851054080);
        iConst3 = (uint64_t *)((uint8_t *)pAddress + 5867831296);
        iAddress = (uint64_t *)((uint8_t *)pAddress + 5884608512);
        op0inv = (FieldElement *)((uint8_t *)pAddress + 5901385728);
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
        freeA = (FieldElement *)((uint8_t *)pAddress + 5918162944);
        freeB = (FieldElement *)((uint8_t *)pAddress + 5934940160);
        gateType = (FieldElement *)((uint8_t *)pAddress + 5951717376);
        freeANorm = (FieldElement *)((uint8_t *)pAddress + 5968494592);
        freeBNorm = (FieldElement *)((uint8_t *)pAddress + 5985271808);
        freeCNorm = (FieldElement *)((uint8_t *)pAddress + 6002049024);
        a = (FieldElement *)((uint8_t *)pAddress + 6018826240);
        b = (FieldElement *)((uint8_t *)pAddress + 6035603456);
        c = (FieldElement *)((uint8_t *)pAddress + 6052380672);
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
        a = (FieldElement *)((uint8_t *)pAddress + 6069157888);
        b = (FieldElement *)((uint8_t *)pAddress + 6085935104);
        c = (FieldElement *)((uint8_t *)pAddress + 6102712320);
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
        bit = (FieldElement *)((uint8_t *)pAddress + 6119489536);
        field9 = (FieldElement *)((uint8_t *)pAddress + 6136266752);
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
        rBit = (FieldElement *)((uint8_t *)pAddress + 6153043968);
        sOutBit = (FieldElement *)((uint8_t *)pAddress + 6169821184);
        r8 = (FieldElement *)((uint8_t *)pAddress + 6186598400);
        connected = (FieldElement *)((uint8_t *)pAddress + 6203375616);
        sOut0 = (FieldElement *)((uint8_t *)pAddress + 6220152832);
        sOut1 = (FieldElement *)((uint8_t *)pAddress + 6236930048);
        sOut2 = (FieldElement *)((uint8_t *)pAddress + 6253707264);
        sOut3 = (FieldElement *)((uint8_t *)pAddress + 6270484480);
        sOut4 = (FieldElement *)((uint8_t *)pAddress + 6287261696);
        sOut5 = (FieldElement *)((uint8_t *)pAddress + 6304038912);
        sOut6 = (FieldElement *)((uint8_t *)pAddress + 6320816128);
        sOut7 = (FieldElement *)((uint8_t *)pAddress + 6337593344);
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
        freeIn = (FieldElement *)((uint8_t *)pAddress + 6354370560);
        connected = (FieldElement *)((uint8_t *)pAddress + 6371147776);
        addr = (FieldElement *)((uint8_t *)pAddress + 6387924992);
        rem = (FieldElement *)((uint8_t *)pAddress + 6404702208);
        remInv = (FieldElement *)((uint8_t *)pAddress + 6421479424);
        spare = (FieldElement *)((uint8_t *)pAddress + 6438256640);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 6455033856);
        len = (FieldElement *)((uint8_t *)pAddress + 6471811072);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 6488588288);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 6505365504);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 6522142720);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 6538919936);
        hash4 = (FieldElement *)((uint8_t *)pAddress + 6555697152);
        hash5 = (FieldElement *)((uint8_t *)pAddress + 6572474368);
        hash6 = (FieldElement *)((uint8_t *)pAddress + 6589251584);
        hash7 = (FieldElement *)((uint8_t *)pAddress + 6606028800);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 6622806016);
        crLen = (FieldElement *)((uint8_t *)pAddress + 6639583232);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 6656360448);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 6673137664);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 6689914880);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 6706692096);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 6723469312);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 6740246528);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 6757023744);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 6773800960);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 6790578176);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 6807355392);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 6824132608);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 6840909824);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 6857687040);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 6874464256);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 6891241472);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 6908018688);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 6924795904);
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
        addr = (FieldElement *)((uint8_t *)pAddress + 6941573120);
        step = (FieldElement *)((uint8_t *)pAddress + 6958350336);
        mOp = (FieldElement *)((uint8_t *)pAddress + 6975127552);
        mWr = (FieldElement *)((uint8_t *)pAddress + 6991904768);
        val[0] = (FieldElement *)((uint8_t *)pAddress + 7008681984);
        val[1] = (FieldElement *)((uint8_t *)pAddress + 7025459200);
        val[2] = (FieldElement *)((uint8_t *)pAddress + 7042236416);
        val[3] = (FieldElement *)((uint8_t *)pAddress + 7059013632);
        val[4] = (FieldElement *)((uint8_t *)pAddress + 7075790848);
        val[5] = (FieldElement *)((uint8_t *)pAddress + 7092568064);
        val[6] = (FieldElement *)((uint8_t *)pAddress + 7109345280);
        val[7] = (FieldElement *)((uint8_t *)pAddress + 7126122496);
        lastAccess = (FieldElement *)((uint8_t *)pAddress + 7142899712);
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
    uint8_t * memAlignWR8;
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
        A7 = (uint32_t *)((uint8_t *)pAddress + 7159676928);
        A6 = (uint32_t *)((uint8_t *)pAddress + 7168065536);
        A5 = (uint32_t *)((uint8_t *)pAddress + 7176454144);
        A4 = (uint32_t *)((uint8_t *)pAddress + 7184842752);
        A3 = (uint32_t *)((uint8_t *)pAddress + 7193231360);
        A2 = (uint32_t *)((uint8_t *)pAddress + 7201619968);
        A1 = (uint32_t *)((uint8_t *)pAddress + 7210008576);
        A0 = (FieldElement *)((uint8_t *)pAddress + 7218397184);
        B7 = (uint32_t *)((uint8_t *)pAddress + 7235174400);
        B6 = (uint32_t *)((uint8_t *)pAddress + 7243563008);
        B5 = (uint32_t *)((uint8_t *)pAddress + 7251951616);
        B4 = (uint32_t *)((uint8_t *)pAddress + 7260340224);
        B3 = (uint32_t *)((uint8_t *)pAddress + 7268728832);
        B2 = (uint32_t *)((uint8_t *)pAddress + 7277117440);
        B1 = (uint32_t *)((uint8_t *)pAddress + 7285506048);
        B0 = (FieldElement *)((uint8_t *)pAddress + 7293894656);
        C7 = (uint32_t *)((uint8_t *)pAddress + 7310671872);
        C6 = (uint32_t *)((uint8_t *)pAddress + 7319060480);
        C5 = (uint32_t *)((uint8_t *)pAddress + 7327449088);
        C4 = (uint32_t *)((uint8_t *)pAddress + 7335837696);
        C3 = (uint32_t *)((uint8_t *)pAddress + 7344226304);
        C2 = (uint32_t *)((uint8_t *)pAddress + 7352614912);
        C1 = (uint32_t *)((uint8_t *)pAddress + 7361003520);
        C0 = (FieldElement *)((uint8_t *)pAddress + 7369392128);
        D7 = (uint32_t *)((uint8_t *)pAddress + 7386169344);
        D6 = (uint32_t *)((uint8_t *)pAddress + 7394557952);
        D5 = (uint32_t *)((uint8_t *)pAddress + 7402946560);
        D4 = (uint32_t *)((uint8_t *)pAddress + 7411335168);
        D3 = (uint32_t *)((uint8_t *)pAddress + 7419723776);
        D2 = (uint32_t *)((uint8_t *)pAddress + 7428112384);
        D1 = (uint32_t *)((uint8_t *)pAddress + 7436500992);
        D0 = (FieldElement *)((uint8_t *)pAddress + 7444889600);
        E7 = (uint32_t *)((uint8_t *)pAddress + 7461666816);
        E6 = (uint32_t *)((uint8_t *)pAddress + 7470055424);
        E5 = (uint32_t *)((uint8_t *)pAddress + 7478444032);
        E4 = (uint32_t *)((uint8_t *)pAddress + 7486832640);
        E3 = (uint32_t *)((uint8_t *)pAddress + 7495221248);
        E2 = (uint32_t *)((uint8_t *)pAddress + 7503609856);
        E1 = (uint32_t *)((uint8_t *)pAddress + 7511998464);
        E0 = (FieldElement *)((uint8_t *)pAddress + 7520387072);
        SR7 = (uint32_t *)((uint8_t *)pAddress + 7537164288);
        SR6 = (uint32_t *)((uint8_t *)pAddress + 7545552896);
        SR5 = (uint32_t *)((uint8_t *)pAddress + 7553941504);
        SR4 = (uint32_t *)((uint8_t *)pAddress + 7562330112);
        SR3 = (uint32_t *)((uint8_t *)pAddress + 7570718720);
        SR2 = (uint32_t *)((uint8_t *)pAddress + 7579107328);
        SR1 = (uint32_t *)((uint8_t *)pAddress + 7587495936);
        SR0 = (uint32_t *)((uint8_t *)pAddress + 7595884544);
        CTX = (uint32_t *)((uint8_t *)pAddress + 7604273152);
        SP = (uint16_t *)((uint8_t *)pAddress + 7612661760);
        PC = (uint32_t *)((uint8_t *)pAddress + 7616856064);
        GAS = (uint64_t *)((uint8_t *)pAddress + 7625244672);
        MAXMEM = (uint32_t *)((uint8_t *)pAddress + 7642021888);
        zkPC = (uint32_t *)((uint8_t *)pAddress + 7650410496);
        RR = (uint32_t *)((uint8_t *)pAddress + 7658799104);
        HASHPOS = (uint32_t *)((uint8_t *)pAddress + 7667187712);
        CONST7 = (FieldElement *)((uint8_t *)pAddress + 7675576320);
        CONST6 = (FieldElement *)((uint8_t *)pAddress + 7692353536);
        CONST5 = (FieldElement *)((uint8_t *)pAddress + 7709130752);
        CONST4 = (FieldElement *)((uint8_t *)pAddress + 7725907968);
        CONST3 = (FieldElement *)((uint8_t *)pAddress + 7742685184);
        CONST2 = (FieldElement *)((uint8_t *)pAddress + 7759462400);
        CONST1 = (FieldElement *)((uint8_t *)pAddress + 7776239616);
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 7793016832);
        FREE7 = (FieldElement *)((uint8_t *)pAddress + 7809794048);
        FREE6 = (FieldElement *)((uint8_t *)pAddress + 7826571264);
        FREE5 = (FieldElement *)((uint8_t *)pAddress + 7843348480);
        FREE4 = (FieldElement *)((uint8_t *)pAddress + 7860125696);
        FREE3 = (FieldElement *)((uint8_t *)pAddress + 7876902912);
        FREE2 = (FieldElement *)((uint8_t *)pAddress + 7893680128);
        FREE1 = (FieldElement *)((uint8_t *)pAddress + 7910457344);
        FREE0 = (FieldElement *)((uint8_t *)pAddress + 7927234560);
        inA = (FieldElement *)((uint8_t *)pAddress + 7944011776);
        inB = (FieldElement *)((uint8_t *)pAddress + 7960788992);
        inC = (FieldElement *)((uint8_t *)pAddress + 7977566208);
        inD = (FieldElement *)((uint8_t *)pAddress + 7994343424);
        inE = (FieldElement *)((uint8_t *)pAddress + 8011120640);
        inSR = (FieldElement *)((uint8_t *)pAddress + 8027897856);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 8044675072);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 8061452288);
        inSP = (FieldElement *)((uint8_t *)pAddress + 8078229504);
        inPC = (FieldElement *)((uint8_t *)pAddress + 8095006720);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 8111783936);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 8128561152);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 8145338368);
        inRR = (FieldElement *)((uint8_t *)pAddress + 8162115584);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 8178892800);
        setA = (uint8_t *)((uint8_t *)pAddress + 8195670016);
        setB = (uint8_t *)((uint8_t *)pAddress + 8197767168);
        setC = (uint8_t *)((uint8_t *)pAddress + 8199864320);
        setD = (uint8_t *)((uint8_t *)pAddress + 8201961472);
        setE = (uint8_t *)((uint8_t *)pAddress + 8204058624);
        setSR = (uint8_t *)((uint8_t *)pAddress + 8206155776);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 8208252928);
        setSP = (uint8_t *)((uint8_t *)pAddress + 8210350080);
        setPC = (uint8_t *)((uint8_t *)pAddress + 8212447232);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 8214544384);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 8216641536);
        JMP = (uint8_t *)((uint8_t *)pAddress + 8218738688);
        JMPN = (uint8_t *)((uint8_t *)pAddress + 8220835840);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 8222932992);
        setRR = (uint8_t *)((uint8_t *)pAddress + 8225030144);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 8227127296);
        offset = (uint32_t *)((uint8_t *)pAddress + 8229224448);
        incStack = (int32_t *)((uint8_t *)pAddress + 8237613056);
        incCode = (int32_t *)((uint8_t *)pAddress + 8246001664);
        isStack = (uint8_t *)((uint8_t *)pAddress + 8254390272);
        isCode = (uint8_t *)((uint8_t *)pAddress + 8256487424);
        isMem = (uint8_t *)((uint8_t *)pAddress + 8258584576);
        ind = (uint8_t *)((uint8_t *)pAddress + 8260681728);
        indRR = (uint8_t *)((uint8_t *)pAddress + 8262778880);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 8264876032);
        carry = (uint8_t *)((uint8_t *)pAddress + 8266973184);
        mOp = (uint8_t *)((uint8_t *)pAddress + 8269070336);
        mWR = (uint8_t *)((uint8_t *)pAddress + 8271167488);
        sWR = (uint8_t *)((uint8_t *)pAddress + 8273264640);
        sRD = (uint8_t *)((uint8_t *)pAddress + 8275361792);
        arith = (uint8_t *)((uint8_t *)pAddress + 8277458944);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 8279556096);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 8281653248);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 8283750400);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 8285847552);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 8287944704);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 8290041856);
        memAlignWR8 = (uint8_t *)((uint8_t *)pAddress + 8292139008);
        hashK = (uint8_t *)((uint8_t *)pAddress + 8294236160);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 8296333312);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 8298430464);
        hashP = (uint8_t *)((uint8_t *)pAddress + 8300527616);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 8302624768);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 8304721920);
        bin = (uint8_t *)((uint8_t *)pAddress + 8306819072);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 8308916224);
        assert = (uint8_t *)((uint8_t *)pAddress + 8311013376);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 8313110528);
        isNeg = (uint8_t *)((uint8_t *)pAddress + 8315207680);
        isMaxMem = (uint8_t *)((uint8_t *)pAddress + 8317304832);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 8319401984);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 8336179200);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 8352956416);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 8369733632);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 8386510848);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 8403288064);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 8420065280);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 8436842496);
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
        memAlignWR8 = (uint8_t *)((uint8_t *)pAddress + 540*degree);
        hashK = (uint8_t *)((uint8_t *)pAddress + 541*degree);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 542*degree);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 543*degree);
        hashP = (uint8_t *)((uint8_t *)pAddress + 544*degree);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 545*degree);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 546*degree);
        bin = (uint8_t *)((uint8_t *)pAddress + 547*degree);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 548*degree);
        assert = (uint8_t *)((uint8_t *)pAddress + 549*degree);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 550*degree);
        isNeg = (uint8_t *)((uint8_t *)pAddress + 551*degree);
        isMaxMem = (uint8_t *)((uint8_t *)pAddress + 552*degree);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 553*degree);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 561*degree);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 569*degree);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 577*degree);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 585*degree);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 593*degree);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 601*degree);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 609*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 617; }
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

    static uint64_t size (void) { return 8453619712; }
};

#endif // COMMIT_POLS_HPP
