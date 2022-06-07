#ifndef COMMIT_POLS_HPP
#define COMMIT_POLS_HPP

#include <cstdint>
#include "ff/ff.hpp"

class GeneratedPol
{
private:
    FieldElement * pData;
public:
    GeneratedPol() : pData(NULL) {};
    FieldElement & operator[](int i) { return pData[i*619]; };
    FieldElement * operator=(FieldElement * pAddress) { pData = pAddress; return pData; };
};

class Byte4CommitPols
{
public:
    GeneratedPol freeIN;
    GeneratedPol out;

    Byte4CommitPols (void * pAddress)
    {
        freeIN = (FieldElement *)((uint8_t *)pAddress + 0);
        out = (FieldElement *)((uint8_t *)pAddress + 8);
    }

    Byte4CommitPols (void * pAddress, uint64_t degree)
    {
        freeIN = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        out = (FieldElement *)((uint8_t *)pAddress + 8*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 16; }
};

class MemAlignCommitPols
{
public:
    GeneratedPol inM[2];
    GeneratedPol inV;
    GeneratedPol wr256;
    GeneratedPol wr8;
    GeneratedPol m0[8];
    GeneratedPol m1[8];
    GeneratedPol w0[8];
    GeneratedPol w1[8];
    GeneratedPol v[8];
    GeneratedPol selM1;
    GeneratedPol factorV[8];
    GeneratedPol offset;

    MemAlignCommitPols (void * pAddress)
    {
        inM[0] = (FieldElement *)((uint8_t *)pAddress + 16);
        inM[1] = (FieldElement *)((uint8_t *)pAddress + 24);
        inV = (FieldElement *)((uint8_t *)pAddress + 32);
        wr256 = (FieldElement *)((uint8_t *)pAddress + 40);
        wr8 = (FieldElement *)((uint8_t *)pAddress + 48);
        m0[0] = (FieldElement *)((uint8_t *)pAddress + 56);
        m0[1] = (FieldElement *)((uint8_t *)pAddress + 64);
        m0[2] = (FieldElement *)((uint8_t *)pAddress + 72);
        m0[3] = (FieldElement *)((uint8_t *)pAddress + 80);
        m0[4] = (FieldElement *)((uint8_t *)pAddress + 88);
        m0[5] = (FieldElement *)((uint8_t *)pAddress + 96);
        m0[6] = (FieldElement *)((uint8_t *)pAddress + 104);
        m0[7] = (FieldElement *)((uint8_t *)pAddress + 112);
        m1[0] = (FieldElement *)((uint8_t *)pAddress + 120);
        m1[1] = (FieldElement *)((uint8_t *)pAddress + 128);
        m1[2] = (FieldElement *)((uint8_t *)pAddress + 136);
        m1[3] = (FieldElement *)((uint8_t *)pAddress + 144);
        m1[4] = (FieldElement *)((uint8_t *)pAddress + 152);
        m1[5] = (FieldElement *)((uint8_t *)pAddress + 160);
        m1[6] = (FieldElement *)((uint8_t *)pAddress + 168);
        m1[7] = (FieldElement *)((uint8_t *)pAddress + 176);
        w0[0] = (FieldElement *)((uint8_t *)pAddress + 184);
        w0[1] = (FieldElement *)((uint8_t *)pAddress + 192);
        w0[2] = (FieldElement *)((uint8_t *)pAddress + 200);
        w0[3] = (FieldElement *)((uint8_t *)pAddress + 208);
        w0[4] = (FieldElement *)((uint8_t *)pAddress + 216);
        w0[5] = (FieldElement *)((uint8_t *)pAddress + 224);
        w0[6] = (FieldElement *)((uint8_t *)pAddress + 232);
        w0[7] = (FieldElement *)((uint8_t *)pAddress + 240);
        w1[0] = (FieldElement *)((uint8_t *)pAddress + 248);
        w1[1] = (FieldElement *)((uint8_t *)pAddress + 256);
        w1[2] = (FieldElement *)((uint8_t *)pAddress + 264);
        w1[3] = (FieldElement *)((uint8_t *)pAddress + 272);
        w1[4] = (FieldElement *)((uint8_t *)pAddress + 280);
        w1[5] = (FieldElement *)((uint8_t *)pAddress + 288);
        w1[6] = (FieldElement *)((uint8_t *)pAddress + 296);
        w1[7] = (FieldElement *)((uint8_t *)pAddress + 304);
        v[0] = (FieldElement *)((uint8_t *)pAddress + 312);
        v[1] = (FieldElement *)((uint8_t *)pAddress + 320);
        v[2] = (FieldElement *)((uint8_t *)pAddress + 328);
        v[3] = (FieldElement *)((uint8_t *)pAddress + 336);
        v[4] = (FieldElement *)((uint8_t *)pAddress + 344);
        v[5] = (FieldElement *)((uint8_t *)pAddress + 352);
        v[6] = (FieldElement *)((uint8_t *)pAddress + 360);
        v[7] = (FieldElement *)((uint8_t *)pAddress + 368);
        selM1 = (FieldElement *)((uint8_t *)pAddress + 376);
        factorV[0] = (FieldElement *)((uint8_t *)pAddress + 384);
        factorV[1] = (FieldElement *)((uint8_t *)pAddress + 392);
        factorV[2] = (FieldElement *)((uint8_t *)pAddress + 400);
        factorV[3] = (FieldElement *)((uint8_t *)pAddress + 408);
        factorV[4] = (FieldElement *)((uint8_t *)pAddress + 416);
        factorV[5] = (FieldElement *)((uint8_t *)pAddress + 424);
        factorV[6] = (FieldElement *)((uint8_t *)pAddress + 432);
        factorV[7] = (FieldElement *)((uint8_t *)pAddress + 440);
        offset = (FieldElement *)((uint8_t *)pAddress + 448);
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
    GeneratedPol x1[16];
    GeneratedPol y1[16];
    GeneratedPol x2[16];
    GeneratedPol y2[16];
    GeneratedPol x3[16];
    GeneratedPol y3[16];
    GeneratedPol s[16];
    GeneratedPol q0[16];
    GeneratedPol q1[16];
    GeneratedPol q2[16];
    GeneratedPol selEq[4];
    GeneratedPol carryL[3];
    GeneratedPol carryH[3];

    ArithCommitPols (void * pAddress)
    {
        x1[0] = (FieldElement *)((uint8_t *)pAddress + 456);
        x1[1] = (FieldElement *)((uint8_t *)pAddress + 464);
        x1[2] = (FieldElement *)((uint8_t *)pAddress + 472);
        x1[3] = (FieldElement *)((uint8_t *)pAddress + 480);
        x1[4] = (FieldElement *)((uint8_t *)pAddress + 488);
        x1[5] = (FieldElement *)((uint8_t *)pAddress + 496);
        x1[6] = (FieldElement *)((uint8_t *)pAddress + 504);
        x1[7] = (FieldElement *)((uint8_t *)pAddress + 512);
        x1[8] = (FieldElement *)((uint8_t *)pAddress + 520);
        x1[9] = (FieldElement *)((uint8_t *)pAddress + 528);
        x1[10] = (FieldElement *)((uint8_t *)pAddress + 536);
        x1[11] = (FieldElement *)((uint8_t *)pAddress + 544);
        x1[12] = (FieldElement *)((uint8_t *)pAddress + 552);
        x1[13] = (FieldElement *)((uint8_t *)pAddress + 560);
        x1[14] = (FieldElement *)((uint8_t *)pAddress + 568);
        x1[15] = (FieldElement *)((uint8_t *)pAddress + 576);
        y1[0] = (FieldElement *)((uint8_t *)pAddress + 584);
        y1[1] = (FieldElement *)((uint8_t *)pAddress + 592);
        y1[2] = (FieldElement *)((uint8_t *)pAddress + 600);
        y1[3] = (FieldElement *)((uint8_t *)pAddress + 608);
        y1[4] = (FieldElement *)((uint8_t *)pAddress + 616);
        y1[5] = (FieldElement *)((uint8_t *)pAddress + 624);
        y1[6] = (FieldElement *)((uint8_t *)pAddress + 632);
        y1[7] = (FieldElement *)((uint8_t *)pAddress + 640);
        y1[8] = (FieldElement *)((uint8_t *)pAddress + 648);
        y1[9] = (FieldElement *)((uint8_t *)pAddress + 656);
        y1[10] = (FieldElement *)((uint8_t *)pAddress + 664);
        y1[11] = (FieldElement *)((uint8_t *)pAddress + 672);
        y1[12] = (FieldElement *)((uint8_t *)pAddress + 680);
        y1[13] = (FieldElement *)((uint8_t *)pAddress + 688);
        y1[14] = (FieldElement *)((uint8_t *)pAddress + 696);
        y1[15] = (FieldElement *)((uint8_t *)pAddress + 704);
        x2[0] = (FieldElement *)((uint8_t *)pAddress + 712);
        x2[1] = (FieldElement *)((uint8_t *)pAddress + 720);
        x2[2] = (FieldElement *)((uint8_t *)pAddress + 728);
        x2[3] = (FieldElement *)((uint8_t *)pAddress + 736);
        x2[4] = (FieldElement *)((uint8_t *)pAddress + 744);
        x2[5] = (FieldElement *)((uint8_t *)pAddress + 752);
        x2[6] = (FieldElement *)((uint8_t *)pAddress + 760);
        x2[7] = (FieldElement *)((uint8_t *)pAddress + 768);
        x2[8] = (FieldElement *)((uint8_t *)pAddress + 776);
        x2[9] = (FieldElement *)((uint8_t *)pAddress + 784);
        x2[10] = (FieldElement *)((uint8_t *)pAddress + 792);
        x2[11] = (FieldElement *)((uint8_t *)pAddress + 800);
        x2[12] = (FieldElement *)((uint8_t *)pAddress + 808);
        x2[13] = (FieldElement *)((uint8_t *)pAddress + 816);
        x2[14] = (FieldElement *)((uint8_t *)pAddress + 824);
        x2[15] = (FieldElement *)((uint8_t *)pAddress + 832);
        y2[0] = (FieldElement *)((uint8_t *)pAddress + 840);
        y2[1] = (FieldElement *)((uint8_t *)pAddress + 848);
        y2[2] = (FieldElement *)((uint8_t *)pAddress + 856);
        y2[3] = (FieldElement *)((uint8_t *)pAddress + 864);
        y2[4] = (FieldElement *)((uint8_t *)pAddress + 872);
        y2[5] = (FieldElement *)((uint8_t *)pAddress + 880);
        y2[6] = (FieldElement *)((uint8_t *)pAddress + 888);
        y2[7] = (FieldElement *)((uint8_t *)pAddress + 896);
        y2[8] = (FieldElement *)((uint8_t *)pAddress + 904);
        y2[9] = (FieldElement *)((uint8_t *)pAddress + 912);
        y2[10] = (FieldElement *)((uint8_t *)pAddress + 920);
        y2[11] = (FieldElement *)((uint8_t *)pAddress + 928);
        y2[12] = (FieldElement *)((uint8_t *)pAddress + 936);
        y2[13] = (FieldElement *)((uint8_t *)pAddress + 944);
        y2[14] = (FieldElement *)((uint8_t *)pAddress + 952);
        y2[15] = (FieldElement *)((uint8_t *)pAddress + 960);
        x3[0] = (FieldElement *)((uint8_t *)pAddress + 968);
        x3[1] = (FieldElement *)((uint8_t *)pAddress + 976);
        x3[2] = (FieldElement *)((uint8_t *)pAddress + 984);
        x3[3] = (FieldElement *)((uint8_t *)pAddress + 992);
        x3[4] = (FieldElement *)((uint8_t *)pAddress + 1000);
        x3[5] = (FieldElement *)((uint8_t *)pAddress + 1008);
        x3[6] = (FieldElement *)((uint8_t *)pAddress + 1016);
        x3[7] = (FieldElement *)((uint8_t *)pAddress + 1024);
        x3[8] = (FieldElement *)((uint8_t *)pAddress + 1032);
        x3[9] = (FieldElement *)((uint8_t *)pAddress + 1040);
        x3[10] = (FieldElement *)((uint8_t *)pAddress + 1048);
        x3[11] = (FieldElement *)((uint8_t *)pAddress + 1056);
        x3[12] = (FieldElement *)((uint8_t *)pAddress + 1064);
        x3[13] = (FieldElement *)((uint8_t *)pAddress + 1072);
        x3[14] = (FieldElement *)((uint8_t *)pAddress + 1080);
        x3[15] = (FieldElement *)((uint8_t *)pAddress + 1088);
        y3[0] = (FieldElement *)((uint8_t *)pAddress + 1096);
        y3[1] = (FieldElement *)((uint8_t *)pAddress + 1104);
        y3[2] = (FieldElement *)((uint8_t *)pAddress + 1112);
        y3[3] = (FieldElement *)((uint8_t *)pAddress + 1120);
        y3[4] = (FieldElement *)((uint8_t *)pAddress + 1128);
        y3[5] = (FieldElement *)((uint8_t *)pAddress + 1136);
        y3[6] = (FieldElement *)((uint8_t *)pAddress + 1144);
        y3[7] = (FieldElement *)((uint8_t *)pAddress + 1152);
        y3[8] = (FieldElement *)((uint8_t *)pAddress + 1160);
        y3[9] = (FieldElement *)((uint8_t *)pAddress + 1168);
        y3[10] = (FieldElement *)((uint8_t *)pAddress + 1176);
        y3[11] = (FieldElement *)((uint8_t *)pAddress + 1184);
        y3[12] = (FieldElement *)((uint8_t *)pAddress + 1192);
        y3[13] = (FieldElement *)((uint8_t *)pAddress + 1200);
        y3[14] = (FieldElement *)((uint8_t *)pAddress + 1208);
        y3[15] = (FieldElement *)((uint8_t *)pAddress + 1216);
        s[0] = (FieldElement *)((uint8_t *)pAddress + 1224);
        s[1] = (FieldElement *)((uint8_t *)pAddress + 1232);
        s[2] = (FieldElement *)((uint8_t *)pAddress + 1240);
        s[3] = (FieldElement *)((uint8_t *)pAddress + 1248);
        s[4] = (FieldElement *)((uint8_t *)pAddress + 1256);
        s[5] = (FieldElement *)((uint8_t *)pAddress + 1264);
        s[6] = (FieldElement *)((uint8_t *)pAddress + 1272);
        s[7] = (FieldElement *)((uint8_t *)pAddress + 1280);
        s[8] = (FieldElement *)((uint8_t *)pAddress + 1288);
        s[9] = (FieldElement *)((uint8_t *)pAddress + 1296);
        s[10] = (FieldElement *)((uint8_t *)pAddress + 1304);
        s[11] = (FieldElement *)((uint8_t *)pAddress + 1312);
        s[12] = (FieldElement *)((uint8_t *)pAddress + 1320);
        s[13] = (FieldElement *)((uint8_t *)pAddress + 1328);
        s[14] = (FieldElement *)((uint8_t *)pAddress + 1336);
        s[15] = (FieldElement *)((uint8_t *)pAddress + 1344);
        q0[0] = (FieldElement *)((uint8_t *)pAddress + 1352);
        q0[1] = (FieldElement *)((uint8_t *)pAddress + 1360);
        q0[2] = (FieldElement *)((uint8_t *)pAddress + 1368);
        q0[3] = (FieldElement *)((uint8_t *)pAddress + 1376);
        q0[4] = (FieldElement *)((uint8_t *)pAddress + 1384);
        q0[5] = (FieldElement *)((uint8_t *)pAddress + 1392);
        q0[6] = (FieldElement *)((uint8_t *)pAddress + 1400);
        q0[7] = (FieldElement *)((uint8_t *)pAddress + 1408);
        q0[8] = (FieldElement *)((uint8_t *)pAddress + 1416);
        q0[9] = (FieldElement *)((uint8_t *)pAddress + 1424);
        q0[10] = (FieldElement *)((uint8_t *)pAddress + 1432);
        q0[11] = (FieldElement *)((uint8_t *)pAddress + 1440);
        q0[12] = (FieldElement *)((uint8_t *)pAddress + 1448);
        q0[13] = (FieldElement *)((uint8_t *)pAddress + 1456);
        q0[14] = (FieldElement *)((uint8_t *)pAddress + 1464);
        q0[15] = (FieldElement *)((uint8_t *)pAddress + 1472);
        q1[0] = (FieldElement *)((uint8_t *)pAddress + 1480);
        q1[1] = (FieldElement *)((uint8_t *)pAddress + 1488);
        q1[2] = (FieldElement *)((uint8_t *)pAddress + 1496);
        q1[3] = (FieldElement *)((uint8_t *)pAddress + 1504);
        q1[4] = (FieldElement *)((uint8_t *)pAddress + 1512);
        q1[5] = (FieldElement *)((uint8_t *)pAddress + 1520);
        q1[6] = (FieldElement *)((uint8_t *)pAddress + 1528);
        q1[7] = (FieldElement *)((uint8_t *)pAddress + 1536);
        q1[8] = (FieldElement *)((uint8_t *)pAddress + 1544);
        q1[9] = (FieldElement *)((uint8_t *)pAddress + 1552);
        q1[10] = (FieldElement *)((uint8_t *)pAddress + 1560);
        q1[11] = (FieldElement *)((uint8_t *)pAddress + 1568);
        q1[12] = (FieldElement *)((uint8_t *)pAddress + 1576);
        q1[13] = (FieldElement *)((uint8_t *)pAddress + 1584);
        q1[14] = (FieldElement *)((uint8_t *)pAddress + 1592);
        q1[15] = (FieldElement *)((uint8_t *)pAddress + 1600);
        q2[0] = (FieldElement *)((uint8_t *)pAddress + 1608);
        q2[1] = (FieldElement *)((uint8_t *)pAddress + 1616);
        q2[2] = (FieldElement *)((uint8_t *)pAddress + 1624);
        q2[3] = (FieldElement *)((uint8_t *)pAddress + 1632);
        q2[4] = (FieldElement *)((uint8_t *)pAddress + 1640);
        q2[5] = (FieldElement *)((uint8_t *)pAddress + 1648);
        q2[6] = (FieldElement *)((uint8_t *)pAddress + 1656);
        q2[7] = (FieldElement *)((uint8_t *)pAddress + 1664);
        q2[8] = (FieldElement *)((uint8_t *)pAddress + 1672);
        q2[9] = (FieldElement *)((uint8_t *)pAddress + 1680);
        q2[10] = (FieldElement *)((uint8_t *)pAddress + 1688);
        q2[11] = (FieldElement *)((uint8_t *)pAddress + 1696);
        q2[12] = (FieldElement *)((uint8_t *)pAddress + 1704);
        q2[13] = (FieldElement *)((uint8_t *)pAddress + 1712);
        q2[14] = (FieldElement *)((uint8_t *)pAddress + 1720);
        q2[15] = (FieldElement *)((uint8_t *)pAddress + 1728);
        selEq[0] = (FieldElement *)((uint8_t *)pAddress + 1736);
        selEq[1] = (FieldElement *)((uint8_t *)pAddress + 1744);
        selEq[2] = (FieldElement *)((uint8_t *)pAddress + 1752);
        selEq[3] = (FieldElement *)((uint8_t *)pAddress + 1760);
        carryL[0] = (FieldElement *)((uint8_t *)pAddress + 1768);
        carryL[1] = (FieldElement *)((uint8_t *)pAddress + 1776);
        carryL[2] = (FieldElement *)((uint8_t *)pAddress + 1784);
        carryH[0] = (FieldElement *)((uint8_t *)pAddress + 1792);
        carryH[1] = (FieldElement *)((uint8_t *)pAddress + 1800);
        carryH[2] = (FieldElement *)((uint8_t *)pAddress + 1808);
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
    GeneratedPol freeInA;
    GeneratedPol freeInB;
    GeneratedPol freeInC;
    GeneratedPol a0;
    GeneratedPol a1;
    GeneratedPol a2;
    GeneratedPol a3;
    GeneratedPol a4;
    GeneratedPol a5;
    GeneratedPol a6;
    GeneratedPol a7;
    GeneratedPol b0;
    GeneratedPol b1;
    GeneratedPol b2;
    GeneratedPol b3;
    GeneratedPol b4;
    GeneratedPol b5;
    GeneratedPol b6;
    GeneratedPol b7;
    GeneratedPol c0;
    GeneratedPol c1;
    GeneratedPol c2;
    GeneratedPol c3;
    GeneratedPol c4;
    GeneratedPol c5;
    GeneratedPol c6;
    GeneratedPol c7;
    GeneratedPol opcode;
    GeneratedPol cIn;
    GeneratedPol cOut;
    GeneratedPol lCout;
    GeneratedPol lOpcode;
    GeneratedPol last;
    GeneratedPol useCarry;

    BinaryCommitPols (void * pAddress)
    {
        freeInA = (FieldElement *)((uint8_t *)pAddress + 1816);
        freeInB = (FieldElement *)((uint8_t *)pAddress + 1824);
        freeInC = (FieldElement *)((uint8_t *)pAddress + 1832);
        a0 = (FieldElement *)((uint8_t *)pAddress + 1840);
        a1 = (FieldElement *)((uint8_t *)pAddress + 1848);
        a2 = (FieldElement *)((uint8_t *)pAddress + 1856);
        a3 = (FieldElement *)((uint8_t *)pAddress + 1864);
        a4 = (FieldElement *)((uint8_t *)pAddress + 1872);
        a5 = (FieldElement *)((uint8_t *)pAddress + 1880);
        a6 = (FieldElement *)((uint8_t *)pAddress + 1888);
        a7 = (FieldElement *)((uint8_t *)pAddress + 1896);
        b0 = (FieldElement *)((uint8_t *)pAddress + 1904);
        b1 = (FieldElement *)((uint8_t *)pAddress + 1912);
        b2 = (FieldElement *)((uint8_t *)pAddress + 1920);
        b3 = (FieldElement *)((uint8_t *)pAddress + 1928);
        b4 = (FieldElement *)((uint8_t *)pAddress + 1936);
        b5 = (FieldElement *)((uint8_t *)pAddress + 1944);
        b6 = (FieldElement *)((uint8_t *)pAddress + 1952);
        b7 = (FieldElement *)((uint8_t *)pAddress + 1960);
        c0 = (FieldElement *)((uint8_t *)pAddress + 1968);
        c1 = (FieldElement *)((uint8_t *)pAddress + 1976);
        c2 = (FieldElement *)((uint8_t *)pAddress + 1984);
        c3 = (FieldElement *)((uint8_t *)pAddress + 1992);
        c4 = (FieldElement *)((uint8_t *)pAddress + 2000);
        c5 = (FieldElement *)((uint8_t *)pAddress + 2008);
        c6 = (FieldElement *)((uint8_t *)pAddress + 2016);
        c7 = (FieldElement *)((uint8_t *)pAddress + 2024);
        opcode = (FieldElement *)((uint8_t *)pAddress + 2032);
        cIn = (FieldElement *)((uint8_t *)pAddress + 2040);
        cOut = (FieldElement *)((uint8_t *)pAddress + 2048);
        lCout = (FieldElement *)((uint8_t *)pAddress + 2056);
        lOpcode = (FieldElement *)((uint8_t *)pAddress + 2064);
        last = (FieldElement *)((uint8_t *)pAddress + 2072);
        useCarry = (FieldElement *)((uint8_t *)pAddress + 2080);
    }

    BinaryCommitPols (void * pAddress, uint64_t degree)
    {
        freeInA = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        freeInB = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        freeInC = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        a0 = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        a1 = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        a2 = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        a3 = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        a4 = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        a5 = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        a6 = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        a7 = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        b0 = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        b1 = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        b2 = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        b3 = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        b4 = (FieldElement *)((uint8_t *)pAddress + 120*degree);
        b5 = (FieldElement *)((uint8_t *)pAddress + 128*degree);
        b6 = (FieldElement *)((uint8_t *)pAddress + 136*degree);
        b7 = (FieldElement *)((uint8_t *)pAddress + 144*degree);
        c0 = (FieldElement *)((uint8_t *)pAddress + 152*degree);
        c1 = (FieldElement *)((uint8_t *)pAddress + 160*degree);
        c2 = (FieldElement *)((uint8_t *)pAddress + 168*degree);
        c3 = (FieldElement *)((uint8_t *)pAddress + 176*degree);
        c4 = (FieldElement *)((uint8_t *)pAddress + 184*degree);
        c5 = (FieldElement *)((uint8_t *)pAddress + 192*degree);
        c6 = (FieldElement *)((uint8_t *)pAddress + 200*degree);
        c7 = (FieldElement *)((uint8_t *)pAddress + 208*degree);
        opcode = (FieldElement *)((uint8_t *)pAddress + 216*degree);
        cIn = (FieldElement *)((uint8_t *)pAddress + 224*degree);
        cOut = (FieldElement *)((uint8_t *)pAddress + 232*degree);
        lCout = (FieldElement *)((uint8_t *)pAddress + 240*degree);
        lOpcode = (FieldElement *)((uint8_t *)pAddress + 248*degree);
        last = (FieldElement *)((uint8_t *)pAddress + 256*degree);
        useCarry = (FieldElement *)((uint8_t *)pAddress + 264*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 272; }
};

class PoseidonGCommitPols
{
public:
    GeneratedPol in0;
    GeneratedPol in1;
    GeneratedPol in2;
    GeneratedPol in3;
    GeneratedPol in4;
    GeneratedPol in5;
    GeneratedPol in6;
    GeneratedPol in7;
    GeneratedPol hashType;
    GeneratedPol cap1;
    GeneratedPol cap2;
    GeneratedPol cap3;
    GeneratedPol hash0;
    GeneratedPol hash1;
    GeneratedPol hash2;
    GeneratedPol hash3;

    PoseidonGCommitPols (void * pAddress)
    {
        in0 = (FieldElement *)((uint8_t *)pAddress + 2088);
        in1 = (FieldElement *)((uint8_t *)pAddress + 2096);
        in2 = (FieldElement *)((uint8_t *)pAddress + 2104);
        in3 = (FieldElement *)((uint8_t *)pAddress + 2112);
        in4 = (FieldElement *)((uint8_t *)pAddress + 2120);
        in5 = (FieldElement *)((uint8_t *)pAddress + 2128);
        in6 = (FieldElement *)((uint8_t *)pAddress + 2136);
        in7 = (FieldElement *)((uint8_t *)pAddress + 2144);
        hashType = (FieldElement *)((uint8_t *)pAddress + 2152);
        cap1 = (FieldElement *)((uint8_t *)pAddress + 2160);
        cap2 = (FieldElement *)((uint8_t *)pAddress + 2168);
        cap3 = (FieldElement *)((uint8_t *)pAddress + 2176);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 2184);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 2192);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 2200);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 2208);
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
    GeneratedPol acc[8];
    GeneratedPol freeIn;
    GeneratedPol addr;
    GeneratedPol rem;
    GeneratedPol remInv;
    GeneratedPol spare;
    GeneratedPol firstHash;
    GeneratedPol curHash0;
    GeneratedPol curHash1;
    GeneratedPol curHash2;
    GeneratedPol curHash3;
    GeneratedPol prevHash0;
    GeneratedPol prevHash1;
    GeneratedPol prevHash2;
    GeneratedPol prevHash3;
    GeneratedPol len;
    GeneratedPol crOffset;
    GeneratedPol crLen;
    GeneratedPol crOffsetInv;
    GeneratedPol crF0;
    GeneratedPol crF1;
    GeneratedPol crF2;
    GeneratedPol crF3;
    GeneratedPol crF4;
    GeneratedPol crF5;
    GeneratedPol crF6;
    GeneratedPol crF7;
    GeneratedPol crV0;
    GeneratedPol crV1;
    GeneratedPol crV2;
    GeneratedPol crV3;
    GeneratedPol crV4;
    GeneratedPol crV5;
    GeneratedPol crV6;
    GeneratedPol crV7;

    PaddingPGCommitPols (void * pAddress)
    {
        acc[0] = (FieldElement *)((uint8_t *)pAddress + 2216);
        acc[1] = (FieldElement *)((uint8_t *)pAddress + 2224);
        acc[2] = (FieldElement *)((uint8_t *)pAddress + 2232);
        acc[3] = (FieldElement *)((uint8_t *)pAddress + 2240);
        acc[4] = (FieldElement *)((uint8_t *)pAddress + 2248);
        acc[5] = (FieldElement *)((uint8_t *)pAddress + 2256);
        acc[6] = (FieldElement *)((uint8_t *)pAddress + 2264);
        acc[7] = (FieldElement *)((uint8_t *)pAddress + 2272);
        freeIn = (FieldElement *)((uint8_t *)pAddress + 2280);
        addr = (FieldElement *)((uint8_t *)pAddress + 2288);
        rem = (FieldElement *)((uint8_t *)pAddress + 2296);
        remInv = (FieldElement *)((uint8_t *)pAddress + 2304);
        spare = (FieldElement *)((uint8_t *)pAddress + 2312);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 2320);
        curHash0 = (FieldElement *)((uint8_t *)pAddress + 2328);
        curHash1 = (FieldElement *)((uint8_t *)pAddress + 2336);
        curHash2 = (FieldElement *)((uint8_t *)pAddress + 2344);
        curHash3 = (FieldElement *)((uint8_t *)pAddress + 2352);
        prevHash0 = (FieldElement *)((uint8_t *)pAddress + 2360);
        prevHash1 = (FieldElement *)((uint8_t *)pAddress + 2368);
        prevHash2 = (FieldElement *)((uint8_t *)pAddress + 2376);
        prevHash3 = (FieldElement *)((uint8_t *)pAddress + 2384);
        len = (FieldElement *)((uint8_t *)pAddress + 2392);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 2400);
        crLen = (FieldElement *)((uint8_t *)pAddress + 2408);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 2416);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 2424);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 2432);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 2440);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 2448);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 2456);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 2464);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 2472);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 2480);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 2488);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 2496);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 2504);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 2512);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 2520);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 2528);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 2536);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 2544);
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
    GeneratedPol free0;
    GeneratedPol free1;
    GeneratedPol free2;
    GeneratedPol free3;
    GeneratedPol hashLeft0;
    GeneratedPol hashLeft1;
    GeneratedPol hashLeft2;
    GeneratedPol hashLeft3;
    GeneratedPol hashRight0;
    GeneratedPol hashRight1;
    GeneratedPol hashRight2;
    GeneratedPol hashRight3;
    GeneratedPol oldRoot0;
    GeneratedPol oldRoot1;
    GeneratedPol oldRoot2;
    GeneratedPol oldRoot3;
    GeneratedPol newRoot0;
    GeneratedPol newRoot1;
    GeneratedPol newRoot2;
    GeneratedPol newRoot3;
    GeneratedPol valueLow0;
    GeneratedPol valueLow1;
    GeneratedPol valueLow2;
    GeneratedPol valueLow3;
    GeneratedPol valueHigh0;
    GeneratedPol valueHigh1;
    GeneratedPol valueHigh2;
    GeneratedPol valueHigh3;
    GeneratedPol siblingValueHash0;
    GeneratedPol siblingValueHash1;
    GeneratedPol siblingValueHash2;
    GeneratedPol siblingValueHash3;
    GeneratedPol rkey0;
    GeneratedPol rkey1;
    GeneratedPol rkey2;
    GeneratedPol rkey3;
    GeneratedPol siblingRkey0;
    GeneratedPol siblingRkey1;
    GeneratedPol siblingRkey2;
    GeneratedPol siblingRkey3;
    GeneratedPol rkeyBit;
    GeneratedPol level0;
    GeneratedPol level1;
    GeneratedPol level2;
    GeneratedPol level3;
    GeneratedPol pc;
    GeneratedPol selOldRoot;
    GeneratedPol selNewRoot;
    GeneratedPol selValueLow;
    GeneratedPol selValueHigh;
    GeneratedPol selSiblingValueHash;
    GeneratedPol selRkey;
    GeneratedPol selRkeyBit;
    GeneratedPol selSiblingRkey;
    GeneratedPol selFree;
    GeneratedPol setHashLeft;
    GeneratedPol setHashRight;
    GeneratedPol setOldRoot;
    GeneratedPol setNewRoot;
    GeneratedPol setValueLow;
    GeneratedPol setValueHigh;
    GeneratedPol setSiblingValueHash;
    GeneratedPol setRkey;
    GeneratedPol setSiblingRkey;
    GeneratedPol setRkeyBit;
    GeneratedPol setLevel;
    GeneratedPol iHash;
    GeneratedPol iHashType;
    GeneratedPol iLatchSet;
    GeneratedPol iLatchGet;
    GeneratedPol iClimbRkey;
    GeneratedPol iClimbSiblingRkey;
    GeneratedPol iClimbSiblingRkeyN;
    GeneratedPol iRotateLevel;
    GeneratedPol iJmpz;
    GeneratedPol iJmp;
    GeneratedPol iConst0;
    GeneratedPol iConst1;
    GeneratedPol iConst2;
    GeneratedPol iConst3;
    GeneratedPol iAddress;
    GeneratedPol op0inv;

    StorageCommitPols (void * pAddress)
    {
        free0 = (FieldElement *)((uint8_t *)pAddress + 2552);
        free1 = (FieldElement *)((uint8_t *)pAddress + 2560);
        free2 = (FieldElement *)((uint8_t *)pAddress + 2568);
        free3 = (FieldElement *)((uint8_t *)pAddress + 2576);
        hashLeft0 = (FieldElement *)((uint8_t *)pAddress + 2584);
        hashLeft1 = (FieldElement *)((uint8_t *)pAddress + 2592);
        hashLeft2 = (FieldElement *)((uint8_t *)pAddress + 2600);
        hashLeft3 = (FieldElement *)((uint8_t *)pAddress + 2608);
        hashRight0 = (FieldElement *)((uint8_t *)pAddress + 2616);
        hashRight1 = (FieldElement *)((uint8_t *)pAddress + 2624);
        hashRight2 = (FieldElement *)((uint8_t *)pAddress + 2632);
        hashRight3 = (FieldElement *)((uint8_t *)pAddress + 2640);
        oldRoot0 = (FieldElement *)((uint8_t *)pAddress + 2648);
        oldRoot1 = (FieldElement *)((uint8_t *)pAddress + 2656);
        oldRoot2 = (FieldElement *)((uint8_t *)pAddress + 2664);
        oldRoot3 = (FieldElement *)((uint8_t *)pAddress + 2672);
        newRoot0 = (FieldElement *)((uint8_t *)pAddress + 2680);
        newRoot1 = (FieldElement *)((uint8_t *)pAddress + 2688);
        newRoot2 = (FieldElement *)((uint8_t *)pAddress + 2696);
        newRoot3 = (FieldElement *)((uint8_t *)pAddress + 2704);
        valueLow0 = (FieldElement *)((uint8_t *)pAddress + 2712);
        valueLow1 = (FieldElement *)((uint8_t *)pAddress + 2720);
        valueLow2 = (FieldElement *)((uint8_t *)pAddress + 2728);
        valueLow3 = (FieldElement *)((uint8_t *)pAddress + 2736);
        valueHigh0 = (FieldElement *)((uint8_t *)pAddress + 2744);
        valueHigh1 = (FieldElement *)((uint8_t *)pAddress + 2752);
        valueHigh2 = (FieldElement *)((uint8_t *)pAddress + 2760);
        valueHigh3 = (FieldElement *)((uint8_t *)pAddress + 2768);
        siblingValueHash0 = (FieldElement *)((uint8_t *)pAddress + 2776);
        siblingValueHash1 = (FieldElement *)((uint8_t *)pAddress + 2784);
        siblingValueHash2 = (FieldElement *)((uint8_t *)pAddress + 2792);
        siblingValueHash3 = (FieldElement *)((uint8_t *)pAddress + 2800);
        rkey0 = (FieldElement *)((uint8_t *)pAddress + 2808);
        rkey1 = (FieldElement *)((uint8_t *)pAddress + 2816);
        rkey2 = (FieldElement *)((uint8_t *)pAddress + 2824);
        rkey3 = (FieldElement *)((uint8_t *)pAddress + 2832);
        siblingRkey0 = (FieldElement *)((uint8_t *)pAddress + 2840);
        siblingRkey1 = (FieldElement *)((uint8_t *)pAddress + 2848);
        siblingRkey2 = (FieldElement *)((uint8_t *)pAddress + 2856);
        siblingRkey3 = (FieldElement *)((uint8_t *)pAddress + 2864);
        rkeyBit = (FieldElement *)((uint8_t *)pAddress + 2872);
        level0 = (FieldElement *)((uint8_t *)pAddress + 2880);
        level1 = (FieldElement *)((uint8_t *)pAddress + 2888);
        level2 = (FieldElement *)((uint8_t *)pAddress + 2896);
        level3 = (FieldElement *)((uint8_t *)pAddress + 2904);
        pc = (FieldElement *)((uint8_t *)pAddress + 2912);
        selOldRoot = (FieldElement *)((uint8_t *)pAddress + 2920);
        selNewRoot = (FieldElement *)((uint8_t *)pAddress + 2928);
        selValueLow = (FieldElement *)((uint8_t *)pAddress + 2936);
        selValueHigh = (FieldElement *)((uint8_t *)pAddress + 2944);
        selSiblingValueHash = (FieldElement *)((uint8_t *)pAddress + 2952);
        selRkey = (FieldElement *)((uint8_t *)pAddress + 2960);
        selRkeyBit = (FieldElement *)((uint8_t *)pAddress + 2968);
        selSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 2976);
        selFree = (FieldElement *)((uint8_t *)pAddress + 2984);
        setHashLeft = (FieldElement *)((uint8_t *)pAddress + 2992);
        setHashRight = (FieldElement *)((uint8_t *)pAddress + 3000);
        setOldRoot = (FieldElement *)((uint8_t *)pAddress + 3008);
        setNewRoot = (FieldElement *)((uint8_t *)pAddress + 3016);
        setValueLow = (FieldElement *)((uint8_t *)pAddress + 3024);
        setValueHigh = (FieldElement *)((uint8_t *)pAddress + 3032);
        setSiblingValueHash = (FieldElement *)((uint8_t *)pAddress + 3040);
        setRkey = (FieldElement *)((uint8_t *)pAddress + 3048);
        setSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 3056);
        setRkeyBit = (FieldElement *)((uint8_t *)pAddress + 3064);
        setLevel = (FieldElement *)((uint8_t *)pAddress + 3072);
        iHash = (FieldElement *)((uint8_t *)pAddress + 3080);
        iHashType = (FieldElement *)((uint8_t *)pAddress + 3088);
        iLatchSet = (FieldElement *)((uint8_t *)pAddress + 3096);
        iLatchGet = (FieldElement *)((uint8_t *)pAddress + 3104);
        iClimbRkey = (FieldElement *)((uint8_t *)pAddress + 3112);
        iClimbSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 3120);
        iClimbSiblingRkeyN = (FieldElement *)((uint8_t *)pAddress + 3128);
        iRotateLevel = (FieldElement *)((uint8_t *)pAddress + 3136);
        iJmpz = (FieldElement *)((uint8_t *)pAddress + 3144);
        iJmp = (FieldElement *)((uint8_t *)pAddress + 3152);
        iConst0 = (FieldElement *)((uint8_t *)pAddress + 3160);
        iConst1 = (FieldElement *)((uint8_t *)pAddress + 3168);
        iConst2 = (FieldElement *)((uint8_t *)pAddress + 3176);
        iConst3 = (FieldElement *)((uint8_t *)pAddress + 3184);
        iAddress = (FieldElement *)((uint8_t *)pAddress + 3192);
        op0inv = (FieldElement *)((uint8_t *)pAddress + 3200);
    }

    StorageCommitPols (void * pAddress, uint64_t degree)
    {
        free0 = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        free1 = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        free2 = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        free3 = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        hashLeft0 = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        hashLeft1 = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        hashLeft2 = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        hashLeft3 = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        hashRight0 = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        hashRight1 = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        hashRight2 = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        hashRight3 = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        oldRoot0 = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        oldRoot1 = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        oldRoot2 = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        oldRoot3 = (FieldElement *)((uint8_t *)pAddress + 120*degree);
        newRoot0 = (FieldElement *)((uint8_t *)pAddress + 128*degree);
        newRoot1 = (FieldElement *)((uint8_t *)pAddress + 136*degree);
        newRoot2 = (FieldElement *)((uint8_t *)pAddress + 144*degree);
        newRoot3 = (FieldElement *)((uint8_t *)pAddress + 152*degree);
        valueLow0 = (FieldElement *)((uint8_t *)pAddress + 160*degree);
        valueLow1 = (FieldElement *)((uint8_t *)pAddress + 168*degree);
        valueLow2 = (FieldElement *)((uint8_t *)pAddress + 176*degree);
        valueLow3 = (FieldElement *)((uint8_t *)pAddress + 184*degree);
        valueHigh0 = (FieldElement *)((uint8_t *)pAddress + 192*degree);
        valueHigh1 = (FieldElement *)((uint8_t *)pAddress + 200*degree);
        valueHigh2 = (FieldElement *)((uint8_t *)pAddress + 208*degree);
        valueHigh3 = (FieldElement *)((uint8_t *)pAddress + 216*degree);
        siblingValueHash0 = (FieldElement *)((uint8_t *)pAddress + 224*degree);
        siblingValueHash1 = (FieldElement *)((uint8_t *)pAddress + 232*degree);
        siblingValueHash2 = (FieldElement *)((uint8_t *)pAddress + 240*degree);
        siblingValueHash3 = (FieldElement *)((uint8_t *)pAddress + 248*degree);
        rkey0 = (FieldElement *)((uint8_t *)pAddress + 256*degree);
        rkey1 = (FieldElement *)((uint8_t *)pAddress + 264*degree);
        rkey2 = (FieldElement *)((uint8_t *)pAddress + 272*degree);
        rkey3 = (FieldElement *)((uint8_t *)pAddress + 280*degree);
        siblingRkey0 = (FieldElement *)((uint8_t *)pAddress + 288*degree);
        siblingRkey1 = (FieldElement *)((uint8_t *)pAddress + 296*degree);
        siblingRkey2 = (FieldElement *)((uint8_t *)pAddress + 304*degree);
        siblingRkey3 = (FieldElement *)((uint8_t *)pAddress + 312*degree);
        rkeyBit = (FieldElement *)((uint8_t *)pAddress + 320*degree);
        level0 = (FieldElement *)((uint8_t *)pAddress + 328*degree);
        level1 = (FieldElement *)((uint8_t *)pAddress + 336*degree);
        level2 = (FieldElement *)((uint8_t *)pAddress + 344*degree);
        level3 = (FieldElement *)((uint8_t *)pAddress + 352*degree);
        pc = (FieldElement *)((uint8_t *)pAddress + 360*degree);
        selOldRoot = (FieldElement *)((uint8_t *)pAddress + 368*degree);
        selNewRoot = (FieldElement *)((uint8_t *)pAddress + 376*degree);
        selValueLow = (FieldElement *)((uint8_t *)pAddress + 384*degree);
        selValueHigh = (FieldElement *)((uint8_t *)pAddress + 392*degree);
        selSiblingValueHash = (FieldElement *)((uint8_t *)pAddress + 400*degree);
        selRkey = (FieldElement *)((uint8_t *)pAddress + 408*degree);
        selRkeyBit = (FieldElement *)((uint8_t *)pAddress + 416*degree);
        selSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 424*degree);
        selFree = (FieldElement *)((uint8_t *)pAddress + 432*degree);
        setHashLeft = (FieldElement *)((uint8_t *)pAddress + 440*degree);
        setHashRight = (FieldElement *)((uint8_t *)pAddress + 448*degree);
        setOldRoot = (FieldElement *)((uint8_t *)pAddress + 456*degree);
        setNewRoot = (FieldElement *)((uint8_t *)pAddress + 464*degree);
        setValueLow = (FieldElement *)((uint8_t *)pAddress + 472*degree);
        setValueHigh = (FieldElement *)((uint8_t *)pAddress + 480*degree);
        setSiblingValueHash = (FieldElement *)((uint8_t *)pAddress + 488*degree);
        setRkey = (FieldElement *)((uint8_t *)pAddress + 496*degree);
        setSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 504*degree);
        setRkeyBit = (FieldElement *)((uint8_t *)pAddress + 512*degree);
        setLevel = (FieldElement *)((uint8_t *)pAddress + 520*degree);
        iHash = (FieldElement *)((uint8_t *)pAddress + 528*degree);
        iHashType = (FieldElement *)((uint8_t *)pAddress + 536*degree);
        iLatchSet = (FieldElement *)((uint8_t *)pAddress + 544*degree);
        iLatchGet = (FieldElement *)((uint8_t *)pAddress + 552*degree);
        iClimbRkey = (FieldElement *)((uint8_t *)pAddress + 560*degree);
        iClimbSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 568*degree);
        iClimbSiblingRkeyN = (FieldElement *)((uint8_t *)pAddress + 576*degree);
        iRotateLevel = (FieldElement *)((uint8_t *)pAddress + 584*degree);
        iJmpz = (FieldElement *)((uint8_t *)pAddress + 592*degree);
        iJmp = (FieldElement *)((uint8_t *)pAddress + 600*degree);
        iConst0 = (FieldElement *)((uint8_t *)pAddress + 608*degree);
        iConst1 = (FieldElement *)((uint8_t *)pAddress + 616*degree);
        iConst2 = (FieldElement *)((uint8_t *)pAddress + 624*degree);
        iConst3 = (FieldElement *)((uint8_t *)pAddress + 632*degree);
        iAddress = (FieldElement *)((uint8_t *)pAddress + 640*degree);
        op0inv = (FieldElement *)((uint8_t *)pAddress + 648*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 656; }
};

class NormGate9CommitPols
{
public:
    GeneratedPol freeA;
    GeneratedPol freeB;
    GeneratedPol gateType;
    GeneratedPol freeANorm;
    GeneratedPol freeBNorm;
    GeneratedPol freeCNorm;
    GeneratedPol a;
    GeneratedPol b;
    GeneratedPol c;

    NormGate9CommitPols (void * pAddress)
    {
        freeA = (FieldElement *)((uint8_t *)pAddress + 3208);
        freeB = (FieldElement *)((uint8_t *)pAddress + 3216);
        gateType = (FieldElement *)((uint8_t *)pAddress + 3224);
        freeANorm = (FieldElement *)((uint8_t *)pAddress + 3232);
        freeBNorm = (FieldElement *)((uint8_t *)pAddress + 3240);
        freeCNorm = (FieldElement *)((uint8_t *)pAddress + 3248);
        a = (FieldElement *)((uint8_t *)pAddress + 3256);
        b = (FieldElement *)((uint8_t *)pAddress + 3264);
        c = (FieldElement *)((uint8_t *)pAddress + 3272);
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
    GeneratedPol a;
    GeneratedPol b;
    GeneratedPol c;

    KeccakFCommitPols (void * pAddress)
    {
        a = (FieldElement *)((uint8_t *)pAddress + 3280);
        b = (FieldElement *)((uint8_t *)pAddress + 3288);
        c = (FieldElement *)((uint8_t *)pAddress + 3296);
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
    GeneratedPol bit;
    GeneratedPol field9;

    Nine2OneCommitPols (void * pAddress)
    {
        bit = (FieldElement *)((uint8_t *)pAddress + 3304);
        field9 = (FieldElement *)((uint8_t *)pAddress + 3312);
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
    GeneratedPol rBit;
    GeneratedPol sOutBit;
    GeneratedPol r8;
    GeneratedPol connected;
    GeneratedPol sOut0;
    GeneratedPol sOut1;
    GeneratedPol sOut2;
    GeneratedPol sOut3;
    GeneratedPol sOut4;
    GeneratedPol sOut5;
    GeneratedPol sOut6;
    GeneratedPol sOut7;

    PaddingKKBitCommitPols (void * pAddress)
    {
        rBit = (FieldElement *)((uint8_t *)pAddress + 3320);
        sOutBit = (FieldElement *)((uint8_t *)pAddress + 3328);
        r8 = (FieldElement *)((uint8_t *)pAddress + 3336);
        connected = (FieldElement *)((uint8_t *)pAddress + 3344);
        sOut0 = (FieldElement *)((uint8_t *)pAddress + 3352);
        sOut1 = (FieldElement *)((uint8_t *)pAddress + 3360);
        sOut2 = (FieldElement *)((uint8_t *)pAddress + 3368);
        sOut3 = (FieldElement *)((uint8_t *)pAddress + 3376);
        sOut4 = (FieldElement *)((uint8_t *)pAddress + 3384);
        sOut5 = (FieldElement *)((uint8_t *)pAddress + 3392);
        sOut6 = (FieldElement *)((uint8_t *)pAddress + 3400);
        sOut7 = (FieldElement *)((uint8_t *)pAddress + 3408);
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
    GeneratedPol freeIn;
    GeneratedPol connected;
    GeneratedPol addr;
    GeneratedPol rem;
    GeneratedPol remInv;
    GeneratedPol spare;
    GeneratedPol firstHash;
    GeneratedPol len;
    GeneratedPol hash0;
    GeneratedPol hash1;
    GeneratedPol hash2;
    GeneratedPol hash3;
    GeneratedPol hash4;
    GeneratedPol hash5;
    GeneratedPol hash6;
    GeneratedPol hash7;
    GeneratedPol crOffset;
    GeneratedPol crLen;
    GeneratedPol crOffsetInv;
    GeneratedPol crF0;
    GeneratedPol crF1;
    GeneratedPol crF2;
    GeneratedPol crF3;
    GeneratedPol crF4;
    GeneratedPol crF5;
    GeneratedPol crF6;
    GeneratedPol crF7;
    GeneratedPol crV0;
    GeneratedPol crV1;
    GeneratedPol crV2;
    GeneratedPol crV3;
    GeneratedPol crV4;
    GeneratedPol crV5;
    GeneratedPol crV6;
    GeneratedPol crV7;

    PaddingKKCommitPols (void * pAddress)
    {
        freeIn = (FieldElement *)((uint8_t *)pAddress + 3416);
        connected = (FieldElement *)((uint8_t *)pAddress + 3424);
        addr = (FieldElement *)((uint8_t *)pAddress + 3432);
        rem = (FieldElement *)((uint8_t *)pAddress + 3440);
        remInv = (FieldElement *)((uint8_t *)pAddress + 3448);
        spare = (FieldElement *)((uint8_t *)pAddress + 3456);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 3464);
        len = (FieldElement *)((uint8_t *)pAddress + 3472);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 3480);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 3488);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 3496);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 3504);
        hash4 = (FieldElement *)((uint8_t *)pAddress + 3512);
        hash5 = (FieldElement *)((uint8_t *)pAddress + 3520);
        hash6 = (FieldElement *)((uint8_t *)pAddress + 3528);
        hash7 = (FieldElement *)((uint8_t *)pAddress + 3536);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 3544);
        crLen = (FieldElement *)((uint8_t *)pAddress + 3552);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 3560);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 3568);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 3576);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 3584);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 3592);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 3600);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 3608);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 3616);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 3624);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 3632);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 3640);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 3648);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 3656);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 3664);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 3672);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 3680);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 3688);
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
    GeneratedPol addr;
    GeneratedPol step;
    GeneratedPol mOp;
    GeneratedPol mWr;
    GeneratedPol val[8];
    GeneratedPol lastAccess;

    MemCommitPols (void * pAddress)
    {
        addr = (FieldElement *)((uint8_t *)pAddress + 3696);
        step = (FieldElement *)((uint8_t *)pAddress + 3704);
        mOp = (FieldElement *)((uint8_t *)pAddress + 3712);
        mWr = (FieldElement *)((uint8_t *)pAddress + 3720);
        val[0] = (FieldElement *)((uint8_t *)pAddress + 3728);
        val[1] = (FieldElement *)((uint8_t *)pAddress + 3736);
        val[2] = (FieldElement *)((uint8_t *)pAddress + 3744);
        val[3] = (FieldElement *)((uint8_t *)pAddress + 3752);
        val[4] = (FieldElement *)((uint8_t *)pAddress + 3760);
        val[5] = (FieldElement *)((uint8_t *)pAddress + 3768);
        val[6] = (FieldElement *)((uint8_t *)pAddress + 3776);
        val[7] = (FieldElement *)((uint8_t *)pAddress + 3784);
        lastAccess = (FieldElement *)((uint8_t *)pAddress + 3792);
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
    GeneratedPol A7;
    GeneratedPol A6;
    GeneratedPol A5;
    GeneratedPol A4;
    GeneratedPol A3;
    GeneratedPol A2;
    GeneratedPol A1;
    GeneratedPol A0;
    GeneratedPol B7;
    GeneratedPol B6;
    GeneratedPol B5;
    GeneratedPol B4;
    GeneratedPol B3;
    GeneratedPol B2;
    GeneratedPol B1;
    GeneratedPol B0;
    GeneratedPol C7;
    GeneratedPol C6;
    GeneratedPol C5;
    GeneratedPol C4;
    GeneratedPol C3;
    GeneratedPol C2;
    GeneratedPol C1;
    GeneratedPol C0;
    GeneratedPol D7;
    GeneratedPol D6;
    GeneratedPol D5;
    GeneratedPol D4;
    GeneratedPol D3;
    GeneratedPol D2;
    GeneratedPol D1;
    GeneratedPol D0;
    GeneratedPol E7;
    GeneratedPol E6;
    GeneratedPol E5;
    GeneratedPol E4;
    GeneratedPol E3;
    GeneratedPol E2;
    GeneratedPol E1;
    GeneratedPol E0;
    GeneratedPol SR7;
    GeneratedPol SR6;
    GeneratedPol SR5;
    GeneratedPol SR4;
    GeneratedPol SR3;
    GeneratedPol SR2;
    GeneratedPol SR1;
    GeneratedPol SR0;
    GeneratedPol CTX;
    GeneratedPol SP;
    GeneratedPol PC;
    GeneratedPol GAS;
    GeneratedPol MAXMEM;
    GeneratedPol zkPC;
    GeneratedPol RR;
    GeneratedPol HASHPOS;
    GeneratedPol CONST7;
    GeneratedPol CONST6;
    GeneratedPol CONST5;
    GeneratedPol CONST4;
    GeneratedPol CONST3;
    GeneratedPol CONST2;
    GeneratedPol CONST1;
    GeneratedPol CONST0;
    GeneratedPol FREE7;
    GeneratedPol FREE6;
    GeneratedPol FREE5;
    GeneratedPol FREE4;
    GeneratedPol FREE3;
    GeneratedPol FREE2;
    GeneratedPol FREE1;
    GeneratedPol FREE0;
    GeneratedPol inA;
    GeneratedPol inB;
    GeneratedPol inC;
    GeneratedPol inD;
    GeneratedPol inE;
    GeneratedPol inSR;
    GeneratedPol inFREE;
    GeneratedPol inCTX;
    GeneratedPol inSP;
    GeneratedPol inPC;
    GeneratedPol inGAS;
    GeneratedPol inMAXMEM;
    GeneratedPol inSTEP;
    GeneratedPol inRR;
    GeneratedPol inHASHPOS;
    GeneratedPol setA;
    GeneratedPol setB;
    GeneratedPol setC;
    GeneratedPol setD;
    GeneratedPol setE;
    GeneratedPol setSR;
    GeneratedPol setCTX;
    GeneratedPol setSP;
    GeneratedPol setPC;
    GeneratedPol setGAS;
    GeneratedPol setMAXMEM;
    GeneratedPol JMP;
    GeneratedPol JMPN;
    GeneratedPol JMPC;
    GeneratedPol setRR;
    GeneratedPol setHASHPOS;
    GeneratedPol offset;
    GeneratedPol incStack;
    GeneratedPol incCode;
    GeneratedPol isStack;
    GeneratedPol isCode;
    GeneratedPol isMem;
    GeneratedPol ind;
    GeneratedPol indRR;
    GeneratedPol useCTX;
    GeneratedPol carry;
    GeneratedPol mOp;
    GeneratedPol mWR;
    GeneratedPol sWR;
    GeneratedPol sRD;
    GeneratedPol arith;
    GeneratedPol arithEq0;
    GeneratedPol arithEq1;
    GeneratedPol arithEq2;
    GeneratedPol arithEq3;
    GeneratedPol memAlign;
    GeneratedPol memAlignWR;
    GeneratedPol memAlignWR8;
    GeneratedPol hashK;
    GeneratedPol hashKLen;
    GeneratedPol hashKDigest;
    GeneratedPol hashP;
    GeneratedPol hashPLen;
    GeneratedPol hashPDigest;
    GeneratedPol bin;
    GeneratedPol binOpcode;
    GeneratedPol assert;
    GeneratedPol isNeg;
    GeneratedPol isMaxMem;
    GeneratedPol sKeyI[4];
    GeneratedPol sKey[4];

    MainCommitPols (void * pAddress)
    {
        A7 = (FieldElement *)((uint8_t *)pAddress + 3800);
        A6 = (FieldElement *)((uint8_t *)pAddress + 3808);
        A5 = (FieldElement *)((uint8_t *)pAddress + 3816);
        A4 = (FieldElement *)((uint8_t *)pAddress + 3824);
        A3 = (FieldElement *)((uint8_t *)pAddress + 3832);
        A2 = (FieldElement *)((uint8_t *)pAddress + 3840);
        A1 = (FieldElement *)((uint8_t *)pAddress + 3848);
        A0 = (FieldElement *)((uint8_t *)pAddress + 3856);
        B7 = (FieldElement *)((uint8_t *)pAddress + 3864);
        B6 = (FieldElement *)((uint8_t *)pAddress + 3872);
        B5 = (FieldElement *)((uint8_t *)pAddress + 3880);
        B4 = (FieldElement *)((uint8_t *)pAddress + 3888);
        B3 = (FieldElement *)((uint8_t *)pAddress + 3896);
        B2 = (FieldElement *)((uint8_t *)pAddress + 3904);
        B1 = (FieldElement *)((uint8_t *)pAddress + 3912);
        B0 = (FieldElement *)((uint8_t *)pAddress + 3920);
        C7 = (FieldElement *)((uint8_t *)pAddress + 3928);
        C6 = (FieldElement *)((uint8_t *)pAddress + 3936);
        C5 = (FieldElement *)((uint8_t *)pAddress + 3944);
        C4 = (FieldElement *)((uint8_t *)pAddress + 3952);
        C3 = (FieldElement *)((uint8_t *)pAddress + 3960);
        C2 = (FieldElement *)((uint8_t *)pAddress + 3968);
        C1 = (FieldElement *)((uint8_t *)pAddress + 3976);
        C0 = (FieldElement *)((uint8_t *)pAddress + 3984);
        D7 = (FieldElement *)((uint8_t *)pAddress + 3992);
        D6 = (FieldElement *)((uint8_t *)pAddress + 4000);
        D5 = (FieldElement *)((uint8_t *)pAddress + 4008);
        D4 = (FieldElement *)((uint8_t *)pAddress + 4016);
        D3 = (FieldElement *)((uint8_t *)pAddress + 4024);
        D2 = (FieldElement *)((uint8_t *)pAddress + 4032);
        D1 = (FieldElement *)((uint8_t *)pAddress + 4040);
        D0 = (FieldElement *)((uint8_t *)pAddress + 4048);
        E7 = (FieldElement *)((uint8_t *)pAddress + 4056);
        E6 = (FieldElement *)((uint8_t *)pAddress + 4064);
        E5 = (FieldElement *)((uint8_t *)pAddress + 4072);
        E4 = (FieldElement *)((uint8_t *)pAddress + 4080);
        E3 = (FieldElement *)((uint8_t *)pAddress + 4088);
        E2 = (FieldElement *)((uint8_t *)pAddress + 4096);
        E1 = (FieldElement *)((uint8_t *)pAddress + 4104);
        E0 = (FieldElement *)((uint8_t *)pAddress + 4112);
        SR7 = (FieldElement *)((uint8_t *)pAddress + 4120);
        SR6 = (FieldElement *)((uint8_t *)pAddress + 4128);
        SR5 = (FieldElement *)((uint8_t *)pAddress + 4136);
        SR4 = (FieldElement *)((uint8_t *)pAddress + 4144);
        SR3 = (FieldElement *)((uint8_t *)pAddress + 4152);
        SR2 = (FieldElement *)((uint8_t *)pAddress + 4160);
        SR1 = (FieldElement *)((uint8_t *)pAddress + 4168);
        SR0 = (FieldElement *)((uint8_t *)pAddress + 4176);
        CTX = (FieldElement *)((uint8_t *)pAddress + 4184);
        SP = (FieldElement *)((uint8_t *)pAddress + 4192);
        PC = (FieldElement *)((uint8_t *)pAddress + 4200);
        GAS = (FieldElement *)((uint8_t *)pAddress + 4208);
        MAXMEM = (FieldElement *)((uint8_t *)pAddress + 4216);
        zkPC = (FieldElement *)((uint8_t *)pAddress + 4224);
        RR = (FieldElement *)((uint8_t *)pAddress + 4232);
        HASHPOS = (FieldElement *)((uint8_t *)pAddress + 4240);
        CONST7 = (FieldElement *)((uint8_t *)pAddress + 4248);
        CONST6 = (FieldElement *)((uint8_t *)pAddress + 4256);
        CONST5 = (FieldElement *)((uint8_t *)pAddress + 4264);
        CONST4 = (FieldElement *)((uint8_t *)pAddress + 4272);
        CONST3 = (FieldElement *)((uint8_t *)pAddress + 4280);
        CONST2 = (FieldElement *)((uint8_t *)pAddress + 4288);
        CONST1 = (FieldElement *)((uint8_t *)pAddress + 4296);
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 4304);
        FREE7 = (FieldElement *)((uint8_t *)pAddress + 4312);
        FREE6 = (FieldElement *)((uint8_t *)pAddress + 4320);
        FREE5 = (FieldElement *)((uint8_t *)pAddress + 4328);
        FREE4 = (FieldElement *)((uint8_t *)pAddress + 4336);
        FREE3 = (FieldElement *)((uint8_t *)pAddress + 4344);
        FREE2 = (FieldElement *)((uint8_t *)pAddress + 4352);
        FREE1 = (FieldElement *)((uint8_t *)pAddress + 4360);
        FREE0 = (FieldElement *)((uint8_t *)pAddress + 4368);
        inA = (FieldElement *)((uint8_t *)pAddress + 4376);
        inB = (FieldElement *)((uint8_t *)pAddress + 4384);
        inC = (FieldElement *)((uint8_t *)pAddress + 4392);
        inD = (FieldElement *)((uint8_t *)pAddress + 4400);
        inE = (FieldElement *)((uint8_t *)pAddress + 4408);
        inSR = (FieldElement *)((uint8_t *)pAddress + 4416);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 4424);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 4432);
        inSP = (FieldElement *)((uint8_t *)pAddress + 4440);
        inPC = (FieldElement *)((uint8_t *)pAddress + 4448);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 4456);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 4464);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 4472);
        inRR = (FieldElement *)((uint8_t *)pAddress + 4480);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 4488);
        setA = (FieldElement *)((uint8_t *)pAddress + 4496);
        setB = (FieldElement *)((uint8_t *)pAddress + 4504);
        setC = (FieldElement *)((uint8_t *)pAddress + 4512);
        setD = (FieldElement *)((uint8_t *)pAddress + 4520);
        setE = (FieldElement *)((uint8_t *)pAddress + 4528);
        setSR = (FieldElement *)((uint8_t *)pAddress + 4536);
        setCTX = (FieldElement *)((uint8_t *)pAddress + 4544);
        setSP = (FieldElement *)((uint8_t *)pAddress + 4552);
        setPC = (FieldElement *)((uint8_t *)pAddress + 4560);
        setGAS = (FieldElement *)((uint8_t *)pAddress + 4568);
        setMAXMEM = (FieldElement *)((uint8_t *)pAddress + 4576);
        JMP = (FieldElement *)((uint8_t *)pAddress + 4584);
        JMPN = (FieldElement *)((uint8_t *)pAddress + 4592);
        JMPC = (FieldElement *)((uint8_t *)pAddress + 4600);
        setRR = (FieldElement *)((uint8_t *)pAddress + 4608);
        setHASHPOS = (FieldElement *)((uint8_t *)pAddress + 4616);
        offset = (FieldElement *)((uint8_t *)pAddress + 4624);
        incStack = (FieldElement *)((uint8_t *)pAddress + 4632);
        incCode = (FieldElement *)((uint8_t *)pAddress + 4640);
        isStack = (FieldElement *)((uint8_t *)pAddress + 4648);
        isCode = (FieldElement *)((uint8_t *)pAddress + 4656);
        isMem = (FieldElement *)((uint8_t *)pAddress + 4664);
        ind = (FieldElement *)((uint8_t *)pAddress + 4672);
        indRR = (FieldElement *)((uint8_t *)pAddress + 4680);
        useCTX = (FieldElement *)((uint8_t *)pAddress + 4688);
        carry = (FieldElement *)((uint8_t *)pAddress + 4696);
        mOp = (FieldElement *)((uint8_t *)pAddress + 4704);
        mWR = (FieldElement *)((uint8_t *)pAddress + 4712);
        sWR = (FieldElement *)((uint8_t *)pAddress + 4720);
        sRD = (FieldElement *)((uint8_t *)pAddress + 4728);
        arith = (FieldElement *)((uint8_t *)pAddress + 4736);
        arithEq0 = (FieldElement *)((uint8_t *)pAddress + 4744);
        arithEq1 = (FieldElement *)((uint8_t *)pAddress + 4752);
        arithEq2 = (FieldElement *)((uint8_t *)pAddress + 4760);
        arithEq3 = (FieldElement *)((uint8_t *)pAddress + 4768);
        memAlign = (FieldElement *)((uint8_t *)pAddress + 4776);
        memAlignWR = (FieldElement *)((uint8_t *)pAddress + 4784);
        memAlignWR8 = (FieldElement *)((uint8_t *)pAddress + 4792);
        hashK = (FieldElement *)((uint8_t *)pAddress + 4800);
        hashKLen = (FieldElement *)((uint8_t *)pAddress + 4808);
        hashKDigest = (FieldElement *)((uint8_t *)pAddress + 4816);
        hashP = (FieldElement *)((uint8_t *)pAddress + 4824);
        hashPLen = (FieldElement *)((uint8_t *)pAddress + 4832);
        hashPDigest = (FieldElement *)((uint8_t *)pAddress + 4840);
        bin = (FieldElement *)((uint8_t *)pAddress + 4848);
        binOpcode = (FieldElement *)((uint8_t *)pAddress + 4856);
        assert = (FieldElement *)((uint8_t *)pAddress + 4864);
        isNeg = (FieldElement *)((uint8_t *)pAddress + 4872);
        isMaxMem = (FieldElement *)((uint8_t *)pAddress + 4880);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 4888);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 4896);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 4904);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 4912);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 4920);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 4928);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 4936);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 4944);
    }

    MainCommitPols (void * pAddress, uint64_t degree)
    {
        A7 = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        A6 = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        A5 = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        A4 = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        A3 = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        A2 = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        A1 = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        A0 = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        B7 = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        B6 = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        B5 = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        B4 = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        B3 = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        B2 = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        B1 = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        B0 = (FieldElement *)((uint8_t *)pAddress + 120*degree);
        C7 = (FieldElement *)((uint8_t *)pAddress + 128*degree);
        C6 = (FieldElement *)((uint8_t *)pAddress + 136*degree);
        C5 = (FieldElement *)((uint8_t *)pAddress + 144*degree);
        C4 = (FieldElement *)((uint8_t *)pAddress + 152*degree);
        C3 = (FieldElement *)((uint8_t *)pAddress + 160*degree);
        C2 = (FieldElement *)((uint8_t *)pAddress + 168*degree);
        C1 = (FieldElement *)((uint8_t *)pAddress + 176*degree);
        C0 = (FieldElement *)((uint8_t *)pAddress + 184*degree);
        D7 = (FieldElement *)((uint8_t *)pAddress + 192*degree);
        D6 = (FieldElement *)((uint8_t *)pAddress + 200*degree);
        D5 = (FieldElement *)((uint8_t *)pAddress + 208*degree);
        D4 = (FieldElement *)((uint8_t *)pAddress + 216*degree);
        D3 = (FieldElement *)((uint8_t *)pAddress + 224*degree);
        D2 = (FieldElement *)((uint8_t *)pAddress + 232*degree);
        D1 = (FieldElement *)((uint8_t *)pAddress + 240*degree);
        D0 = (FieldElement *)((uint8_t *)pAddress + 248*degree);
        E7 = (FieldElement *)((uint8_t *)pAddress + 256*degree);
        E6 = (FieldElement *)((uint8_t *)pAddress + 264*degree);
        E5 = (FieldElement *)((uint8_t *)pAddress + 272*degree);
        E4 = (FieldElement *)((uint8_t *)pAddress + 280*degree);
        E3 = (FieldElement *)((uint8_t *)pAddress + 288*degree);
        E2 = (FieldElement *)((uint8_t *)pAddress + 296*degree);
        E1 = (FieldElement *)((uint8_t *)pAddress + 304*degree);
        E0 = (FieldElement *)((uint8_t *)pAddress + 312*degree);
        SR7 = (FieldElement *)((uint8_t *)pAddress + 320*degree);
        SR6 = (FieldElement *)((uint8_t *)pAddress + 328*degree);
        SR5 = (FieldElement *)((uint8_t *)pAddress + 336*degree);
        SR4 = (FieldElement *)((uint8_t *)pAddress + 344*degree);
        SR3 = (FieldElement *)((uint8_t *)pAddress + 352*degree);
        SR2 = (FieldElement *)((uint8_t *)pAddress + 360*degree);
        SR1 = (FieldElement *)((uint8_t *)pAddress + 368*degree);
        SR0 = (FieldElement *)((uint8_t *)pAddress + 376*degree);
        CTX = (FieldElement *)((uint8_t *)pAddress + 384*degree);
        SP = (FieldElement *)((uint8_t *)pAddress + 392*degree);
        PC = (FieldElement *)((uint8_t *)pAddress + 400*degree);
        GAS = (FieldElement *)((uint8_t *)pAddress + 408*degree);
        MAXMEM = (FieldElement *)((uint8_t *)pAddress + 416*degree);
        zkPC = (FieldElement *)((uint8_t *)pAddress + 424*degree);
        RR = (FieldElement *)((uint8_t *)pAddress + 432*degree);
        HASHPOS = (FieldElement *)((uint8_t *)pAddress + 440*degree);
        CONST7 = (FieldElement *)((uint8_t *)pAddress + 448*degree);
        CONST6 = (FieldElement *)((uint8_t *)pAddress + 456*degree);
        CONST5 = (FieldElement *)((uint8_t *)pAddress + 464*degree);
        CONST4 = (FieldElement *)((uint8_t *)pAddress + 472*degree);
        CONST3 = (FieldElement *)((uint8_t *)pAddress + 480*degree);
        CONST2 = (FieldElement *)((uint8_t *)pAddress + 488*degree);
        CONST1 = (FieldElement *)((uint8_t *)pAddress + 496*degree);
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 504*degree);
        FREE7 = (FieldElement *)((uint8_t *)pAddress + 512*degree);
        FREE6 = (FieldElement *)((uint8_t *)pAddress + 520*degree);
        FREE5 = (FieldElement *)((uint8_t *)pAddress + 528*degree);
        FREE4 = (FieldElement *)((uint8_t *)pAddress + 536*degree);
        FREE3 = (FieldElement *)((uint8_t *)pAddress + 544*degree);
        FREE2 = (FieldElement *)((uint8_t *)pAddress + 552*degree);
        FREE1 = (FieldElement *)((uint8_t *)pAddress + 560*degree);
        FREE0 = (FieldElement *)((uint8_t *)pAddress + 568*degree);
        inA = (FieldElement *)((uint8_t *)pAddress + 576*degree);
        inB = (FieldElement *)((uint8_t *)pAddress + 584*degree);
        inC = (FieldElement *)((uint8_t *)pAddress + 592*degree);
        inD = (FieldElement *)((uint8_t *)pAddress + 600*degree);
        inE = (FieldElement *)((uint8_t *)pAddress + 608*degree);
        inSR = (FieldElement *)((uint8_t *)pAddress + 616*degree);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 624*degree);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 632*degree);
        inSP = (FieldElement *)((uint8_t *)pAddress + 640*degree);
        inPC = (FieldElement *)((uint8_t *)pAddress + 648*degree);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 656*degree);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 664*degree);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 672*degree);
        inRR = (FieldElement *)((uint8_t *)pAddress + 680*degree);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 688*degree);
        setA = (FieldElement *)((uint8_t *)pAddress + 696*degree);
        setB = (FieldElement *)((uint8_t *)pAddress + 704*degree);
        setC = (FieldElement *)((uint8_t *)pAddress + 712*degree);
        setD = (FieldElement *)((uint8_t *)pAddress + 720*degree);
        setE = (FieldElement *)((uint8_t *)pAddress + 728*degree);
        setSR = (FieldElement *)((uint8_t *)pAddress + 736*degree);
        setCTX = (FieldElement *)((uint8_t *)pAddress + 744*degree);
        setSP = (FieldElement *)((uint8_t *)pAddress + 752*degree);
        setPC = (FieldElement *)((uint8_t *)pAddress + 760*degree);
        setGAS = (FieldElement *)((uint8_t *)pAddress + 768*degree);
        setMAXMEM = (FieldElement *)((uint8_t *)pAddress + 776*degree);
        JMP = (FieldElement *)((uint8_t *)pAddress + 784*degree);
        JMPN = (FieldElement *)((uint8_t *)pAddress + 792*degree);
        JMPC = (FieldElement *)((uint8_t *)pAddress + 800*degree);
        setRR = (FieldElement *)((uint8_t *)pAddress + 808*degree);
        setHASHPOS = (FieldElement *)((uint8_t *)pAddress + 816*degree);
        offset = (FieldElement *)((uint8_t *)pAddress + 824*degree);
        incStack = (FieldElement *)((uint8_t *)pAddress + 832*degree);
        incCode = (FieldElement *)((uint8_t *)pAddress + 840*degree);
        isStack = (FieldElement *)((uint8_t *)pAddress + 848*degree);
        isCode = (FieldElement *)((uint8_t *)pAddress + 856*degree);
        isMem = (FieldElement *)((uint8_t *)pAddress + 864*degree);
        ind = (FieldElement *)((uint8_t *)pAddress + 872*degree);
        indRR = (FieldElement *)((uint8_t *)pAddress + 880*degree);
        useCTX = (FieldElement *)((uint8_t *)pAddress + 888*degree);
        carry = (FieldElement *)((uint8_t *)pAddress + 896*degree);
        mOp = (FieldElement *)((uint8_t *)pAddress + 904*degree);
        mWR = (FieldElement *)((uint8_t *)pAddress + 912*degree);
        sWR = (FieldElement *)((uint8_t *)pAddress + 920*degree);
        sRD = (FieldElement *)((uint8_t *)pAddress + 928*degree);
        arith = (FieldElement *)((uint8_t *)pAddress + 936*degree);
        arithEq0 = (FieldElement *)((uint8_t *)pAddress + 944*degree);
        arithEq1 = (FieldElement *)((uint8_t *)pAddress + 952*degree);
        arithEq2 = (FieldElement *)((uint8_t *)pAddress + 960*degree);
        arithEq3 = (FieldElement *)((uint8_t *)pAddress + 968*degree);
        memAlign = (FieldElement *)((uint8_t *)pAddress + 976*degree);
        memAlignWR = (FieldElement *)((uint8_t *)pAddress + 984*degree);
        memAlignWR8 = (FieldElement *)((uint8_t *)pAddress + 992*degree);
        hashK = (FieldElement *)((uint8_t *)pAddress + 1000*degree);
        hashKLen = (FieldElement *)((uint8_t *)pAddress + 1008*degree);
        hashKDigest = (FieldElement *)((uint8_t *)pAddress + 1016*degree);
        hashP = (FieldElement *)((uint8_t *)pAddress + 1024*degree);
        hashPLen = (FieldElement *)((uint8_t *)pAddress + 1032*degree);
        hashPDigest = (FieldElement *)((uint8_t *)pAddress + 1040*degree);
        bin = (FieldElement *)((uint8_t *)pAddress + 1048*degree);
        binOpcode = (FieldElement *)((uint8_t *)pAddress + 1056*degree);
        assert = (FieldElement *)((uint8_t *)pAddress + 1064*degree);
        isNeg = (FieldElement *)((uint8_t *)pAddress + 1072*degree);
        isMaxMem = (FieldElement *)((uint8_t *)pAddress + 1080*degree);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 1088*degree);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 1096*degree);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 1104*degree);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 1112*degree);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 1120*degree);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 1128*degree);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 1136*degree);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 1144*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 1152; }
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

    static uint64_t size (void) { return 10385096704; }
};

#endif // COMMIT_POLS_HPP
