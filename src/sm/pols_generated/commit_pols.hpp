#ifndef COMMIT_POLS_HPP
#define COMMIT_POLS_HPP

#include <cstdint>
#include "goldilocks/goldilocks_base_field.hpp"

class CommitGeneratedPol
{
private:
    Goldilocks::Element * pData;
public:
    CommitGeneratedPol() : pData(NULL) {};
    Goldilocks::Element & operator[](int i) { return pData[i*636]; };
    Goldilocks::Element * operator=(Goldilocks::Element * pAddress) { pData = pAddress; return pData; };
};

class Byte4CommitPols
{
public:
    CommitGeneratedPol freeIN;
    CommitGeneratedPol out;

    Byte4CommitPols (void * pAddress)
    {
        freeIN = (Goldilocks::Element *)((uint8_t *)pAddress + 0);
        out = (Goldilocks::Element *)((uint8_t *)pAddress + 8);
    }

    Byte4CommitPols (void * pAddress, uint64_t degree)
    {
        freeIN = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        out = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 16; }
};

class MemAlignCommitPols
{
public:
    CommitGeneratedPol inM[2];
    CommitGeneratedPol inV;
    CommitGeneratedPol wr256;
    CommitGeneratedPol wr8;
    CommitGeneratedPol m0[8];
    CommitGeneratedPol m1[8];
    CommitGeneratedPol w0[8];
    CommitGeneratedPol w1[8];
    CommitGeneratedPol v[8];
    CommitGeneratedPol selM1;
    CommitGeneratedPol factorV[8];
    CommitGeneratedPol offset;

    MemAlignCommitPols (void * pAddress)
    {
        inM[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 16);
        inM[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 24);
        inV = (Goldilocks::Element *)((uint8_t *)pAddress + 32);
        wr256 = (Goldilocks::Element *)((uint8_t *)pAddress + 40);
        wr8 = (Goldilocks::Element *)((uint8_t *)pAddress + 48);
        m0[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 56);
        m0[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 64);
        m0[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 72);
        m0[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 80);
        m0[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 88);
        m0[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 96);
        m0[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 104);
        m0[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 112);
        m1[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 120);
        m1[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 128);
        m1[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 136);
        m1[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 144);
        m1[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 152);
        m1[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 160);
        m1[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 168);
        m1[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 176);
        w0[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 184);
        w0[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 192);
        w0[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 200);
        w0[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 208);
        w0[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 216);
        w0[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 224);
        w0[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 232);
        w0[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 240);
        w1[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 248);
        w1[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 256);
        w1[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 264);
        w1[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 272);
        w1[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 280);
        w1[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 288);
        w1[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 296);
        w1[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 304);
        v[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 312);
        v[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 320);
        v[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 328);
        v[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 336);
        v[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 344);
        v[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 352);
        v[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 360);
        v[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 368);
        selM1 = (Goldilocks::Element *)((uint8_t *)pAddress + 376);
        factorV[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 384);
        factorV[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 392);
        factorV[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 400);
        factorV[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 408);
        factorV[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 416);
        factorV[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 424);
        factorV[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 432);
        factorV[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 440);
        offset = (Goldilocks::Element *)((uint8_t *)pAddress + 448);
    }

    MemAlignCommitPols (void * pAddress, uint64_t degree)
    {
        inM[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        inM[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        inV = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        wr256 = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        wr8 = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        m0[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        m0[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        m0[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        m0[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        m0[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        m0[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        m0[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        m0[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        m1[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        m1[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        m1[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        m1[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
        m1[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 136*degree);
        m1[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 144*degree);
        m1[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 152*degree);
        m1[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 160*degree);
        w0[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 168*degree);
        w0[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 176*degree);
        w0[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 184*degree);
        w0[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 192*degree);
        w0[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 200*degree);
        w0[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 208*degree);
        w0[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 216*degree);
        w0[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 224*degree);
        w1[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 232*degree);
        w1[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 240*degree);
        w1[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 248*degree);
        w1[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 256*degree);
        w1[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 264*degree);
        w1[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 272*degree);
        w1[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 280*degree);
        w1[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 288*degree);
        v[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 296*degree);
        v[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 304*degree);
        v[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 312*degree);
        v[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 320*degree);
        v[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 328*degree);
        v[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 336*degree);
        v[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 344*degree);
        v[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 352*degree);
        selM1 = (Goldilocks::Element *)((uint8_t *)pAddress + 360*degree);
        factorV[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 368*degree);
        factorV[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 376*degree);
        factorV[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 384*degree);
        factorV[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 392*degree);
        factorV[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 400*degree);
        factorV[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 408*degree);
        factorV[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 416*degree);
        factorV[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 424*degree);
        offset = (Goldilocks::Element *)((uint8_t *)pAddress + 432*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 440; }
};

class ArithCommitPols
{
public:
    CommitGeneratedPol x1[16];
    CommitGeneratedPol y1[16];
    CommitGeneratedPol x2[16];
    CommitGeneratedPol y2[16];
    CommitGeneratedPol x3[16];
    CommitGeneratedPol y3[16];
    CommitGeneratedPol s[16];
    CommitGeneratedPol q0[16];
    CommitGeneratedPol q1[16];
    CommitGeneratedPol q2[16];
    CommitGeneratedPol selEq[4];
    CommitGeneratedPol carryL[3];
    CommitGeneratedPol carryH[3];

    ArithCommitPols (void * pAddress)
    {
        x1[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 456);
        x1[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 464);
        x1[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 472);
        x1[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 480);
        x1[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 488);
        x1[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 496);
        x1[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 504);
        x1[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 512);
        x1[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 520);
        x1[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 528);
        x1[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 536);
        x1[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 544);
        x1[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 552);
        x1[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 560);
        x1[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 568);
        x1[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 576);
        y1[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 584);
        y1[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 592);
        y1[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 600);
        y1[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 608);
        y1[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 616);
        y1[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 624);
        y1[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 632);
        y1[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 640);
        y1[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 648);
        y1[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 656);
        y1[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 664);
        y1[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 672);
        y1[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 680);
        y1[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 688);
        y1[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 696);
        y1[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 704);
        x2[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 712);
        x2[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 720);
        x2[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 728);
        x2[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 736);
        x2[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 744);
        x2[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 752);
        x2[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 760);
        x2[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 768);
        x2[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 776);
        x2[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 784);
        x2[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 792);
        x2[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 800);
        x2[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 808);
        x2[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 816);
        x2[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 824);
        x2[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 832);
        y2[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 840);
        y2[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 848);
        y2[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 856);
        y2[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 864);
        y2[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 872);
        y2[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 880);
        y2[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 888);
        y2[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 896);
        y2[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 904);
        y2[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 912);
        y2[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 920);
        y2[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 928);
        y2[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 936);
        y2[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 944);
        y2[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 952);
        y2[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 960);
        x3[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 968);
        x3[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 976);
        x3[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 984);
        x3[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 992);
        x3[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1000);
        x3[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1008);
        x3[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1016);
        x3[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1024);
        x3[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 1032);
        x3[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 1040);
        x3[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 1048);
        x3[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 1056);
        x3[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 1064);
        x3[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 1072);
        x3[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 1080);
        x3[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 1088);
        y3[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1096);
        y3[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1104);
        y3[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1112);
        y3[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1120);
        y3[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1128);
        y3[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1136);
        y3[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1144);
        y3[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1152);
        y3[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 1160);
        y3[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 1168);
        y3[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 1176);
        y3[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 1184);
        y3[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 1192);
        y3[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 1200);
        y3[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 1208);
        y3[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 1216);
        s[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1224);
        s[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1232);
        s[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1240);
        s[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1248);
        s[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1256);
        s[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1264);
        s[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1272);
        s[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1280);
        s[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 1288);
        s[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 1296);
        s[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 1304);
        s[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 1312);
        s[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 1320);
        s[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 1328);
        s[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 1336);
        s[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 1344);
        q0[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1352);
        q0[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1360);
        q0[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1368);
        q0[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1376);
        q0[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1384);
        q0[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1392);
        q0[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1400);
        q0[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1408);
        q0[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 1416);
        q0[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 1424);
        q0[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 1432);
        q0[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 1440);
        q0[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 1448);
        q0[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 1456);
        q0[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 1464);
        q0[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 1472);
        q1[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1480);
        q1[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1488);
        q1[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1496);
        q1[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1504);
        q1[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1512);
        q1[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1520);
        q1[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1528);
        q1[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1536);
        q1[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 1544);
        q1[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 1552);
        q1[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 1560);
        q1[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 1568);
        q1[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 1576);
        q1[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 1584);
        q1[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 1592);
        q1[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 1600);
        q2[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1608);
        q2[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1616);
        q2[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1624);
        q2[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1632);
        q2[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1640);
        q2[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1648);
        q2[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1656);
        q2[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1664);
        q2[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 1672);
        q2[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 1680);
        q2[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 1688);
        q2[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 1696);
        q2[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 1704);
        q2[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 1712);
        q2[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 1720);
        q2[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 1728);
        selEq[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1736);
        selEq[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1744);
        selEq[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1752);
        selEq[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1760);
        carryL[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1768);
        carryL[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1776);
        carryL[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1784);
        carryH[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1792);
        carryH[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1800);
        carryH[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1808);
    }

    ArithCommitPols (void * pAddress, uint64_t degree)
    {
        x1[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        x1[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        x1[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        x1[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        x1[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        x1[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        x1[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        x1[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        x1[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        x1[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        x1[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        x1[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        x1[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        x1[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        x1[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        x1[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        y1[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
        y1[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 136*degree);
        y1[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 144*degree);
        y1[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 152*degree);
        y1[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 160*degree);
        y1[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 168*degree);
        y1[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 176*degree);
        y1[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 184*degree);
        y1[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 192*degree);
        y1[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 200*degree);
        y1[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 208*degree);
        y1[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 216*degree);
        y1[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 224*degree);
        y1[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 232*degree);
        y1[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 240*degree);
        y1[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 248*degree);
        x2[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 256*degree);
        x2[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 264*degree);
        x2[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 272*degree);
        x2[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 280*degree);
        x2[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 288*degree);
        x2[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 296*degree);
        x2[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 304*degree);
        x2[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 312*degree);
        x2[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 320*degree);
        x2[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 328*degree);
        x2[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 336*degree);
        x2[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 344*degree);
        x2[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 352*degree);
        x2[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 360*degree);
        x2[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 368*degree);
        x2[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 376*degree);
        y2[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 384*degree);
        y2[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 392*degree);
        y2[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 400*degree);
        y2[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 408*degree);
        y2[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 416*degree);
        y2[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 424*degree);
        y2[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 432*degree);
        y2[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 440*degree);
        y2[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 448*degree);
        y2[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 456*degree);
        y2[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 464*degree);
        y2[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 472*degree);
        y2[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 480*degree);
        y2[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 488*degree);
        y2[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 496*degree);
        y2[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 504*degree);
        x3[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 512*degree);
        x3[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 520*degree);
        x3[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 528*degree);
        x3[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 536*degree);
        x3[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 544*degree);
        x3[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 552*degree);
        x3[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 560*degree);
        x3[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 568*degree);
        x3[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 576*degree);
        x3[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 584*degree);
        x3[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 592*degree);
        x3[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 600*degree);
        x3[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 608*degree);
        x3[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 616*degree);
        x3[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 624*degree);
        x3[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 632*degree);
        y3[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 640*degree);
        y3[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 648*degree);
        y3[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 656*degree);
        y3[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 664*degree);
        y3[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 672*degree);
        y3[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 680*degree);
        y3[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 688*degree);
        y3[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 696*degree);
        y3[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 704*degree);
        y3[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 712*degree);
        y3[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 720*degree);
        y3[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 728*degree);
        y3[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 736*degree);
        y3[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 744*degree);
        y3[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 752*degree);
        y3[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 760*degree);
        s[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 768*degree);
        s[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 776*degree);
        s[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 784*degree);
        s[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 792*degree);
        s[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 800*degree);
        s[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 808*degree);
        s[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 816*degree);
        s[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 824*degree);
        s[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 832*degree);
        s[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 840*degree);
        s[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 848*degree);
        s[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 856*degree);
        s[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 864*degree);
        s[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 872*degree);
        s[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 880*degree);
        s[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 888*degree);
        q0[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 896*degree);
        q0[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 904*degree);
        q0[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 912*degree);
        q0[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 920*degree);
        q0[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 928*degree);
        q0[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 936*degree);
        q0[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 944*degree);
        q0[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 952*degree);
        q0[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 960*degree);
        q0[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 968*degree);
        q0[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 976*degree);
        q0[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 984*degree);
        q0[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 992*degree);
        q0[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 1000*degree);
        q0[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 1008*degree);
        q0[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 1016*degree);
        q1[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1024*degree);
        q1[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1032*degree);
        q1[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1040*degree);
        q1[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1048*degree);
        q1[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1056*degree);
        q1[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1064*degree);
        q1[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1072*degree);
        q1[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1080*degree);
        q1[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 1088*degree);
        q1[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 1096*degree);
        q1[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 1104*degree);
        q1[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 1112*degree);
        q1[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 1120*degree);
        q1[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 1128*degree);
        q1[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 1136*degree);
        q1[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 1144*degree);
        q2[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1152*degree);
        q2[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1160*degree);
        q2[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1168*degree);
        q2[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1176*degree);
        q2[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1184*degree);
        q2[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1192*degree);
        q2[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1200*degree);
        q2[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1208*degree);
        q2[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 1216*degree);
        q2[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 1224*degree);
        q2[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 1232*degree);
        q2[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 1240*degree);
        q2[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 1248*degree);
        q2[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 1256*degree);
        q2[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 1264*degree);
        q2[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 1272*degree);
        selEq[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1280*degree);
        selEq[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1288*degree);
        selEq[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1296*degree);
        selEq[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1304*degree);
        carryL[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1312*degree);
        carryL[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1320*degree);
        carryL[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1328*degree);
        carryH[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1336*degree);
        carryH[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1344*degree);
        carryH[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1352*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 1360; }
};

class BinaryCommitPols
{
public:
    CommitGeneratedPol freeInA;
    CommitGeneratedPol freeInB;
    CommitGeneratedPol freeInC;
    CommitGeneratedPol a0;
    CommitGeneratedPol a1;
    CommitGeneratedPol a2;
    CommitGeneratedPol a3;
    CommitGeneratedPol a4;
    CommitGeneratedPol a5;
    CommitGeneratedPol a6;
    CommitGeneratedPol a7;
    CommitGeneratedPol b0;
    CommitGeneratedPol b1;
    CommitGeneratedPol b2;
    CommitGeneratedPol b3;
    CommitGeneratedPol b4;
    CommitGeneratedPol b5;
    CommitGeneratedPol b6;
    CommitGeneratedPol b7;
    CommitGeneratedPol c0;
    CommitGeneratedPol c1;
    CommitGeneratedPol c2;
    CommitGeneratedPol c3;
    CommitGeneratedPol c4;
    CommitGeneratedPol c5;
    CommitGeneratedPol c6;
    CommitGeneratedPol c7;
    CommitGeneratedPol opcode;
    CommitGeneratedPol cIn;
    CommitGeneratedPol cOut;
    CommitGeneratedPol lCout;
    CommitGeneratedPol lOpcode;
    CommitGeneratedPol last;
    CommitGeneratedPol useCarry;

    BinaryCommitPols (void * pAddress)
    {
        freeInA = (Goldilocks::Element *)((uint8_t *)pAddress + 1816);
        freeInB = (Goldilocks::Element *)((uint8_t *)pAddress + 1824);
        freeInC = (Goldilocks::Element *)((uint8_t *)pAddress + 1832);
        a0 = (Goldilocks::Element *)((uint8_t *)pAddress + 1840);
        a1 = (Goldilocks::Element *)((uint8_t *)pAddress + 1848);
        a2 = (Goldilocks::Element *)((uint8_t *)pAddress + 1856);
        a3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1864);
        a4 = (Goldilocks::Element *)((uint8_t *)pAddress + 1872);
        a5 = (Goldilocks::Element *)((uint8_t *)pAddress + 1880);
        a6 = (Goldilocks::Element *)((uint8_t *)pAddress + 1888);
        a7 = (Goldilocks::Element *)((uint8_t *)pAddress + 1896);
        b0 = (Goldilocks::Element *)((uint8_t *)pAddress + 1904);
        b1 = (Goldilocks::Element *)((uint8_t *)pAddress + 1912);
        b2 = (Goldilocks::Element *)((uint8_t *)pAddress + 1920);
        b3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1928);
        b4 = (Goldilocks::Element *)((uint8_t *)pAddress + 1936);
        b5 = (Goldilocks::Element *)((uint8_t *)pAddress + 1944);
        b6 = (Goldilocks::Element *)((uint8_t *)pAddress + 1952);
        b7 = (Goldilocks::Element *)((uint8_t *)pAddress + 1960);
        c0 = (Goldilocks::Element *)((uint8_t *)pAddress + 1968);
        c1 = (Goldilocks::Element *)((uint8_t *)pAddress + 1976);
        c2 = (Goldilocks::Element *)((uint8_t *)pAddress + 1984);
        c3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1992);
        c4 = (Goldilocks::Element *)((uint8_t *)pAddress + 2000);
        c5 = (Goldilocks::Element *)((uint8_t *)pAddress + 2008);
        c6 = (Goldilocks::Element *)((uint8_t *)pAddress + 2016);
        c7 = (Goldilocks::Element *)((uint8_t *)pAddress + 2024);
        opcode = (Goldilocks::Element *)((uint8_t *)pAddress + 2032);
        cIn = (Goldilocks::Element *)((uint8_t *)pAddress + 2040);
        cOut = (Goldilocks::Element *)((uint8_t *)pAddress + 2048);
        lCout = (Goldilocks::Element *)((uint8_t *)pAddress + 2056);
        lOpcode = (Goldilocks::Element *)((uint8_t *)pAddress + 2064);
        last = (Goldilocks::Element *)((uint8_t *)pAddress + 2072);
        useCarry = (Goldilocks::Element *)((uint8_t *)pAddress + 2080);
    }

    BinaryCommitPols (void * pAddress, uint64_t degree)
    {
        freeInA = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        freeInB = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        freeInC = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        a0 = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        a1 = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        a2 = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        a3 = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        a4 = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        a5 = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        a6 = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        a7 = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        b0 = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        b1 = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        b2 = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        b3 = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        b4 = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        b5 = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
        b6 = (Goldilocks::Element *)((uint8_t *)pAddress + 136*degree);
        b7 = (Goldilocks::Element *)((uint8_t *)pAddress + 144*degree);
        c0 = (Goldilocks::Element *)((uint8_t *)pAddress + 152*degree);
        c1 = (Goldilocks::Element *)((uint8_t *)pAddress + 160*degree);
        c2 = (Goldilocks::Element *)((uint8_t *)pAddress + 168*degree);
        c3 = (Goldilocks::Element *)((uint8_t *)pAddress + 176*degree);
        c4 = (Goldilocks::Element *)((uint8_t *)pAddress + 184*degree);
        c5 = (Goldilocks::Element *)((uint8_t *)pAddress + 192*degree);
        c6 = (Goldilocks::Element *)((uint8_t *)pAddress + 200*degree);
        c7 = (Goldilocks::Element *)((uint8_t *)pAddress + 208*degree);
        opcode = (Goldilocks::Element *)((uint8_t *)pAddress + 216*degree);
        cIn = (Goldilocks::Element *)((uint8_t *)pAddress + 224*degree);
        cOut = (Goldilocks::Element *)((uint8_t *)pAddress + 232*degree);
        lCout = (Goldilocks::Element *)((uint8_t *)pAddress + 240*degree);
        lOpcode = (Goldilocks::Element *)((uint8_t *)pAddress + 248*degree);
        last = (Goldilocks::Element *)((uint8_t *)pAddress + 256*degree);
        useCarry = (Goldilocks::Element *)((uint8_t *)pAddress + 264*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 272; }
};

class PoseidonGCommitPols
{
public:
    CommitGeneratedPol in0;
    CommitGeneratedPol in1;
    CommitGeneratedPol in2;
    CommitGeneratedPol in3;
    CommitGeneratedPol in4;
    CommitGeneratedPol in5;
    CommitGeneratedPol in6;
    CommitGeneratedPol in7;
    CommitGeneratedPol hashType;
    CommitGeneratedPol cap1;
    CommitGeneratedPol cap2;
    CommitGeneratedPol cap3;
    CommitGeneratedPol hash0;
    CommitGeneratedPol hash1;
    CommitGeneratedPol hash2;
    CommitGeneratedPol hash3;

    PoseidonGCommitPols (void * pAddress)
    {
        in0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2088);
        in1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2096);
        in2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2104);
        in3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2112);
        in4 = (Goldilocks::Element *)((uint8_t *)pAddress + 2120);
        in5 = (Goldilocks::Element *)((uint8_t *)pAddress + 2128);
        in6 = (Goldilocks::Element *)((uint8_t *)pAddress + 2136);
        in7 = (Goldilocks::Element *)((uint8_t *)pAddress + 2144);
        hashType = (Goldilocks::Element *)((uint8_t *)pAddress + 2152);
        cap1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2160);
        cap2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2168);
        cap3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2176);
        hash0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2184);
        hash1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2192);
        hash2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2200);
        hash3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2208);
    }

    PoseidonGCommitPols (void * pAddress, uint64_t degree)
    {
        in0 = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        in1 = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        in2 = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        in3 = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        in4 = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        in5 = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        in6 = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        in7 = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        hashType = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        cap1 = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        cap2 = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        cap3 = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        hash0 = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        hash1 = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        hash2 = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        hash3 = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 128; }
};

class PaddingPGCommitPols
{
public:
    CommitGeneratedPol acc[8];
    CommitGeneratedPol freeIn;
    CommitGeneratedPol addr;
    CommitGeneratedPol rem;
    CommitGeneratedPol remInv;
    CommitGeneratedPol spare;
    CommitGeneratedPol firstHash;
    CommitGeneratedPol curHash0;
    CommitGeneratedPol curHash1;
    CommitGeneratedPol curHash2;
    CommitGeneratedPol curHash3;
    CommitGeneratedPol prevHash0;
    CommitGeneratedPol prevHash1;
    CommitGeneratedPol prevHash2;
    CommitGeneratedPol prevHash3;
    CommitGeneratedPol incCounter;
    CommitGeneratedPol len;
    CommitGeneratedPol crOffset;
    CommitGeneratedPol crLen;
    CommitGeneratedPol crOffsetInv;
    CommitGeneratedPol crF0;
    CommitGeneratedPol crF1;
    CommitGeneratedPol crF2;
    CommitGeneratedPol crF3;
    CommitGeneratedPol crF4;
    CommitGeneratedPol crF5;
    CommitGeneratedPol crF6;
    CommitGeneratedPol crF7;
    CommitGeneratedPol crV0;
    CommitGeneratedPol crV1;
    CommitGeneratedPol crV2;
    CommitGeneratedPol crV3;
    CommitGeneratedPol crV4;
    CommitGeneratedPol crV5;
    CommitGeneratedPol crV6;
    CommitGeneratedPol crV7;

    PaddingPGCommitPols (void * pAddress)
    {
        acc[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 2216);
        acc[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 2224);
        acc[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 2232);
        acc[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 2240);
        acc[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 2248);
        acc[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 2256);
        acc[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 2264);
        acc[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 2272);
        freeIn = (Goldilocks::Element *)((uint8_t *)pAddress + 2280);
        addr = (Goldilocks::Element *)((uint8_t *)pAddress + 2288);
        rem = (Goldilocks::Element *)((uint8_t *)pAddress + 2296);
        remInv = (Goldilocks::Element *)((uint8_t *)pAddress + 2304);
        spare = (Goldilocks::Element *)((uint8_t *)pAddress + 2312);
        firstHash = (Goldilocks::Element *)((uint8_t *)pAddress + 2320);
        curHash0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2328);
        curHash1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2336);
        curHash2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2344);
        curHash3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2352);
        prevHash0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2360);
        prevHash1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2368);
        prevHash2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2376);
        prevHash3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2384);
        incCounter = (Goldilocks::Element *)((uint8_t *)pAddress + 2392);
        len = (Goldilocks::Element *)((uint8_t *)pAddress + 2400);
        crOffset = (Goldilocks::Element *)((uint8_t *)pAddress + 2408);
        crLen = (Goldilocks::Element *)((uint8_t *)pAddress + 2416);
        crOffsetInv = (Goldilocks::Element *)((uint8_t *)pAddress + 2424);
        crF0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2432);
        crF1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2440);
        crF2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2448);
        crF3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2456);
        crF4 = (Goldilocks::Element *)((uint8_t *)pAddress + 2464);
        crF5 = (Goldilocks::Element *)((uint8_t *)pAddress + 2472);
        crF6 = (Goldilocks::Element *)((uint8_t *)pAddress + 2480);
        crF7 = (Goldilocks::Element *)((uint8_t *)pAddress + 2488);
        crV0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2496);
        crV1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2504);
        crV2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2512);
        crV3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2520);
        crV4 = (Goldilocks::Element *)((uint8_t *)pAddress + 2528);
        crV5 = (Goldilocks::Element *)((uint8_t *)pAddress + 2536);
        crV6 = (Goldilocks::Element *)((uint8_t *)pAddress + 2544);
        crV7 = (Goldilocks::Element *)((uint8_t *)pAddress + 2552);
    }

    PaddingPGCommitPols (void * pAddress, uint64_t degree)
    {
        acc[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        acc[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        acc[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        acc[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        acc[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        acc[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        acc[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        acc[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        freeIn = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        addr = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        rem = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        remInv = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        spare = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        firstHash = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        curHash0 = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        curHash1 = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        curHash2 = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
        curHash3 = (Goldilocks::Element *)((uint8_t *)pAddress + 136*degree);
        prevHash0 = (Goldilocks::Element *)((uint8_t *)pAddress + 144*degree);
        prevHash1 = (Goldilocks::Element *)((uint8_t *)pAddress + 152*degree);
        prevHash2 = (Goldilocks::Element *)((uint8_t *)pAddress + 160*degree);
        prevHash3 = (Goldilocks::Element *)((uint8_t *)pAddress + 168*degree);
        incCounter = (Goldilocks::Element *)((uint8_t *)pAddress + 176*degree);
        len = (Goldilocks::Element *)((uint8_t *)pAddress + 184*degree);
        crOffset = (Goldilocks::Element *)((uint8_t *)pAddress + 192*degree);
        crLen = (Goldilocks::Element *)((uint8_t *)pAddress + 200*degree);
        crOffsetInv = (Goldilocks::Element *)((uint8_t *)pAddress + 208*degree);
        crF0 = (Goldilocks::Element *)((uint8_t *)pAddress + 216*degree);
        crF1 = (Goldilocks::Element *)((uint8_t *)pAddress + 224*degree);
        crF2 = (Goldilocks::Element *)((uint8_t *)pAddress + 232*degree);
        crF3 = (Goldilocks::Element *)((uint8_t *)pAddress + 240*degree);
        crF4 = (Goldilocks::Element *)((uint8_t *)pAddress + 248*degree);
        crF5 = (Goldilocks::Element *)((uint8_t *)pAddress + 256*degree);
        crF6 = (Goldilocks::Element *)((uint8_t *)pAddress + 264*degree);
        crF7 = (Goldilocks::Element *)((uint8_t *)pAddress + 272*degree);
        crV0 = (Goldilocks::Element *)((uint8_t *)pAddress + 280*degree);
        crV1 = (Goldilocks::Element *)((uint8_t *)pAddress + 288*degree);
        crV2 = (Goldilocks::Element *)((uint8_t *)pAddress + 296*degree);
        crV3 = (Goldilocks::Element *)((uint8_t *)pAddress + 304*degree);
        crV4 = (Goldilocks::Element *)((uint8_t *)pAddress + 312*degree);
        crV5 = (Goldilocks::Element *)((uint8_t *)pAddress + 320*degree);
        crV6 = (Goldilocks::Element *)((uint8_t *)pAddress + 328*degree);
        crV7 = (Goldilocks::Element *)((uint8_t *)pAddress + 336*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 344; }
};

class StorageCommitPols
{
public:
    CommitGeneratedPol free0;
    CommitGeneratedPol free1;
    CommitGeneratedPol free2;
    CommitGeneratedPol free3;
    CommitGeneratedPol hashLeft0;
    CommitGeneratedPol hashLeft1;
    CommitGeneratedPol hashLeft2;
    CommitGeneratedPol hashLeft3;
    CommitGeneratedPol hashRight0;
    CommitGeneratedPol hashRight1;
    CommitGeneratedPol hashRight2;
    CommitGeneratedPol hashRight3;
    CommitGeneratedPol oldRoot0;
    CommitGeneratedPol oldRoot1;
    CommitGeneratedPol oldRoot2;
    CommitGeneratedPol oldRoot3;
    CommitGeneratedPol newRoot0;
    CommitGeneratedPol newRoot1;
    CommitGeneratedPol newRoot2;
    CommitGeneratedPol newRoot3;
    CommitGeneratedPol valueLow0;
    CommitGeneratedPol valueLow1;
    CommitGeneratedPol valueLow2;
    CommitGeneratedPol valueLow3;
    CommitGeneratedPol valueHigh0;
    CommitGeneratedPol valueHigh1;
    CommitGeneratedPol valueHigh2;
    CommitGeneratedPol valueHigh3;
    CommitGeneratedPol siblingValueHash0;
    CommitGeneratedPol siblingValueHash1;
    CommitGeneratedPol siblingValueHash2;
    CommitGeneratedPol siblingValueHash3;
    CommitGeneratedPol rkey0;
    CommitGeneratedPol rkey1;
    CommitGeneratedPol rkey2;
    CommitGeneratedPol rkey3;
    CommitGeneratedPol siblingRkey0;
    CommitGeneratedPol siblingRkey1;
    CommitGeneratedPol siblingRkey2;
    CommitGeneratedPol siblingRkey3;
    CommitGeneratedPol rkeyBit;
    CommitGeneratedPol level0;
    CommitGeneratedPol level1;
    CommitGeneratedPol level2;
    CommitGeneratedPol level3;
    CommitGeneratedPol pc;
    CommitGeneratedPol selOldRoot;
    CommitGeneratedPol selNewRoot;
    CommitGeneratedPol selValueLow;
    CommitGeneratedPol selValueHigh;
    CommitGeneratedPol selSiblingValueHash;
    CommitGeneratedPol selRkey;
    CommitGeneratedPol selRkeyBit;
    CommitGeneratedPol selSiblingRkey;
    CommitGeneratedPol selFree;
    CommitGeneratedPol setHashLeft;
    CommitGeneratedPol setHashRight;
    CommitGeneratedPol setOldRoot;
    CommitGeneratedPol setNewRoot;
    CommitGeneratedPol setValueLow;
    CommitGeneratedPol setValueHigh;
    CommitGeneratedPol setSiblingValueHash;
    CommitGeneratedPol setRkey;
    CommitGeneratedPol setSiblingRkey;
    CommitGeneratedPol setRkeyBit;
    CommitGeneratedPol setLevel;
    CommitGeneratedPol iHash;
    CommitGeneratedPol iHashType;
    CommitGeneratedPol iLatchSet;
    CommitGeneratedPol iLatchGet;
    CommitGeneratedPol iClimbRkey;
    CommitGeneratedPol iClimbSiblingRkey;
    CommitGeneratedPol iClimbSiblingRkeyN;
    CommitGeneratedPol iRotateLevel;
    CommitGeneratedPol iJmpz;
    CommitGeneratedPol iJmp;
    CommitGeneratedPol iConst0;
    CommitGeneratedPol iConst1;
    CommitGeneratedPol iConst2;
    CommitGeneratedPol iConst3;
    CommitGeneratedPol iAddress;
    CommitGeneratedPol incCounter;
    CommitGeneratedPol op0inv;

    StorageCommitPols (void * pAddress)
    {
        free0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2560);
        free1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2568);
        free2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2576);
        free3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2584);
        hashLeft0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2592);
        hashLeft1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2600);
        hashLeft2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2608);
        hashLeft3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2616);
        hashRight0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2624);
        hashRight1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2632);
        hashRight2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2640);
        hashRight3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2648);
        oldRoot0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2656);
        oldRoot1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2664);
        oldRoot2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2672);
        oldRoot3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2680);
        newRoot0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2688);
        newRoot1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2696);
        newRoot2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2704);
        newRoot3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2712);
        valueLow0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2720);
        valueLow1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2728);
        valueLow2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2736);
        valueLow3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2744);
        valueHigh0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2752);
        valueHigh1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2760);
        valueHigh2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2768);
        valueHigh3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2776);
        siblingValueHash0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2784);
        siblingValueHash1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2792);
        siblingValueHash2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2800);
        siblingValueHash3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2808);
        rkey0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2816);
        rkey1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2824);
        rkey2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2832);
        rkey3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2840);
        siblingRkey0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2848);
        siblingRkey1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2856);
        siblingRkey2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2864);
        siblingRkey3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2872);
        rkeyBit = (Goldilocks::Element *)((uint8_t *)pAddress + 2880);
        level0 = (Goldilocks::Element *)((uint8_t *)pAddress + 2888);
        level1 = (Goldilocks::Element *)((uint8_t *)pAddress + 2896);
        level2 = (Goldilocks::Element *)((uint8_t *)pAddress + 2904);
        level3 = (Goldilocks::Element *)((uint8_t *)pAddress + 2912);
        pc = (Goldilocks::Element *)((uint8_t *)pAddress + 2920);
        selOldRoot = (Goldilocks::Element *)((uint8_t *)pAddress + 2928);
        selNewRoot = (Goldilocks::Element *)((uint8_t *)pAddress + 2936);
        selValueLow = (Goldilocks::Element *)((uint8_t *)pAddress + 2944);
        selValueHigh = (Goldilocks::Element *)((uint8_t *)pAddress + 2952);
        selSiblingValueHash = (Goldilocks::Element *)((uint8_t *)pAddress + 2960);
        selRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 2968);
        selRkeyBit = (Goldilocks::Element *)((uint8_t *)pAddress + 2976);
        selSiblingRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 2984);
        selFree = (Goldilocks::Element *)((uint8_t *)pAddress + 2992);
        setHashLeft = (Goldilocks::Element *)((uint8_t *)pAddress + 3000);
        setHashRight = (Goldilocks::Element *)((uint8_t *)pAddress + 3008);
        setOldRoot = (Goldilocks::Element *)((uint8_t *)pAddress + 3016);
        setNewRoot = (Goldilocks::Element *)((uint8_t *)pAddress + 3024);
        setValueLow = (Goldilocks::Element *)((uint8_t *)pAddress + 3032);
        setValueHigh = (Goldilocks::Element *)((uint8_t *)pAddress + 3040);
        setSiblingValueHash = (Goldilocks::Element *)((uint8_t *)pAddress + 3048);
        setRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 3056);
        setSiblingRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 3064);
        setRkeyBit = (Goldilocks::Element *)((uint8_t *)pAddress + 3072);
        setLevel = (Goldilocks::Element *)((uint8_t *)pAddress + 3080);
        iHash = (Goldilocks::Element *)((uint8_t *)pAddress + 3088);
        iHashType = (Goldilocks::Element *)((uint8_t *)pAddress + 3096);
        iLatchSet = (Goldilocks::Element *)((uint8_t *)pAddress + 3104);
        iLatchGet = (Goldilocks::Element *)((uint8_t *)pAddress + 3112);
        iClimbRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 3120);
        iClimbSiblingRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 3128);
        iClimbSiblingRkeyN = (Goldilocks::Element *)((uint8_t *)pAddress + 3136);
        iRotateLevel = (Goldilocks::Element *)((uint8_t *)pAddress + 3144);
        iJmpz = (Goldilocks::Element *)((uint8_t *)pAddress + 3152);
        iJmp = (Goldilocks::Element *)((uint8_t *)pAddress + 3160);
        iConst0 = (Goldilocks::Element *)((uint8_t *)pAddress + 3168);
        iConst1 = (Goldilocks::Element *)((uint8_t *)pAddress + 3176);
        iConst2 = (Goldilocks::Element *)((uint8_t *)pAddress + 3184);
        iConst3 = (Goldilocks::Element *)((uint8_t *)pAddress + 3192);
        iAddress = (Goldilocks::Element *)((uint8_t *)pAddress + 3200);
        incCounter = (Goldilocks::Element *)((uint8_t *)pAddress + 3208);
        op0inv = (Goldilocks::Element *)((uint8_t *)pAddress + 3216);
    }

    StorageCommitPols (void * pAddress, uint64_t degree)
    {
        free0 = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        free1 = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        free2 = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        free3 = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        hashLeft0 = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        hashLeft1 = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        hashLeft2 = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        hashLeft3 = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        hashRight0 = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        hashRight1 = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        hashRight2 = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        hashRight3 = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        oldRoot0 = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        oldRoot1 = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        oldRoot2 = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        oldRoot3 = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        newRoot0 = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
        newRoot1 = (Goldilocks::Element *)((uint8_t *)pAddress + 136*degree);
        newRoot2 = (Goldilocks::Element *)((uint8_t *)pAddress + 144*degree);
        newRoot3 = (Goldilocks::Element *)((uint8_t *)pAddress + 152*degree);
        valueLow0 = (Goldilocks::Element *)((uint8_t *)pAddress + 160*degree);
        valueLow1 = (Goldilocks::Element *)((uint8_t *)pAddress + 168*degree);
        valueLow2 = (Goldilocks::Element *)((uint8_t *)pAddress + 176*degree);
        valueLow3 = (Goldilocks::Element *)((uint8_t *)pAddress + 184*degree);
        valueHigh0 = (Goldilocks::Element *)((uint8_t *)pAddress + 192*degree);
        valueHigh1 = (Goldilocks::Element *)((uint8_t *)pAddress + 200*degree);
        valueHigh2 = (Goldilocks::Element *)((uint8_t *)pAddress + 208*degree);
        valueHigh3 = (Goldilocks::Element *)((uint8_t *)pAddress + 216*degree);
        siblingValueHash0 = (Goldilocks::Element *)((uint8_t *)pAddress + 224*degree);
        siblingValueHash1 = (Goldilocks::Element *)((uint8_t *)pAddress + 232*degree);
        siblingValueHash2 = (Goldilocks::Element *)((uint8_t *)pAddress + 240*degree);
        siblingValueHash3 = (Goldilocks::Element *)((uint8_t *)pAddress + 248*degree);
        rkey0 = (Goldilocks::Element *)((uint8_t *)pAddress + 256*degree);
        rkey1 = (Goldilocks::Element *)((uint8_t *)pAddress + 264*degree);
        rkey2 = (Goldilocks::Element *)((uint8_t *)pAddress + 272*degree);
        rkey3 = (Goldilocks::Element *)((uint8_t *)pAddress + 280*degree);
        siblingRkey0 = (Goldilocks::Element *)((uint8_t *)pAddress + 288*degree);
        siblingRkey1 = (Goldilocks::Element *)((uint8_t *)pAddress + 296*degree);
        siblingRkey2 = (Goldilocks::Element *)((uint8_t *)pAddress + 304*degree);
        siblingRkey3 = (Goldilocks::Element *)((uint8_t *)pAddress + 312*degree);
        rkeyBit = (Goldilocks::Element *)((uint8_t *)pAddress + 320*degree);
        level0 = (Goldilocks::Element *)((uint8_t *)pAddress + 328*degree);
        level1 = (Goldilocks::Element *)((uint8_t *)pAddress + 336*degree);
        level2 = (Goldilocks::Element *)((uint8_t *)pAddress + 344*degree);
        level3 = (Goldilocks::Element *)((uint8_t *)pAddress + 352*degree);
        pc = (Goldilocks::Element *)((uint8_t *)pAddress + 360*degree);
        selOldRoot = (Goldilocks::Element *)((uint8_t *)pAddress + 368*degree);
        selNewRoot = (Goldilocks::Element *)((uint8_t *)pAddress + 376*degree);
        selValueLow = (Goldilocks::Element *)((uint8_t *)pAddress + 384*degree);
        selValueHigh = (Goldilocks::Element *)((uint8_t *)pAddress + 392*degree);
        selSiblingValueHash = (Goldilocks::Element *)((uint8_t *)pAddress + 400*degree);
        selRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 408*degree);
        selRkeyBit = (Goldilocks::Element *)((uint8_t *)pAddress + 416*degree);
        selSiblingRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 424*degree);
        selFree = (Goldilocks::Element *)((uint8_t *)pAddress + 432*degree);
        setHashLeft = (Goldilocks::Element *)((uint8_t *)pAddress + 440*degree);
        setHashRight = (Goldilocks::Element *)((uint8_t *)pAddress + 448*degree);
        setOldRoot = (Goldilocks::Element *)((uint8_t *)pAddress + 456*degree);
        setNewRoot = (Goldilocks::Element *)((uint8_t *)pAddress + 464*degree);
        setValueLow = (Goldilocks::Element *)((uint8_t *)pAddress + 472*degree);
        setValueHigh = (Goldilocks::Element *)((uint8_t *)pAddress + 480*degree);
        setSiblingValueHash = (Goldilocks::Element *)((uint8_t *)pAddress + 488*degree);
        setRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 496*degree);
        setSiblingRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 504*degree);
        setRkeyBit = (Goldilocks::Element *)((uint8_t *)pAddress + 512*degree);
        setLevel = (Goldilocks::Element *)((uint8_t *)pAddress + 520*degree);
        iHash = (Goldilocks::Element *)((uint8_t *)pAddress + 528*degree);
        iHashType = (Goldilocks::Element *)((uint8_t *)pAddress + 536*degree);
        iLatchSet = (Goldilocks::Element *)((uint8_t *)pAddress + 544*degree);
        iLatchGet = (Goldilocks::Element *)((uint8_t *)pAddress + 552*degree);
        iClimbRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 560*degree);
        iClimbSiblingRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 568*degree);
        iClimbSiblingRkeyN = (Goldilocks::Element *)((uint8_t *)pAddress + 576*degree);
        iRotateLevel = (Goldilocks::Element *)((uint8_t *)pAddress + 584*degree);
        iJmpz = (Goldilocks::Element *)((uint8_t *)pAddress + 592*degree);
        iJmp = (Goldilocks::Element *)((uint8_t *)pAddress + 600*degree);
        iConst0 = (Goldilocks::Element *)((uint8_t *)pAddress + 608*degree);
        iConst1 = (Goldilocks::Element *)((uint8_t *)pAddress + 616*degree);
        iConst2 = (Goldilocks::Element *)((uint8_t *)pAddress + 624*degree);
        iConst3 = (Goldilocks::Element *)((uint8_t *)pAddress + 632*degree);
        iAddress = (Goldilocks::Element *)((uint8_t *)pAddress + 640*degree);
        incCounter = (Goldilocks::Element *)((uint8_t *)pAddress + 648*degree);
        op0inv = (Goldilocks::Element *)((uint8_t *)pAddress + 656*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 664; }
};

class NormGate9CommitPols
{
public:
    CommitGeneratedPol freeA;
    CommitGeneratedPol freeB;
    CommitGeneratedPol gateType;
    CommitGeneratedPol freeANorm;
    CommitGeneratedPol freeBNorm;
    CommitGeneratedPol freeCNorm;
    CommitGeneratedPol a;
    CommitGeneratedPol b;
    CommitGeneratedPol c;

    NormGate9CommitPols (void * pAddress)
    {
        freeA = (Goldilocks::Element *)((uint8_t *)pAddress + 3224);
        freeB = (Goldilocks::Element *)((uint8_t *)pAddress + 3232);
        gateType = (Goldilocks::Element *)((uint8_t *)pAddress + 3240);
        freeANorm = (Goldilocks::Element *)((uint8_t *)pAddress + 3248);
        freeBNorm = (Goldilocks::Element *)((uint8_t *)pAddress + 3256);
        freeCNorm = (Goldilocks::Element *)((uint8_t *)pAddress + 3264);
        a = (Goldilocks::Element *)((uint8_t *)pAddress + 3272);
        b = (Goldilocks::Element *)((uint8_t *)pAddress + 3280);
        c = (Goldilocks::Element *)((uint8_t *)pAddress + 3288);
    }

    NormGate9CommitPols (void * pAddress, uint64_t degree)
    {
        freeA = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        freeB = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        gateType = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        freeANorm = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        freeBNorm = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        freeCNorm = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        a = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        b = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        c = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 72; }
};

class KeccakFCommitPols
{
public:
    CommitGeneratedPol a;
    CommitGeneratedPol b;
    CommitGeneratedPol c;

    KeccakFCommitPols (void * pAddress)
    {
        a = (Goldilocks::Element *)((uint8_t *)pAddress + 3296);
        b = (Goldilocks::Element *)((uint8_t *)pAddress + 3304);
        c = (Goldilocks::Element *)((uint8_t *)pAddress + 3312);
    }

    KeccakFCommitPols (void * pAddress, uint64_t degree)
    {
        a = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        b = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        c = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 24; }
};

class Nine2OneCommitPols
{
public:
    CommitGeneratedPol bit;
    CommitGeneratedPol field9;

    Nine2OneCommitPols (void * pAddress)
    {
        bit = (Goldilocks::Element *)((uint8_t *)pAddress + 3320);
        field9 = (Goldilocks::Element *)((uint8_t *)pAddress + 3328);
    }

    Nine2OneCommitPols (void * pAddress, uint64_t degree)
    {
        bit = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        field9 = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 16; }
};

class PaddingKKBitCommitPols
{
public:
    CommitGeneratedPol rBit;
    CommitGeneratedPol sOutBit;
    CommitGeneratedPol r8;
    CommitGeneratedPol connected;
    CommitGeneratedPol sOut0;
    CommitGeneratedPol sOut1;
    CommitGeneratedPol sOut2;
    CommitGeneratedPol sOut3;
    CommitGeneratedPol sOut4;
    CommitGeneratedPol sOut5;
    CommitGeneratedPol sOut6;
    CommitGeneratedPol sOut7;

    PaddingKKBitCommitPols (void * pAddress)
    {
        rBit = (Goldilocks::Element *)((uint8_t *)pAddress + 3336);
        sOutBit = (Goldilocks::Element *)((uint8_t *)pAddress + 3344);
        r8 = (Goldilocks::Element *)((uint8_t *)pAddress + 3352);
        connected = (Goldilocks::Element *)((uint8_t *)pAddress + 3360);
        sOut0 = (Goldilocks::Element *)((uint8_t *)pAddress + 3368);
        sOut1 = (Goldilocks::Element *)((uint8_t *)pAddress + 3376);
        sOut2 = (Goldilocks::Element *)((uint8_t *)pAddress + 3384);
        sOut3 = (Goldilocks::Element *)((uint8_t *)pAddress + 3392);
        sOut4 = (Goldilocks::Element *)((uint8_t *)pAddress + 3400);
        sOut5 = (Goldilocks::Element *)((uint8_t *)pAddress + 3408);
        sOut6 = (Goldilocks::Element *)((uint8_t *)pAddress + 3416);
        sOut7 = (Goldilocks::Element *)((uint8_t *)pAddress + 3424);
    }

    PaddingKKBitCommitPols (void * pAddress, uint64_t degree)
    {
        rBit = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        sOutBit = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        r8 = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        connected = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        sOut0 = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        sOut1 = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        sOut2 = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        sOut3 = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        sOut4 = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        sOut5 = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        sOut6 = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        sOut7 = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 96; }
};

class PaddingKKCommitPols
{
public:
    CommitGeneratedPol freeIn;
    CommitGeneratedPol connected;
    CommitGeneratedPol addr;
    CommitGeneratedPol rem;
    CommitGeneratedPol remInv;
    CommitGeneratedPol spare;
    CommitGeneratedPol firstHash;
    CommitGeneratedPol len;
    CommitGeneratedPol hash0;
    CommitGeneratedPol hash1;
    CommitGeneratedPol hash2;
    CommitGeneratedPol hash3;
    CommitGeneratedPol hash4;
    CommitGeneratedPol hash5;
    CommitGeneratedPol hash6;
    CommitGeneratedPol hash7;
    CommitGeneratedPol incCounter;
    CommitGeneratedPol crOffset;
    CommitGeneratedPol crLen;
    CommitGeneratedPol crOffsetInv;
    CommitGeneratedPol crF0;
    CommitGeneratedPol crF1;
    CommitGeneratedPol crF2;
    CommitGeneratedPol crF3;
    CommitGeneratedPol crF4;
    CommitGeneratedPol crF5;
    CommitGeneratedPol crF6;
    CommitGeneratedPol crF7;
    CommitGeneratedPol crV0;
    CommitGeneratedPol crV1;
    CommitGeneratedPol crV2;
    CommitGeneratedPol crV3;
    CommitGeneratedPol crV4;
    CommitGeneratedPol crV5;
    CommitGeneratedPol crV6;
    CommitGeneratedPol crV7;

    PaddingKKCommitPols (void * pAddress)
    {
        freeIn = (Goldilocks::Element *)((uint8_t *)pAddress + 3432);
        connected = (Goldilocks::Element *)((uint8_t *)pAddress + 3440);
        addr = (Goldilocks::Element *)((uint8_t *)pAddress + 3448);
        rem = (Goldilocks::Element *)((uint8_t *)pAddress + 3456);
        remInv = (Goldilocks::Element *)((uint8_t *)pAddress + 3464);
        spare = (Goldilocks::Element *)((uint8_t *)pAddress + 3472);
        firstHash = (Goldilocks::Element *)((uint8_t *)pAddress + 3480);
        len = (Goldilocks::Element *)((uint8_t *)pAddress + 3488);
        hash0 = (Goldilocks::Element *)((uint8_t *)pAddress + 3496);
        hash1 = (Goldilocks::Element *)((uint8_t *)pAddress + 3504);
        hash2 = (Goldilocks::Element *)((uint8_t *)pAddress + 3512);
        hash3 = (Goldilocks::Element *)((uint8_t *)pAddress + 3520);
        hash4 = (Goldilocks::Element *)((uint8_t *)pAddress + 3528);
        hash5 = (Goldilocks::Element *)((uint8_t *)pAddress + 3536);
        hash6 = (Goldilocks::Element *)((uint8_t *)pAddress + 3544);
        hash7 = (Goldilocks::Element *)((uint8_t *)pAddress + 3552);
        incCounter = (Goldilocks::Element *)((uint8_t *)pAddress + 3560);
        crOffset = (Goldilocks::Element *)((uint8_t *)pAddress + 3568);
        crLen = (Goldilocks::Element *)((uint8_t *)pAddress + 3576);
        crOffsetInv = (Goldilocks::Element *)((uint8_t *)pAddress + 3584);
        crF0 = (Goldilocks::Element *)((uint8_t *)pAddress + 3592);
        crF1 = (Goldilocks::Element *)((uint8_t *)pAddress + 3600);
        crF2 = (Goldilocks::Element *)((uint8_t *)pAddress + 3608);
        crF3 = (Goldilocks::Element *)((uint8_t *)pAddress + 3616);
        crF4 = (Goldilocks::Element *)((uint8_t *)pAddress + 3624);
        crF5 = (Goldilocks::Element *)((uint8_t *)pAddress + 3632);
        crF6 = (Goldilocks::Element *)((uint8_t *)pAddress + 3640);
        crF7 = (Goldilocks::Element *)((uint8_t *)pAddress + 3648);
        crV0 = (Goldilocks::Element *)((uint8_t *)pAddress + 3656);
        crV1 = (Goldilocks::Element *)((uint8_t *)pAddress + 3664);
        crV2 = (Goldilocks::Element *)((uint8_t *)pAddress + 3672);
        crV3 = (Goldilocks::Element *)((uint8_t *)pAddress + 3680);
        crV4 = (Goldilocks::Element *)((uint8_t *)pAddress + 3688);
        crV5 = (Goldilocks::Element *)((uint8_t *)pAddress + 3696);
        crV6 = (Goldilocks::Element *)((uint8_t *)pAddress + 3704);
        crV7 = (Goldilocks::Element *)((uint8_t *)pAddress + 3712);
    }

    PaddingKKCommitPols (void * pAddress, uint64_t degree)
    {
        freeIn = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        connected = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        addr = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        rem = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        remInv = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        spare = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        firstHash = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        len = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        hash0 = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        hash1 = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        hash2 = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        hash3 = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        hash4 = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        hash5 = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        hash6 = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        hash7 = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        incCounter = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
        crOffset = (Goldilocks::Element *)((uint8_t *)pAddress + 136*degree);
        crLen = (Goldilocks::Element *)((uint8_t *)pAddress + 144*degree);
        crOffsetInv = (Goldilocks::Element *)((uint8_t *)pAddress + 152*degree);
        crF0 = (Goldilocks::Element *)((uint8_t *)pAddress + 160*degree);
        crF1 = (Goldilocks::Element *)((uint8_t *)pAddress + 168*degree);
        crF2 = (Goldilocks::Element *)((uint8_t *)pAddress + 176*degree);
        crF3 = (Goldilocks::Element *)((uint8_t *)pAddress + 184*degree);
        crF4 = (Goldilocks::Element *)((uint8_t *)pAddress + 192*degree);
        crF5 = (Goldilocks::Element *)((uint8_t *)pAddress + 200*degree);
        crF6 = (Goldilocks::Element *)((uint8_t *)pAddress + 208*degree);
        crF7 = (Goldilocks::Element *)((uint8_t *)pAddress + 216*degree);
        crV0 = (Goldilocks::Element *)((uint8_t *)pAddress + 224*degree);
        crV1 = (Goldilocks::Element *)((uint8_t *)pAddress + 232*degree);
        crV2 = (Goldilocks::Element *)((uint8_t *)pAddress + 240*degree);
        crV3 = (Goldilocks::Element *)((uint8_t *)pAddress + 248*degree);
        crV4 = (Goldilocks::Element *)((uint8_t *)pAddress + 256*degree);
        crV5 = (Goldilocks::Element *)((uint8_t *)pAddress + 264*degree);
        crV6 = (Goldilocks::Element *)((uint8_t *)pAddress + 272*degree);
        crV7 = (Goldilocks::Element *)((uint8_t *)pAddress + 280*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 288; }
};

class MemCommitPols
{
public:
    CommitGeneratedPol addr;
    CommitGeneratedPol step;
    CommitGeneratedPol mOp;
    CommitGeneratedPol mWr;
    CommitGeneratedPol val[8];
    CommitGeneratedPol lastAccess;

    MemCommitPols (void * pAddress)
    {
        addr = (Goldilocks::Element *)((uint8_t *)pAddress + 3720);
        step = (Goldilocks::Element *)((uint8_t *)pAddress + 3728);
        mOp = (Goldilocks::Element *)((uint8_t *)pAddress + 3736);
        mWr = (Goldilocks::Element *)((uint8_t *)pAddress + 3744);
        val[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 3752);
        val[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 3760);
        val[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 3768);
        val[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 3776);
        val[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 3784);
        val[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 3792);
        val[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 3800);
        val[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 3808);
        lastAccess = (Goldilocks::Element *)((uint8_t *)pAddress + 3816);
    }

    MemCommitPols (void * pAddress, uint64_t degree)
    {
        addr = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        step = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        mOp = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        mWr = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        val[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        val[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        val[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        val[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        val[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        val[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        val[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        val[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        lastAccess = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 104; }
};

class MainCommitPols
{
public:
    CommitGeneratedPol A7;
    CommitGeneratedPol A6;
    CommitGeneratedPol A5;
    CommitGeneratedPol A4;
    CommitGeneratedPol A3;
    CommitGeneratedPol A2;
    CommitGeneratedPol A1;
    CommitGeneratedPol A0;
    CommitGeneratedPol B7;
    CommitGeneratedPol B6;
    CommitGeneratedPol B5;
    CommitGeneratedPol B4;
    CommitGeneratedPol B3;
    CommitGeneratedPol B2;
    CommitGeneratedPol B1;
    CommitGeneratedPol B0;
    CommitGeneratedPol C7;
    CommitGeneratedPol C6;
    CommitGeneratedPol C5;
    CommitGeneratedPol C4;
    CommitGeneratedPol C3;
    CommitGeneratedPol C2;
    CommitGeneratedPol C1;
    CommitGeneratedPol C0;
    CommitGeneratedPol D7;
    CommitGeneratedPol D6;
    CommitGeneratedPol D5;
    CommitGeneratedPol D4;
    CommitGeneratedPol D3;
    CommitGeneratedPol D2;
    CommitGeneratedPol D1;
    CommitGeneratedPol D0;
    CommitGeneratedPol E7;
    CommitGeneratedPol E6;
    CommitGeneratedPol E5;
    CommitGeneratedPol E4;
    CommitGeneratedPol E3;
    CommitGeneratedPol E2;
    CommitGeneratedPol E1;
    CommitGeneratedPol E0;
    CommitGeneratedPol SR7;
    CommitGeneratedPol SR6;
    CommitGeneratedPol SR5;
    CommitGeneratedPol SR4;
    CommitGeneratedPol SR3;
    CommitGeneratedPol SR2;
    CommitGeneratedPol SR1;
    CommitGeneratedPol SR0;
    CommitGeneratedPol CTX;
    CommitGeneratedPol SP;
    CommitGeneratedPol PC;
    CommitGeneratedPol GAS;
    CommitGeneratedPol MAXMEM;
    CommitGeneratedPol zkPC;
    CommitGeneratedPol RR;
    CommitGeneratedPol HASHPOS;
    CommitGeneratedPol CONST7;
    CommitGeneratedPol CONST6;
    CommitGeneratedPol CONST5;
    CommitGeneratedPol CONST4;
    CommitGeneratedPol CONST3;
    CommitGeneratedPol CONST2;
    CommitGeneratedPol CONST1;
    CommitGeneratedPol CONST0;
    CommitGeneratedPol FREE7;
    CommitGeneratedPol FREE6;
    CommitGeneratedPol FREE5;
    CommitGeneratedPol FREE4;
    CommitGeneratedPol FREE3;
    CommitGeneratedPol FREE2;
    CommitGeneratedPol FREE1;
    CommitGeneratedPol FREE0;
    CommitGeneratedPol inA;
    CommitGeneratedPol inB;
    CommitGeneratedPol inC;
    CommitGeneratedPol inROTL_C;
    CommitGeneratedPol inD;
    CommitGeneratedPol inE;
    CommitGeneratedPol inSR;
    CommitGeneratedPol inFREE;
    CommitGeneratedPol inCTX;
    CommitGeneratedPol inSP;
    CommitGeneratedPol inPC;
    CommitGeneratedPol inGAS;
    CommitGeneratedPol inMAXMEM;
    CommitGeneratedPol inSTEP;
    CommitGeneratedPol inRR;
    CommitGeneratedPol inHASHPOS;
    CommitGeneratedPol setA;
    CommitGeneratedPol setB;
    CommitGeneratedPol setC;
    CommitGeneratedPol setD;
    CommitGeneratedPol setE;
    CommitGeneratedPol setSR;
    CommitGeneratedPol setCTX;
    CommitGeneratedPol setSP;
    CommitGeneratedPol setPC;
    CommitGeneratedPol setGAS;
    CommitGeneratedPol setMAXMEM;
    CommitGeneratedPol JMP;
    CommitGeneratedPol JMPN;
    CommitGeneratedPol JMPC;
    CommitGeneratedPol setRR;
    CommitGeneratedPol setHASHPOS;
    CommitGeneratedPol offset;
    CommitGeneratedPol incStack;
    CommitGeneratedPol incCode;
    CommitGeneratedPol isStack;
    CommitGeneratedPol isCode;
    CommitGeneratedPol isMem;
    CommitGeneratedPol ind;
    CommitGeneratedPol indRR;
    CommitGeneratedPol useCTX;
    CommitGeneratedPol carry;
    CommitGeneratedPol mOp;
    CommitGeneratedPol mWR;
    CommitGeneratedPol sWR;
    CommitGeneratedPol sRD;
    CommitGeneratedPol arith;
    CommitGeneratedPol arithEq0;
    CommitGeneratedPol arithEq1;
    CommitGeneratedPol arithEq2;
    CommitGeneratedPol arithEq3;
    CommitGeneratedPol memAlign;
    CommitGeneratedPol memAlignWR;
    CommitGeneratedPol memAlignWR8;
    CommitGeneratedPol hashK;
    CommitGeneratedPol hashKLen;
    CommitGeneratedPol hashKDigest;
    CommitGeneratedPol hashP;
    CommitGeneratedPol hashPLen;
    CommitGeneratedPol hashPDigest;
    CommitGeneratedPol bin;
    CommitGeneratedPol binOpcode;
    CommitGeneratedPol assert;
    CommitGeneratedPol isNeg;
    CommitGeneratedPol isMaxMem;
    CommitGeneratedPol cntArith;
    CommitGeneratedPol cntBinary;
    CommitGeneratedPol cntMemAlign;
    CommitGeneratedPol cntKeccakF;
    CommitGeneratedPol cntPoseidonG;
    CommitGeneratedPol cntPaddingPG;
    CommitGeneratedPol inCntArith;
    CommitGeneratedPol inCntBinary;
    CommitGeneratedPol inCntMemAlign;
    CommitGeneratedPol inCntKeccakF;
    CommitGeneratedPol inCntPoseidonG;
    CommitGeneratedPol inCntPaddingPG;
    CommitGeneratedPol incCounter;
    CommitGeneratedPol sKeyI[4];
    CommitGeneratedPol sKey[4];

    MainCommitPols (void * pAddress)
    {
        A7 = (Goldilocks::Element *)((uint8_t *)pAddress + 3824);
        A6 = (Goldilocks::Element *)((uint8_t *)pAddress + 3832);
        A5 = (Goldilocks::Element *)((uint8_t *)pAddress + 3840);
        A4 = (Goldilocks::Element *)((uint8_t *)pAddress + 3848);
        A3 = (Goldilocks::Element *)((uint8_t *)pAddress + 3856);
        A2 = (Goldilocks::Element *)((uint8_t *)pAddress + 3864);
        A1 = (Goldilocks::Element *)((uint8_t *)pAddress + 3872);
        A0 = (Goldilocks::Element *)((uint8_t *)pAddress + 3880);
        B7 = (Goldilocks::Element *)((uint8_t *)pAddress + 3888);
        B6 = (Goldilocks::Element *)((uint8_t *)pAddress + 3896);
        B5 = (Goldilocks::Element *)((uint8_t *)pAddress + 3904);
        B4 = (Goldilocks::Element *)((uint8_t *)pAddress + 3912);
        B3 = (Goldilocks::Element *)((uint8_t *)pAddress + 3920);
        B2 = (Goldilocks::Element *)((uint8_t *)pAddress + 3928);
        B1 = (Goldilocks::Element *)((uint8_t *)pAddress + 3936);
        B0 = (Goldilocks::Element *)((uint8_t *)pAddress + 3944);
        C7 = (Goldilocks::Element *)((uint8_t *)pAddress + 3952);
        C6 = (Goldilocks::Element *)((uint8_t *)pAddress + 3960);
        C5 = (Goldilocks::Element *)((uint8_t *)pAddress + 3968);
        C4 = (Goldilocks::Element *)((uint8_t *)pAddress + 3976);
        C3 = (Goldilocks::Element *)((uint8_t *)pAddress + 3984);
        C2 = (Goldilocks::Element *)((uint8_t *)pAddress + 3992);
        C1 = (Goldilocks::Element *)((uint8_t *)pAddress + 4000);
        C0 = (Goldilocks::Element *)((uint8_t *)pAddress + 4008);
        D7 = (Goldilocks::Element *)((uint8_t *)pAddress + 4016);
        D6 = (Goldilocks::Element *)((uint8_t *)pAddress + 4024);
        D5 = (Goldilocks::Element *)((uint8_t *)pAddress + 4032);
        D4 = (Goldilocks::Element *)((uint8_t *)pAddress + 4040);
        D3 = (Goldilocks::Element *)((uint8_t *)pAddress + 4048);
        D2 = (Goldilocks::Element *)((uint8_t *)pAddress + 4056);
        D1 = (Goldilocks::Element *)((uint8_t *)pAddress + 4064);
        D0 = (Goldilocks::Element *)((uint8_t *)pAddress + 4072);
        E7 = (Goldilocks::Element *)((uint8_t *)pAddress + 4080);
        E6 = (Goldilocks::Element *)((uint8_t *)pAddress + 4088);
        E5 = (Goldilocks::Element *)((uint8_t *)pAddress + 4096);
        E4 = (Goldilocks::Element *)((uint8_t *)pAddress + 4104);
        E3 = (Goldilocks::Element *)((uint8_t *)pAddress + 4112);
        E2 = (Goldilocks::Element *)((uint8_t *)pAddress + 4120);
        E1 = (Goldilocks::Element *)((uint8_t *)pAddress + 4128);
        E0 = (Goldilocks::Element *)((uint8_t *)pAddress + 4136);
        SR7 = (Goldilocks::Element *)((uint8_t *)pAddress + 4144);
        SR6 = (Goldilocks::Element *)((uint8_t *)pAddress + 4152);
        SR5 = (Goldilocks::Element *)((uint8_t *)pAddress + 4160);
        SR4 = (Goldilocks::Element *)((uint8_t *)pAddress + 4168);
        SR3 = (Goldilocks::Element *)((uint8_t *)pAddress + 4176);
        SR2 = (Goldilocks::Element *)((uint8_t *)pAddress + 4184);
        SR1 = (Goldilocks::Element *)((uint8_t *)pAddress + 4192);
        SR0 = (Goldilocks::Element *)((uint8_t *)pAddress + 4200);
        CTX = (Goldilocks::Element *)((uint8_t *)pAddress + 4208);
        SP = (Goldilocks::Element *)((uint8_t *)pAddress + 4216);
        PC = (Goldilocks::Element *)((uint8_t *)pAddress + 4224);
        GAS = (Goldilocks::Element *)((uint8_t *)pAddress + 4232);
        MAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 4240);
        zkPC = (Goldilocks::Element *)((uint8_t *)pAddress + 4248);
        RR = (Goldilocks::Element *)((uint8_t *)pAddress + 4256);
        HASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 4264);
        CONST7 = (Goldilocks::Element *)((uint8_t *)pAddress + 4272);
        CONST6 = (Goldilocks::Element *)((uint8_t *)pAddress + 4280);
        CONST5 = (Goldilocks::Element *)((uint8_t *)pAddress + 4288);
        CONST4 = (Goldilocks::Element *)((uint8_t *)pAddress + 4296);
        CONST3 = (Goldilocks::Element *)((uint8_t *)pAddress + 4304);
        CONST2 = (Goldilocks::Element *)((uint8_t *)pAddress + 4312);
        CONST1 = (Goldilocks::Element *)((uint8_t *)pAddress + 4320);
        CONST0 = (Goldilocks::Element *)((uint8_t *)pAddress + 4328);
        FREE7 = (Goldilocks::Element *)((uint8_t *)pAddress + 4336);
        FREE6 = (Goldilocks::Element *)((uint8_t *)pAddress + 4344);
        FREE5 = (Goldilocks::Element *)((uint8_t *)pAddress + 4352);
        FREE4 = (Goldilocks::Element *)((uint8_t *)pAddress + 4360);
        FREE3 = (Goldilocks::Element *)((uint8_t *)pAddress + 4368);
        FREE2 = (Goldilocks::Element *)((uint8_t *)pAddress + 4376);
        FREE1 = (Goldilocks::Element *)((uint8_t *)pAddress + 4384);
        FREE0 = (Goldilocks::Element *)((uint8_t *)pAddress + 4392);
        inA = (Goldilocks::Element *)((uint8_t *)pAddress + 4400);
        inB = (Goldilocks::Element *)((uint8_t *)pAddress + 4408);
        inC = (Goldilocks::Element *)((uint8_t *)pAddress + 4416);
        inROTL_C = (Goldilocks::Element *)((uint8_t *)pAddress + 4424);
        inD = (Goldilocks::Element *)((uint8_t *)pAddress + 4432);
        inE = (Goldilocks::Element *)((uint8_t *)pAddress + 4440);
        inSR = (Goldilocks::Element *)((uint8_t *)pAddress + 4448);
        inFREE = (Goldilocks::Element *)((uint8_t *)pAddress + 4456);
        inCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 4464);
        inSP = (Goldilocks::Element *)((uint8_t *)pAddress + 4472);
        inPC = (Goldilocks::Element *)((uint8_t *)pAddress + 4480);
        inGAS = (Goldilocks::Element *)((uint8_t *)pAddress + 4488);
        inMAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 4496);
        inSTEP = (Goldilocks::Element *)((uint8_t *)pAddress + 4504);
        inRR = (Goldilocks::Element *)((uint8_t *)pAddress + 4512);
        inHASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 4520);
        setA = (Goldilocks::Element *)((uint8_t *)pAddress + 4528);
        setB = (Goldilocks::Element *)((uint8_t *)pAddress + 4536);
        setC = (Goldilocks::Element *)((uint8_t *)pAddress + 4544);
        setD = (Goldilocks::Element *)((uint8_t *)pAddress + 4552);
        setE = (Goldilocks::Element *)((uint8_t *)pAddress + 4560);
        setSR = (Goldilocks::Element *)((uint8_t *)pAddress + 4568);
        setCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 4576);
        setSP = (Goldilocks::Element *)((uint8_t *)pAddress + 4584);
        setPC = (Goldilocks::Element *)((uint8_t *)pAddress + 4592);
        setGAS = (Goldilocks::Element *)((uint8_t *)pAddress + 4600);
        setMAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 4608);
        JMP = (Goldilocks::Element *)((uint8_t *)pAddress + 4616);
        JMPN = (Goldilocks::Element *)((uint8_t *)pAddress + 4624);
        JMPC = (Goldilocks::Element *)((uint8_t *)pAddress + 4632);
        setRR = (Goldilocks::Element *)((uint8_t *)pAddress + 4640);
        setHASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 4648);
        offset = (Goldilocks::Element *)((uint8_t *)pAddress + 4656);
        incStack = (Goldilocks::Element *)((uint8_t *)pAddress + 4664);
        incCode = (Goldilocks::Element *)((uint8_t *)pAddress + 4672);
        isStack = (Goldilocks::Element *)((uint8_t *)pAddress + 4680);
        isCode = (Goldilocks::Element *)((uint8_t *)pAddress + 4688);
        isMem = (Goldilocks::Element *)((uint8_t *)pAddress + 4696);
        ind = (Goldilocks::Element *)((uint8_t *)pAddress + 4704);
        indRR = (Goldilocks::Element *)((uint8_t *)pAddress + 4712);
        useCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 4720);
        carry = (Goldilocks::Element *)((uint8_t *)pAddress + 4728);
        mOp = (Goldilocks::Element *)((uint8_t *)pAddress + 4736);
        mWR = (Goldilocks::Element *)((uint8_t *)pAddress + 4744);
        sWR = (Goldilocks::Element *)((uint8_t *)pAddress + 4752);
        sRD = (Goldilocks::Element *)((uint8_t *)pAddress + 4760);
        arith = (Goldilocks::Element *)((uint8_t *)pAddress + 4768);
        arithEq0 = (Goldilocks::Element *)((uint8_t *)pAddress + 4776);
        arithEq1 = (Goldilocks::Element *)((uint8_t *)pAddress + 4784);
        arithEq2 = (Goldilocks::Element *)((uint8_t *)pAddress + 4792);
        arithEq3 = (Goldilocks::Element *)((uint8_t *)pAddress + 4800);
        memAlign = (Goldilocks::Element *)((uint8_t *)pAddress + 4808);
        memAlignWR = (Goldilocks::Element *)((uint8_t *)pAddress + 4816);
        memAlignWR8 = (Goldilocks::Element *)((uint8_t *)pAddress + 4824);
        hashK = (Goldilocks::Element *)((uint8_t *)pAddress + 4832);
        hashKLen = (Goldilocks::Element *)((uint8_t *)pAddress + 4840);
        hashKDigest = (Goldilocks::Element *)((uint8_t *)pAddress + 4848);
        hashP = (Goldilocks::Element *)((uint8_t *)pAddress + 4856);
        hashPLen = (Goldilocks::Element *)((uint8_t *)pAddress + 4864);
        hashPDigest = (Goldilocks::Element *)((uint8_t *)pAddress + 4872);
        bin = (Goldilocks::Element *)((uint8_t *)pAddress + 4880);
        binOpcode = (Goldilocks::Element *)((uint8_t *)pAddress + 4888);
        assert = (Goldilocks::Element *)((uint8_t *)pAddress + 4896);
        isNeg = (Goldilocks::Element *)((uint8_t *)pAddress + 4904);
        isMaxMem = (Goldilocks::Element *)((uint8_t *)pAddress + 4912);
        cntArith = (Goldilocks::Element *)((uint8_t *)pAddress + 4920);
        cntBinary = (Goldilocks::Element *)((uint8_t *)pAddress + 4928);
        cntMemAlign = (Goldilocks::Element *)((uint8_t *)pAddress + 4936);
        cntKeccakF = (Goldilocks::Element *)((uint8_t *)pAddress + 4944);
        cntPoseidonG = (Goldilocks::Element *)((uint8_t *)pAddress + 4952);
        cntPaddingPG = (Goldilocks::Element *)((uint8_t *)pAddress + 4960);
        inCntArith = (Goldilocks::Element *)((uint8_t *)pAddress + 4968);
        inCntBinary = (Goldilocks::Element *)((uint8_t *)pAddress + 4976);
        inCntMemAlign = (Goldilocks::Element *)((uint8_t *)pAddress + 4984);
        inCntKeccakF = (Goldilocks::Element *)((uint8_t *)pAddress + 4992);
        inCntPoseidonG = (Goldilocks::Element *)((uint8_t *)pAddress + 5000);
        inCntPaddingPG = (Goldilocks::Element *)((uint8_t *)pAddress + 5008);
        incCounter = (Goldilocks::Element *)((uint8_t *)pAddress + 5016);
        sKeyI[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 5024);
        sKeyI[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 5032);
        sKeyI[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 5040);
        sKeyI[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 5048);
        sKey[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 5056);
        sKey[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 5064);
        sKey[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 5072);
        sKey[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 5080);
    }

    MainCommitPols (void * pAddress, uint64_t degree)
    {
        A7 = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        A6 = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        A5 = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        A4 = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        A3 = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        A2 = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        A1 = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        A0 = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        B7 = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        B6 = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        B5 = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        B4 = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        B3 = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        B2 = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        B1 = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        B0 = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        C7 = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
        C6 = (Goldilocks::Element *)((uint8_t *)pAddress + 136*degree);
        C5 = (Goldilocks::Element *)((uint8_t *)pAddress + 144*degree);
        C4 = (Goldilocks::Element *)((uint8_t *)pAddress + 152*degree);
        C3 = (Goldilocks::Element *)((uint8_t *)pAddress + 160*degree);
        C2 = (Goldilocks::Element *)((uint8_t *)pAddress + 168*degree);
        C1 = (Goldilocks::Element *)((uint8_t *)pAddress + 176*degree);
        C0 = (Goldilocks::Element *)((uint8_t *)pAddress + 184*degree);
        D7 = (Goldilocks::Element *)((uint8_t *)pAddress + 192*degree);
        D6 = (Goldilocks::Element *)((uint8_t *)pAddress + 200*degree);
        D5 = (Goldilocks::Element *)((uint8_t *)pAddress + 208*degree);
        D4 = (Goldilocks::Element *)((uint8_t *)pAddress + 216*degree);
        D3 = (Goldilocks::Element *)((uint8_t *)pAddress + 224*degree);
        D2 = (Goldilocks::Element *)((uint8_t *)pAddress + 232*degree);
        D1 = (Goldilocks::Element *)((uint8_t *)pAddress + 240*degree);
        D0 = (Goldilocks::Element *)((uint8_t *)pAddress + 248*degree);
        E7 = (Goldilocks::Element *)((uint8_t *)pAddress + 256*degree);
        E6 = (Goldilocks::Element *)((uint8_t *)pAddress + 264*degree);
        E5 = (Goldilocks::Element *)((uint8_t *)pAddress + 272*degree);
        E4 = (Goldilocks::Element *)((uint8_t *)pAddress + 280*degree);
        E3 = (Goldilocks::Element *)((uint8_t *)pAddress + 288*degree);
        E2 = (Goldilocks::Element *)((uint8_t *)pAddress + 296*degree);
        E1 = (Goldilocks::Element *)((uint8_t *)pAddress + 304*degree);
        E0 = (Goldilocks::Element *)((uint8_t *)pAddress + 312*degree);
        SR7 = (Goldilocks::Element *)((uint8_t *)pAddress + 320*degree);
        SR6 = (Goldilocks::Element *)((uint8_t *)pAddress + 328*degree);
        SR5 = (Goldilocks::Element *)((uint8_t *)pAddress + 336*degree);
        SR4 = (Goldilocks::Element *)((uint8_t *)pAddress + 344*degree);
        SR3 = (Goldilocks::Element *)((uint8_t *)pAddress + 352*degree);
        SR2 = (Goldilocks::Element *)((uint8_t *)pAddress + 360*degree);
        SR1 = (Goldilocks::Element *)((uint8_t *)pAddress + 368*degree);
        SR0 = (Goldilocks::Element *)((uint8_t *)pAddress + 376*degree);
        CTX = (Goldilocks::Element *)((uint8_t *)pAddress + 384*degree);
        SP = (Goldilocks::Element *)((uint8_t *)pAddress + 392*degree);
        PC = (Goldilocks::Element *)((uint8_t *)pAddress + 400*degree);
        GAS = (Goldilocks::Element *)((uint8_t *)pAddress + 408*degree);
        MAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 416*degree);
        zkPC = (Goldilocks::Element *)((uint8_t *)pAddress + 424*degree);
        RR = (Goldilocks::Element *)((uint8_t *)pAddress + 432*degree);
        HASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 440*degree);
        CONST7 = (Goldilocks::Element *)((uint8_t *)pAddress + 448*degree);
        CONST6 = (Goldilocks::Element *)((uint8_t *)pAddress + 456*degree);
        CONST5 = (Goldilocks::Element *)((uint8_t *)pAddress + 464*degree);
        CONST4 = (Goldilocks::Element *)((uint8_t *)pAddress + 472*degree);
        CONST3 = (Goldilocks::Element *)((uint8_t *)pAddress + 480*degree);
        CONST2 = (Goldilocks::Element *)((uint8_t *)pAddress + 488*degree);
        CONST1 = (Goldilocks::Element *)((uint8_t *)pAddress + 496*degree);
        CONST0 = (Goldilocks::Element *)((uint8_t *)pAddress + 504*degree);
        FREE7 = (Goldilocks::Element *)((uint8_t *)pAddress + 512*degree);
        FREE6 = (Goldilocks::Element *)((uint8_t *)pAddress + 520*degree);
        FREE5 = (Goldilocks::Element *)((uint8_t *)pAddress + 528*degree);
        FREE4 = (Goldilocks::Element *)((uint8_t *)pAddress + 536*degree);
        FREE3 = (Goldilocks::Element *)((uint8_t *)pAddress + 544*degree);
        FREE2 = (Goldilocks::Element *)((uint8_t *)pAddress + 552*degree);
        FREE1 = (Goldilocks::Element *)((uint8_t *)pAddress + 560*degree);
        FREE0 = (Goldilocks::Element *)((uint8_t *)pAddress + 568*degree);
        inA = (Goldilocks::Element *)((uint8_t *)pAddress + 576*degree);
        inB = (Goldilocks::Element *)((uint8_t *)pAddress + 584*degree);
        inC = (Goldilocks::Element *)((uint8_t *)pAddress + 592*degree);
        inROTL_C = (Goldilocks::Element *)((uint8_t *)pAddress + 600*degree);
        inD = (Goldilocks::Element *)((uint8_t *)pAddress + 608*degree);
        inE = (Goldilocks::Element *)((uint8_t *)pAddress + 616*degree);
        inSR = (Goldilocks::Element *)((uint8_t *)pAddress + 624*degree);
        inFREE = (Goldilocks::Element *)((uint8_t *)pAddress + 632*degree);
        inCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 640*degree);
        inSP = (Goldilocks::Element *)((uint8_t *)pAddress + 648*degree);
        inPC = (Goldilocks::Element *)((uint8_t *)pAddress + 656*degree);
        inGAS = (Goldilocks::Element *)((uint8_t *)pAddress + 664*degree);
        inMAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 672*degree);
        inSTEP = (Goldilocks::Element *)((uint8_t *)pAddress + 680*degree);
        inRR = (Goldilocks::Element *)((uint8_t *)pAddress + 688*degree);
        inHASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 696*degree);
        setA = (Goldilocks::Element *)((uint8_t *)pAddress + 704*degree);
        setB = (Goldilocks::Element *)((uint8_t *)pAddress + 712*degree);
        setC = (Goldilocks::Element *)((uint8_t *)pAddress + 720*degree);
        setD = (Goldilocks::Element *)((uint8_t *)pAddress + 728*degree);
        setE = (Goldilocks::Element *)((uint8_t *)pAddress + 736*degree);
        setSR = (Goldilocks::Element *)((uint8_t *)pAddress + 744*degree);
        setCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 752*degree);
        setSP = (Goldilocks::Element *)((uint8_t *)pAddress + 760*degree);
        setPC = (Goldilocks::Element *)((uint8_t *)pAddress + 768*degree);
        setGAS = (Goldilocks::Element *)((uint8_t *)pAddress + 776*degree);
        setMAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 784*degree);
        JMP = (Goldilocks::Element *)((uint8_t *)pAddress + 792*degree);
        JMPN = (Goldilocks::Element *)((uint8_t *)pAddress + 800*degree);
        JMPC = (Goldilocks::Element *)((uint8_t *)pAddress + 808*degree);
        setRR = (Goldilocks::Element *)((uint8_t *)pAddress + 816*degree);
        setHASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 824*degree);
        offset = (Goldilocks::Element *)((uint8_t *)pAddress + 832*degree);
        incStack = (Goldilocks::Element *)((uint8_t *)pAddress + 840*degree);
        incCode = (Goldilocks::Element *)((uint8_t *)pAddress + 848*degree);
        isStack = (Goldilocks::Element *)((uint8_t *)pAddress + 856*degree);
        isCode = (Goldilocks::Element *)((uint8_t *)pAddress + 864*degree);
        isMem = (Goldilocks::Element *)((uint8_t *)pAddress + 872*degree);
        ind = (Goldilocks::Element *)((uint8_t *)pAddress + 880*degree);
        indRR = (Goldilocks::Element *)((uint8_t *)pAddress + 888*degree);
        useCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 896*degree);
        carry = (Goldilocks::Element *)((uint8_t *)pAddress + 904*degree);
        mOp = (Goldilocks::Element *)((uint8_t *)pAddress + 912*degree);
        mWR = (Goldilocks::Element *)((uint8_t *)pAddress + 920*degree);
        sWR = (Goldilocks::Element *)((uint8_t *)pAddress + 928*degree);
        sRD = (Goldilocks::Element *)((uint8_t *)pAddress + 936*degree);
        arith = (Goldilocks::Element *)((uint8_t *)pAddress + 944*degree);
        arithEq0 = (Goldilocks::Element *)((uint8_t *)pAddress + 952*degree);
        arithEq1 = (Goldilocks::Element *)((uint8_t *)pAddress + 960*degree);
        arithEq2 = (Goldilocks::Element *)((uint8_t *)pAddress + 968*degree);
        arithEq3 = (Goldilocks::Element *)((uint8_t *)pAddress + 976*degree);
        memAlign = (Goldilocks::Element *)((uint8_t *)pAddress + 984*degree);
        memAlignWR = (Goldilocks::Element *)((uint8_t *)pAddress + 992*degree);
        memAlignWR8 = (Goldilocks::Element *)((uint8_t *)pAddress + 1000*degree);
        hashK = (Goldilocks::Element *)((uint8_t *)pAddress + 1008*degree);
        hashKLen = (Goldilocks::Element *)((uint8_t *)pAddress + 1016*degree);
        hashKDigest = (Goldilocks::Element *)((uint8_t *)pAddress + 1024*degree);
        hashP = (Goldilocks::Element *)((uint8_t *)pAddress + 1032*degree);
        hashPLen = (Goldilocks::Element *)((uint8_t *)pAddress + 1040*degree);
        hashPDigest = (Goldilocks::Element *)((uint8_t *)pAddress + 1048*degree);
        bin = (Goldilocks::Element *)((uint8_t *)pAddress + 1056*degree);
        binOpcode = (Goldilocks::Element *)((uint8_t *)pAddress + 1064*degree);
        assert = (Goldilocks::Element *)((uint8_t *)pAddress + 1072*degree);
        isNeg = (Goldilocks::Element *)((uint8_t *)pAddress + 1080*degree);
        isMaxMem = (Goldilocks::Element *)((uint8_t *)pAddress + 1088*degree);
        cntArith = (Goldilocks::Element *)((uint8_t *)pAddress + 1096*degree);
        cntBinary = (Goldilocks::Element *)((uint8_t *)pAddress + 1104*degree);
        cntMemAlign = (Goldilocks::Element *)((uint8_t *)pAddress + 1112*degree);
        cntKeccakF = (Goldilocks::Element *)((uint8_t *)pAddress + 1120*degree);
        cntPoseidonG = (Goldilocks::Element *)((uint8_t *)pAddress + 1128*degree);
        cntPaddingPG = (Goldilocks::Element *)((uint8_t *)pAddress + 1136*degree);
        inCntArith = (Goldilocks::Element *)((uint8_t *)pAddress + 1144*degree);
        inCntBinary = (Goldilocks::Element *)((uint8_t *)pAddress + 1152*degree);
        inCntMemAlign = (Goldilocks::Element *)((uint8_t *)pAddress + 1160*degree);
        inCntKeccakF = (Goldilocks::Element *)((uint8_t *)pAddress + 1168*degree);
        inCntPoseidonG = (Goldilocks::Element *)((uint8_t *)pAddress + 1176*degree);
        inCntPaddingPG = (Goldilocks::Element *)((uint8_t *)pAddress + 1184*degree);
        incCounter = (Goldilocks::Element *)((uint8_t *)pAddress + 1192*degree);
        sKeyI[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1200*degree);
        sKeyI[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1208*degree);
        sKeyI[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1216*degree);
        sKeyI[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1224*degree);
        sKey[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1232*degree);
        sKey[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1240*degree);
        sKey[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1248*degree);
        sKey[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1256*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 1264; }
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

    static uint64_t size (void) { return 10670309376; }
};

#endif // COMMIT_POLS_HPP
