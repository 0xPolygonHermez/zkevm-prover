#ifndef COMMIT_POLS_HPP
#define COMMIT_POLS_HPP

#include <cstdint>
#include "ff/ff.hpp"

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

    MemAlignCommitPols (void * pAddress)
    {
        inM = (uint8_t *)((uint8_t *)pAddress + 0);
        inV = (uint8_t *)((uint8_t *)pAddress + 4194304);
        wr = (uint8_t *)((uint8_t *)pAddress + 8388608);
        m0[0] = (uint32_t *)((uint8_t *)pAddress + 12582912);
        m0[1] = (uint32_t *)((uint8_t *)pAddress + 46137344);
        m0[2] = (uint32_t *)((uint8_t *)pAddress + 79691776);
        m0[3] = (uint32_t *)((uint8_t *)pAddress + 113246208);
        m0[4] = (uint32_t *)((uint8_t *)pAddress + 146800640);
        m0[5] = (uint32_t *)((uint8_t *)pAddress + 180355072);
        m0[6] = (uint32_t *)((uint8_t *)pAddress + 213909504);
        m0[7] = (uint32_t *)((uint8_t *)pAddress + 247463936);
        m1[0] = (uint32_t *)((uint8_t *)pAddress + 281018368);
        m1[1] = (uint32_t *)((uint8_t *)pAddress + 314572800);
        m1[2] = (uint32_t *)((uint8_t *)pAddress + 348127232);
        m1[3] = (uint32_t *)((uint8_t *)pAddress + 381681664);
        m1[4] = (uint32_t *)((uint8_t *)pAddress + 415236096);
        m1[5] = (uint32_t *)((uint8_t *)pAddress + 448790528);
        m1[6] = (uint32_t *)((uint8_t *)pAddress + 482344960);
        m1[7] = (uint32_t *)((uint8_t *)pAddress + 515899392);
        w0[0] = (uint32_t *)((uint8_t *)pAddress + 549453824);
        w0[1] = (uint32_t *)((uint8_t *)pAddress + 583008256);
        w0[2] = (uint32_t *)((uint8_t *)pAddress + 616562688);
        w0[3] = (uint32_t *)((uint8_t *)pAddress + 650117120);
        w0[4] = (uint32_t *)((uint8_t *)pAddress + 683671552);
        w0[5] = (uint32_t *)((uint8_t *)pAddress + 717225984);
        w0[6] = (uint32_t *)((uint8_t *)pAddress + 750780416);
        w0[7] = (uint32_t *)((uint8_t *)pAddress + 784334848);
        w1[0] = (uint32_t *)((uint8_t *)pAddress + 817889280);
        w1[1] = (uint32_t *)((uint8_t *)pAddress + 851443712);
        w1[2] = (uint32_t *)((uint8_t *)pAddress + 884998144);
        w1[3] = (uint32_t *)((uint8_t *)pAddress + 918552576);
        w1[4] = (uint32_t *)((uint8_t *)pAddress + 952107008);
        w1[5] = (uint32_t *)((uint8_t *)pAddress + 985661440);
        w1[6] = (uint32_t *)((uint8_t *)pAddress + 1019215872);
        w1[7] = (uint32_t *)((uint8_t *)pAddress + 1052770304);
        v[0] = (uint32_t *)((uint8_t *)pAddress + 1086324736);
        v[1] = (uint32_t *)((uint8_t *)pAddress + 1119879168);
        v[2] = (uint32_t *)((uint8_t *)pAddress + 1153433600);
        v[3] = (uint32_t *)((uint8_t *)pAddress + 1186988032);
        v[4] = (uint32_t *)((uint8_t *)pAddress + 1220542464);
        v[5] = (uint32_t *)((uint8_t *)pAddress + 1254096896);
        v[6] = (uint32_t *)((uint8_t *)pAddress + 1287651328);
        v[7] = (uint32_t *)((uint8_t *)pAddress + 1321205760);
        offset = (uint8_t *)((uint8_t *)pAddress + 1354760192);
        selW = (uint8_t *)((uint8_t *)pAddress + 1358954496);
        factorV[0] = (FieldElement *)((uint8_t *)pAddress + 1363148800);
        factorV[1] = (FieldElement *)((uint8_t *)pAddress + 1396703232);
        factorV[2] = (FieldElement *)((uint8_t *)pAddress + 1430257664);
        factorV[3] = (FieldElement *)((uint8_t *)pAddress + 1463812096);
        factorV[4] = (FieldElement *)((uint8_t *)pAddress + 1497366528);
        factorV[5] = (FieldElement *)((uint8_t *)pAddress + 1530920960);
        factorV[6] = (FieldElement *)((uint8_t *)pAddress + 1564475392);
        factorV[7] = (FieldElement *)((uint8_t *)pAddress + 1598029824);
    }

    static uint64_t degree (void) { return 4194304; }
};

class Byte4CommitPols
{
public:
    uint16_t * freeIN;
    uint32_t * out;

    Byte4CommitPols (void * pAddress)
    {
        freeIN = (uint16_t *)((uint8_t *)pAddress + 1631584256);
        out = (uint32_t *)((uint8_t *)pAddress + 1639972864);
    }

    static uint64_t degree (void) { return 4194304; }
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
        x1[0] = (FieldElement *)((uint8_t *)pAddress + 1673527296);
        x1[1] = (FieldElement *)((uint8_t *)pAddress + 1707081728);
        x1[2] = (FieldElement *)((uint8_t *)pAddress + 1740636160);
        x1[3] = (FieldElement *)((uint8_t *)pAddress + 1774190592);
        x1[4] = (FieldElement *)((uint8_t *)pAddress + 1807745024);
        x1[5] = (FieldElement *)((uint8_t *)pAddress + 1841299456);
        x1[6] = (FieldElement *)((uint8_t *)pAddress + 1874853888);
        x1[7] = (FieldElement *)((uint8_t *)pAddress + 1908408320);
        x1[8] = (FieldElement *)((uint8_t *)pAddress + 1941962752);
        x1[9] = (FieldElement *)((uint8_t *)pAddress + 1975517184);
        x1[10] = (FieldElement *)((uint8_t *)pAddress + 2009071616);
        x1[11] = (FieldElement *)((uint8_t *)pAddress + 2042626048);
        x1[12] = (FieldElement *)((uint8_t *)pAddress + 2076180480);
        x1[13] = (FieldElement *)((uint8_t *)pAddress + 2109734912);
        x1[14] = (FieldElement *)((uint8_t *)pAddress + 2143289344);
        x1[15] = (FieldElement *)((uint8_t *)pAddress + 2176843776);
        y1[0] = (FieldElement *)((uint8_t *)pAddress + 2210398208);
        y1[1] = (FieldElement *)((uint8_t *)pAddress + 2243952640);
        y1[2] = (FieldElement *)((uint8_t *)pAddress + 2277507072);
        y1[3] = (FieldElement *)((uint8_t *)pAddress + 2311061504);
        y1[4] = (FieldElement *)((uint8_t *)pAddress + 2344615936);
        y1[5] = (FieldElement *)((uint8_t *)pAddress + 2378170368);
        y1[6] = (FieldElement *)((uint8_t *)pAddress + 2411724800);
        y1[7] = (FieldElement *)((uint8_t *)pAddress + 2445279232);
        y1[8] = (FieldElement *)((uint8_t *)pAddress + 2478833664);
        y1[9] = (FieldElement *)((uint8_t *)pAddress + 2512388096);
        y1[10] = (FieldElement *)((uint8_t *)pAddress + 2545942528);
        y1[11] = (FieldElement *)((uint8_t *)pAddress + 2579496960);
        y1[12] = (FieldElement *)((uint8_t *)pAddress + 2613051392);
        y1[13] = (FieldElement *)((uint8_t *)pAddress + 2646605824);
        y1[14] = (FieldElement *)((uint8_t *)pAddress + 2680160256);
        y1[15] = (FieldElement *)((uint8_t *)pAddress + 2713714688);
        x2[0] = (FieldElement *)((uint8_t *)pAddress + 2747269120);
        x2[1] = (FieldElement *)((uint8_t *)pAddress + 2780823552);
        x2[2] = (FieldElement *)((uint8_t *)pAddress + 2814377984);
        x2[3] = (FieldElement *)((uint8_t *)pAddress + 2847932416);
        x2[4] = (FieldElement *)((uint8_t *)pAddress + 2881486848);
        x2[5] = (FieldElement *)((uint8_t *)pAddress + 2915041280);
        x2[6] = (FieldElement *)((uint8_t *)pAddress + 2948595712);
        x2[7] = (FieldElement *)((uint8_t *)pAddress + 2982150144);
        x2[8] = (FieldElement *)((uint8_t *)pAddress + 3015704576);
        x2[9] = (FieldElement *)((uint8_t *)pAddress + 3049259008);
        x2[10] = (FieldElement *)((uint8_t *)pAddress + 3082813440);
        x2[11] = (FieldElement *)((uint8_t *)pAddress + 3116367872);
        x2[12] = (FieldElement *)((uint8_t *)pAddress + 3149922304);
        x2[13] = (FieldElement *)((uint8_t *)pAddress + 3183476736);
        x2[14] = (FieldElement *)((uint8_t *)pAddress + 3217031168);
        x2[15] = (FieldElement *)((uint8_t *)pAddress + 3250585600);
        y2[0] = (FieldElement *)((uint8_t *)pAddress + 3284140032);
        y2[1] = (FieldElement *)((uint8_t *)pAddress + 3317694464);
        y2[2] = (FieldElement *)((uint8_t *)pAddress + 3351248896);
        y2[3] = (FieldElement *)((uint8_t *)pAddress + 3384803328);
        y2[4] = (FieldElement *)((uint8_t *)pAddress + 3418357760);
        y2[5] = (FieldElement *)((uint8_t *)pAddress + 3451912192);
        y2[6] = (FieldElement *)((uint8_t *)pAddress + 3485466624);
        y2[7] = (FieldElement *)((uint8_t *)pAddress + 3519021056);
        y2[8] = (FieldElement *)((uint8_t *)pAddress + 3552575488);
        y2[9] = (FieldElement *)((uint8_t *)pAddress + 3586129920);
        y2[10] = (FieldElement *)((uint8_t *)pAddress + 3619684352);
        y2[11] = (FieldElement *)((uint8_t *)pAddress + 3653238784);
        y2[12] = (FieldElement *)((uint8_t *)pAddress + 3686793216);
        y2[13] = (FieldElement *)((uint8_t *)pAddress + 3720347648);
        y2[14] = (FieldElement *)((uint8_t *)pAddress + 3753902080);
        y2[15] = (FieldElement *)((uint8_t *)pAddress + 3787456512);
        x3[0] = (FieldElement *)((uint8_t *)pAddress + 3821010944);
        x3[1] = (FieldElement *)((uint8_t *)pAddress + 3854565376);
        x3[2] = (FieldElement *)((uint8_t *)pAddress + 3888119808);
        x3[3] = (FieldElement *)((uint8_t *)pAddress + 3921674240);
        x3[4] = (FieldElement *)((uint8_t *)pAddress + 3955228672);
        x3[5] = (FieldElement *)((uint8_t *)pAddress + 3988783104);
        x3[6] = (FieldElement *)((uint8_t *)pAddress + 4022337536);
        x3[7] = (FieldElement *)((uint8_t *)pAddress + 4055891968);
        x3[8] = (FieldElement *)((uint8_t *)pAddress + 4089446400);
        x3[9] = (FieldElement *)((uint8_t *)pAddress + 4123000832);
        x3[10] = (FieldElement *)((uint8_t *)pAddress + 4156555264);
        x3[11] = (FieldElement *)((uint8_t *)pAddress + 4190109696);
        x3[12] = (FieldElement *)((uint8_t *)pAddress + 4223664128);
        x3[13] = (FieldElement *)((uint8_t *)pAddress + 4257218560);
        x3[14] = (FieldElement *)((uint8_t *)pAddress + 4290772992);
        x3[15] = (FieldElement *)((uint8_t *)pAddress + 4324327424);
        y3[0] = (FieldElement *)((uint8_t *)pAddress + 4357881856);
        y3[1] = (FieldElement *)((uint8_t *)pAddress + 4391436288);
        y3[2] = (FieldElement *)((uint8_t *)pAddress + 4424990720);
        y3[3] = (FieldElement *)((uint8_t *)pAddress + 4458545152);
        y3[4] = (FieldElement *)((uint8_t *)pAddress + 4492099584);
        y3[5] = (FieldElement *)((uint8_t *)pAddress + 4525654016);
        y3[6] = (FieldElement *)((uint8_t *)pAddress + 4559208448);
        y3[7] = (FieldElement *)((uint8_t *)pAddress + 4592762880);
        y3[8] = (FieldElement *)((uint8_t *)pAddress + 4626317312);
        y3[9] = (FieldElement *)((uint8_t *)pAddress + 4659871744);
        y3[10] = (FieldElement *)((uint8_t *)pAddress + 4693426176);
        y3[11] = (FieldElement *)((uint8_t *)pAddress + 4726980608);
        y3[12] = (FieldElement *)((uint8_t *)pAddress + 4760535040);
        y3[13] = (FieldElement *)((uint8_t *)pAddress + 4794089472);
        y3[14] = (FieldElement *)((uint8_t *)pAddress + 4827643904);
        y3[15] = (FieldElement *)((uint8_t *)pAddress + 4861198336);
        s[0] = (FieldElement *)((uint8_t *)pAddress + 4894752768);
        s[1] = (FieldElement *)((uint8_t *)pAddress + 4928307200);
        s[2] = (FieldElement *)((uint8_t *)pAddress + 4961861632);
        s[3] = (FieldElement *)((uint8_t *)pAddress + 4995416064);
        s[4] = (FieldElement *)((uint8_t *)pAddress + 5028970496);
        s[5] = (FieldElement *)((uint8_t *)pAddress + 5062524928);
        s[6] = (FieldElement *)((uint8_t *)pAddress + 5096079360);
        s[7] = (FieldElement *)((uint8_t *)pAddress + 5129633792);
        s[8] = (FieldElement *)((uint8_t *)pAddress + 5163188224);
        s[9] = (FieldElement *)((uint8_t *)pAddress + 5196742656);
        s[10] = (FieldElement *)((uint8_t *)pAddress + 5230297088);
        s[11] = (FieldElement *)((uint8_t *)pAddress + 5263851520);
        s[12] = (FieldElement *)((uint8_t *)pAddress + 5297405952);
        s[13] = (FieldElement *)((uint8_t *)pAddress + 5330960384);
        s[14] = (FieldElement *)((uint8_t *)pAddress + 5364514816);
        s[15] = (FieldElement *)((uint8_t *)pAddress + 5398069248);
        q0[0] = (FieldElement *)((uint8_t *)pAddress + 5431623680);
        q0[1] = (FieldElement *)((uint8_t *)pAddress + 5465178112);
        q0[2] = (FieldElement *)((uint8_t *)pAddress + 5498732544);
        q0[3] = (FieldElement *)((uint8_t *)pAddress + 5532286976);
        q0[4] = (FieldElement *)((uint8_t *)pAddress + 5565841408);
        q0[5] = (FieldElement *)((uint8_t *)pAddress + 5599395840);
        q0[6] = (FieldElement *)((uint8_t *)pAddress + 5632950272);
        q0[7] = (FieldElement *)((uint8_t *)pAddress + 5666504704);
        q0[8] = (FieldElement *)((uint8_t *)pAddress + 5700059136);
        q0[9] = (FieldElement *)((uint8_t *)pAddress + 5733613568);
        q0[10] = (FieldElement *)((uint8_t *)pAddress + 5767168000);
        q0[11] = (FieldElement *)((uint8_t *)pAddress + 5800722432);
        q0[12] = (FieldElement *)((uint8_t *)pAddress + 5834276864);
        q0[13] = (FieldElement *)((uint8_t *)pAddress + 5867831296);
        q0[14] = (FieldElement *)((uint8_t *)pAddress + 5901385728);
        q0[15] = (FieldElement *)((uint8_t *)pAddress + 5934940160);
        q1[0] = (FieldElement *)((uint8_t *)pAddress + 5968494592);
        q1[1] = (FieldElement *)((uint8_t *)pAddress + 6002049024);
        q1[2] = (FieldElement *)((uint8_t *)pAddress + 6035603456);
        q1[3] = (FieldElement *)((uint8_t *)pAddress + 6069157888);
        q1[4] = (FieldElement *)((uint8_t *)pAddress + 6102712320);
        q1[5] = (FieldElement *)((uint8_t *)pAddress + 6136266752);
        q1[6] = (FieldElement *)((uint8_t *)pAddress + 6169821184);
        q1[7] = (FieldElement *)((uint8_t *)pAddress + 6203375616);
        q1[8] = (FieldElement *)((uint8_t *)pAddress + 6236930048);
        q1[9] = (FieldElement *)((uint8_t *)pAddress + 6270484480);
        q1[10] = (FieldElement *)((uint8_t *)pAddress + 6304038912);
        q1[11] = (FieldElement *)((uint8_t *)pAddress + 6337593344);
        q1[12] = (FieldElement *)((uint8_t *)pAddress + 6371147776);
        q1[13] = (FieldElement *)((uint8_t *)pAddress + 6404702208);
        q1[14] = (FieldElement *)((uint8_t *)pAddress + 6438256640);
        q1[15] = (FieldElement *)((uint8_t *)pAddress + 6471811072);
        q2[0] = (FieldElement *)((uint8_t *)pAddress + 6505365504);
        q2[1] = (FieldElement *)((uint8_t *)pAddress + 6538919936);
        q2[2] = (FieldElement *)((uint8_t *)pAddress + 6572474368);
        q2[3] = (FieldElement *)((uint8_t *)pAddress + 6606028800);
        q2[4] = (FieldElement *)((uint8_t *)pAddress + 6639583232);
        q2[5] = (FieldElement *)((uint8_t *)pAddress + 6673137664);
        q2[6] = (FieldElement *)((uint8_t *)pAddress + 6706692096);
        q2[7] = (FieldElement *)((uint8_t *)pAddress + 6740246528);
        q2[8] = (FieldElement *)((uint8_t *)pAddress + 6773800960);
        q2[9] = (FieldElement *)((uint8_t *)pAddress + 6807355392);
        q2[10] = (FieldElement *)((uint8_t *)pAddress + 6840909824);
        q2[11] = (FieldElement *)((uint8_t *)pAddress + 6874464256);
        q2[12] = (FieldElement *)((uint8_t *)pAddress + 6908018688);
        q2[13] = (FieldElement *)((uint8_t *)pAddress + 6941573120);
        q2[14] = (FieldElement *)((uint8_t *)pAddress + 6975127552);
        q2[15] = (FieldElement *)((uint8_t *)pAddress + 7008681984);
        selEq[0] = (FieldElement *)((uint8_t *)pAddress + 7042236416);
        selEq[1] = (FieldElement *)((uint8_t *)pAddress + 7075790848);
        selEq[2] = (FieldElement *)((uint8_t *)pAddress + 7109345280);
        selEq[3] = (FieldElement *)((uint8_t *)pAddress + 7142899712);
        carryL[0] = (FieldElement *)((uint8_t *)pAddress + 7176454144);
        carryL[1] = (FieldElement *)((uint8_t *)pAddress + 7210008576);
        carryL[2] = (FieldElement *)((uint8_t *)pAddress + 7243563008);
        carryH[0] = (FieldElement *)((uint8_t *)pAddress + 7277117440);
        carryH[1] = (FieldElement *)((uint8_t *)pAddress + 7310671872);
        carryH[2] = (FieldElement *)((uint8_t *)pAddress + 7344226304);
    }

    static uint64_t degree (void) { return 4194304; }
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
    uint32_t * c0Temp;
    uint8_t * opcode;
    uint8_t * cIn;
    uint8_t * cOut;
    uint8_t * last;
    uint8_t * useCarry;

    BinaryCommitPols (void * pAddress)
    {
        freeInA = (uint8_t *)((uint8_t *)pAddress + 7377780736);
        freeInB = (uint8_t *)((uint8_t *)pAddress + 7381975040);
        freeInC = (uint8_t *)((uint8_t *)pAddress + 7386169344);
        a0 = (uint32_t *)((uint8_t *)pAddress + 7390363648);
        a1 = (uint32_t *)((uint8_t *)pAddress + 7423918080);
        a2 = (uint32_t *)((uint8_t *)pAddress + 7457472512);
        a3 = (uint32_t *)((uint8_t *)pAddress + 7491026944);
        a4 = (uint32_t *)((uint8_t *)pAddress + 7524581376);
        a5 = (uint32_t *)((uint8_t *)pAddress + 7558135808);
        a6 = (uint32_t *)((uint8_t *)pAddress + 7591690240);
        a7 = (uint32_t *)((uint8_t *)pAddress + 7625244672);
        b0 = (uint32_t *)((uint8_t *)pAddress + 7658799104);
        b1 = (uint32_t *)((uint8_t *)pAddress + 7692353536);
        b2 = (uint32_t *)((uint8_t *)pAddress + 7725907968);
        b3 = (uint32_t *)((uint8_t *)pAddress + 7759462400);
        b4 = (uint32_t *)((uint8_t *)pAddress + 7793016832);
        b5 = (uint32_t *)((uint8_t *)pAddress + 7826571264);
        b6 = (uint32_t *)((uint8_t *)pAddress + 7860125696);
        b7 = (uint32_t *)((uint8_t *)pAddress + 7893680128);
        c0 = (uint32_t *)((uint8_t *)pAddress + 7927234560);
        c1 = (uint32_t *)((uint8_t *)pAddress + 7960788992);
        c2 = (uint32_t *)((uint8_t *)pAddress + 7994343424);
        c3 = (uint32_t *)((uint8_t *)pAddress + 8027897856);
        c4 = (uint32_t *)((uint8_t *)pAddress + 8061452288);
        c5 = (uint32_t *)((uint8_t *)pAddress + 8095006720);
        c6 = (uint32_t *)((uint8_t *)pAddress + 8128561152);
        c7 = (uint32_t *)((uint8_t *)pAddress + 8162115584);
        c0Temp = (uint32_t *)((uint8_t *)pAddress + 8195670016);
        opcode = (uint8_t *)((uint8_t *)pAddress + 8229224448);
        cIn = (uint8_t *)((uint8_t *)pAddress + 8233418752);
        cOut = (uint8_t *)((uint8_t *)pAddress + 8237613056);
        last = (uint8_t *)((uint8_t *)pAddress + 8241807360);
        useCarry = (uint8_t *)((uint8_t *)pAddress + 8246001664);
    }

    static uint64_t degree (void) { return 4194304; }
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
        addr = (FieldElement *)((uint8_t *)pAddress + 8250195968);
        step = (FieldElement *)((uint8_t *)pAddress + 8283750400);
        mOp = (FieldElement *)((uint8_t *)pAddress + 8317304832);
        mWr = (FieldElement *)((uint8_t *)pAddress + 8350859264);
        val[0] = (FieldElement *)((uint8_t *)pAddress + 8384413696);
        val[1] = (FieldElement *)((uint8_t *)pAddress + 8417968128);
        val[2] = (FieldElement *)((uint8_t *)pAddress + 8451522560);
        val[3] = (FieldElement *)((uint8_t *)pAddress + 8485076992);
        val[4] = (FieldElement *)((uint8_t *)pAddress + 8518631424);
        val[5] = (FieldElement *)((uint8_t *)pAddress + 8552185856);
        val[6] = (FieldElement *)((uint8_t *)pAddress + 8585740288);
        val[7] = (FieldElement *)((uint8_t *)pAddress + 8619294720);
        lastAccess = (FieldElement *)((uint8_t *)pAddress + 8652849152);
    }

    static uint64_t degree (void) { return 4194304; }
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
        in0 = (FieldElement *)((uint8_t *)pAddress + 8686403584);
        in1 = (FieldElement *)((uint8_t *)pAddress + 8719958016);
        in2 = (FieldElement *)((uint8_t *)pAddress + 8753512448);
        in3 = (FieldElement *)((uint8_t *)pAddress + 8787066880);
        in4 = (FieldElement *)((uint8_t *)pAddress + 8820621312);
        in5 = (FieldElement *)((uint8_t *)pAddress + 8854175744);
        in6 = (FieldElement *)((uint8_t *)pAddress + 8887730176);
        in7 = (FieldElement *)((uint8_t *)pAddress + 8921284608);
        hashType = (FieldElement *)((uint8_t *)pAddress + 8954839040);
        cap1 = (FieldElement *)((uint8_t *)pAddress + 8988393472);
        cap2 = (FieldElement *)((uint8_t *)pAddress + 9021947904);
        cap3 = (FieldElement *)((uint8_t *)pAddress + 9055502336);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 9089056768);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 9122611200);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 9156165632);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 9189720064);
    }

    static uint64_t degree (void) { return 4194304; }
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
        free0 = (uint64_t *)((uint8_t *)pAddress + 9223274496);
        free1 = (uint64_t *)((uint8_t *)pAddress + 9256828928);
        free2 = (uint64_t *)((uint8_t *)pAddress + 9290383360);
        free3 = (uint64_t *)((uint8_t *)pAddress + 9323937792);
        hashLeft0 = (uint64_t *)((uint8_t *)pAddress + 9357492224);
        hashLeft1 = (uint64_t *)((uint8_t *)pAddress + 9391046656);
        hashLeft2 = (uint64_t *)((uint8_t *)pAddress + 9424601088);
        hashLeft3 = (uint64_t *)((uint8_t *)pAddress + 9458155520);
        hashRight0 = (uint64_t *)((uint8_t *)pAddress + 9491709952);
        hashRight1 = (uint64_t *)((uint8_t *)pAddress + 9525264384);
        hashRight2 = (uint64_t *)((uint8_t *)pAddress + 9558818816);
        hashRight3 = (uint64_t *)((uint8_t *)pAddress + 9592373248);
        oldRoot0 = (uint64_t *)((uint8_t *)pAddress + 9625927680);
        oldRoot1 = (uint64_t *)((uint8_t *)pAddress + 9659482112);
        oldRoot2 = (uint64_t *)((uint8_t *)pAddress + 9693036544);
        oldRoot3 = (uint64_t *)((uint8_t *)pAddress + 9726590976);
        newRoot0 = (uint64_t *)((uint8_t *)pAddress + 9760145408);
        newRoot1 = (uint64_t *)((uint8_t *)pAddress + 9793699840);
        newRoot2 = (uint64_t *)((uint8_t *)pAddress + 9827254272);
        newRoot3 = (uint64_t *)((uint8_t *)pAddress + 9860808704);
        valueLow0 = (uint64_t *)((uint8_t *)pAddress + 9894363136);
        valueLow1 = (uint64_t *)((uint8_t *)pAddress + 9927917568);
        valueLow2 = (uint64_t *)((uint8_t *)pAddress + 9961472000);
        valueLow3 = (uint64_t *)((uint8_t *)pAddress + 9995026432);
        valueHigh0 = (uint64_t *)((uint8_t *)pAddress + 10028580864);
        valueHigh1 = (uint64_t *)((uint8_t *)pAddress + 10062135296);
        valueHigh2 = (uint64_t *)((uint8_t *)pAddress + 10095689728);
        valueHigh3 = (uint64_t *)((uint8_t *)pAddress + 10129244160);
        siblingValueHash0 = (uint64_t *)((uint8_t *)pAddress + 10162798592);
        siblingValueHash1 = (uint64_t *)((uint8_t *)pAddress + 10196353024);
        siblingValueHash2 = (uint64_t *)((uint8_t *)pAddress + 10229907456);
        siblingValueHash3 = (uint64_t *)((uint8_t *)pAddress + 10263461888);
        rkey0 = (uint64_t *)((uint8_t *)pAddress + 10297016320);
        rkey1 = (uint64_t *)((uint8_t *)pAddress + 10330570752);
        rkey2 = (uint64_t *)((uint8_t *)pAddress + 10364125184);
        rkey3 = (uint64_t *)((uint8_t *)pAddress + 10397679616);
        siblingRkey0 = (uint64_t *)((uint8_t *)pAddress + 10431234048);
        siblingRkey1 = (uint64_t *)((uint8_t *)pAddress + 10464788480);
        siblingRkey2 = (uint64_t *)((uint8_t *)pAddress + 10498342912);
        siblingRkey3 = (uint64_t *)((uint8_t *)pAddress + 10531897344);
        rkeyBit = (uint64_t *)((uint8_t *)pAddress + 10565451776);
        level0 = (uint64_t *)((uint8_t *)pAddress + 10599006208);
        level1 = (uint64_t *)((uint8_t *)pAddress + 10632560640);
        level2 = (uint64_t *)((uint8_t *)pAddress + 10666115072);
        level3 = (uint64_t *)((uint8_t *)pAddress + 10699669504);
        pc = (uint64_t *)((uint8_t *)pAddress + 10733223936);
        selOldRoot = (uint8_t *)((uint8_t *)pAddress + 10766778368);
        selNewRoot = (uint8_t *)((uint8_t *)pAddress + 10770972672);
        selValueLow = (uint8_t *)((uint8_t *)pAddress + 10775166976);
        selValueHigh = (uint8_t *)((uint8_t *)pAddress + 10779361280);
        selSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 10783555584);
        selRkey = (uint8_t *)((uint8_t *)pAddress + 10787749888);
        selRkeyBit = (uint8_t *)((uint8_t *)pAddress + 10791944192);
        selSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 10796138496);
        selFree = (uint8_t *)((uint8_t *)pAddress + 10800332800);
        setHashLeft = (uint8_t *)((uint8_t *)pAddress + 10804527104);
        setHashRight = (uint8_t *)((uint8_t *)pAddress + 10808721408);
        setOldRoot = (uint8_t *)((uint8_t *)pAddress + 10812915712);
        setNewRoot = (uint8_t *)((uint8_t *)pAddress + 10817110016);
        setValueLow = (uint8_t *)((uint8_t *)pAddress + 10821304320);
        setValueHigh = (uint8_t *)((uint8_t *)pAddress + 10825498624);
        setSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 10829692928);
        setRkey = (uint8_t *)((uint8_t *)pAddress + 10833887232);
        setSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 10838081536);
        setRkeyBit = (uint8_t *)((uint8_t *)pAddress + 10842275840);
        setLevel = (uint8_t *)((uint8_t *)pAddress + 10846470144);
        iHash = (uint8_t *)((uint8_t *)pAddress + 10850664448);
        iHashType = (uint8_t *)((uint8_t *)pAddress + 10854858752);
        iLatchSet = (uint8_t *)((uint8_t *)pAddress + 10859053056);
        iLatchGet = (uint8_t *)((uint8_t *)pAddress + 10863247360);
        iClimbRkey = (uint8_t *)((uint8_t *)pAddress + 10867441664);
        iClimbSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 10871635968);
        iClimbSiblingRkeyN = (uint8_t *)((uint8_t *)pAddress + 10875830272);
        iRotateLevel = (uint8_t *)((uint8_t *)pAddress + 10880024576);
        iJmpz = (uint8_t *)((uint8_t *)pAddress + 10884218880);
        iJmp = (uint8_t *)((uint8_t *)pAddress + 10888413184);
        iConst0 = (uint64_t *)((uint8_t *)pAddress + 10892607488);
        iConst1 = (uint64_t *)((uint8_t *)pAddress + 10926161920);
        iConst2 = (uint64_t *)((uint8_t *)pAddress + 10959716352);
        iConst3 = (uint64_t *)((uint8_t *)pAddress + 10993270784);
        iAddress = (uint64_t *)((uint8_t *)pAddress + 11026825216);
        op0inv = (FieldElement *)((uint8_t *)pAddress + 11060379648);
    }

    static uint64_t degree (void) { return 4194304; }
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
        freeA = (FieldElement *)((uint8_t *)pAddress + 11093934080);
        freeB = (FieldElement *)((uint8_t *)pAddress + 11127488512);
        gateType = (FieldElement *)((uint8_t *)pAddress + 11161042944);
        freeANorm = (FieldElement *)((uint8_t *)pAddress + 11194597376);
        freeBNorm = (FieldElement *)((uint8_t *)pAddress + 11228151808);
        freeCNorm = (FieldElement *)((uint8_t *)pAddress + 11261706240);
        a = (FieldElement *)((uint8_t *)pAddress + 11295260672);
        b = (FieldElement *)((uint8_t *)pAddress + 11328815104);
        c = (FieldElement *)((uint8_t *)pAddress + 11362369536);
    }

    static uint64_t degree (void) { return 4194304; }
};

class KeccakFCommitPols
{
public:
    FieldElement * a;
    FieldElement * b;
    FieldElement * c;

    KeccakFCommitPols (void * pAddress)
    {
        a = (FieldElement *)((uint8_t *)pAddress + 11395923968);
        b = (FieldElement *)((uint8_t *)pAddress + 11429478400);
        c = (FieldElement *)((uint8_t *)pAddress + 11463032832);
    }

    static uint64_t degree (void) { return 4194304; }
};

class Nine2OneCommitPols
{
public:
    FieldElement * bit;
    FieldElement * field9;

    Nine2OneCommitPols (void * pAddress)
    {
        bit = (FieldElement *)((uint8_t *)pAddress + 11496587264);
        field9 = (FieldElement *)((uint8_t *)pAddress + 11530141696);
    }

    static uint64_t degree (void) { return 4194304; }
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
        rBit = (FieldElement *)((uint8_t *)pAddress + 11563696128);
        sOutBit = (FieldElement *)((uint8_t *)pAddress + 11597250560);
        r8 = (FieldElement *)((uint8_t *)pAddress + 11630804992);
        connected = (FieldElement *)((uint8_t *)pAddress + 11664359424);
        sOut0 = (FieldElement *)((uint8_t *)pAddress + 11697913856);
        sOut1 = (FieldElement *)((uint8_t *)pAddress + 11731468288);
        sOut2 = (FieldElement *)((uint8_t *)pAddress + 11765022720);
        sOut3 = (FieldElement *)((uint8_t *)pAddress + 11798577152);
        sOut4 = (FieldElement *)((uint8_t *)pAddress + 11832131584);
        sOut5 = (FieldElement *)((uint8_t *)pAddress + 11865686016);
        sOut6 = (FieldElement *)((uint8_t *)pAddress + 11899240448);
        sOut7 = (FieldElement *)((uint8_t *)pAddress + 11932794880);
    }

    static uint64_t degree (void) { return 4194304; }
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
        freeIn = (FieldElement *)((uint8_t *)pAddress + 11966349312);
        connected = (FieldElement *)((uint8_t *)pAddress + 11999903744);
        addr = (FieldElement *)((uint8_t *)pAddress + 12033458176);
        rem = (FieldElement *)((uint8_t *)pAddress + 12067012608);
        remInv = (FieldElement *)((uint8_t *)pAddress + 12100567040);
        spare = (FieldElement *)((uint8_t *)pAddress + 12134121472);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 12167675904);
        len = (FieldElement *)((uint8_t *)pAddress + 12201230336);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 12234784768);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 12268339200);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 12301893632);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 12335448064);
        hash4 = (FieldElement *)((uint8_t *)pAddress + 12369002496);
        hash5 = (FieldElement *)((uint8_t *)pAddress + 12402556928);
        hash6 = (FieldElement *)((uint8_t *)pAddress + 12436111360);
        hash7 = (FieldElement *)((uint8_t *)pAddress + 12469665792);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 12503220224);
        crLen = (FieldElement *)((uint8_t *)pAddress + 12536774656);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 12570329088);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 12603883520);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 12637437952);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 12670992384);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 12704546816);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 12738101248);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 12771655680);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 12805210112);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 12838764544);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 12872318976);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 12905873408);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 12939427840);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 12972982272);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 13006536704);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 13040091136);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 13073645568);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 13107200000);
    }

    static uint64_t degree (void) { return 4194304; }
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
        acc[0] = (FieldElement *)((uint8_t *)pAddress + 13140754432);
        acc[1] = (FieldElement *)((uint8_t *)pAddress + 13174308864);
        acc[2] = (FieldElement *)((uint8_t *)pAddress + 13207863296);
        acc[3] = (FieldElement *)((uint8_t *)pAddress + 13241417728);
        acc[4] = (FieldElement *)((uint8_t *)pAddress + 13274972160);
        acc[5] = (FieldElement *)((uint8_t *)pAddress + 13308526592);
        acc[6] = (FieldElement *)((uint8_t *)pAddress + 13342081024);
        acc[7] = (FieldElement *)((uint8_t *)pAddress + 13375635456);
        freeIn = (FieldElement *)((uint8_t *)pAddress + 13409189888);
        addr = (FieldElement *)((uint8_t *)pAddress + 13442744320);
        rem = (FieldElement *)((uint8_t *)pAddress + 13476298752);
        remInv = (FieldElement *)((uint8_t *)pAddress + 13509853184);
        spare = (FieldElement *)((uint8_t *)pAddress + 13543407616);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 13576962048);
        curHash0 = (FieldElement *)((uint8_t *)pAddress + 13610516480);
        curHash1 = (FieldElement *)((uint8_t *)pAddress + 13644070912);
        curHash2 = (FieldElement *)((uint8_t *)pAddress + 13677625344);
        curHash3 = (FieldElement *)((uint8_t *)pAddress + 13711179776);
        prevHash0 = (FieldElement *)((uint8_t *)pAddress + 13744734208);
        prevHash1 = (FieldElement *)((uint8_t *)pAddress + 13778288640);
        prevHash2 = (FieldElement *)((uint8_t *)pAddress + 13811843072);
        prevHash3 = (FieldElement *)((uint8_t *)pAddress + 13845397504);
        len = (FieldElement *)((uint8_t *)pAddress + 13878951936);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 13912506368);
        crLen = (FieldElement *)((uint8_t *)pAddress + 13946060800);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 13979615232);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 14013169664);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 14046724096);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 14080278528);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 14113832960);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 14147387392);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 14180941824);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 14214496256);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 14248050688);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 14281605120);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 14315159552);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 14348713984);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 14382268416);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 14415822848);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 14449377280);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 14482931712);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 14516486144);
    }

    static uint64_t degree (void) { return 4194304; }
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
    uint8_t * useCTX;
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
        A7 = (uint32_t *)((uint8_t *)pAddress + 14550040576);
        A6 = (uint32_t *)((uint8_t *)pAddress + 14583595008);
        A5 = (uint32_t *)((uint8_t *)pAddress + 14617149440);
        A4 = (uint32_t *)((uint8_t *)pAddress + 14650703872);
        A3 = (uint32_t *)((uint8_t *)pAddress + 14684258304);
        A2 = (uint32_t *)((uint8_t *)pAddress + 14717812736);
        A1 = (uint32_t *)((uint8_t *)pAddress + 14751367168);
        A0 = (FieldElement *)((uint8_t *)pAddress + 14784921600);
        B7 = (uint32_t *)((uint8_t *)pAddress + 14818476032);
        B6 = (uint32_t *)((uint8_t *)pAddress + 14852030464);
        B5 = (uint32_t *)((uint8_t *)pAddress + 14885584896);
        B4 = (uint32_t *)((uint8_t *)pAddress + 14919139328);
        B3 = (uint32_t *)((uint8_t *)pAddress + 14952693760);
        B2 = (uint32_t *)((uint8_t *)pAddress + 14986248192);
        B1 = (uint32_t *)((uint8_t *)pAddress + 15019802624);
        B0 = (FieldElement *)((uint8_t *)pAddress + 15053357056);
        C7 = (uint32_t *)((uint8_t *)pAddress + 15086911488);
        C6 = (uint32_t *)((uint8_t *)pAddress + 15120465920);
        C5 = (uint32_t *)((uint8_t *)pAddress + 15154020352);
        C4 = (uint32_t *)((uint8_t *)pAddress + 15187574784);
        C3 = (uint32_t *)((uint8_t *)pAddress + 15221129216);
        C2 = (uint32_t *)((uint8_t *)pAddress + 15254683648);
        C1 = (uint32_t *)((uint8_t *)pAddress + 15288238080);
        C0 = (FieldElement *)((uint8_t *)pAddress + 15321792512);
        D7 = (uint32_t *)((uint8_t *)pAddress + 15355346944);
        D6 = (uint32_t *)((uint8_t *)pAddress + 15388901376);
        D5 = (uint32_t *)((uint8_t *)pAddress + 15422455808);
        D4 = (uint32_t *)((uint8_t *)pAddress + 15456010240);
        D3 = (uint32_t *)((uint8_t *)pAddress + 15489564672);
        D2 = (uint32_t *)((uint8_t *)pAddress + 15523119104);
        D1 = (uint32_t *)((uint8_t *)pAddress + 15556673536);
        D0 = (FieldElement *)((uint8_t *)pAddress + 15590227968);
        E7 = (uint32_t *)((uint8_t *)pAddress + 15623782400);
        E6 = (uint32_t *)((uint8_t *)pAddress + 15657336832);
        E5 = (uint32_t *)((uint8_t *)pAddress + 15690891264);
        E4 = (uint32_t *)((uint8_t *)pAddress + 15724445696);
        E3 = (uint32_t *)((uint8_t *)pAddress + 15758000128);
        E2 = (uint32_t *)((uint8_t *)pAddress + 15791554560);
        E1 = (uint32_t *)((uint8_t *)pAddress + 15825108992);
        E0 = (FieldElement *)((uint8_t *)pAddress + 15858663424);
        SR7 = (uint32_t *)((uint8_t *)pAddress + 15892217856);
        SR6 = (uint32_t *)((uint8_t *)pAddress + 15925772288);
        SR5 = (uint32_t *)((uint8_t *)pAddress + 15959326720);
        SR4 = (uint32_t *)((uint8_t *)pAddress + 15992881152);
        SR3 = (uint32_t *)((uint8_t *)pAddress + 16026435584);
        SR2 = (uint32_t *)((uint8_t *)pAddress + 16059990016);
        SR1 = (uint32_t *)((uint8_t *)pAddress + 16093544448);
        SR0 = (uint32_t *)((uint8_t *)pAddress + 16127098880);
        CTX = (uint32_t *)((uint8_t *)pAddress + 16160653312);
        SP = (uint16_t *)((uint8_t *)pAddress + 16194207744);
        PC = (uint32_t *)((uint8_t *)pAddress + 16202596352);
        GAS = (uint64_t *)((uint8_t *)pAddress + 16236150784);
        MAXMEM = (uint32_t *)((uint8_t *)pAddress + 16269705216);
        zkPC = (uint32_t *)((uint8_t *)pAddress + 16303259648);
        RR = (uint32_t *)((uint8_t *)pAddress + 16336814080);
        HASHPOS = (uint32_t *)((uint8_t *)pAddress + 16370368512);
        CONST7 = (FieldElement *)((uint8_t *)pAddress + 16403922944);
        CONST6 = (FieldElement *)((uint8_t *)pAddress + 16437477376);
        CONST5 = (FieldElement *)((uint8_t *)pAddress + 16471031808);
        CONST4 = (FieldElement *)((uint8_t *)pAddress + 16504586240);
        CONST3 = (FieldElement *)((uint8_t *)pAddress + 16538140672);
        CONST2 = (FieldElement *)((uint8_t *)pAddress + 16571695104);
        CONST1 = (FieldElement *)((uint8_t *)pAddress + 16605249536);
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 16638803968);
        FREE7 = (FieldElement *)((uint8_t *)pAddress + 16672358400);
        FREE6 = (FieldElement *)((uint8_t *)pAddress + 16705912832);
        FREE5 = (FieldElement *)((uint8_t *)pAddress + 16739467264);
        FREE4 = (FieldElement *)((uint8_t *)pAddress + 16773021696);
        FREE3 = (FieldElement *)((uint8_t *)pAddress + 16806576128);
        FREE2 = (FieldElement *)((uint8_t *)pAddress + 16840130560);
        FREE1 = (FieldElement *)((uint8_t *)pAddress + 16873684992);
        FREE0 = (FieldElement *)((uint8_t *)pAddress + 16907239424);
        inA = (FieldElement *)((uint8_t *)pAddress + 16940793856);
        inB = (FieldElement *)((uint8_t *)pAddress + 16974348288);
        inC = (FieldElement *)((uint8_t *)pAddress + 17007902720);
        inD = (FieldElement *)((uint8_t *)pAddress + 17041457152);
        inE = (FieldElement *)((uint8_t *)pAddress + 17075011584);
        inSR = (FieldElement *)((uint8_t *)pAddress + 17108566016);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 17142120448);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 17175674880);
        inSP = (FieldElement *)((uint8_t *)pAddress + 17209229312);
        inPC = (FieldElement *)((uint8_t *)pAddress + 17242783744);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 17276338176);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 17309892608);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 17343447040);
        inRR = (FieldElement *)((uint8_t *)pAddress + 17377001472);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 17410555904);
        setA = (uint8_t *)((uint8_t *)pAddress + 17444110336);
        setB = (uint8_t *)((uint8_t *)pAddress + 17448304640);
        setC = (uint8_t *)((uint8_t *)pAddress + 17452498944);
        setD = (uint8_t *)((uint8_t *)pAddress + 17456693248);
        setE = (uint8_t *)((uint8_t *)pAddress + 17460887552);
        setSR = (uint8_t *)((uint8_t *)pAddress + 17465081856);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 17469276160);
        setSP = (uint8_t *)((uint8_t *)pAddress + 17473470464);
        setPC = (uint8_t *)((uint8_t *)pAddress + 17477664768);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 17481859072);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 17486053376);
        JMP = (uint8_t *)((uint8_t *)pAddress + 17490247680);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 17494441984);
        setRR = (uint8_t *)((uint8_t *)pAddress + 17498636288);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 17502830592);
        offset = (uint32_t *)((uint8_t *)pAddress + 17507024896);
        incStack = (int32_t *)((uint8_t *)pAddress + 17540579328);
        incCode = (int32_t *)((uint8_t *)pAddress + 17557356544);
        isStack = (uint8_t *)((uint8_t *)pAddress + 17574133760);
        isCode = (uint8_t *)((uint8_t *)pAddress + 17578328064);
        isMem = (uint8_t *)((uint8_t *)pAddress + 17582522368);
        ind = (uint8_t *)((uint8_t *)pAddress + 17586716672);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 17590910976);
        mOp = (uint8_t *)((uint8_t *)pAddress + 17595105280);
        mWR = (uint8_t *)((uint8_t *)pAddress + 17599299584);
        sWR = (uint8_t *)((uint8_t *)pAddress + 17603493888);
        sRD = (uint8_t *)((uint8_t *)pAddress + 17607688192);
        arith = (uint8_t *)((uint8_t *)pAddress + 17611882496);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 17616076800);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 17620271104);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 17624465408);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 17628659712);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 17632854016);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 17637048320);
        hashK = (uint8_t *)((uint8_t *)pAddress + 17641242624);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 17645436928);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 17649631232);
        hashP = (uint8_t *)((uint8_t *)pAddress + 17653825536);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 17658019840);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 17662214144);
        bin = (uint8_t *)((uint8_t *)pAddress + 17666408448);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 17670602752);
        assert = (uint8_t *)((uint8_t *)pAddress + 17674797056);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 17678991360);
        isNeg = (uint8_t *)((uint8_t *)pAddress + 17683185664);
        isMaxMem = (uint8_t *)((uint8_t *)pAddress + 17687379968);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 17691574272);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 17725128704);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 17758683136);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 17792237568);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 17825792000);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 17859346432);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 17892900864);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 17926455296);
    }

    static uint64_t degree (void) { return 4194304; }
};

class CommitPols
{
public:
    MemAlignCommitPols MemAlign;
    Byte4CommitPols Byte4;
    ArithCommitPols Arith;
    BinaryCommitPols Binary;
    MemCommitPols Mem;
    PoseidonGCommitPols PoseidonG;
    StorageCommitPols Storage;
    NormGate9CommitPols NormGate9;
    KeccakFCommitPols KeccakF;
    Nine2OneCommitPols Nine2One;
    PaddingKKBitCommitPols PaddingKKBit;
    PaddingKKCommitPols PaddingKK;
    PaddingPGCommitPols PaddingPG;
    MainCommitPols Main;

    CommitPols (void * pAddress) : MemAlign(pAddress), Byte4(pAddress), Arith(pAddress), Binary(pAddress), Mem(pAddress), PoseidonG(pAddress), Storage(pAddress), NormGate9(pAddress), KeccakF(pAddress), Nine2One(pAddress), PaddingKKBit(pAddress), PaddingKK(pAddress), PaddingPG(pAddress), Main(pAddress) {}

    static uint64_t size (void) { return 17960009728; }
};

#endif // COMMIT_POLS_HPP
