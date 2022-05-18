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
        out = (uint32_t *)((uint8_t *)pAddress + 8388608);
    }

    Byte4CommitPols (void * pAddress, uint64_t degree)
    {
        freeIN = (uint16_t *)((uint8_t *)pAddress + 0*degree);
        out = (uint32_t *)((uint8_t *)pAddress + 2*degree);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 10; }
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

    MemAlignCommitPols (void * pAddress)
    {
        inM = (uint8_t *)((uint8_t *)pAddress + 41943040);
        inV = (uint8_t *)((uint8_t *)pAddress + 46137344);
        wr = (uint8_t *)((uint8_t *)pAddress + 50331648);
        m0[0] = (uint32_t *)((uint8_t *)pAddress + 54525952);
        m0[1] = (uint32_t *)((uint8_t *)pAddress + 88080384);
        m0[2] = (uint32_t *)((uint8_t *)pAddress + 121634816);
        m0[3] = (uint32_t *)((uint8_t *)pAddress + 155189248);
        m0[4] = (uint32_t *)((uint8_t *)pAddress + 188743680);
        m0[5] = (uint32_t *)((uint8_t *)pAddress + 222298112);
        m0[6] = (uint32_t *)((uint8_t *)pAddress + 255852544);
        m0[7] = (uint32_t *)((uint8_t *)pAddress + 289406976);
        m1[0] = (uint32_t *)((uint8_t *)pAddress + 322961408);
        m1[1] = (uint32_t *)((uint8_t *)pAddress + 356515840);
        m1[2] = (uint32_t *)((uint8_t *)pAddress + 390070272);
        m1[3] = (uint32_t *)((uint8_t *)pAddress + 423624704);
        m1[4] = (uint32_t *)((uint8_t *)pAddress + 457179136);
        m1[5] = (uint32_t *)((uint8_t *)pAddress + 490733568);
        m1[6] = (uint32_t *)((uint8_t *)pAddress + 524288000);
        m1[7] = (uint32_t *)((uint8_t *)pAddress + 557842432);
        w0[0] = (uint32_t *)((uint8_t *)pAddress + 591396864);
        w0[1] = (uint32_t *)((uint8_t *)pAddress + 624951296);
        w0[2] = (uint32_t *)((uint8_t *)pAddress + 658505728);
        w0[3] = (uint32_t *)((uint8_t *)pAddress + 692060160);
        w0[4] = (uint32_t *)((uint8_t *)pAddress + 725614592);
        w0[5] = (uint32_t *)((uint8_t *)pAddress + 759169024);
        w0[6] = (uint32_t *)((uint8_t *)pAddress + 792723456);
        w0[7] = (uint32_t *)((uint8_t *)pAddress + 826277888);
        w1[0] = (uint32_t *)((uint8_t *)pAddress + 859832320);
        w1[1] = (uint32_t *)((uint8_t *)pAddress + 893386752);
        w1[2] = (uint32_t *)((uint8_t *)pAddress + 926941184);
        w1[3] = (uint32_t *)((uint8_t *)pAddress + 960495616);
        w1[4] = (uint32_t *)((uint8_t *)pAddress + 994050048);
        w1[5] = (uint32_t *)((uint8_t *)pAddress + 1027604480);
        w1[6] = (uint32_t *)((uint8_t *)pAddress + 1061158912);
        w1[7] = (uint32_t *)((uint8_t *)pAddress + 1094713344);
        v[0] = (uint32_t *)((uint8_t *)pAddress + 1128267776);
        v[1] = (uint32_t *)((uint8_t *)pAddress + 1161822208);
        v[2] = (uint32_t *)((uint8_t *)pAddress + 1195376640);
        v[3] = (uint32_t *)((uint8_t *)pAddress + 1228931072);
        v[4] = (uint32_t *)((uint8_t *)pAddress + 1262485504);
        v[5] = (uint32_t *)((uint8_t *)pAddress + 1296039936);
        v[6] = (uint32_t *)((uint8_t *)pAddress + 1329594368);
        v[7] = (uint32_t *)((uint8_t *)pAddress + 1363148800);
        offset = (uint8_t *)((uint8_t *)pAddress + 1396703232);
        selW = (uint8_t *)((uint8_t *)pAddress + 1400897536);
        factorV[0] = (FieldElement *)((uint8_t *)pAddress + 1405091840);
        factorV[1] = (FieldElement *)((uint8_t *)pAddress + 1438646272);
        factorV[2] = (FieldElement *)((uint8_t *)pAddress + 1472200704);
        factorV[3] = (FieldElement *)((uint8_t *)pAddress + 1505755136);
        factorV[4] = (FieldElement *)((uint8_t *)pAddress + 1539309568);
        factorV[5] = (FieldElement *)((uint8_t *)pAddress + 1572864000);
        factorV[6] = (FieldElement *)((uint8_t *)pAddress + 1606418432);
        factorV[7] = (FieldElement *)((uint8_t *)pAddress + 1639972864);
    }

    MemAlignCommitPols (void * pAddress, uint64_t degree)
    {
        inM = (uint8_t *)((uint8_t *)pAddress + 0*degree);
        inV = (uint8_t *)((uint8_t *)pAddress + 1*degree);
        wr = (uint8_t *)((uint8_t *)pAddress + 2*degree);
        m0[0] = (uint32_t *)((uint8_t *)pAddress + 3*degree);
        m0[1] = (uint32_t *)((uint8_t *)pAddress + 11*degree);
        m0[2] = (uint32_t *)((uint8_t *)pAddress + 19*degree);
        m0[3] = (uint32_t *)((uint8_t *)pAddress + 27*degree);
        m0[4] = (uint32_t *)((uint8_t *)pAddress + 35*degree);
        m0[5] = (uint32_t *)((uint8_t *)pAddress + 43*degree);
        m0[6] = (uint32_t *)((uint8_t *)pAddress + 51*degree);
        m0[7] = (uint32_t *)((uint8_t *)pAddress + 59*degree);
        m1[0] = (uint32_t *)((uint8_t *)pAddress + 67*degree);
        m1[1] = (uint32_t *)((uint8_t *)pAddress + 75*degree);
        m1[2] = (uint32_t *)((uint8_t *)pAddress + 83*degree);
        m1[3] = (uint32_t *)((uint8_t *)pAddress + 91*degree);
        m1[4] = (uint32_t *)((uint8_t *)pAddress + 99*degree);
        m1[5] = (uint32_t *)((uint8_t *)pAddress + 107*degree);
        m1[6] = (uint32_t *)((uint8_t *)pAddress + 115*degree);
        m1[7] = (uint32_t *)((uint8_t *)pAddress + 123*degree);
        w0[0] = (uint32_t *)((uint8_t *)pAddress + 131*degree);
        w0[1] = (uint32_t *)((uint8_t *)pAddress + 139*degree);
        w0[2] = (uint32_t *)((uint8_t *)pAddress + 147*degree);
        w0[3] = (uint32_t *)((uint8_t *)pAddress + 155*degree);
        w0[4] = (uint32_t *)((uint8_t *)pAddress + 163*degree);
        w0[5] = (uint32_t *)((uint8_t *)pAddress + 171*degree);
        w0[6] = (uint32_t *)((uint8_t *)pAddress + 179*degree);
        w0[7] = (uint32_t *)((uint8_t *)pAddress + 187*degree);
        w1[0] = (uint32_t *)((uint8_t *)pAddress + 195*degree);
        w1[1] = (uint32_t *)((uint8_t *)pAddress + 203*degree);
        w1[2] = (uint32_t *)((uint8_t *)pAddress + 211*degree);
        w1[3] = (uint32_t *)((uint8_t *)pAddress + 219*degree);
        w1[4] = (uint32_t *)((uint8_t *)pAddress + 227*degree);
        w1[5] = (uint32_t *)((uint8_t *)pAddress + 235*degree);
        w1[6] = (uint32_t *)((uint8_t *)pAddress + 243*degree);
        w1[7] = (uint32_t *)((uint8_t *)pAddress + 251*degree);
        v[0] = (uint32_t *)((uint8_t *)pAddress + 259*degree);
        v[1] = (uint32_t *)((uint8_t *)pAddress + 267*degree);
        v[2] = (uint32_t *)((uint8_t *)pAddress + 275*degree);
        v[3] = (uint32_t *)((uint8_t *)pAddress + 283*degree);
        v[4] = (uint32_t *)((uint8_t *)pAddress + 291*degree);
        v[5] = (uint32_t *)((uint8_t *)pAddress + 299*degree);
        v[6] = (uint32_t *)((uint8_t *)pAddress + 307*degree);
        v[7] = (uint32_t *)((uint8_t *)pAddress + 315*degree);
        offset = (uint8_t *)((uint8_t *)pAddress + 323*degree);
        selW = (uint8_t *)((uint8_t *)pAddress + 324*degree);
        factorV[0] = (FieldElement *)((uint8_t *)pAddress + 325*degree);
        factorV[1] = (FieldElement *)((uint8_t *)pAddress + 333*degree);
        factorV[2] = (FieldElement *)((uint8_t *)pAddress + 341*degree);
        factorV[3] = (FieldElement *)((uint8_t *)pAddress + 349*degree);
        factorV[4] = (FieldElement *)((uint8_t *)pAddress + 357*degree);
        factorV[5] = (FieldElement *)((uint8_t *)pAddress + 365*degree);
        factorV[6] = (FieldElement *)((uint8_t *)pAddress + 373*degree);
        factorV[7] = (FieldElement *)((uint8_t *)pAddress + 381*degree);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 389; }
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

    static uint64_t degree (void) { return 4194304; }
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
    uint32_t * c0Temp;
    uint8_t * opcode;
    uint8_t * cIn;
    uint8_t * cOut;
    uint8_t * lCout;
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
        lCout = (uint8_t *)((uint8_t *)pAddress + 8241807360);
        last = (uint8_t *)((uint8_t *)pAddress + 8246001664);
        useCarry = (uint8_t *)((uint8_t *)pAddress + 8250195968);
    }

    BinaryCommitPols (void * pAddress, uint64_t degree)
    {
        freeInA = (uint8_t *)((uint8_t *)pAddress + 0*degree);
        freeInB = (uint8_t *)((uint8_t *)pAddress + 1*degree);
        freeInC = (uint8_t *)((uint8_t *)pAddress + 2*degree);
        a0 = (uint32_t *)((uint8_t *)pAddress + 3*degree);
        a1 = (uint32_t *)((uint8_t *)pAddress + 11*degree);
        a2 = (uint32_t *)((uint8_t *)pAddress + 19*degree);
        a3 = (uint32_t *)((uint8_t *)pAddress + 27*degree);
        a4 = (uint32_t *)((uint8_t *)pAddress + 35*degree);
        a5 = (uint32_t *)((uint8_t *)pAddress + 43*degree);
        a6 = (uint32_t *)((uint8_t *)pAddress + 51*degree);
        a7 = (uint32_t *)((uint8_t *)pAddress + 59*degree);
        b0 = (uint32_t *)((uint8_t *)pAddress + 67*degree);
        b1 = (uint32_t *)((uint8_t *)pAddress + 75*degree);
        b2 = (uint32_t *)((uint8_t *)pAddress + 83*degree);
        b3 = (uint32_t *)((uint8_t *)pAddress + 91*degree);
        b4 = (uint32_t *)((uint8_t *)pAddress + 99*degree);
        b5 = (uint32_t *)((uint8_t *)pAddress + 107*degree);
        b6 = (uint32_t *)((uint8_t *)pAddress + 115*degree);
        b7 = (uint32_t *)((uint8_t *)pAddress + 123*degree);
        c0 = (uint32_t *)((uint8_t *)pAddress + 131*degree);
        c1 = (uint32_t *)((uint8_t *)pAddress + 139*degree);
        c2 = (uint32_t *)((uint8_t *)pAddress + 147*degree);
        c3 = (uint32_t *)((uint8_t *)pAddress + 155*degree);
        c4 = (uint32_t *)((uint8_t *)pAddress + 163*degree);
        c5 = (uint32_t *)((uint8_t *)pAddress + 171*degree);
        c6 = (uint32_t *)((uint8_t *)pAddress + 179*degree);
        c7 = (uint32_t *)((uint8_t *)pAddress + 187*degree);
        c0Temp = (uint32_t *)((uint8_t *)pAddress + 195*degree);
        opcode = (uint8_t *)((uint8_t *)pAddress + 203*degree);
        cIn = (uint8_t *)((uint8_t *)pAddress + 204*degree);
        cOut = (uint8_t *)((uint8_t *)pAddress + 205*degree);
        lCout = (uint8_t *)((uint8_t *)pAddress + 206*degree);
        last = (uint8_t *)((uint8_t *)pAddress + 207*degree);
        useCarry = (uint8_t *)((uint8_t *)pAddress + 208*degree);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 209; }
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
        in0 = (FieldElement *)((uint8_t *)pAddress + 8254390272);
        in1 = (FieldElement *)((uint8_t *)pAddress + 8287944704);
        in2 = (FieldElement *)((uint8_t *)pAddress + 8321499136);
        in3 = (FieldElement *)((uint8_t *)pAddress + 8355053568);
        in4 = (FieldElement *)((uint8_t *)pAddress + 8388608000);
        in5 = (FieldElement *)((uint8_t *)pAddress + 8422162432);
        in6 = (FieldElement *)((uint8_t *)pAddress + 8455716864);
        in7 = (FieldElement *)((uint8_t *)pAddress + 8489271296);
        hashType = (FieldElement *)((uint8_t *)pAddress + 8522825728);
        cap1 = (FieldElement *)((uint8_t *)pAddress + 8556380160);
        cap2 = (FieldElement *)((uint8_t *)pAddress + 8589934592);
        cap3 = (FieldElement *)((uint8_t *)pAddress + 8623489024);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 8657043456);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 8690597888);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 8724152320);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 8757706752);
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

    static uint64_t degree (void) { return 4194304; }
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
        acc[0] = (FieldElement *)((uint8_t *)pAddress + 8791261184);
        acc[1] = (FieldElement *)((uint8_t *)pAddress + 8824815616);
        acc[2] = (FieldElement *)((uint8_t *)pAddress + 8858370048);
        acc[3] = (FieldElement *)((uint8_t *)pAddress + 8891924480);
        acc[4] = (FieldElement *)((uint8_t *)pAddress + 8925478912);
        acc[5] = (FieldElement *)((uint8_t *)pAddress + 8959033344);
        acc[6] = (FieldElement *)((uint8_t *)pAddress + 8992587776);
        acc[7] = (FieldElement *)((uint8_t *)pAddress + 9026142208);
        freeIn = (FieldElement *)((uint8_t *)pAddress + 9059696640);
        addr = (FieldElement *)((uint8_t *)pAddress + 9093251072);
        rem = (FieldElement *)((uint8_t *)pAddress + 9126805504);
        remInv = (FieldElement *)((uint8_t *)pAddress + 9160359936);
        spare = (FieldElement *)((uint8_t *)pAddress + 9193914368);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 9227468800);
        curHash0 = (FieldElement *)((uint8_t *)pAddress + 9261023232);
        curHash1 = (FieldElement *)((uint8_t *)pAddress + 9294577664);
        curHash2 = (FieldElement *)((uint8_t *)pAddress + 9328132096);
        curHash3 = (FieldElement *)((uint8_t *)pAddress + 9361686528);
        prevHash0 = (FieldElement *)((uint8_t *)pAddress + 9395240960);
        prevHash1 = (FieldElement *)((uint8_t *)pAddress + 9428795392);
        prevHash2 = (FieldElement *)((uint8_t *)pAddress + 9462349824);
        prevHash3 = (FieldElement *)((uint8_t *)pAddress + 9495904256);
        len = (FieldElement *)((uint8_t *)pAddress + 9529458688);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 9563013120);
        crLen = (FieldElement *)((uint8_t *)pAddress + 9596567552);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 9630121984);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 9663676416);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 9697230848);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 9730785280);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 9764339712);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 9797894144);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 9831448576);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 9865003008);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 9898557440);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 9932111872);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 9965666304);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 9999220736);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 10032775168);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 10066329600);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 10099884032);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 10133438464);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 10166992896);
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

    static uint64_t degree (void) { return 4194304; }
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
        free0 = (uint64_t *)((uint8_t *)pAddress + 10200547328);
        free1 = (uint64_t *)((uint8_t *)pAddress + 10234101760);
        free2 = (uint64_t *)((uint8_t *)pAddress + 10267656192);
        free3 = (uint64_t *)((uint8_t *)pAddress + 10301210624);
        hashLeft0 = (uint64_t *)((uint8_t *)pAddress + 10334765056);
        hashLeft1 = (uint64_t *)((uint8_t *)pAddress + 10368319488);
        hashLeft2 = (uint64_t *)((uint8_t *)pAddress + 10401873920);
        hashLeft3 = (uint64_t *)((uint8_t *)pAddress + 10435428352);
        hashRight0 = (uint64_t *)((uint8_t *)pAddress + 10468982784);
        hashRight1 = (uint64_t *)((uint8_t *)pAddress + 10502537216);
        hashRight2 = (uint64_t *)((uint8_t *)pAddress + 10536091648);
        hashRight3 = (uint64_t *)((uint8_t *)pAddress + 10569646080);
        oldRoot0 = (uint64_t *)((uint8_t *)pAddress + 10603200512);
        oldRoot1 = (uint64_t *)((uint8_t *)pAddress + 10636754944);
        oldRoot2 = (uint64_t *)((uint8_t *)pAddress + 10670309376);
        oldRoot3 = (uint64_t *)((uint8_t *)pAddress + 10703863808);
        newRoot0 = (uint64_t *)((uint8_t *)pAddress + 10737418240);
        newRoot1 = (uint64_t *)((uint8_t *)pAddress + 10770972672);
        newRoot2 = (uint64_t *)((uint8_t *)pAddress + 10804527104);
        newRoot3 = (uint64_t *)((uint8_t *)pAddress + 10838081536);
        valueLow0 = (uint64_t *)((uint8_t *)pAddress + 10871635968);
        valueLow1 = (uint64_t *)((uint8_t *)pAddress + 10905190400);
        valueLow2 = (uint64_t *)((uint8_t *)pAddress + 10938744832);
        valueLow3 = (uint64_t *)((uint8_t *)pAddress + 10972299264);
        valueHigh0 = (uint64_t *)((uint8_t *)pAddress + 11005853696);
        valueHigh1 = (uint64_t *)((uint8_t *)pAddress + 11039408128);
        valueHigh2 = (uint64_t *)((uint8_t *)pAddress + 11072962560);
        valueHigh3 = (uint64_t *)((uint8_t *)pAddress + 11106516992);
        siblingValueHash0 = (uint64_t *)((uint8_t *)pAddress + 11140071424);
        siblingValueHash1 = (uint64_t *)((uint8_t *)pAddress + 11173625856);
        siblingValueHash2 = (uint64_t *)((uint8_t *)pAddress + 11207180288);
        siblingValueHash3 = (uint64_t *)((uint8_t *)pAddress + 11240734720);
        rkey0 = (uint64_t *)((uint8_t *)pAddress + 11274289152);
        rkey1 = (uint64_t *)((uint8_t *)pAddress + 11307843584);
        rkey2 = (uint64_t *)((uint8_t *)pAddress + 11341398016);
        rkey3 = (uint64_t *)((uint8_t *)pAddress + 11374952448);
        siblingRkey0 = (uint64_t *)((uint8_t *)pAddress + 11408506880);
        siblingRkey1 = (uint64_t *)((uint8_t *)pAddress + 11442061312);
        siblingRkey2 = (uint64_t *)((uint8_t *)pAddress + 11475615744);
        siblingRkey3 = (uint64_t *)((uint8_t *)pAddress + 11509170176);
        rkeyBit = (uint64_t *)((uint8_t *)pAddress + 11542724608);
        level0 = (uint64_t *)((uint8_t *)pAddress + 11576279040);
        level1 = (uint64_t *)((uint8_t *)pAddress + 11609833472);
        level2 = (uint64_t *)((uint8_t *)pAddress + 11643387904);
        level3 = (uint64_t *)((uint8_t *)pAddress + 11676942336);
        pc = (uint64_t *)((uint8_t *)pAddress + 11710496768);
        selOldRoot = (uint8_t *)((uint8_t *)pAddress + 11744051200);
        selNewRoot = (uint8_t *)((uint8_t *)pAddress + 11748245504);
        selValueLow = (uint8_t *)((uint8_t *)pAddress + 11752439808);
        selValueHigh = (uint8_t *)((uint8_t *)pAddress + 11756634112);
        selSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 11760828416);
        selRkey = (uint8_t *)((uint8_t *)pAddress + 11765022720);
        selRkeyBit = (uint8_t *)((uint8_t *)pAddress + 11769217024);
        selSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 11773411328);
        selFree = (uint8_t *)((uint8_t *)pAddress + 11777605632);
        setHashLeft = (uint8_t *)((uint8_t *)pAddress + 11781799936);
        setHashRight = (uint8_t *)((uint8_t *)pAddress + 11785994240);
        setOldRoot = (uint8_t *)((uint8_t *)pAddress + 11790188544);
        setNewRoot = (uint8_t *)((uint8_t *)pAddress + 11794382848);
        setValueLow = (uint8_t *)((uint8_t *)pAddress + 11798577152);
        setValueHigh = (uint8_t *)((uint8_t *)pAddress + 11802771456);
        setSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 11806965760);
        setRkey = (uint8_t *)((uint8_t *)pAddress + 11811160064);
        setSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 11815354368);
        setRkeyBit = (uint8_t *)((uint8_t *)pAddress + 11819548672);
        setLevel = (uint8_t *)((uint8_t *)pAddress + 11823742976);
        iHash = (uint8_t *)((uint8_t *)pAddress + 11827937280);
        iHashType = (uint8_t *)((uint8_t *)pAddress + 11832131584);
        iLatchSet = (uint8_t *)((uint8_t *)pAddress + 11836325888);
        iLatchGet = (uint8_t *)((uint8_t *)pAddress + 11840520192);
        iClimbRkey = (uint8_t *)((uint8_t *)pAddress + 11844714496);
        iClimbSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 11848908800);
        iClimbSiblingRkeyN = (uint8_t *)((uint8_t *)pAddress + 11853103104);
        iRotateLevel = (uint8_t *)((uint8_t *)pAddress + 11857297408);
        iJmpz = (uint8_t *)((uint8_t *)pAddress + 11861491712);
        iJmp = (uint8_t *)((uint8_t *)pAddress + 11865686016);
        iConst0 = (uint64_t *)((uint8_t *)pAddress + 11869880320);
        iConst1 = (uint64_t *)((uint8_t *)pAddress + 11903434752);
        iConst2 = (uint64_t *)((uint8_t *)pAddress + 11936989184);
        iConst3 = (uint64_t *)((uint8_t *)pAddress + 11970543616);
        iAddress = (uint64_t *)((uint8_t *)pAddress + 12004098048);
        op0inv = (FieldElement *)((uint8_t *)pAddress + 12037652480);
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

    static uint64_t degree (void) { return 4194304; }
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
        freeA = (FieldElement *)((uint8_t *)pAddress + 12071206912);
        freeB = (FieldElement *)((uint8_t *)pAddress + 12104761344);
        gateType = (FieldElement *)((uint8_t *)pAddress + 12138315776);
        freeANorm = (FieldElement *)((uint8_t *)pAddress + 12171870208);
        freeBNorm = (FieldElement *)((uint8_t *)pAddress + 12205424640);
        freeCNorm = (FieldElement *)((uint8_t *)pAddress + 12238979072);
        a = (FieldElement *)((uint8_t *)pAddress + 12272533504);
        b = (FieldElement *)((uint8_t *)pAddress + 12306087936);
        c = (FieldElement *)((uint8_t *)pAddress + 12339642368);
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

    static uint64_t degree (void) { return 4194304; }
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
        a = (FieldElement *)((uint8_t *)pAddress + 12373196800);
        b = (FieldElement *)((uint8_t *)pAddress + 12406751232);
        c = (FieldElement *)((uint8_t *)pAddress + 12440305664);
    }

    KeccakFCommitPols (void * pAddress, uint64_t degree)
    {
        a = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        b = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        c = (FieldElement *)((uint8_t *)pAddress + 16*degree);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 24; }
};

class Nine2OneCommitPols
{
public:
    FieldElement * bit;
    FieldElement * field9;

    Nine2OneCommitPols (void * pAddress)
    {
        bit = (FieldElement *)((uint8_t *)pAddress + 12473860096);
        field9 = (FieldElement *)((uint8_t *)pAddress + 12507414528);
    }

    Nine2OneCommitPols (void * pAddress, uint64_t degree)
    {
        bit = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        field9 = (FieldElement *)((uint8_t *)pAddress + 8*degree);
    }

    static uint64_t degree (void) { return 4194304; }
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
        rBit = (FieldElement *)((uint8_t *)pAddress + 12540968960);
        sOutBit = (FieldElement *)((uint8_t *)pAddress + 12574523392);
        r8 = (FieldElement *)((uint8_t *)pAddress + 12608077824);
        connected = (FieldElement *)((uint8_t *)pAddress + 12641632256);
        sOut0 = (FieldElement *)((uint8_t *)pAddress + 12675186688);
        sOut1 = (FieldElement *)((uint8_t *)pAddress + 12708741120);
        sOut2 = (FieldElement *)((uint8_t *)pAddress + 12742295552);
        sOut3 = (FieldElement *)((uint8_t *)pAddress + 12775849984);
        sOut4 = (FieldElement *)((uint8_t *)pAddress + 12809404416);
        sOut5 = (FieldElement *)((uint8_t *)pAddress + 12842958848);
        sOut6 = (FieldElement *)((uint8_t *)pAddress + 12876513280);
        sOut7 = (FieldElement *)((uint8_t *)pAddress + 12910067712);
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

    static uint64_t degree (void) { return 4194304; }
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
        freeIn = (FieldElement *)((uint8_t *)pAddress + 12943622144);
        connected = (FieldElement *)((uint8_t *)pAddress + 12977176576);
        addr = (FieldElement *)((uint8_t *)pAddress + 13010731008);
        rem = (FieldElement *)((uint8_t *)pAddress + 13044285440);
        remInv = (FieldElement *)((uint8_t *)pAddress + 13077839872);
        spare = (FieldElement *)((uint8_t *)pAddress + 13111394304);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 13144948736);
        len = (FieldElement *)((uint8_t *)pAddress + 13178503168);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 13212057600);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 13245612032);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 13279166464);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 13312720896);
        hash4 = (FieldElement *)((uint8_t *)pAddress + 13346275328);
        hash5 = (FieldElement *)((uint8_t *)pAddress + 13379829760);
        hash6 = (FieldElement *)((uint8_t *)pAddress + 13413384192);
        hash7 = (FieldElement *)((uint8_t *)pAddress + 13446938624);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 13480493056);
        crLen = (FieldElement *)((uint8_t *)pAddress + 13514047488);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 13547601920);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 13581156352);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 13614710784);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 13648265216);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 13681819648);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 13715374080);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 13748928512);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 13782482944);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 13816037376);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 13849591808);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 13883146240);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 13916700672);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 13950255104);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 13983809536);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 14017363968);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 14050918400);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 14084472832);
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

    static uint64_t degree (void) { return 4194304; }
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
        addr = (FieldElement *)((uint8_t *)pAddress + 14118027264);
        step = (FieldElement *)((uint8_t *)pAddress + 14151581696);
        mOp = (FieldElement *)((uint8_t *)pAddress + 14185136128);
        mWr = (FieldElement *)((uint8_t *)pAddress + 14218690560);
        val[0] = (FieldElement *)((uint8_t *)pAddress + 14252244992);
        val[1] = (FieldElement *)((uint8_t *)pAddress + 14285799424);
        val[2] = (FieldElement *)((uint8_t *)pAddress + 14319353856);
        val[3] = (FieldElement *)((uint8_t *)pAddress + 14352908288);
        val[4] = (FieldElement *)((uint8_t *)pAddress + 14386462720);
        val[5] = (FieldElement *)((uint8_t *)pAddress + 14420017152);
        val[6] = (FieldElement *)((uint8_t *)pAddress + 14453571584);
        val[7] = (FieldElement *)((uint8_t *)pAddress + 14487126016);
        lastAccess = (FieldElement *)((uint8_t *)pAddress + 14520680448);
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

    static uint64_t degree (void) { return 4194304; }
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
        A7 = (uint32_t *)((uint8_t *)pAddress + 14554234880);
        A6 = (uint32_t *)((uint8_t *)pAddress + 14587789312);
        A5 = (uint32_t *)((uint8_t *)pAddress + 14621343744);
        A4 = (uint32_t *)((uint8_t *)pAddress + 14654898176);
        A3 = (uint32_t *)((uint8_t *)pAddress + 14688452608);
        A2 = (uint32_t *)((uint8_t *)pAddress + 14722007040);
        A1 = (uint32_t *)((uint8_t *)pAddress + 14755561472);
        A0 = (FieldElement *)((uint8_t *)pAddress + 14789115904);
        B7 = (uint32_t *)((uint8_t *)pAddress + 14822670336);
        B6 = (uint32_t *)((uint8_t *)pAddress + 14856224768);
        B5 = (uint32_t *)((uint8_t *)pAddress + 14889779200);
        B4 = (uint32_t *)((uint8_t *)pAddress + 14923333632);
        B3 = (uint32_t *)((uint8_t *)pAddress + 14956888064);
        B2 = (uint32_t *)((uint8_t *)pAddress + 14990442496);
        B1 = (uint32_t *)((uint8_t *)pAddress + 15023996928);
        B0 = (FieldElement *)((uint8_t *)pAddress + 15057551360);
        C7 = (uint32_t *)((uint8_t *)pAddress + 15091105792);
        C6 = (uint32_t *)((uint8_t *)pAddress + 15124660224);
        C5 = (uint32_t *)((uint8_t *)pAddress + 15158214656);
        C4 = (uint32_t *)((uint8_t *)pAddress + 15191769088);
        C3 = (uint32_t *)((uint8_t *)pAddress + 15225323520);
        C2 = (uint32_t *)((uint8_t *)pAddress + 15258877952);
        C1 = (uint32_t *)((uint8_t *)pAddress + 15292432384);
        C0 = (FieldElement *)((uint8_t *)pAddress + 15325986816);
        D7 = (uint32_t *)((uint8_t *)pAddress + 15359541248);
        D6 = (uint32_t *)((uint8_t *)pAddress + 15393095680);
        D5 = (uint32_t *)((uint8_t *)pAddress + 15426650112);
        D4 = (uint32_t *)((uint8_t *)pAddress + 15460204544);
        D3 = (uint32_t *)((uint8_t *)pAddress + 15493758976);
        D2 = (uint32_t *)((uint8_t *)pAddress + 15527313408);
        D1 = (uint32_t *)((uint8_t *)pAddress + 15560867840);
        D0 = (FieldElement *)((uint8_t *)pAddress + 15594422272);
        E7 = (uint32_t *)((uint8_t *)pAddress + 15627976704);
        E6 = (uint32_t *)((uint8_t *)pAddress + 15661531136);
        E5 = (uint32_t *)((uint8_t *)pAddress + 15695085568);
        E4 = (uint32_t *)((uint8_t *)pAddress + 15728640000);
        E3 = (uint32_t *)((uint8_t *)pAddress + 15762194432);
        E2 = (uint32_t *)((uint8_t *)pAddress + 15795748864);
        E1 = (uint32_t *)((uint8_t *)pAddress + 15829303296);
        E0 = (FieldElement *)((uint8_t *)pAddress + 15862857728);
        SR7 = (uint32_t *)((uint8_t *)pAddress + 15896412160);
        SR6 = (uint32_t *)((uint8_t *)pAddress + 15929966592);
        SR5 = (uint32_t *)((uint8_t *)pAddress + 15963521024);
        SR4 = (uint32_t *)((uint8_t *)pAddress + 15997075456);
        SR3 = (uint32_t *)((uint8_t *)pAddress + 16030629888);
        SR2 = (uint32_t *)((uint8_t *)pAddress + 16064184320);
        SR1 = (uint32_t *)((uint8_t *)pAddress + 16097738752);
        SR0 = (uint32_t *)((uint8_t *)pAddress + 16131293184);
        CTX = (uint32_t *)((uint8_t *)pAddress + 16164847616);
        SP = (uint16_t *)((uint8_t *)pAddress + 16198402048);
        PC = (uint32_t *)((uint8_t *)pAddress + 16206790656);
        GAS = (uint64_t *)((uint8_t *)pAddress + 16240345088);
        MAXMEM = (uint32_t *)((uint8_t *)pAddress + 16273899520);
        zkPC = (uint32_t *)((uint8_t *)pAddress + 16307453952);
        RR = (uint32_t *)((uint8_t *)pAddress + 16341008384);
        HASHPOS = (uint32_t *)((uint8_t *)pAddress + 16374562816);
        CONST7 = (FieldElement *)((uint8_t *)pAddress + 16408117248);
        CONST6 = (FieldElement *)((uint8_t *)pAddress + 16441671680);
        CONST5 = (FieldElement *)((uint8_t *)pAddress + 16475226112);
        CONST4 = (FieldElement *)((uint8_t *)pAddress + 16508780544);
        CONST3 = (FieldElement *)((uint8_t *)pAddress + 16542334976);
        CONST2 = (FieldElement *)((uint8_t *)pAddress + 16575889408);
        CONST1 = (FieldElement *)((uint8_t *)pAddress + 16609443840);
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 16642998272);
        FREE7 = (FieldElement *)((uint8_t *)pAddress + 16676552704);
        FREE6 = (FieldElement *)((uint8_t *)pAddress + 16710107136);
        FREE5 = (FieldElement *)((uint8_t *)pAddress + 16743661568);
        FREE4 = (FieldElement *)((uint8_t *)pAddress + 16777216000);
        FREE3 = (FieldElement *)((uint8_t *)pAddress + 16810770432);
        FREE2 = (FieldElement *)((uint8_t *)pAddress + 16844324864);
        FREE1 = (FieldElement *)((uint8_t *)pAddress + 16877879296);
        FREE0 = (FieldElement *)((uint8_t *)pAddress + 16911433728);
        inA = (FieldElement *)((uint8_t *)pAddress + 16944988160);
        inB = (FieldElement *)((uint8_t *)pAddress + 16978542592);
        inC = (FieldElement *)((uint8_t *)pAddress + 17012097024);
        inD = (FieldElement *)((uint8_t *)pAddress + 17045651456);
        inE = (FieldElement *)((uint8_t *)pAddress + 17079205888);
        inSR = (FieldElement *)((uint8_t *)pAddress + 17112760320);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 17146314752);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 17179869184);
        inSP = (FieldElement *)((uint8_t *)pAddress + 17213423616);
        inPC = (FieldElement *)((uint8_t *)pAddress + 17246978048);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 17280532480);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 17314086912);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 17347641344);
        inRR = (FieldElement *)((uint8_t *)pAddress + 17381195776);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 17414750208);
        setA = (uint8_t *)((uint8_t *)pAddress + 17448304640);
        setB = (uint8_t *)((uint8_t *)pAddress + 17452498944);
        setC = (uint8_t *)((uint8_t *)pAddress + 17456693248);
        setD = (uint8_t *)((uint8_t *)pAddress + 17460887552);
        setE = (uint8_t *)((uint8_t *)pAddress + 17465081856);
        setSR = (uint8_t *)((uint8_t *)pAddress + 17469276160);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 17473470464);
        setSP = (uint8_t *)((uint8_t *)pAddress + 17477664768);
        setPC = (uint8_t *)((uint8_t *)pAddress + 17481859072);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 17486053376);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 17490247680);
        JMP = (uint8_t *)((uint8_t *)pAddress + 17494441984);
        JMPN = (uint8_t *)((uint8_t *)pAddress + 17498636288);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 17502830592);
        setRR = (uint8_t *)((uint8_t *)pAddress + 17507024896);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 17511219200);
        offset = (uint32_t *)((uint8_t *)pAddress + 17515413504);
        incStack = (int32_t *)((uint8_t *)pAddress + 17548967936);
        incCode = (int32_t *)((uint8_t *)pAddress + 17565745152);
        isStack = (uint8_t *)((uint8_t *)pAddress + 17582522368);
        isCode = (uint8_t *)((uint8_t *)pAddress + 17586716672);
        isMem = (uint8_t *)((uint8_t *)pAddress + 17590910976);
        ind = (uint8_t *)((uint8_t *)pAddress + 17595105280);
        indRR = (uint8_t *)((uint8_t *)pAddress + 17599299584);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 17603493888);
        carry = (uint8_t *)((uint8_t *)pAddress + 17607688192);
        mOp = (uint8_t *)((uint8_t *)pAddress + 17611882496);
        mWR = (uint8_t *)((uint8_t *)pAddress + 17616076800);
        sWR = (uint8_t *)((uint8_t *)pAddress + 17620271104);
        sRD = (uint8_t *)((uint8_t *)pAddress + 17624465408);
        arith = (uint8_t *)((uint8_t *)pAddress + 17628659712);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 17632854016);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 17637048320);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 17641242624);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 17645436928);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 17649631232);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 17653825536);
        hashK = (uint8_t *)((uint8_t *)pAddress + 17658019840);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 17662214144);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 17666408448);
        hashP = (uint8_t *)((uint8_t *)pAddress + 17670602752);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 17674797056);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 17678991360);
        bin = (uint8_t *)((uint8_t *)pAddress + 17683185664);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 17687379968);
        assert = (uint8_t *)((uint8_t *)pAddress + 17691574272);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 17695768576);
        isNeg = (uint8_t *)((uint8_t *)pAddress + 17699962880);
        isMaxMem = (uint8_t *)((uint8_t *)pAddress + 17704157184);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 17708351488);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 17741905920);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 17775460352);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 17809014784);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 17842569216);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 17876123648);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 17909678080);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 17943232512);
    }

    MainCommitPols (void * pAddress, uint64_t degree)
    {
        A7 = (uint32_t *)((uint8_t *)pAddress + 0*degree);
        A6 = (uint32_t *)((uint8_t *)pAddress + 8*degree);
        A5 = (uint32_t *)((uint8_t *)pAddress + 16*degree);
        A4 = (uint32_t *)((uint8_t *)pAddress + 24*degree);
        A3 = (uint32_t *)((uint8_t *)pAddress + 32*degree);
        A2 = (uint32_t *)((uint8_t *)pAddress + 40*degree);
        A1 = (uint32_t *)((uint8_t *)pAddress + 48*degree);
        A0 = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        B7 = (uint32_t *)((uint8_t *)pAddress + 64*degree);
        B6 = (uint32_t *)((uint8_t *)pAddress + 72*degree);
        B5 = (uint32_t *)((uint8_t *)pAddress + 80*degree);
        B4 = (uint32_t *)((uint8_t *)pAddress + 88*degree);
        B3 = (uint32_t *)((uint8_t *)pAddress + 96*degree);
        B2 = (uint32_t *)((uint8_t *)pAddress + 104*degree);
        B1 = (uint32_t *)((uint8_t *)pAddress + 112*degree);
        B0 = (FieldElement *)((uint8_t *)pAddress + 120*degree);
        C7 = (uint32_t *)((uint8_t *)pAddress + 128*degree);
        C6 = (uint32_t *)((uint8_t *)pAddress + 136*degree);
        C5 = (uint32_t *)((uint8_t *)pAddress + 144*degree);
        C4 = (uint32_t *)((uint8_t *)pAddress + 152*degree);
        C3 = (uint32_t *)((uint8_t *)pAddress + 160*degree);
        C2 = (uint32_t *)((uint8_t *)pAddress + 168*degree);
        C1 = (uint32_t *)((uint8_t *)pAddress + 176*degree);
        C0 = (FieldElement *)((uint8_t *)pAddress + 184*degree);
        D7 = (uint32_t *)((uint8_t *)pAddress + 192*degree);
        D6 = (uint32_t *)((uint8_t *)pAddress + 200*degree);
        D5 = (uint32_t *)((uint8_t *)pAddress + 208*degree);
        D4 = (uint32_t *)((uint8_t *)pAddress + 216*degree);
        D3 = (uint32_t *)((uint8_t *)pAddress + 224*degree);
        D2 = (uint32_t *)((uint8_t *)pAddress + 232*degree);
        D1 = (uint32_t *)((uint8_t *)pAddress + 240*degree);
        D0 = (FieldElement *)((uint8_t *)pAddress + 248*degree);
        E7 = (uint32_t *)((uint8_t *)pAddress + 256*degree);
        E6 = (uint32_t *)((uint8_t *)pAddress + 264*degree);
        E5 = (uint32_t *)((uint8_t *)pAddress + 272*degree);
        E4 = (uint32_t *)((uint8_t *)pAddress + 280*degree);
        E3 = (uint32_t *)((uint8_t *)pAddress + 288*degree);
        E2 = (uint32_t *)((uint8_t *)pAddress + 296*degree);
        E1 = (uint32_t *)((uint8_t *)pAddress + 304*degree);
        E0 = (FieldElement *)((uint8_t *)pAddress + 312*degree);
        SR7 = (uint32_t *)((uint8_t *)pAddress + 320*degree);
        SR6 = (uint32_t *)((uint8_t *)pAddress + 328*degree);
        SR5 = (uint32_t *)((uint8_t *)pAddress + 336*degree);
        SR4 = (uint32_t *)((uint8_t *)pAddress + 344*degree);
        SR3 = (uint32_t *)((uint8_t *)pAddress + 352*degree);
        SR2 = (uint32_t *)((uint8_t *)pAddress + 360*degree);
        SR1 = (uint32_t *)((uint8_t *)pAddress + 368*degree);
        SR0 = (uint32_t *)((uint8_t *)pAddress + 376*degree);
        CTX = (uint32_t *)((uint8_t *)pAddress + 384*degree);
        SP = (uint16_t *)((uint8_t *)pAddress + 392*degree);
        PC = (uint32_t *)((uint8_t *)pAddress + 394*degree);
        GAS = (uint64_t *)((uint8_t *)pAddress + 402*degree);
        MAXMEM = (uint32_t *)((uint8_t *)pAddress + 410*degree);
        zkPC = (uint32_t *)((uint8_t *)pAddress + 418*degree);
        RR = (uint32_t *)((uint8_t *)pAddress + 426*degree);
        HASHPOS = (uint32_t *)((uint8_t *)pAddress + 434*degree);
        CONST7 = (FieldElement *)((uint8_t *)pAddress + 442*degree);
        CONST6 = (FieldElement *)((uint8_t *)pAddress + 450*degree);
        CONST5 = (FieldElement *)((uint8_t *)pAddress + 458*degree);
        CONST4 = (FieldElement *)((uint8_t *)pAddress + 466*degree);
        CONST3 = (FieldElement *)((uint8_t *)pAddress + 474*degree);
        CONST2 = (FieldElement *)((uint8_t *)pAddress + 482*degree);
        CONST1 = (FieldElement *)((uint8_t *)pAddress + 490*degree);
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 498*degree);
        FREE7 = (FieldElement *)((uint8_t *)pAddress + 506*degree);
        FREE6 = (FieldElement *)((uint8_t *)pAddress + 514*degree);
        FREE5 = (FieldElement *)((uint8_t *)pAddress + 522*degree);
        FREE4 = (FieldElement *)((uint8_t *)pAddress + 530*degree);
        FREE3 = (FieldElement *)((uint8_t *)pAddress + 538*degree);
        FREE2 = (FieldElement *)((uint8_t *)pAddress + 546*degree);
        FREE1 = (FieldElement *)((uint8_t *)pAddress + 554*degree);
        FREE0 = (FieldElement *)((uint8_t *)pAddress + 562*degree);
        inA = (FieldElement *)((uint8_t *)pAddress + 570*degree);
        inB = (FieldElement *)((uint8_t *)pAddress + 578*degree);
        inC = (FieldElement *)((uint8_t *)pAddress + 586*degree);
        inD = (FieldElement *)((uint8_t *)pAddress + 594*degree);
        inE = (FieldElement *)((uint8_t *)pAddress + 602*degree);
        inSR = (FieldElement *)((uint8_t *)pAddress + 610*degree);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 618*degree);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 626*degree);
        inSP = (FieldElement *)((uint8_t *)pAddress + 634*degree);
        inPC = (FieldElement *)((uint8_t *)pAddress + 642*degree);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 650*degree);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 658*degree);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 666*degree);
        inRR = (FieldElement *)((uint8_t *)pAddress + 674*degree);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 682*degree);
        setA = (uint8_t *)((uint8_t *)pAddress + 690*degree);
        setB = (uint8_t *)((uint8_t *)pAddress + 691*degree);
        setC = (uint8_t *)((uint8_t *)pAddress + 692*degree);
        setD = (uint8_t *)((uint8_t *)pAddress + 693*degree);
        setE = (uint8_t *)((uint8_t *)pAddress + 694*degree);
        setSR = (uint8_t *)((uint8_t *)pAddress + 695*degree);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 696*degree);
        setSP = (uint8_t *)((uint8_t *)pAddress + 697*degree);
        setPC = (uint8_t *)((uint8_t *)pAddress + 698*degree);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 699*degree);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 700*degree);
        JMP = (uint8_t *)((uint8_t *)pAddress + 701*degree);
        JMPN = (uint8_t *)((uint8_t *)pAddress + 702*degree);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 703*degree);
        setRR = (uint8_t *)((uint8_t *)pAddress + 704*degree);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 705*degree);
        offset = (uint32_t *)((uint8_t *)pAddress + 706*degree);
        incStack = (int32_t *)((uint8_t *)pAddress + 714*degree);
        incCode = (int32_t *)((uint8_t *)pAddress + 718*degree);
        isStack = (uint8_t *)((uint8_t *)pAddress + 722*degree);
        isCode = (uint8_t *)((uint8_t *)pAddress + 723*degree);
        isMem = (uint8_t *)((uint8_t *)pAddress + 724*degree);
        ind = (uint8_t *)((uint8_t *)pAddress + 725*degree);
        indRR = (uint8_t *)((uint8_t *)pAddress + 726*degree);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 727*degree);
        carry = (uint8_t *)((uint8_t *)pAddress + 728*degree);
        mOp = (uint8_t *)((uint8_t *)pAddress + 729*degree);
        mWR = (uint8_t *)((uint8_t *)pAddress + 730*degree);
        sWR = (uint8_t *)((uint8_t *)pAddress + 731*degree);
        sRD = (uint8_t *)((uint8_t *)pAddress + 732*degree);
        arith = (uint8_t *)((uint8_t *)pAddress + 733*degree);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 734*degree);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 735*degree);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 736*degree);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 737*degree);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 738*degree);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 739*degree);
        hashK = (uint8_t *)((uint8_t *)pAddress + 740*degree);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 741*degree);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 742*degree);
        hashP = (uint8_t *)((uint8_t *)pAddress + 743*degree);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 744*degree);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 745*degree);
        bin = (uint8_t *)((uint8_t *)pAddress + 746*degree);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 747*degree);
        assert = (uint8_t *)((uint8_t *)pAddress + 748*degree);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 749*degree);
        isNeg = (uint8_t *)((uint8_t *)pAddress + 750*degree);
        isMaxMem = (uint8_t *)((uint8_t *)pAddress + 751*degree);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 752*degree);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 760*degree);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 768*degree);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 776*degree);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 784*degree);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 792*degree);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 800*degree);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 808*degree);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 816; }
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

    static uint64_t size (void) { return 17976786944; }
};

#endif // COMMIT_POLS_HPP
