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
        last = (uint8_t *)((uint8_t *)pAddress + 206*degree);
        useCarry = (uint8_t *)((uint8_t *)pAddress + 207*degree);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 208; }
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
        in0 = (FieldElement *)((uint8_t *)pAddress + 8250195968);
        in1 = (FieldElement *)((uint8_t *)pAddress + 8283750400);
        in2 = (FieldElement *)((uint8_t *)pAddress + 8317304832);
        in3 = (FieldElement *)((uint8_t *)pAddress + 8350859264);
        in4 = (FieldElement *)((uint8_t *)pAddress + 8384413696);
        in5 = (FieldElement *)((uint8_t *)pAddress + 8417968128);
        in6 = (FieldElement *)((uint8_t *)pAddress + 8451522560);
        in7 = (FieldElement *)((uint8_t *)pAddress + 8485076992);
        hashType = (FieldElement *)((uint8_t *)pAddress + 8518631424);
        cap1 = (FieldElement *)((uint8_t *)pAddress + 8552185856);
        cap2 = (FieldElement *)((uint8_t *)pAddress + 8585740288);
        cap3 = (FieldElement *)((uint8_t *)pAddress + 8619294720);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 8652849152);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 8686403584);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 8719958016);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 8753512448);
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
        acc[0] = (FieldElement *)((uint8_t *)pAddress + 8787066880);
        acc[1] = (FieldElement *)((uint8_t *)pAddress + 8820621312);
        acc[2] = (FieldElement *)((uint8_t *)pAddress + 8854175744);
        acc[3] = (FieldElement *)((uint8_t *)pAddress + 8887730176);
        acc[4] = (FieldElement *)((uint8_t *)pAddress + 8921284608);
        acc[5] = (FieldElement *)((uint8_t *)pAddress + 8954839040);
        acc[6] = (FieldElement *)((uint8_t *)pAddress + 8988393472);
        acc[7] = (FieldElement *)((uint8_t *)pAddress + 9021947904);
        freeIn = (FieldElement *)((uint8_t *)pAddress + 9055502336);
        addr = (FieldElement *)((uint8_t *)pAddress + 9089056768);
        rem = (FieldElement *)((uint8_t *)pAddress + 9122611200);
        remInv = (FieldElement *)((uint8_t *)pAddress + 9156165632);
        spare = (FieldElement *)((uint8_t *)pAddress + 9189720064);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 9223274496);
        curHash0 = (FieldElement *)((uint8_t *)pAddress + 9256828928);
        curHash1 = (FieldElement *)((uint8_t *)pAddress + 9290383360);
        curHash2 = (FieldElement *)((uint8_t *)pAddress + 9323937792);
        curHash3 = (FieldElement *)((uint8_t *)pAddress + 9357492224);
        prevHash0 = (FieldElement *)((uint8_t *)pAddress + 9391046656);
        prevHash1 = (FieldElement *)((uint8_t *)pAddress + 9424601088);
        prevHash2 = (FieldElement *)((uint8_t *)pAddress + 9458155520);
        prevHash3 = (FieldElement *)((uint8_t *)pAddress + 9491709952);
        len = (FieldElement *)((uint8_t *)pAddress + 9525264384);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 9558818816);
        crLen = (FieldElement *)((uint8_t *)pAddress + 9592373248);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 9625927680);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 9659482112);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 9693036544);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 9726590976);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 9760145408);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 9793699840);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 9827254272);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 9860808704);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 9894363136);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 9927917568);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 9961472000);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 9995026432);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 10028580864);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 10062135296);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 10095689728);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 10129244160);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 10162798592);
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
        free0 = (uint64_t *)((uint8_t *)pAddress + 10196353024);
        free1 = (uint64_t *)((uint8_t *)pAddress + 10229907456);
        free2 = (uint64_t *)((uint8_t *)pAddress + 10263461888);
        free3 = (uint64_t *)((uint8_t *)pAddress + 10297016320);
        hashLeft0 = (uint64_t *)((uint8_t *)pAddress + 10330570752);
        hashLeft1 = (uint64_t *)((uint8_t *)pAddress + 10364125184);
        hashLeft2 = (uint64_t *)((uint8_t *)pAddress + 10397679616);
        hashLeft3 = (uint64_t *)((uint8_t *)pAddress + 10431234048);
        hashRight0 = (uint64_t *)((uint8_t *)pAddress + 10464788480);
        hashRight1 = (uint64_t *)((uint8_t *)pAddress + 10498342912);
        hashRight2 = (uint64_t *)((uint8_t *)pAddress + 10531897344);
        hashRight3 = (uint64_t *)((uint8_t *)pAddress + 10565451776);
        oldRoot0 = (uint64_t *)((uint8_t *)pAddress + 10599006208);
        oldRoot1 = (uint64_t *)((uint8_t *)pAddress + 10632560640);
        oldRoot2 = (uint64_t *)((uint8_t *)pAddress + 10666115072);
        oldRoot3 = (uint64_t *)((uint8_t *)pAddress + 10699669504);
        newRoot0 = (uint64_t *)((uint8_t *)pAddress + 10733223936);
        newRoot1 = (uint64_t *)((uint8_t *)pAddress + 10766778368);
        newRoot2 = (uint64_t *)((uint8_t *)pAddress + 10800332800);
        newRoot3 = (uint64_t *)((uint8_t *)pAddress + 10833887232);
        valueLow0 = (uint64_t *)((uint8_t *)pAddress + 10867441664);
        valueLow1 = (uint64_t *)((uint8_t *)pAddress + 10900996096);
        valueLow2 = (uint64_t *)((uint8_t *)pAddress + 10934550528);
        valueLow3 = (uint64_t *)((uint8_t *)pAddress + 10968104960);
        valueHigh0 = (uint64_t *)((uint8_t *)pAddress + 11001659392);
        valueHigh1 = (uint64_t *)((uint8_t *)pAddress + 11035213824);
        valueHigh2 = (uint64_t *)((uint8_t *)pAddress + 11068768256);
        valueHigh3 = (uint64_t *)((uint8_t *)pAddress + 11102322688);
        siblingValueHash0 = (uint64_t *)((uint8_t *)pAddress + 11135877120);
        siblingValueHash1 = (uint64_t *)((uint8_t *)pAddress + 11169431552);
        siblingValueHash2 = (uint64_t *)((uint8_t *)pAddress + 11202985984);
        siblingValueHash3 = (uint64_t *)((uint8_t *)pAddress + 11236540416);
        rkey0 = (uint64_t *)((uint8_t *)pAddress + 11270094848);
        rkey1 = (uint64_t *)((uint8_t *)pAddress + 11303649280);
        rkey2 = (uint64_t *)((uint8_t *)pAddress + 11337203712);
        rkey3 = (uint64_t *)((uint8_t *)pAddress + 11370758144);
        siblingRkey0 = (uint64_t *)((uint8_t *)pAddress + 11404312576);
        siblingRkey1 = (uint64_t *)((uint8_t *)pAddress + 11437867008);
        siblingRkey2 = (uint64_t *)((uint8_t *)pAddress + 11471421440);
        siblingRkey3 = (uint64_t *)((uint8_t *)pAddress + 11504975872);
        rkeyBit = (uint64_t *)((uint8_t *)pAddress + 11538530304);
        level0 = (uint64_t *)((uint8_t *)pAddress + 11572084736);
        level1 = (uint64_t *)((uint8_t *)pAddress + 11605639168);
        level2 = (uint64_t *)((uint8_t *)pAddress + 11639193600);
        level3 = (uint64_t *)((uint8_t *)pAddress + 11672748032);
        pc = (uint64_t *)((uint8_t *)pAddress + 11706302464);
        selOldRoot = (uint8_t *)((uint8_t *)pAddress + 11739856896);
        selNewRoot = (uint8_t *)((uint8_t *)pAddress + 11744051200);
        selValueLow = (uint8_t *)((uint8_t *)pAddress + 11748245504);
        selValueHigh = (uint8_t *)((uint8_t *)pAddress + 11752439808);
        selSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 11756634112);
        selRkey = (uint8_t *)((uint8_t *)pAddress + 11760828416);
        selRkeyBit = (uint8_t *)((uint8_t *)pAddress + 11765022720);
        selSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 11769217024);
        selFree = (uint8_t *)((uint8_t *)pAddress + 11773411328);
        setHashLeft = (uint8_t *)((uint8_t *)pAddress + 11777605632);
        setHashRight = (uint8_t *)((uint8_t *)pAddress + 11781799936);
        setOldRoot = (uint8_t *)((uint8_t *)pAddress + 11785994240);
        setNewRoot = (uint8_t *)((uint8_t *)pAddress + 11790188544);
        setValueLow = (uint8_t *)((uint8_t *)pAddress + 11794382848);
        setValueHigh = (uint8_t *)((uint8_t *)pAddress + 11798577152);
        setSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 11802771456);
        setRkey = (uint8_t *)((uint8_t *)pAddress + 11806965760);
        setSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 11811160064);
        setRkeyBit = (uint8_t *)((uint8_t *)pAddress + 11815354368);
        setLevel = (uint8_t *)((uint8_t *)pAddress + 11819548672);
        iHash = (uint8_t *)((uint8_t *)pAddress + 11823742976);
        iHashType = (uint8_t *)((uint8_t *)pAddress + 11827937280);
        iLatchSet = (uint8_t *)((uint8_t *)pAddress + 11832131584);
        iLatchGet = (uint8_t *)((uint8_t *)pAddress + 11836325888);
        iClimbRkey = (uint8_t *)((uint8_t *)pAddress + 11840520192);
        iClimbSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 11844714496);
        iClimbSiblingRkeyN = (uint8_t *)((uint8_t *)pAddress + 11848908800);
        iRotateLevel = (uint8_t *)((uint8_t *)pAddress + 11853103104);
        iJmpz = (uint8_t *)((uint8_t *)pAddress + 11857297408);
        iJmp = (uint8_t *)((uint8_t *)pAddress + 11861491712);
        iConst0 = (uint64_t *)((uint8_t *)pAddress + 11865686016);
        iConst1 = (uint64_t *)((uint8_t *)pAddress + 11899240448);
        iConst2 = (uint64_t *)((uint8_t *)pAddress + 11932794880);
        iConst3 = (uint64_t *)((uint8_t *)pAddress + 11966349312);
        iAddress = (uint64_t *)((uint8_t *)pAddress + 11999903744);
        op0inv = (FieldElement *)((uint8_t *)pAddress + 12033458176);
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
        freeA = (FieldElement *)((uint8_t *)pAddress + 12067012608);
        freeB = (FieldElement *)((uint8_t *)pAddress + 12100567040);
        gateType = (FieldElement *)((uint8_t *)pAddress + 12134121472);
        freeANorm = (FieldElement *)((uint8_t *)pAddress + 12167675904);
        freeBNorm = (FieldElement *)((uint8_t *)pAddress + 12201230336);
        freeCNorm = (FieldElement *)((uint8_t *)pAddress + 12234784768);
        a = (FieldElement *)((uint8_t *)pAddress + 12268339200);
        b = (FieldElement *)((uint8_t *)pAddress + 12301893632);
        c = (FieldElement *)((uint8_t *)pAddress + 12335448064);
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
        a = (FieldElement *)((uint8_t *)pAddress + 12369002496);
        b = (FieldElement *)((uint8_t *)pAddress + 12402556928);
        c = (FieldElement *)((uint8_t *)pAddress + 12436111360);
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
        bit = (FieldElement *)((uint8_t *)pAddress + 12469665792);
        field9 = (FieldElement *)((uint8_t *)pAddress + 12503220224);
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
        rBit = (FieldElement *)((uint8_t *)pAddress + 12536774656);
        sOutBit = (FieldElement *)((uint8_t *)pAddress + 12570329088);
        r8 = (FieldElement *)((uint8_t *)pAddress + 12603883520);
        connected = (FieldElement *)((uint8_t *)pAddress + 12637437952);
        sOut0 = (FieldElement *)((uint8_t *)pAddress + 12670992384);
        sOut1 = (FieldElement *)((uint8_t *)pAddress + 12704546816);
        sOut2 = (FieldElement *)((uint8_t *)pAddress + 12738101248);
        sOut3 = (FieldElement *)((uint8_t *)pAddress + 12771655680);
        sOut4 = (FieldElement *)((uint8_t *)pAddress + 12805210112);
        sOut5 = (FieldElement *)((uint8_t *)pAddress + 12838764544);
        sOut6 = (FieldElement *)((uint8_t *)pAddress + 12872318976);
        sOut7 = (FieldElement *)((uint8_t *)pAddress + 12905873408);
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
        freeIn = (FieldElement *)((uint8_t *)pAddress + 12939427840);
        connected = (FieldElement *)((uint8_t *)pAddress + 12972982272);
        addr = (FieldElement *)((uint8_t *)pAddress + 13006536704);
        rem = (FieldElement *)((uint8_t *)pAddress + 13040091136);
        remInv = (FieldElement *)((uint8_t *)pAddress + 13073645568);
        spare = (FieldElement *)((uint8_t *)pAddress + 13107200000);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 13140754432);
        len = (FieldElement *)((uint8_t *)pAddress + 13174308864);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 13207863296);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 13241417728);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 13274972160);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 13308526592);
        hash4 = (FieldElement *)((uint8_t *)pAddress + 13342081024);
        hash5 = (FieldElement *)((uint8_t *)pAddress + 13375635456);
        hash6 = (FieldElement *)((uint8_t *)pAddress + 13409189888);
        hash7 = (FieldElement *)((uint8_t *)pAddress + 13442744320);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 13476298752);
        crLen = (FieldElement *)((uint8_t *)pAddress + 13509853184);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 13543407616);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 13576962048);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 13610516480);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 13644070912);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 13677625344);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 13711179776);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 13744734208);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 13778288640);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 13811843072);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 13845397504);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 13878951936);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 13912506368);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 13946060800);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 13979615232);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 14013169664);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 14046724096);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 14080278528);
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
        addr = (FieldElement *)((uint8_t *)pAddress + 14113832960);
        step = (FieldElement *)((uint8_t *)pAddress + 14147387392);
        mOp = (FieldElement *)((uint8_t *)pAddress + 14180941824);
        mWr = (FieldElement *)((uint8_t *)pAddress + 14214496256);
        val[0] = (FieldElement *)((uint8_t *)pAddress + 14248050688);
        val[1] = (FieldElement *)((uint8_t *)pAddress + 14281605120);
        val[2] = (FieldElement *)((uint8_t *)pAddress + 14315159552);
        val[3] = (FieldElement *)((uint8_t *)pAddress + 14348713984);
        val[4] = (FieldElement *)((uint8_t *)pAddress + 14382268416);
        val[5] = (FieldElement *)((uint8_t *)pAddress + 14415822848);
        val[6] = (FieldElement *)((uint8_t *)pAddress + 14449377280);
        val[7] = (FieldElement *)((uint8_t *)pAddress + 14482931712);
        lastAccess = (FieldElement *)((uint8_t *)pAddress + 14516486144);
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
        indRR = (uint8_t *)((uint8_t *)pAddress + 17590910976);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 17595105280);
        mOp = (uint8_t *)((uint8_t *)pAddress + 17599299584);
        mWR = (uint8_t *)((uint8_t *)pAddress + 17603493888);
        sWR = (uint8_t *)((uint8_t *)pAddress + 17607688192);
        sRD = (uint8_t *)((uint8_t *)pAddress + 17611882496);
        arith = (uint8_t *)((uint8_t *)pAddress + 17616076800);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 17620271104);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 17624465408);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 17628659712);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 17632854016);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 17637048320);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 17641242624);
        hashK = (uint8_t *)((uint8_t *)pAddress + 17645436928);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 17649631232);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 17653825536);
        hashP = (uint8_t *)((uint8_t *)pAddress + 17658019840);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 17662214144);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 17666408448);
        bin = (uint8_t *)((uint8_t *)pAddress + 17670602752);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 17674797056);
        assert = (uint8_t *)((uint8_t *)pAddress + 17678991360);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 17683185664);
        isNeg = (uint8_t *)((uint8_t *)pAddress + 17687379968);
        isMaxMem = (uint8_t *)((uint8_t *)pAddress + 17691574272);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 17695768576);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 17729323008);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 17762877440);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 17796431872);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 17829986304);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 17863540736);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 17897095168);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 17930649600);
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
        JMPC = (uint8_t *)((uint8_t *)pAddress + 702*degree);
        setRR = (uint8_t *)((uint8_t *)pAddress + 703*degree);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 704*degree);
        offset = (uint32_t *)((uint8_t *)pAddress + 705*degree);
        incStack = (int32_t *)((uint8_t *)pAddress + 713*degree);
        incCode = (int32_t *)((uint8_t *)pAddress + 717*degree);
        isStack = (uint8_t *)((uint8_t *)pAddress + 721*degree);
        isCode = (uint8_t *)((uint8_t *)pAddress + 722*degree);
        isMem = (uint8_t *)((uint8_t *)pAddress + 723*degree);
        ind = (uint8_t *)((uint8_t *)pAddress + 724*degree);
        indRR = (uint8_t *)((uint8_t *)pAddress + 725*degree);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 726*degree);
        mOp = (uint8_t *)((uint8_t *)pAddress + 727*degree);
        mWR = (uint8_t *)((uint8_t *)pAddress + 728*degree);
        sWR = (uint8_t *)((uint8_t *)pAddress + 729*degree);
        sRD = (uint8_t *)((uint8_t *)pAddress + 730*degree);
        arith = (uint8_t *)((uint8_t *)pAddress + 731*degree);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 732*degree);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 733*degree);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 734*degree);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 735*degree);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 736*degree);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 737*degree);
        hashK = (uint8_t *)((uint8_t *)pAddress + 738*degree);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 739*degree);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 740*degree);
        hashP = (uint8_t *)((uint8_t *)pAddress + 741*degree);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 742*degree);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 743*degree);
        bin = (uint8_t *)((uint8_t *)pAddress + 744*degree);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 745*degree);
        assert = (uint8_t *)((uint8_t *)pAddress + 746*degree);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 747*degree);
        isNeg = (uint8_t *)((uint8_t *)pAddress + 748*degree);
        isMaxMem = (uint8_t *)((uint8_t *)pAddress + 749*degree);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 750*degree);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 758*degree);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 766*degree);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 774*degree);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 782*degree);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 790*degree);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 798*degree);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 806*degree);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 814; }
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

    static uint64_t size (void) { return 17964204032; }
};

#endif // COMMIT_POLS_HPP
