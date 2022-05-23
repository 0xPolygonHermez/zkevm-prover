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
    FieldElement * latchOffset;
    FieldElement * latchWr;

    MemAlignCommitPols (void * pAddress)
    {
        inM = (uint8_t *)((uint8_t *)pAddress + 20971520);
        inV = (uint8_t *)((uint8_t *)pAddress + 23068672);
        wr = (uint8_t *)((uint8_t *)pAddress + 25165824);
        m0[0] = (uint32_t *)((uint8_t *)pAddress + 27262976);
        m0[1] = (uint32_t *)((uint8_t *)pAddress + 44040192);
        m0[2] = (uint32_t *)((uint8_t *)pAddress + 60817408);
        m0[3] = (uint32_t *)((uint8_t *)pAddress + 77594624);
        m0[4] = (uint32_t *)((uint8_t *)pAddress + 94371840);
        m0[5] = (uint32_t *)((uint8_t *)pAddress + 111149056);
        m0[6] = (uint32_t *)((uint8_t *)pAddress + 127926272);
        m0[7] = (uint32_t *)((uint8_t *)pAddress + 144703488);
        m1[0] = (uint32_t *)((uint8_t *)pAddress + 161480704);
        m1[1] = (uint32_t *)((uint8_t *)pAddress + 178257920);
        m1[2] = (uint32_t *)((uint8_t *)pAddress + 195035136);
        m1[3] = (uint32_t *)((uint8_t *)pAddress + 211812352);
        m1[4] = (uint32_t *)((uint8_t *)pAddress + 228589568);
        m1[5] = (uint32_t *)((uint8_t *)pAddress + 245366784);
        m1[6] = (uint32_t *)((uint8_t *)pAddress + 262144000);
        m1[7] = (uint32_t *)((uint8_t *)pAddress + 278921216);
        w0[0] = (uint32_t *)((uint8_t *)pAddress + 295698432);
        w0[1] = (uint32_t *)((uint8_t *)pAddress + 312475648);
        w0[2] = (uint32_t *)((uint8_t *)pAddress + 329252864);
        w0[3] = (uint32_t *)((uint8_t *)pAddress + 346030080);
        w0[4] = (uint32_t *)((uint8_t *)pAddress + 362807296);
        w0[5] = (uint32_t *)((uint8_t *)pAddress + 379584512);
        w0[6] = (uint32_t *)((uint8_t *)pAddress + 396361728);
        w0[7] = (uint32_t *)((uint8_t *)pAddress + 413138944);
        w1[0] = (uint32_t *)((uint8_t *)pAddress + 429916160);
        w1[1] = (uint32_t *)((uint8_t *)pAddress + 446693376);
        w1[2] = (uint32_t *)((uint8_t *)pAddress + 463470592);
        w1[3] = (uint32_t *)((uint8_t *)pAddress + 480247808);
        w1[4] = (uint32_t *)((uint8_t *)pAddress + 497025024);
        w1[5] = (uint32_t *)((uint8_t *)pAddress + 513802240);
        w1[6] = (uint32_t *)((uint8_t *)pAddress + 530579456);
        w1[7] = (uint32_t *)((uint8_t *)pAddress + 547356672);
        v[0] = (uint32_t *)((uint8_t *)pAddress + 564133888);
        v[1] = (uint32_t *)((uint8_t *)pAddress + 580911104);
        v[2] = (uint32_t *)((uint8_t *)pAddress + 597688320);
        v[3] = (uint32_t *)((uint8_t *)pAddress + 614465536);
        v[4] = (uint32_t *)((uint8_t *)pAddress + 631242752);
        v[5] = (uint32_t *)((uint8_t *)pAddress + 648019968);
        v[6] = (uint32_t *)((uint8_t *)pAddress + 664797184);
        v[7] = (uint32_t *)((uint8_t *)pAddress + 681574400);
        offset = (uint8_t *)((uint8_t *)pAddress + 698351616);
        selW = (uint8_t *)((uint8_t *)pAddress + 700448768);
        factorV[0] = (FieldElement *)((uint8_t *)pAddress + 702545920);
        factorV[1] = (FieldElement *)((uint8_t *)pAddress + 719323136);
        factorV[2] = (FieldElement *)((uint8_t *)pAddress + 736100352);
        factorV[3] = (FieldElement *)((uint8_t *)pAddress + 752877568);
        factorV[4] = (FieldElement *)((uint8_t *)pAddress + 769654784);
        factorV[5] = (FieldElement *)((uint8_t *)pAddress + 786432000);
        factorV[6] = (FieldElement *)((uint8_t *)pAddress + 803209216);
        factorV[7] = (FieldElement *)((uint8_t *)pAddress + 819986432);
        latchOffset = (FieldElement *)((uint8_t *)pAddress + 836763648);
        latchWr = (FieldElement *)((uint8_t *)pAddress + 853540864);
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
        latchOffset = (FieldElement *)((uint8_t *)pAddress + 389*degree);
        latchWr = (FieldElement *)((uint8_t *)pAddress + 397*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 405; }
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
        x1[0] = (FieldElement *)((uint8_t *)pAddress + 870318080);
        x1[1] = (FieldElement *)((uint8_t *)pAddress + 887095296);
        x1[2] = (FieldElement *)((uint8_t *)pAddress + 903872512);
        x1[3] = (FieldElement *)((uint8_t *)pAddress + 920649728);
        x1[4] = (FieldElement *)((uint8_t *)pAddress + 937426944);
        x1[5] = (FieldElement *)((uint8_t *)pAddress + 954204160);
        x1[6] = (FieldElement *)((uint8_t *)pAddress + 970981376);
        x1[7] = (FieldElement *)((uint8_t *)pAddress + 987758592);
        x1[8] = (FieldElement *)((uint8_t *)pAddress + 1004535808);
        x1[9] = (FieldElement *)((uint8_t *)pAddress + 1021313024);
        x1[10] = (FieldElement *)((uint8_t *)pAddress + 1038090240);
        x1[11] = (FieldElement *)((uint8_t *)pAddress + 1054867456);
        x1[12] = (FieldElement *)((uint8_t *)pAddress + 1071644672);
        x1[13] = (FieldElement *)((uint8_t *)pAddress + 1088421888);
        x1[14] = (FieldElement *)((uint8_t *)pAddress + 1105199104);
        x1[15] = (FieldElement *)((uint8_t *)pAddress + 1121976320);
        y1[0] = (FieldElement *)((uint8_t *)pAddress + 1138753536);
        y1[1] = (FieldElement *)((uint8_t *)pAddress + 1155530752);
        y1[2] = (FieldElement *)((uint8_t *)pAddress + 1172307968);
        y1[3] = (FieldElement *)((uint8_t *)pAddress + 1189085184);
        y1[4] = (FieldElement *)((uint8_t *)pAddress + 1205862400);
        y1[5] = (FieldElement *)((uint8_t *)pAddress + 1222639616);
        y1[6] = (FieldElement *)((uint8_t *)pAddress + 1239416832);
        y1[7] = (FieldElement *)((uint8_t *)pAddress + 1256194048);
        y1[8] = (FieldElement *)((uint8_t *)pAddress + 1272971264);
        y1[9] = (FieldElement *)((uint8_t *)pAddress + 1289748480);
        y1[10] = (FieldElement *)((uint8_t *)pAddress + 1306525696);
        y1[11] = (FieldElement *)((uint8_t *)pAddress + 1323302912);
        y1[12] = (FieldElement *)((uint8_t *)pAddress + 1340080128);
        y1[13] = (FieldElement *)((uint8_t *)pAddress + 1356857344);
        y1[14] = (FieldElement *)((uint8_t *)pAddress + 1373634560);
        y1[15] = (FieldElement *)((uint8_t *)pAddress + 1390411776);
        x2[0] = (FieldElement *)((uint8_t *)pAddress + 1407188992);
        x2[1] = (FieldElement *)((uint8_t *)pAddress + 1423966208);
        x2[2] = (FieldElement *)((uint8_t *)pAddress + 1440743424);
        x2[3] = (FieldElement *)((uint8_t *)pAddress + 1457520640);
        x2[4] = (FieldElement *)((uint8_t *)pAddress + 1474297856);
        x2[5] = (FieldElement *)((uint8_t *)pAddress + 1491075072);
        x2[6] = (FieldElement *)((uint8_t *)pAddress + 1507852288);
        x2[7] = (FieldElement *)((uint8_t *)pAddress + 1524629504);
        x2[8] = (FieldElement *)((uint8_t *)pAddress + 1541406720);
        x2[9] = (FieldElement *)((uint8_t *)pAddress + 1558183936);
        x2[10] = (FieldElement *)((uint8_t *)pAddress + 1574961152);
        x2[11] = (FieldElement *)((uint8_t *)pAddress + 1591738368);
        x2[12] = (FieldElement *)((uint8_t *)pAddress + 1608515584);
        x2[13] = (FieldElement *)((uint8_t *)pAddress + 1625292800);
        x2[14] = (FieldElement *)((uint8_t *)pAddress + 1642070016);
        x2[15] = (FieldElement *)((uint8_t *)pAddress + 1658847232);
        y2[0] = (FieldElement *)((uint8_t *)pAddress + 1675624448);
        y2[1] = (FieldElement *)((uint8_t *)pAddress + 1692401664);
        y2[2] = (FieldElement *)((uint8_t *)pAddress + 1709178880);
        y2[3] = (FieldElement *)((uint8_t *)pAddress + 1725956096);
        y2[4] = (FieldElement *)((uint8_t *)pAddress + 1742733312);
        y2[5] = (FieldElement *)((uint8_t *)pAddress + 1759510528);
        y2[6] = (FieldElement *)((uint8_t *)pAddress + 1776287744);
        y2[7] = (FieldElement *)((uint8_t *)pAddress + 1793064960);
        y2[8] = (FieldElement *)((uint8_t *)pAddress + 1809842176);
        y2[9] = (FieldElement *)((uint8_t *)pAddress + 1826619392);
        y2[10] = (FieldElement *)((uint8_t *)pAddress + 1843396608);
        y2[11] = (FieldElement *)((uint8_t *)pAddress + 1860173824);
        y2[12] = (FieldElement *)((uint8_t *)pAddress + 1876951040);
        y2[13] = (FieldElement *)((uint8_t *)pAddress + 1893728256);
        y2[14] = (FieldElement *)((uint8_t *)pAddress + 1910505472);
        y2[15] = (FieldElement *)((uint8_t *)pAddress + 1927282688);
        x3[0] = (FieldElement *)((uint8_t *)pAddress + 1944059904);
        x3[1] = (FieldElement *)((uint8_t *)pAddress + 1960837120);
        x3[2] = (FieldElement *)((uint8_t *)pAddress + 1977614336);
        x3[3] = (FieldElement *)((uint8_t *)pAddress + 1994391552);
        x3[4] = (FieldElement *)((uint8_t *)pAddress + 2011168768);
        x3[5] = (FieldElement *)((uint8_t *)pAddress + 2027945984);
        x3[6] = (FieldElement *)((uint8_t *)pAddress + 2044723200);
        x3[7] = (FieldElement *)((uint8_t *)pAddress + 2061500416);
        x3[8] = (FieldElement *)((uint8_t *)pAddress + 2078277632);
        x3[9] = (FieldElement *)((uint8_t *)pAddress + 2095054848);
        x3[10] = (FieldElement *)((uint8_t *)pAddress + 2111832064);
        x3[11] = (FieldElement *)((uint8_t *)pAddress + 2128609280);
        x3[12] = (FieldElement *)((uint8_t *)pAddress + 2145386496);
        x3[13] = (FieldElement *)((uint8_t *)pAddress + 2162163712);
        x3[14] = (FieldElement *)((uint8_t *)pAddress + 2178940928);
        x3[15] = (FieldElement *)((uint8_t *)pAddress + 2195718144);
        y3[0] = (FieldElement *)((uint8_t *)pAddress + 2212495360);
        y3[1] = (FieldElement *)((uint8_t *)pAddress + 2229272576);
        y3[2] = (FieldElement *)((uint8_t *)pAddress + 2246049792);
        y3[3] = (FieldElement *)((uint8_t *)pAddress + 2262827008);
        y3[4] = (FieldElement *)((uint8_t *)pAddress + 2279604224);
        y3[5] = (FieldElement *)((uint8_t *)pAddress + 2296381440);
        y3[6] = (FieldElement *)((uint8_t *)pAddress + 2313158656);
        y3[7] = (FieldElement *)((uint8_t *)pAddress + 2329935872);
        y3[8] = (FieldElement *)((uint8_t *)pAddress + 2346713088);
        y3[9] = (FieldElement *)((uint8_t *)pAddress + 2363490304);
        y3[10] = (FieldElement *)((uint8_t *)pAddress + 2380267520);
        y3[11] = (FieldElement *)((uint8_t *)pAddress + 2397044736);
        y3[12] = (FieldElement *)((uint8_t *)pAddress + 2413821952);
        y3[13] = (FieldElement *)((uint8_t *)pAddress + 2430599168);
        y3[14] = (FieldElement *)((uint8_t *)pAddress + 2447376384);
        y3[15] = (FieldElement *)((uint8_t *)pAddress + 2464153600);
        s[0] = (FieldElement *)((uint8_t *)pAddress + 2480930816);
        s[1] = (FieldElement *)((uint8_t *)pAddress + 2497708032);
        s[2] = (FieldElement *)((uint8_t *)pAddress + 2514485248);
        s[3] = (FieldElement *)((uint8_t *)pAddress + 2531262464);
        s[4] = (FieldElement *)((uint8_t *)pAddress + 2548039680);
        s[5] = (FieldElement *)((uint8_t *)pAddress + 2564816896);
        s[6] = (FieldElement *)((uint8_t *)pAddress + 2581594112);
        s[7] = (FieldElement *)((uint8_t *)pAddress + 2598371328);
        s[8] = (FieldElement *)((uint8_t *)pAddress + 2615148544);
        s[9] = (FieldElement *)((uint8_t *)pAddress + 2631925760);
        s[10] = (FieldElement *)((uint8_t *)pAddress + 2648702976);
        s[11] = (FieldElement *)((uint8_t *)pAddress + 2665480192);
        s[12] = (FieldElement *)((uint8_t *)pAddress + 2682257408);
        s[13] = (FieldElement *)((uint8_t *)pAddress + 2699034624);
        s[14] = (FieldElement *)((uint8_t *)pAddress + 2715811840);
        s[15] = (FieldElement *)((uint8_t *)pAddress + 2732589056);
        q0[0] = (FieldElement *)((uint8_t *)pAddress + 2749366272);
        q0[1] = (FieldElement *)((uint8_t *)pAddress + 2766143488);
        q0[2] = (FieldElement *)((uint8_t *)pAddress + 2782920704);
        q0[3] = (FieldElement *)((uint8_t *)pAddress + 2799697920);
        q0[4] = (FieldElement *)((uint8_t *)pAddress + 2816475136);
        q0[5] = (FieldElement *)((uint8_t *)pAddress + 2833252352);
        q0[6] = (FieldElement *)((uint8_t *)pAddress + 2850029568);
        q0[7] = (FieldElement *)((uint8_t *)pAddress + 2866806784);
        q0[8] = (FieldElement *)((uint8_t *)pAddress + 2883584000);
        q0[9] = (FieldElement *)((uint8_t *)pAddress + 2900361216);
        q0[10] = (FieldElement *)((uint8_t *)pAddress + 2917138432);
        q0[11] = (FieldElement *)((uint8_t *)pAddress + 2933915648);
        q0[12] = (FieldElement *)((uint8_t *)pAddress + 2950692864);
        q0[13] = (FieldElement *)((uint8_t *)pAddress + 2967470080);
        q0[14] = (FieldElement *)((uint8_t *)pAddress + 2984247296);
        q0[15] = (FieldElement *)((uint8_t *)pAddress + 3001024512);
        q1[0] = (FieldElement *)((uint8_t *)pAddress + 3017801728);
        q1[1] = (FieldElement *)((uint8_t *)pAddress + 3034578944);
        q1[2] = (FieldElement *)((uint8_t *)pAddress + 3051356160);
        q1[3] = (FieldElement *)((uint8_t *)pAddress + 3068133376);
        q1[4] = (FieldElement *)((uint8_t *)pAddress + 3084910592);
        q1[5] = (FieldElement *)((uint8_t *)pAddress + 3101687808);
        q1[6] = (FieldElement *)((uint8_t *)pAddress + 3118465024);
        q1[7] = (FieldElement *)((uint8_t *)pAddress + 3135242240);
        q1[8] = (FieldElement *)((uint8_t *)pAddress + 3152019456);
        q1[9] = (FieldElement *)((uint8_t *)pAddress + 3168796672);
        q1[10] = (FieldElement *)((uint8_t *)pAddress + 3185573888);
        q1[11] = (FieldElement *)((uint8_t *)pAddress + 3202351104);
        q1[12] = (FieldElement *)((uint8_t *)pAddress + 3219128320);
        q1[13] = (FieldElement *)((uint8_t *)pAddress + 3235905536);
        q1[14] = (FieldElement *)((uint8_t *)pAddress + 3252682752);
        q1[15] = (FieldElement *)((uint8_t *)pAddress + 3269459968);
        q2[0] = (FieldElement *)((uint8_t *)pAddress + 3286237184);
        q2[1] = (FieldElement *)((uint8_t *)pAddress + 3303014400);
        q2[2] = (FieldElement *)((uint8_t *)pAddress + 3319791616);
        q2[3] = (FieldElement *)((uint8_t *)pAddress + 3336568832);
        q2[4] = (FieldElement *)((uint8_t *)pAddress + 3353346048);
        q2[5] = (FieldElement *)((uint8_t *)pAddress + 3370123264);
        q2[6] = (FieldElement *)((uint8_t *)pAddress + 3386900480);
        q2[7] = (FieldElement *)((uint8_t *)pAddress + 3403677696);
        q2[8] = (FieldElement *)((uint8_t *)pAddress + 3420454912);
        q2[9] = (FieldElement *)((uint8_t *)pAddress + 3437232128);
        q2[10] = (FieldElement *)((uint8_t *)pAddress + 3454009344);
        q2[11] = (FieldElement *)((uint8_t *)pAddress + 3470786560);
        q2[12] = (FieldElement *)((uint8_t *)pAddress + 3487563776);
        q2[13] = (FieldElement *)((uint8_t *)pAddress + 3504340992);
        q2[14] = (FieldElement *)((uint8_t *)pAddress + 3521118208);
        q2[15] = (FieldElement *)((uint8_t *)pAddress + 3537895424);
        selEq[0] = (FieldElement *)((uint8_t *)pAddress + 3554672640);
        selEq[1] = (FieldElement *)((uint8_t *)pAddress + 3571449856);
        selEq[2] = (FieldElement *)((uint8_t *)pAddress + 3588227072);
        selEq[3] = (FieldElement *)((uint8_t *)pAddress + 3605004288);
        carryL[0] = (FieldElement *)((uint8_t *)pAddress + 3621781504);
        carryL[1] = (FieldElement *)((uint8_t *)pAddress + 3638558720);
        carryL[2] = (FieldElement *)((uint8_t *)pAddress + 3655335936);
        carryH[0] = (FieldElement *)((uint8_t *)pAddress + 3672113152);
        carryH[1] = (FieldElement *)((uint8_t *)pAddress + 3688890368);
        carryH[2] = (FieldElement *)((uint8_t *)pAddress + 3705667584);
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
        freeInA = (uint8_t *)((uint8_t *)pAddress + 3722444800);
        freeInB = (uint8_t *)((uint8_t *)pAddress + 3724541952);
        freeInC = (uint8_t *)((uint8_t *)pAddress + 3726639104);
        a0 = (uint32_t *)((uint8_t *)pAddress + 3728736256);
        a1 = (uint32_t *)((uint8_t *)pAddress + 3745513472);
        a2 = (uint32_t *)((uint8_t *)pAddress + 3762290688);
        a3 = (uint32_t *)((uint8_t *)pAddress + 3779067904);
        a4 = (uint32_t *)((uint8_t *)pAddress + 3795845120);
        a5 = (uint32_t *)((uint8_t *)pAddress + 3812622336);
        a6 = (uint32_t *)((uint8_t *)pAddress + 3829399552);
        a7 = (uint32_t *)((uint8_t *)pAddress + 3846176768);
        b0 = (uint32_t *)((uint8_t *)pAddress + 3862953984);
        b1 = (uint32_t *)((uint8_t *)pAddress + 3879731200);
        b2 = (uint32_t *)((uint8_t *)pAddress + 3896508416);
        b3 = (uint32_t *)((uint8_t *)pAddress + 3913285632);
        b4 = (uint32_t *)((uint8_t *)pAddress + 3930062848);
        b5 = (uint32_t *)((uint8_t *)pAddress + 3946840064);
        b6 = (uint32_t *)((uint8_t *)pAddress + 3963617280);
        b7 = (uint32_t *)((uint8_t *)pAddress + 3980394496);
        c0 = (uint32_t *)((uint8_t *)pAddress + 3997171712);
        c1 = (uint32_t *)((uint8_t *)pAddress + 4013948928);
        c2 = (uint32_t *)((uint8_t *)pAddress + 4030726144);
        c3 = (uint32_t *)((uint8_t *)pAddress + 4047503360);
        c4 = (uint32_t *)((uint8_t *)pAddress + 4064280576);
        c5 = (uint32_t *)((uint8_t *)pAddress + 4081057792);
        c6 = (uint32_t *)((uint8_t *)pAddress + 4097835008);
        c7 = (uint32_t *)((uint8_t *)pAddress + 4114612224);
        opcode = (uint8_t *)((uint8_t *)pAddress + 4131389440);
        cIn = (uint8_t *)((uint8_t *)pAddress + 4133486592);
        cOut = (uint8_t *)((uint8_t *)pAddress + 4135583744);
        lCout = (uint8_t *)((uint8_t *)pAddress + 4137680896);
        lOpcode = (uint8_t *)((uint8_t *)pAddress + 4139778048);
        last = (uint8_t *)((uint8_t *)pAddress + 4141875200);
        useCarry = (uint8_t *)((uint8_t *)pAddress + 4143972352);
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
        opcode = (uint8_t *)((uint8_t *)pAddress + 195*degree);
        cIn = (uint8_t *)((uint8_t *)pAddress + 196*degree);
        cOut = (uint8_t *)((uint8_t *)pAddress + 197*degree);
        lCout = (uint8_t *)((uint8_t *)pAddress + 198*degree);
        lOpcode = (uint8_t *)((uint8_t *)pAddress + 199*degree);
        last = (uint8_t *)((uint8_t *)pAddress + 200*degree);
        useCarry = (uint8_t *)((uint8_t *)pAddress + 201*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 202; }
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
        in0 = (FieldElement *)((uint8_t *)pAddress + 4146069504);
        in1 = (FieldElement *)((uint8_t *)pAddress + 4162846720);
        in2 = (FieldElement *)((uint8_t *)pAddress + 4179623936);
        in3 = (FieldElement *)((uint8_t *)pAddress + 4196401152);
        in4 = (FieldElement *)((uint8_t *)pAddress + 4213178368);
        in5 = (FieldElement *)((uint8_t *)pAddress + 4229955584);
        in6 = (FieldElement *)((uint8_t *)pAddress + 4246732800);
        in7 = (FieldElement *)((uint8_t *)pAddress + 4263510016);
        hashType = (FieldElement *)((uint8_t *)pAddress + 4280287232);
        cap1 = (FieldElement *)((uint8_t *)pAddress + 4297064448);
        cap2 = (FieldElement *)((uint8_t *)pAddress + 4313841664);
        cap3 = (FieldElement *)((uint8_t *)pAddress + 4330618880);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 4347396096);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 4364173312);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 4380950528);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 4397727744);
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
        acc[0] = (FieldElement *)((uint8_t *)pAddress + 4414504960);
        acc[1] = (FieldElement *)((uint8_t *)pAddress + 4431282176);
        acc[2] = (FieldElement *)((uint8_t *)pAddress + 4448059392);
        acc[3] = (FieldElement *)((uint8_t *)pAddress + 4464836608);
        acc[4] = (FieldElement *)((uint8_t *)pAddress + 4481613824);
        acc[5] = (FieldElement *)((uint8_t *)pAddress + 4498391040);
        acc[6] = (FieldElement *)((uint8_t *)pAddress + 4515168256);
        acc[7] = (FieldElement *)((uint8_t *)pAddress + 4531945472);
        freeIn = (FieldElement *)((uint8_t *)pAddress + 4548722688);
        addr = (FieldElement *)((uint8_t *)pAddress + 4565499904);
        rem = (FieldElement *)((uint8_t *)pAddress + 4582277120);
        remInv = (FieldElement *)((uint8_t *)pAddress + 4599054336);
        spare = (FieldElement *)((uint8_t *)pAddress + 4615831552);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 4632608768);
        curHash0 = (FieldElement *)((uint8_t *)pAddress + 4649385984);
        curHash1 = (FieldElement *)((uint8_t *)pAddress + 4666163200);
        curHash2 = (FieldElement *)((uint8_t *)pAddress + 4682940416);
        curHash3 = (FieldElement *)((uint8_t *)pAddress + 4699717632);
        prevHash0 = (FieldElement *)((uint8_t *)pAddress + 4716494848);
        prevHash1 = (FieldElement *)((uint8_t *)pAddress + 4733272064);
        prevHash2 = (FieldElement *)((uint8_t *)pAddress + 4750049280);
        prevHash3 = (FieldElement *)((uint8_t *)pAddress + 4766826496);
        len = (FieldElement *)((uint8_t *)pAddress + 4783603712);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 4800380928);
        crLen = (FieldElement *)((uint8_t *)pAddress + 4817158144);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 4833935360);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 4850712576);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 4867489792);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 4884267008);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 4901044224);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 4917821440);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 4934598656);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 4951375872);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 4968153088);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 4984930304);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 5001707520);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 5018484736);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 5035261952);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 5052039168);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 5068816384);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 5085593600);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 5102370816);
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
        free0 = (uint64_t *)((uint8_t *)pAddress + 5119148032);
        free1 = (uint64_t *)((uint8_t *)pAddress + 5135925248);
        free2 = (uint64_t *)((uint8_t *)pAddress + 5152702464);
        free3 = (uint64_t *)((uint8_t *)pAddress + 5169479680);
        hashLeft0 = (uint64_t *)((uint8_t *)pAddress + 5186256896);
        hashLeft1 = (uint64_t *)((uint8_t *)pAddress + 5203034112);
        hashLeft2 = (uint64_t *)((uint8_t *)pAddress + 5219811328);
        hashLeft3 = (uint64_t *)((uint8_t *)pAddress + 5236588544);
        hashRight0 = (uint64_t *)((uint8_t *)pAddress + 5253365760);
        hashRight1 = (uint64_t *)((uint8_t *)pAddress + 5270142976);
        hashRight2 = (uint64_t *)((uint8_t *)pAddress + 5286920192);
        hashRight3 = (uint64_t *)((uint8_t *)pAddress + 5303697408);
        oldRoot0 = (uint64_t *)((uint8_t *)pAddress + 5320474624);
        oldRoot1 = (uint64_t *)((uint8_t *)pAddress + 5337251840);
        oldRoot2 = (uint64_t *)((uint8_t *)pAddress + 5354029056);
        oldRoot3 = (uint64_t *)((uint8_t *)pAddress + 5370806272);
        newRoot0 = (uint64_t *)((uint8_t *)pAddress + 5387583488);
        newRoot1 = (uint64_t *)((uint8_t *)pAddress + 5404360704);
        newRoot2 = (uint64_t *)((uint8_t *)pAddress + 5421137920);
        newRoot3 = (uint64_t *)((uint8_t *)pAddress + 5437915136);
        valueLow0 = (uint64_t *)((uint8_t *)pAddress + 5454692352);
        valueLow1 = (uint64_t *)((uint8_t *)pAddress + 5471469568);
        valueLow2 = (uint64_t *)((uint8_t *)pAddress + 5488246784);
        valueLow3 = (uint64_t *)((uint8_t *)pAddress + 5505024000);
        valueHigh0 = (uint64_t *)((uint8_t *)pAddress + 5521801216);
        valueHigh1 = (uint64_t *)((uint8_t *)pAddress + 5538578432);
        valueHigh2 = (uint64_t *)((uint8_t *)pAddress + 5555355648);
        valueHigh3 = (uint64_t *)((uint8_t *)pAddress + 5572132864);
        siblingValueHash0 = (uint64_t *)((uint8_t *)pAddress + 5588910080);
        siblingValueHash1 = (uint64_t *)((uint8_t *)pAddress + 5605687296);
        siblingValueHash2 = (uint64_t *)((uint8_t *)pAddress + 5622464512);
        siblingValueHash3 = (uint64_t *)((uint8_t *)pAddress + 5639241728);
        rkey0 = (uint64_t *)((uint8_t *)pAddress + 5656018944);
        rkey1 = (uint64_t *)((uint8_t *)pAddress + 5672796160);
        rkey2 = (uint64_t *)((uint8_t *)pAddress + 5689573376);
        rkey3 = (uint64_t *)((uint8_t *)pAddress + 5706350592);
        siblingRkey0 = (uint64_t *)((uint8_t *)pAddress + 5723127808);
        siblingRkey1 = (uint64_t *)((uint8_t *)pAddress + 5739905024);
        siblingRkey2 = (uint64_t *)((uint8_t *)pAddress + 5756682240);
        siblingRkey3 = (uint64_t *)((uint8_t *)pAddress + 5773459456);
        rkeyBit = (uint64_t *)((uint8_t *)pAddress + 5790236672);
        level0 = (uint64_t *)((uint8_t *)pAddress + 5807013888);
        level1 = (uint64_t *)((uint8_t *)pAddress + 5823791104);
        level2 = (uint64_t *)((uint8_t *)pAddress + 5840568320);
        level3 = (uint64_t *)((uint8_t *)pAddress + 5857345536);
        pc = (uint64_t *)((uint8_t *)pAddress + 5874122752);
        selOldRoot = (uint8_t *)((uint8_t *)pAddress + 5890899968);
        selNewRoot = (uint8_t *)((uint8_t *)pAddress + 5892997120);
        selValueLow = (uint8_t *)((uint8_t *)pAddress + 5895094272);
        selValueHigh = (uint8_t *)((uint8_t *)pAddress + 5897191424);
        selSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 5899288576);
        selRkey = (uint8_t *)((uint8_t *)pAddress + 5901385728);
        selRkeyBit = (uint8_t *)((uint8_t *)pAddress + 5903482880);
        selSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 5905580032);
        selFree = (uint8_t *)((uint8_t *)pAddress + 5907677184);
        setHashLeft = (uint8_t *)((uint8_t *)pAddress + 5909774336);
        setHashRight = (uint8_t *)((uint8_t *)pAddress + 5911871488);
        setOldRoot = (uint8_t *)((uint8_t *)pAddress + 5913968640);
        setNewRoot = (uint8_t *)((uint8_t *)pAddress + 5916065792);
        setValueLow = (uint8_t *)((uint8_t *)pAddress + 5918162944);
        setValueHigh = (uint8_t *)((uint8_t *)pAddress + 5920260096);
        setSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 5922357248);
        setRkey = (uint8_t *)((uint8_t *)pAddress + 5924454400);
        setSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 5926551552);
        setRkeyBit = (uint8_t *)((uint8_t *)pAddress + 5928648704);
        setLevel = (uint8_t *)((uint8_t *)pAddress + 5930745856);
        iHash = (uint8_t *)((uint8_t *)pAddress + 5932843008);
        iHashType = (uint8_t *)((uint8_t *)pAddress + 5934940160);
        iLatchSet = (uint8_t *)((uint8_t *)pAddress + 5937037312);
        iLatchGet = (uint8_t *)((uint8_t *)pAddress + 5939134464);
        iClimbRkey = (uint8_t *)((uint8_t *)pAddress + 5941231616);
        iClimbSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 5943328768);
        iClimbSiblingRkeyN = (uint8_t *)((uint8_t *)pAddress + 5945425920);
        iRotateLevel = (uint8_t *)((uint8_t *)pAddress + 5947523072);
        iJmpz = (uint8_t *)((uint8_t *)pAddress + 5949620224);
        iJmp = (uint8_t *)((uint8_t *)pAddress + 5951717376);
        iConst0 = (uint64_t *)((uint8_t *)pAddress + 5953814528);
        iConst1 = (uint64_t *)((uint8_t *)pAddress + 5970591744);
        iConst2 = (uint64_t *)((uint8_t *)pAddress + 5987368960);
        iConst3 = (uint64_t *)((uint8_t *)pAddress + 6004146176);
        iAddress = (uint64_t *)((uint8_t *)pAddress + 6020923392);
        op0inv = (FieldElement *)((uint8_t *)pAddress + 6037700608);
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
        freeA = (FieldElement *)((uint8_t *)pAddress + 6054477824);
        freeB = (FieldElement *)((uint8_t *)pAddress + 6071255040);
        gateType = (FieldElement *)((uint8_t *)pAddress + 6088032256);
        freeANorm = (FieldElement *)((uint8_t *)pAddress + 6104809472);
        freeBNorm = (FieldElement *)((uint8_t *)pAddress + 6121586688);
        freeCNorm = (FieldElement *)((uint8_t *)pAddress + 6138363904);
        a = (FieldElement *)((uint8_t *)pAddress + 6155141120);
        b = (FieldElement *)((uint8_t *)pAddress + 6171918336);
        c = (FieldElement *)((uint8_t *)pAddress + 6188695552);
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
        a = (FieldElement *)((uint8_t *)pAddress + 6205472768);
        b = (FieldElement *)((uint8_t *)pAddress + 6222249984);
        c = (FieldElement *)((uint8_t *)pAddress + 6239027200);
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
        bit = (FieldElement *)((uint8_t *)pAddress + 6255804416);
        field9 = (FieldElement *)((uint8_t *)pAddress + 6272581632);
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
        rBit = (FieldElement *)((uint8_t *)pAddress + 6289358848);
        sOutBit = (FieldElement *)((uint8_t *)pAddress + 6306136064);
        r8 = (FieldElement *)((uint8_t *)pAddress + 6322913280);
        connected = (FieldElement *)((uint8_t *)pAddress + 6339690496);
        sOut0 = (FieldElement *)((uint8_t *)pAddress + 6356467712);
        sOut1 = (FieldElement *)((uint8_t *)pAddress + 6373244928);
        sOut2 = (FieldElement *)((uint8_t *)pAddress + 6390022144);
        sOut3 = (FieldElement *)((uint8_t *)pAddress + 6406799360);
        sOut4 = (FieldElement *)((uint8_t *)pAddress + 6423576576);
        sOut5 = (FieldElement *)((uint8_t *)pAddress + 6440353792);
        sOut6 = (FieldElement *)((uint8_t *)pAddress + 6457131008);
        sOut7 = (FieldElement *)((uint8_t *)pAddress + 6473908224);
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
        freeIn = (FieldElement *)((uint8_t *)pAddress + 6490685440);
        connected = (FieldElement *)((uint8_t *)pAddress + 6507462656);
        addr = (FieldElement *)((uint8_t *)pAddress + 6524239872);
        rem = (FieldElement *)((uint8_t *)pAddress + 6541017088);
        remInv = (FieldElement *)((uint8_t *)pAddress + 6557794304);
        spare = (FieldElement *)((uint8_t *)pAddress + 6574571520);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 6591348736);
        len = (FieldElement *)((uint8_t *)pAddress + 6608125952);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 6624903168);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 6641680384);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 6658457600);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 6675234816);
        hash4 = (FieldElement *)((uint8_t *)pAddress + 6692012032);
        hash5 = (FieldElement *)((uint8_t *)pAddress + 6708789248);
        hash6 = (FieldElement *)((uint8_t *)pAddress + 6725566464);
        hash7 = (FieldElement *)((uint8_t *)pAddress + 6742343680);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 6759120896);
        crLen = (FieldElement *)((uint8_t *)pAddress + 6775898112);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 6792675328);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 6809452544);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 6826229760);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 6843006976);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 6859784192);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 6876561408);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 6893338624);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 6910115840);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 6926893056);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 6943670272);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 6960447488);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 6977224704);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 6994001920);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 7010779136);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 7027556352);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 7044333568);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 7061110784);
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
        addr = (FieldElement *)((uint8_t *)pAddress + 7077888000);
        step = (FieldElement *)((uint8_t *)pAddress + 7094665216);
        mOp = (FieldElement *)((uint8_t *)pAddress + 7111442432);
        mWr = (FieldElement *)((uint8_t *)pAddress + 7128219648);
        val[0] = (FieldElement *)((uint8_t *)pAddress + 7144996864);
        val[1] = (FieldElement *)((uint8_t *)pAddress + 7161774080);
        val[2] = (FieldElement *)((uint8_t *)pAddress + 7178551296);
        val[3] = (FieldElement *)((uint8_t *)pAddress + 7195328512);
        val[4] = (FieldElement *)((uint8_t *)pAddress + 7212105728);
        val[5] = (FieldElement *)((uint8_t *)pAddress + 7228882944);
        val[6] = (FieldElement *)((uint8_t *)pAddress + 7245660160);
        val[7] = (FieldElement *)((uint8_t *)pAddress + 7262437376);
        lastAccess = (FieldElement *)((uint8_t *)pAddress + 7279214592);
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
        A7 = (uint32_t *)((uint8_t *)pAddress + 7295991808);
        A6 = (uint32_t *)((uint8_t *)pAddress + 7312769024);
        A5 = (uint32_t *)((uint8_t *)pAddress + 7329546240);
        A4 = (uint32_t *)((uint8_t *)pAddress + 7346323456);
        A3 = (uint32_t *)((uint8_t *)pAddress + 7363100672);
        A2 = (uint32_t *)((uint8_t *)pAddress + 7379877888);
        A1 = (uint32_t *)((uint8_t *)pAddress + 7396655104);
        A0 = (FieldElement *)((uint8_t *)pAddress + 7413432320);
        B7 = (uint32_t *)((uint8_t *)pAddress + 7430209536);
        B6 = (uint32_t *)((uint8_t *)pAddress + 7446986752);
        B5 = (uint32_t *)((uint8_t *)pAddress + 7463763968);
        B4 = (uint32_t *)((uint8_t *)pAddress + 7480541184);
        B3 = (uint32_t *)((uint8_t *)pAddress + 7497318400);
        B2 = (uint32_t *)((uint8_t *)pAddress + 7514095616);
        B1 = (uint32_t *)((uint8_t *)pAddress + 7530872832);
        B0 = (FieldElement *)((uint8_t *)pAddress + 7547650048);
        C7 = (uint32_t *)((uint8_t *)pAddress + 7564427264);
        C6 = (uint32_t *)((uint8_t *)pAddress + 7581204480);
        C5 = (uint32_t *)((uint8_t *)pAddress + 7597981696);
        C4 = (uint32_t *)((uint8_t *)pAddress + 7614758912);
        C3 = (uint32_t *)((uint8_t *)pAddress + 7631536128);
        C2 = (uint32_t *)((uint8_t *)pAddress + 7648313344);
        C1 = (uint32_t *)((uint8_t *)pAddress + 7665090560);
        C0 = (FieldElement *)((uint8_t *)pAddress + 7681867776);
        D7 = (uint32_t *)((uint8_t *)pAddress + 7698644992);
        D6 = (uint32_t *)((uint8_t *)pAddress + 7715422208);
        D5 = (uint32_t *)((uint8_t *)pAddress + 7732199424);
        D4 = (uint32_t *)((uint8_t *)pAddress + 7748976640);
        D3 = (uint32_t *)((uint8_t *)pAddress + 7765753856);
        D2 = (uint32_t *)((uint8_t *)pAddress + 7782531072);
        D1 = (uint32_t *)((uint8_t *)pAddress + 7799308288);
        D0 = (FieldElement *)((uint8_t *)pAddress + 7816085504);
        E7 = (uint32_t *)((uint8_t *)pAddress + 7832862720);
        E6 = (uint32_t *)((uint8_t *)pAddress + 7849639936);
        E5 = (uint32_t *)((uint8_t *)pAddress + 7866417152);
        E4 = (uint32_t *)((uint8_t *)pAddress + 7883194368);
        E3 = (uint32_t *)((uint8_t *)pAddress + 7899971584);
        E2 = (uint32_t *)((uint8_t *)pAddress + 7916748800);
        E1 = (uint32_t *)((uint8_t *)pAddress + 7933526016);
        E0 = (FieldElement *)((uint8_t *)pAddress + 7950303232);
        SR7 = (uint32_t *)((uint8_t *)pAddress + 7967080448);
        SR6 = (uint32_t *)((uint8_t *)pAddress + 7983857664);
        SR5 = (uint32_t *)((uint8_t *)pAddress + 8000634880);
        SR4 = (uint32_t *)((uint8_t *)pAddress + 8017412096);
        SR3 = (uint32_t *)((uint8_t *)pAddress + 8034189312);
        SR2 = (uint32_t *)((uint8_t *)pAddress + 8050966528);
        SR1 = (uint32_t *)((uint8_t *)pAddress + 8067743744);
        SR0 = (uint32_t *)((uint8_t *)pAddress + 8084520960);
        CTX = (uint32_t *)((uint8_t *)pAddress + 8101298176);
        SP = (uint16_t *)((uint8_t *)pAddress + 8118075392);
        PC = (uint32_t *)((uint8_t *)pAddress + 8122269696);
        GAS = (uint64_t *)((uint8_t *)pAddress + 8139046912);
        MAXMEM = (uint32_t *)((uint8_t *)pAddress + 8155824128);
        zkPC = (uint32_t *)((uint8_t *)pAddress + 8172601344);
        RR = (uint32_t *)((uint8_t *)pAddress + 8189378560);
        HASHPOS = (uint32_t *)((uint8_t *)pAddress + 8206155776);
        CONST7 = (FieldElement *)((uint8_t *)pAddress + 8222932992);
        CONST6 = (FieldElement *)((uint8_t *)pAddress + 8239710208);
        CONST5 = (FieldElement *)((uint8_t *)pAddress + 8256487424);
        CONST4 = (FieldElement *)((uint8_t *)pAddress + 8273264640);
        CONST3 = (FieldElement *)((uint8_t *)pAddress + 8290041856);
        CONST2 = (FieldElement *)((uint8_t *)pAddress + 8306819072);
        CONST1 = (FieldElement *)((uint8_t *)pAddress + 8323596288);
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 8340373504);
        FREE7 = (FieldElement *)((uint8_t *)pAddress + 8357150720);
        FREE6 = (FieldElement *)((uint8_t *)pAddress + 8373927936);
        FREE5 = (FieldElement *)((uint8_t *)pAddress + 8390705152);
        FREE4 = (FieldElement *)((uint8_t *)pAddress + 8407482368);
        FREE3 = (FieldElement *)((uint8_t *)pAddress + 8424259584);
        FREE2 = (FieldElement *)((uint8_t *)pAddress + 8441036800);
        FREE1 = (FieldElement *)((uint8_t *)pAddress + 8457814016);
        FREE0 = (FieldElement *)((uint8_t *)pAddress + 8474591232);
        inA = (FieldElement *)((uint8_t *)pAddress + 8491368448);
        inB = (FieldElement *)((uint8_t *)pAddress + 8508145664);
        inC = (FieldElement *)((uint8_t *)pAddress + 8524922880);
        inD = (FieldElement *)((uint8_t *)pAddress + 8541700096);
        inE = (FieldElement *)((uint8_t *)pAddress + 8558477312);
        inSR = (FieldElement *)((uint8_t *)pAddress + 8575254528);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 8592031744);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 8608808960);
        inSP = (FieldElement *)((uint8_t *)pAddress + 8625586176);
        inPC = (FieldElement *)((uint8_t *)pAddress + 8642363392);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 8659140608);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 8675917824);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 8692695040);
        inRR = (FieldElement *)((uint8_t *)pAddress + 8709472256);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 8726249472);
        setA = (uint8_t *)((uint8_t *)pAddress + 8743026688);
        setB = (uint8_t *)((uint8_t *)pAddress + 8745123840);
        setC = (uint8_t *)((uint8_t *)pAddress + 8747220992);
        setD = (uint8_t *)((uint8_t *)pAddress + 8749318144);
        setE = (uint8_t *)((uint8_t *)pAddress + 8751415296);
        setSR = (uint8_t *)((uint8_t *)pAddress + 8753512448);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 8755609600);
        setSP = (uint8_t *)((uint8_t *)pAddress + 8757706752);
        setPC = (uint8_t *)((uint8_t *)pAddress + 8759803904);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 8761901056);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 8763998208);
        JMP = (uint8_t *)((uint8_t *)pAddress + 8766095360);
        JMPN = (uint8_t *)((uint8_t *)pAddress + 8768192512);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 8770289664);
        setRR = (uint8_t *)((uint8_t *)pAddress + 8772386816);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 8774483968);
        offset = (uint32_t *)((uint8_t *)pAddress + 8776581120);
        incStack = (int32_t *)((uint8_t *)pAddress + 8793358336);
        incCode = (int32_t *)((uint8_t *)pAddress + 8801746944);
        isStack = (uint8_t *)((uint8_t *)pAddress + 8810135552);
        isCode = (uint8_t *)((uint8_t *)pAddress + 8812232704);
        isMem = (uint8_t *)((uint8_t *)pAddress + 8814329856);
        ind = (uint8_t *)((uint8_t *)pAddress + 8816427008);
        indRR = (uint8_t *)((uint8_t *)pAddress + 8818524160);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 8820621312);
        carry = (uint8_t *)((uint8_t *)pAddress + 8822718464);
        mOp = (uint8_t *)((uint8_t *)pAddress + 8824815616);
        mWR = (uint8_t *)((uint8_t *)pAddress + 8826912768);
        sWR = (uint8_t *)((uint8_t *)pAddress + 8829009920);
        sRD = (uint8_t *)((uint8_t *)pAddress + 8831107072);
        arith = (uint8_t *)((uint8_t *)pAddress + 8833204224);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 8835301376);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 8837398528);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 8839495680);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 8841592832);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 8843689984);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 8845787136);
        hashK = (uint8_t *)((uint8_t *)pAddress + 8847884288);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 8849981440);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 8852078592);
        hashP = (uint8_t *)((uint8_t *)pAddress + 8854175744);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 8856272896);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 8858370048);
        bin = (uint8_t *)((uint8_t *)pAddress + 8860467200);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 8862564352);
        assert = (uint8_t *)((uint8_t *)pAddress + 8864661504);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 8866758656);
        isNeg = (uint8_t *)((uint8_t *)pAddress + 8868855808);
        isMaxMem = (uint8_t *)((uint8_t *)pAddress + 8870952960);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 8873050112);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 8889827328);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 8906604544);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 8923381760);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 8940158976);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 8956936192);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 8973713408);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 8990490624);
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

    static uint64_t degree (void) { return 2097152; }
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

    static uint64_t size (void) { return 9007267840; }
};

#endif // COMMIT_POLS_HPP
