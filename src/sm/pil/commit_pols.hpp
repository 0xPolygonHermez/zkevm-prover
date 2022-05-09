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
        out = (uint32_t *)((uint8_t *)pAddress + 131072);
    }

    static uint64_t degree (void) { return 65536; }
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
        x1[0] = (FieldElement *)((uint8_t *)pAddress + 655360);
        x1[1] = (FieldElement *)((uint8_t *)pAddress + 4849664);
        x1[2] = (FieldElement *)((uint8_t *)pAddress + 9043968);
        x1[3] = (FieldElement *)((uint8_t *)pAddress + 13238272);
        x1[4] = (FieldElement *)((uint8_t *)pAddress + 17432576);
        x1[5] = (FieldElement *)((uint8_t *)pAddress + 21626880);
        x1[6] = (FieldElement *)((uint8_t *)pAddress + 25821184);
        x1[7] = (FieldElement *)((uint8_t *)pAddress + 30015488);
        x1[8] = (FieldElement *)((uint8_t *)pAddress + 34209792);
        x1[9] = (FieldElement *)((uint8_t *)pAddress + 38404096);
        x1[10] = (FieldElement *)((uint8_t *)pAddress + 42598400);
        x1[11] = (FieldElement *)((uint8_t *)pAddress + 46792704);
        x1[12] = (FieldElement *)((uint8_t *)pAddress + 50987008);
        x1[13] = (FieldElement *)((uint8_t *)pAddress + 55181312);
        x1[14] = (FieldElement *)((uint8_t *)pAddress + 59375616);
        x1[15] = (FieldElement *)((uint8_t *)pAddress + 63569920);
        y1[0] = (FieldElement *)((uint8_t *)pAddress + 67764224);
        y1[1] = (FieldElement *)((uint8_t *)pAddress + 71958528);
        y1[2] = (FieldElement *)((uint8_t *)pAddress + 76152832);
        y1[3] = (FieldElement *)((uint8_t *)pAddress + 80347136);
        y1[4] = (FieldElement *)((uint8_t *)pAddress + 84541440);
        y1[5] = (FieldElement *)((uint8_t *)pAddress + 88735744);
        y1[6] = (FieldElement *)((uint8_t *)pAddress + 92930048);
        y1[7] = (FieldElement *)((uint8_t *)pAddress + 97124352);
        y1[8] = (FieldElement *)((uint8_t *)pAddress + 101318656);
        y1[9] = (FieldElement *)((uint8_t *)pAddress + 105512960);
        y1[10] = (FieldElement *)((uint8_t *)pAddress + 109707264);
        y1[11] = (FieldElement *)((uint8_t *)pAddress + 113901568);
        y1[12] = (FieldElement *)((uint8_t *)pAddress + 118095872);
        y1[13] = (FieldElement *)((uint8_t *)pAddress + 122290176);
        y1[14] = (FieldElement *)((uint8_t *)pAddress + 126484480);
        y1[15] = (FieldElement *)((uint8_t *)pAddress + 130678784);
        x2[0] = (FieldElement *)((uint8_t *)pAddress + 134873088);
        x2[1] = (FieldElement *)((uint8_t *)pAddress + 139067392);
        x2[2] = (FieldElement *)((uint8_t *)pAddress + 143261696);
        x2[3] = (FieldElement *)((uint8_t *)pAddress + 147456000);
        x2[4] = (FieldElement *)((uint8_t *)pAddress + 151650304);
        x2[5] = (FieldElement *)((uint8_t *)pAddress + 155844608);
        x2[6] = (FieldElement *)((uint8_t *)pAddress + 160038912);
        x2[7] = (FieldElement *)((uint8_t *)pAddress + 164233216);
        x2[8] = (FieldElement *)((uint8_t *)pAddress + 168427520);
        x2[9] = (FieldElement *)((uint8_t *)pAddress + 172621824);
        x2[10] = (FieldElement *)((uint8_t *)pAddress + 176816128);
        x2[11] = (FieldElement *)((uint8_t *)pAddress + 181010432);
        x2[12] = (FieldElement *)((uint8_t *)pAddress + 185204736);
        x2[13] = (FieldElement *)((uint8_t *)pAddress + 189399040);
        x2[14] = (FieldElement *)((uint8_t *)pAddress + 193593344);
        x2[15] = (FieldElement *)((uint8_t *)pAddress + 197787648);
        y2[0] = (FieldElement *)((uint8_t *)pAddress + 201981952);
        y2[1] = (FieldElement *)((uint8_t *)pAddress + 206176256);
        y2[2] = (FieldElement *)((uint8_t *)pAddress + 210370560);
        y2[3] = (FieldElement *)((uint8_t *)pAddress + 214564864);
        y2[4] = (FieldElement *)((uint8_t *)pAddress + 218759168);
        y2[5] = (FieldElement *)((uint8_t *)pAddress + 222953472);
        y2[6] = (FieldElement *)((uint8_t *)pAddress + 227147776);
        y2[7] = (FieldElement *)((uint8_t *)pAddress + 231342080);
        y2[8] = (FieldElement *)((uint8_t *)pAddress + 235536384);
        y2[9] = (FieldElement *)((uint8_t *)pAddress + 239730688);
        y2[10] = (FieldElement *)((uint8_t *)pAddress + 243924992);
        y2[11] = (FieldElement *)((uint8_t *)pAddress + 248119296);
        y2[12] = (FieldElement *)((uint8_t *)pAddress + 252313600);
        y2[13] = (FieldElement *)((uint8_t *)pAddress + 256507904);
        y2[14] = (FieldElement *)((uint8_t *)pAddress + 260702208);
        y2[15] = (FieldElement *)((uint8_t *)pAddress + 264896512);
        x3[0] = (FieldElement *)((uint8_t *)pAddress + 269090816);
        x3[1] = (FieldElement *)((uint8_t *)pAddress + 273285120);
        x3[2] = (FieldElement *)((uint8_t *)pAddress + 277479424);
        x3[3] = (FieldElement *)((uint8_t *)pAddress + 281673728);
        x3[4] = (FieldElement *)((uint8_t *)pAddress + 285868032);
        x3[5] = (FieldElement *)((uint8_t *)pAddress + 290062336);
        x3[6] = (FieldElement *)((uint8_t *)pAddress + 294256640);
        x3[7] = (FieldElement *)((uint8_t *)pAddress + 298450944);
        x3[8] = (FieldElement *)((uint8_t *)pAddress + 302645248);
        x3[9] = (FieldElement *)((uint8_t *)pAddress + 306839552);
        x3[10] = (FieldElement *)((uint8_t *)pAddress + 311033856);
        x3[11] = (FieldElement *)((uint8_t *)pAddress + 315228160);
        x3[12] = (FieldElement *)((uint8_t *)pAddress + 319422464);
        x3[13] = (FieldElement *)((uint8_t *)pAddress + 323616768);
        x3[14] = (FieldElement *)((uint8_t *)pAddress + 327811072);
        x3[15] = (FieldElement *)((uint8_t *)pAddress + 332005376);
        y3[0] = (FieldElement *)((uint8_t *)pAddress + 336199680);
        y3[1] = (FieldElement *)((uint8_t *)pAddress + 340393984);
        y3[2] = (FieldElement *)((uint8_t *)pAddress + 344588288);
        y3[3] = (FieldElement *)((uint8_t *)pAddress + 348782592);
        y3[4] = (FieldElement *)((uint8_t *)pAddress + 352976896);
        y3[5] = (FieldElement *)((uint8_t *)pAddress + 357171200);
        y3[6] = (FieldElement *)((uint8_t *)pAddress + 361365504);
        y3[7] = (FieldElement *)((uint8_t *)pAddress + 365559808);
        y3[8] = (FieldElement *)((uint8_t *)pAddress + 369754112);
        y3[9] = (FieldElement *)((uint8_t *)pAddress + 373948416);
        y3[10] = (FieldElement *)((uint8_t *)pAddress + 378142720);
        y3[11] = (FieldElement *)((uint8_t *)pAddress + 382337024);
        y3[12] = (FieldElement *)((uint8_t *)pAddress + 386531328);
        y3[13] = (FieldElement *)((uint8_t *)pAddress + 390725632);
        y3[14] = (FieldElement *)((uint8_t *)pAddress + 394919936);
        y3[15] = (FieldElement *)((uint8_t *)pAddress + 399114240);
        s[0] = (FieldElement *)((uint8_t *)pAddress + 403308544);
        s[1] = (FieldElement *)((uint8_t *)pAddress + 407502848);
        s[2] = (FieldElement *)((uint8_t *)pAddress + 411697152);
        s[3] = (FieldElement *)((uint8_t *)pAddress + 415891456);
        s[4] = (FieldElement *)((uint8_t *)pAddress + 420085760);
        s[5] = (FieldElement *)((uint8_t *)pAddress + 424280064);
        s[6] = (FieldElement *)((uint8_t *)pAddress + 428474368);
        s[7] = (FieldElement *)((uint8_t *)pAddress + 432668672);
        s[8] = (FieldElement *)((uint8_t *)pAddress + 436862976);
        s[9] = (FieldElement *)((uint8_t *)pAddress + 441057280);
        s[10] = (FieldElement *)((uint8_t *)pAddress + 445251584);
        s[11] = (FieldElement *)((uint8_t *)pAddress + 449445888);
        s[12] = (FieldElement *)((uint8_t *)pAddress + 453640192);
        s[13] = (FieldElement *)((uint8_t *)pAddress + 457834496);
        s[14] = (FieldElement *)((uint8_t *)pAddress + 462028800);
        s[15] = (FieldElement *)((uint8_t *)pAddress + 466223104);
        q0[0] = (FieldElement *)((uint8_t *)pAddress + 470417408);
        q0[1] = (FieldElement *)((uint8_t *)pAddress + 474611712);
        q0[2] = (FieldElement *)((uint8_t *)pAddress + 478806016);
        q0[3] = (FieldElement *)((uint8_t *)pAddress + 483000320);
        q0[4] = (FieldElement *)((uint8_t *)pAddress + 487194624);
        q0[5] = (FieldElement *)((uint8_t *)pAddress + 491388928);
        q0[6] = (FieldElement *)((uint8_t *)pAddress + 495583232);
        q0[7] = (FieldElement *)((uint8_t *)pAddress + 499777536);
        q0[8] = (FieldElement *)((uint8_t *)pAddress + 503971840);
        q0[9] = (FieldElement *)((uint8_t *)pAddress + 508166144);
        q0[10] = (FieldElement *)((uint8_t *)pAddress + 512360448);
        q0[11] = (FieldElement *)((uint8_t *)pAddress + 516554752);
        q0[12] = (FieldElement *)((uint8_t *)pAddress + 520749056);
        q0[13] = (FieldElement *)((uint8_t *)pAddress + 524943360);
        q0[14] = (FieldElement *)((uint8_t *)pAddress + 529137664);
        q0[15] = (FieldElement *)((uint8_t *)pAddress + 533331968);
        q1[0] = (FieldElement *)((uint8_t *)pAddress + 537526272);
        q1[1] = (FieldElement *)((uint8_t *)pAddress + 541720576);
        q1[2] = (FieldElement *)((uint8_t *)pAddress + 545914880);
        q1[3] = (FieldElement *)((uint8_t *)pAddress + 550109184);
        q1[4] = (FieldElement *)((uint8_t *)pAddress + 554303488);
        q1[5] = (FieldElement *)((uint8_t *)pAddress + 558497792);
        q1[6] = (FieldElement *)((uint8_t *)pAddress + 562692096);
        q1[7] = (FieldElement *)((uint8_t *)pAddress + 566886400);
        q1[8] = (FieldElement *)((uint8_t *)pAddress + 571080704);
        q1[9] = (FieldElement *)((uint8_t *)pAddress + 575275008);
        q1[10] = (FieldElement *)((uint8_t *)pAddress + 579469312);
        q1[11] = (FieldElement *)((uint8_t *)pAddress + 583663616);
        q1[12] = (FieldElement *)((uint8_t *)pAddress + 587857920);
        q1[13] = (FieldElement *)((uint8_t *)pAddress + 592052224);
        q1[14] = (FieldElement *)((uint8_t *)pAddress + 596246528);
        q1[15] = (FieldElement *)((uint8_t *)pAddress + 600440832);
        q2[0] = (FieldElement *)((uint8_t *)pAddress + 604635136);
        q2[1] = (FieldElement *)((uint8_t *)pAddress + 608829440);
        q2[2] = (FieldElement *)((uint8_t *)pAddress + 613023744);
        q2[3] = (FieldElement *)((uint8_t *)pAddress + 617218048);
        q2[4] = (FieldElement *)((uint8_t *)pAddress + 621412352);
        q2[5] = (FieldElement *)((uint8_t *)pAddress + 625606656);
        q2[6] = (FieldElement *)((uint8_t *)pAddress + 629800960);
        q2[7] = (FieldElement *)((uint8_t *)pAddress + 633995264);
        q2[8] = (FieldElement *)((uint8_t *)pAddress + 638189568);
        q2[9] = (FieldElement *)((uint8_t *)pAddress + 642383872);
        q2[10] = (FieldElement *)((uint8_t *)pAddress + 646578176);
        q2[11] = (FieldElement *)((uint8_t *)pAddress + 650772480);
        q2[12] = (FieldElement *)((uint8_t *)pAddress + 654966784);
        q2[13] = (FieldElement *)((uint8_t *)pAddress + 659161088);
        q2[14] = (FieldElement *)((uint8_t *)pAddress + 663355392);
        q2[15] = (FieldElement *)((uint8_t *)pAddress + 667549696);
        selEq[0] = (FieldElement *)((uint8_t *)pAddress + 671744000);
        selEq[1] = (FieldElement *)((uint8_t *)pAddress + 675938304);
        selEq[2] = (FieldElement *)((uint8_t *)pAddress + 680132608);
        selEq[3] = (FieldElement *)((uint8_t *)pAddress + 684326912);
        carryL[0] = (FieldElement *)((uint8_t *)pAddress + 688521216);
        carryL[1] = (FieldElement *)((uint8_t *)pAddress + 692715520);
        carryL[2] = (FieldElement *)((uint8_t *)pAddress + 696909824);
        carryH[0] = (FieldElement *)((uint8_t *)pAddress + 701104128);
        carryH[1] = (FieldElement *)((uint8_t *)pAddress + 705298432);
        carryH[2] = (FieldElement *)((uint8_t *)pAddress + 709492736);
    }

    static uint64_t degree (void) { return 524288; }
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
        freeInA = (uint8_t *)((uint8_t *)pAddress + 713687040);
        freeInB = (uint8_t *)((uint8_t *)pAddress + 713752576);
        freeInC = (uint8_t *)((uint8_t *)pAddress + 713818112);
        a0 = (uint32_t *)((uint8_t *)pAddress + 713883648);
        a1 = (uint32_t *)((uint8_t *)pAddress + 714407936);
        a2 = (uint32_t *)((uint8_t *)pAddress + 714932224);
        a3 = (uint32_t *)((uint8_t *)pAddress + 715456512);
        a4 = (uint32_t *)((uint8_t *)pAddress + 715980800);
        a5 = (uint32_t *)((uint8_t *)pAddress + 716505088);
        a6 = (uint32_t *)((uint8_t *)pAddress + 717029376);
        a7 = (uint32_t *)((uint8_t *)pAddress + 717553664);
        b0 = (uint32_t *)((uint8_t *)pAddress + 718077952);
        b1 = (uint32_t *)((uint8_t *)pAddress + 718602240);
        b2 = (uint32_t *)((uint8_t *)pAddress + 719126528);
        b3 = (uint32_t *)((uint8_t *)pAddress + 719650816);
        b4 = (uint32_t *)((uint8_t *)pAddress + 720175104);
        b5 = (uint32_t *)((uint8_t *)pAddress + 720699392);
        b6 = (uint32_t *)((uint8_t *)pAddress + 721223680);
        b7 = (uint32_t *)((uint8_t *)pAddress + 721747968);
        c0 = (uint32_t *)((uint8_t *)pAddress + 722272256);
        c1 = (uint32_t *)((uint8_t *)pAddress + 722796544);
        c2 = (uint32_t *)((uint8_t *)pAddress + 723320832);
        c3 = (uint32_t *)((uint8_t *)pAddress + 723845120);
        c4 = (uint32_t *)((uint8_t *)pAddress + 724369408);
        c5 = (uint32_t *)((uint8_t *)pAddress + 724893696);
        c6 = (uint32_t *)((uint8_t *)pAddress + 725417984);
        c7 = (uint32_t *)((uint8_t *)pAddress + 725942272);
        c0Temp = (uint32_t *)((uint8_t *)pAddress + 726466560);
        opcode = (uint8_t *)((uint8_t *)pAddress + 726990848);
        cIn = (uint8_t *)((uint8_t *)pAddress + 727056384);
        cOut = (uint8_t *)((uint8_t *)pAddress + 727121920);
        last = (uint8_t *)((uint8_t *)pAddress + 727187456);
        useCarry = (uint8_t *)((uint8_t *)pAddress + 727252992);
    }

    static uint64_t degree (void) { return 65536; }
};

class RamCommitPols
{
public:
    FieldElement * addr;
    FieldElement * step;
    FieldElement * mOp;
    FieldElement * mWr;
    FieldElement * val[8];
    FieldElement * lastAccess;

    RamCommitPols (void * pAddress)
    {
        addr = (FieldElement *)((uint8_t *)pAddress + 727318528);
        step = (FieldElement *)((uint8_t *)pAddress + 727842816);
        mOp = (FieldElement *)((uint8_t *)pAddress + 728367104);
        mWr = (FieldElement *)((uint8_t *)pAddress + 728891392);
        val[0] = (FieldElement *)((uint8_t *)pAddress + 729415680);
        val[1] = (FieldElement *)((uint8_t *)pAddress + 729939968);
        val[2] = (FieldElement *)((uint8_t *)pAddress + 730464256);
        val[3] = (FieldElement *)((uint8_t *)pAddress + 730988544);
        val[4] = (FieldElement *)((uint8_t *)pAddress + 731512832);
        val[5] = (FieldElement *)((uint8_t *)pAddress + 732037120);
        val[6] = (FieldElement *)((uint8_t *)pAddress + 732561408);
        val[7] = (FieldElement *)((uint8_t *)pAddress + 733085696);
        lastAccess = (FieldElement *)((uint8_t *)pAddress + 733609984);
    }

    static uint64_t degree (void) { return 65536; }
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
        in0 = (FieldElement *)((uint8_t *)pAddress + 734134272);
        in1 = (FieldElement *)((uint8_t *)pAddress + 734658560);
        in2 = (FieldElement *)((uint8_t *)pAddress + 735182848);
        in3 = (FieldElement *)((uint8_t *)pAddress + 735707136);
        in4 = (FieldElement *)((uint8_t *)pAddress + 736231424);
        in5 = (FieldElement *)((uint8_t *)pAddress + 736755712);
        in6 = (FieldElement *)((uint8_t *)pAddress + 737280000);
        in7 = (FieldElement *)((uint8_t *)pAddress + 737804288);
        hashType = (FieldElement *)((uint8_t *)pAddress + 738328576);
        cap1 = (FieldElement *)((uint8_t *)pAddress + 738852864);
        cap2 = (FieldElement *)((uint8_t *)pAddress + 739377152);
        cap3 = (FieldElement *)((uint8_t *)pAddress + 739901440);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 740425728);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 740950016);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 741474304);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 741998592);
    }

    static uint64_t degree (void) { return 65536; }
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
    uint8_t * iRotateLevel;
    uint8_t * iJmpz;
    uint8_t * iJmp;
    uint64_t * iConst0;
    uint64_t * iConst1;
    uint64_t * iConst2;
    uint64_t * iConst3;
    uint64_t * iAddress;
    FieldElement * op0Inv;
    FieldElement * op0inv;

    StorageCommitPols (void * pAddress)
    {
        free0 = (uint64_t *)((uint8_t *)pAddress + 742522880);
        free1 = (uint64_t *)((uint8_t *)pAddress + 743047168);
        free2 = (uint64_t *)((uint8_t *)pAddress + 743571456);
        free3 = (uint64_t *)((uint8_t *)pAddress + 744095744);
        hashLeft0 = (uint64_t *)((uint8_t *)pAddress + 744620032);
        hashLeft1 = (uint64_t *)((uint8_t *)pAddress + 745144320);
        hashLeft2 = (uint64_t *)((uint8_t *)pAddress + 745668608);
        hashLeft3 = (uint64_t *)((uint8_t *)pAddress + 746192896);
        hashRight0 = (uint64_t *)((uint8_t *)pAddress + 746717184);
        hashRight1 = (uint64_t *)((uint8_t *)pAddress + 747241472);
        hashRight2 = (uint64_t *)((uint8_t *)pAddress + 747765760);
        hashRight3 = (uint64_t *)((uint8_t *)pAddress + 748290048);
        oldRoot0 = (uint64_t *)((uint8_t *)pAddress + 748814336);
        oldRoot1 = (uint64_t *)((uint8_t *)pAddress + 749338624);
        oldRoot2 = (uint64_t *)((uint8_t *)pAddress + 749862912);
        oldRoot3 = (uint64_t *)((uint8_t *)pAddress + 750387200);
        newRoot0 = (uint64_t *)((uint8_t *)pAddress + 750911488);
        newRoot1 = (uint64_t *)((uint8_t *)pAddress + 751435776);
        newRoot2 = (uint64_t *)((uint8_t *)pAddress + 751960064);
        newRoot3 = (uint64_t *)((uint8_t *)pAddress + 752484352);
        valueLow0 = (uint64_t *)((uint8_t *)pAddress + 753008640);
        valueLow1 = (uint64_t *)((uint8_t *)pAddress + 753532928);
        valueLow2 = (uint64_t *)((uint8_t *)pAddress + 754057216);
        valueLow3 = (uint64_t *)((uint8_t *)pAddress + 754581504);
        valueHigh0 = (uint64_t *)((uint8_t *)pAddress + 755105792);
        valueHigh1 = (uint64_t *)((uint8_t *)pAddress + 755630080);
        valueHigh2 = (uint64_t *)((uint8_t *)pAddress + 756154368);
        valueHigh3 = (uint64_t *)((uint8_t *)pAddress + 756678656);
        siblingValueHash0 = (uint64_t *)((uint8_t *)pAddress + 757202944);
        siblingValueHash1 = (uint64_t *)((uint8_t *)pAddress + 757727232);
        siblingValueHash2 = (uint64_t *)((uint8_t *)pAddress + 758251520);
        siblingValueHash3 = (uint64_t *)((uint8_t *)pAddress + 758775808);
        rkey0 = (uint64_t *)((uint8_t *)pAddress + 759300096);
        rkey1 = (uint64_t *)((uint8_t *)pAddress + 759824384);
        rkey2 = (uint64_t *)((uint8_t *)pAddress + 760348672);
        rkey3 = (uint64_t *)((uint8_t *)pAddress + 760872960);
        siblingRkey0 = (uint64_t *)((uint8_t *)pAddress + 761397248);
        siblingRkey1 = (uint64_t *)((uint8_t *)pAddress + 761921536);
        siblingRkey2 = (uint64_t *)((uint8_t *)pAddress + 762445824);
        siblingRkey3 = (uint64_t *)((uint8_t *)pAddress + 762970112);
        rkeyBit = (uint64_t *)((uint8_t *)pAddress + 763494400);
        level0 = (uint64_t *)((uint8_t *)pAddress + 764018688);
        level1 = (uint64_t *)((uint8_t *)pAddress + 764542976);
        level2 = (uint64_t *)((uint8_t *)pAddress + 765067264);
        level3 = (uint64_t *)((uint8_t *)pAddress + 765591552);
        pc = (uint64_t *)((uint8_t *)pAddress + 766115840);
        selOldRoot = (uint8_t *)((uint8_t *)pAddress + 766640128);
        selNewRoot = (uint8_t *)((uint8_t *)pAddress + 766705664);
        selValueLow = (uint8_t *)((uint8_t *)pAddress + 766771200);
        selValueHigh = (uint8_t *)((uint8_t *)pAddress + 766836736);
        selSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 766902272);
        selRkey = (uint8_t *)((uint8_t *)pAddress + 766967808);
        selRkeyBit = (uint8_t *)((uint8_t *)pAddress + 767033344);
        selSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 767098880);
        selFree = (uint8_t *)((uint8_t *)pAddress + 767164416);
        setHashLeft = (uint8_t *)((uint8_t *)pAddress + 767229952);
        setHashRight = (uint8_t *)((uint8_t *)pAddress + 767295488);
        setOldRoot = (uint8_t *)((uint8_t *)pAddress + 767361024);
        setNewRoot = (uint8_t *)((uint8_t *)pAddress + 767426560);
        setValueLow = (uint8_t *)((uint8_t *)pAddress + 767492096);
        setValueHigh = (uint8_t *)((uint8_t *)pAddress + 767557632);
        setSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 767623168);
        setRkey = (uint8_t *)((uint8_t *)pAddress + 767688704);
        setSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 767754240);
        setRkeyBit = (uint8_t *)((uint8_t *)pAddress + 767819776);
        setLevel = (uint8_t *)((uint8_t *)pAddress + 767885312);
        iHash = (uint8_t *)((uint8_t *)pAddress + 767950848);
        iHashType = (uint8_t *)((uint8_t *)pAddress + 768016384);
        iLatchSet = (uint8_t *)((uint8_t *)pAddress + 768081920);
        iLatchGet = (uint8_t *)((uint8_t *)pAddress + 768147456);
        iClimbRkey = (uint8_t *)((uint8_t *)pAddress + 768212992);
        iClimbSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 768278528);
        iRotateLevel = (uint8_t *)((uint8_t *)pAddress + 768344064);
        iJmpz = (uint8_t *)((uint8_t *)pAddress + 768409600);
        iJmp = (uint8_t *)((uint8_t *)pAddress + 768475136);
        iConst0 = (uint64_t *)((uint8_t *)pAddress + 768540672);
        iConst1 = (uint64_t *)((uint8_t *)pAddress + 769064960);
        iConst2 = (uint64_t *)((uint8_t *)pAddress + 769589248);
        iConst3 = (uint64_t *)((uint8_t *)pAddress + 770113536);
        iAddress = (uint64_t *)((uint8_t *)pAddress + 770637824);
        op0Inv = (FieldElement *)((uint8_t *)pAddress + 771162112);
        op0inv = (FieldElement *)((uint8_t *)pAddress + 771686400);
    }

    static uint64_t degree (void) { return 65536; }
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
    uint8_t * mWR;
    uint8_t * mRD;
    uint8_t * sWR;
    uint8_t * sRD;
    uint8_t * arith;
    uint8_t * arithEq0;
    uint8_t * arithEq1;
    uint8_t * arithEq2;
    uint8_t * arithEq3;
    uint8_t * shl;
    uint8_t * shr;
    uint8_t * hashK;
    uint8_t * hashKLen;
    uint8_t * hashKDigest;
    uint8_t * hashP;
    uint8_t * hashPLen;
    uint8_t * hashPDigest;
    uint8_t * ecRecover;
    uint8_t * comparator;
    uint8_t * bin;
    uint8_t * binOpcode;
    uint8_t * assert;
    uint8_t * opcodeRomMap;
    uint8_t * isNeg;
    uint8_t * isMaxMem;

    MainCommitPols (void * pAddress)
    {
        A7 = (uint32_t *)((uint8_t *)pAddress + 772210688);
        A6 = (uint32_t *)((uint8_t *)pAddress + 772734976);
        A5 = (uint32_t *)((uint8_t *)pAddress + 773259264);
        A4 = (uint32_t *)((uint8_t *)pAddress + 773783552);
        A3 = (uint32_t *)((uint8_t *)pAddress + 774307840);
        A2 = (uint32_t *)((uint8_t *)pAddress + 774832128);
        A1 = (uint32_t *)((uint8_t *)pAddress + 775356416);
        A0 = (FieldElement *)((uint8_t *)pAddress + 775880704);
        B7 = (uint32_t *)((uint8_t *)pAddress + 776404992);
        B6 = (uint32_t *)((uint8_t *)pAddress + 776929280);
        B5 = (uint32_t *)((uint8_t *)pAddress + 777453568);
        B4 = (uint32_t *)((uint8_t *)pAddress + 777977856);
        B3 = (uint32_t *)((uint8_t *)pAddress + 778502144);
        B2 = (uint32_t *)((uint8_t *)pAddress + 779026432);
        B1 = (uint32_t *)((uint8_t *)pAddress + 779550720);
        B0 = (FieldElement *)((uint8_t *)pAddress + 780075008);
        C7 = (uint32_t *)((uint8_t *)pAddress + 780599296);
        C6 = (uint32_t *)((uint8_t *)pAddress + 781123584);
        C5 = (uint32_t *)((uint8_t *)pAddress + 781647872);
        C4 = (uint32_t *)((uint8_t *)pAddress + 782172160);
        C3 = (uint32_t *)((uint8_t *)pAddress + 782696448);
        C2 = (uint32_t *)((uint8_t *)pAddress + 783220736);
        C1 = (uint32_t *)((uint8_t *)pAddress + 783745024);
        C0 = (FieldElement *)((uint8_t *)pAddress + 784269312);
        D7 = (uint32_t *)((uint8_t *)pAddress + 784793600);
        D6 = (uint32_t *)((uint8_t *)pAddress + 785317888);
        D5 = (uint32_t *)((uint8_t *)pAddress + 785842176);
        D4 = (uint32_t *)((uint8_t *)pAddress + 786366464);
        D3 = (uint32_t *)((uint8_t *)pAddress + 786890752);
        D2 = (uint32_t *)((uint8_t *)pAddress + 787415040);
        D1 = (uint32_t *)((uint8_t *)pAddress + 787939328);
        D0 = (FieldElement *)((uint8_t *)pAddress + 788463616);
        E7 = (uint32_t *)((uint8_t *)pAddress + 788987904);
        E6 = (uint32_t *)((uint8_t *)pAddress + 789512192);
        E5 = (uint32_t *)((uint8_t *)pAddress + 790036480);
        E4 = (uint32_t *)((uint8_t *)pAddress + 790560768);
        E3 = (uint32_t *)((uint8_t *)pAddress + 791085056);
        E2 = (uint32_t *)((uint8_t *)pAddress + 791609344);
        E1 = (uint32_t *)((uint8_t *)pAddress + 792133632);
        E0 = (FieldElement *)((uint8_t *)pAddress + 792657920);
        SR7 = (uint32_t *)((uint8_t *)pAddress + 793182208);
        SR6 = (uint32_t *)((uint8_t *)pAddress + 793706496);
        SR5 = (uint32_t *)((uint8_t *)pAddress + 794230784);
        SR4 = (uint32_t *)((uint8_t *)pAddress + 794755072);
        SR3 = (uint32_t *)((uint8_t *)pAddress + 795279360);
        SR2 = (uint32_t *)((uint8_t *)pAddress + 795803648);
        SR1 = (uint32_t *)((uint8_t *)pAddress + 796327936);
        SR0 = (uint32_t *)((uint8_t *)pAddress + 796852224);
        CTX = (uint32_t *)((uint8_t *)pAddress + 797376512);
        SP = (uint16_t *)((uint8_t *)pAddress + 797900800);
        PC = (uint32_t *)((uint8_t *)pAddress + 798031872);
        GAS = (uint64_t *)((uint8_t *)pAddress + 798556160);
        MAXMEM = (uint32_t *)((uint8_t *)pAddress + 799080448);
        zkPC = (uint32_t *)((uint8_t *)pAddress + 799604736);
        RR = (uint32_t *)((uint8_t *)pAddress + 800129024);
        HASHPOS = (uint32_t *)((uint8_t *)pAddress + 800653312);
        CONST7 = (FieldElement *)((uint8_t *)pAddress + 801177600);
        CONST6 = (FieldElement *)((uint8_t *)pAddress + 801701888);
        CONST5 = (FieldElement *)((uint8_t *)pAddress + 802226176);
        CONST4 = (FieldElement *)((uint8_t *)pAddress + 802750464);
        CONST3 = (FieldElement *)((uint8_t *)pAddress + 803274752);
        CONST2 = (FieldElement *)((uint8_t *)pAddress + 803799040);
        CONST1 = (FieldElement *)((uint8_t *)pAddress + 804323328);
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 804847616);
        FREE7 = (FieldElement *)((uint8_t *)pAddress + 805371904);
        FREE6 = (FieldElement *)((uint8_t *)pAddress + 805896192);
        FREE5 = (FieldElement *)((uint8_t *)pAddress + 806420480);
        FREE4 = (FieldElement *)((uint8_t *)pAddress + 806944768);
        FREE3 = (FieldElement *)((uint8_t *)pAddress + 807469056);
        FREE2 = (FieldElement *)((uint8_t *)pAddress + 807993344);
        FREE1 = (FieldElement *)((uint8_t *)pAddress + 808517632);
        FREE0 = (FieldElement *)((uint8_t *)pAddress + 809041920);
        inA = (FieldElement *)((uint8_t *)pAddress + 809566208);
        inB = (FieldElement *)((uint8_t *)pAddress + 810090496);
        inC = (FieldElement *)((uint8_t *)pAddress + 810614784);
        inD = (FieldElement *)((uint8_t *)pAddress + 811139072);
        inE = (FieldElement *)((uint8_t *)pAddress + 811663360);
        inSR = (FieldElement *)((uint8_t *)pAddress + 812187648);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 812711936);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 813236224);
        inSP = (FieldElement *)((uint8_t *)pAddress + 813760512);
        inPC = (FieldElement *)((uint8_t *)pAddress + 814284800);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 814809088);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 815333376);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 815857664);
        inRR = (FieldElement *)((uint8_t *)pAddress + 816381952);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 816906240);
        setA = (uint8_t *)((uint8_t *)pAddress + 817430528);
        setB = (uint8_t *)((uint8_t *)pAddress + 817496064);
        setC = (uint8_t *)((uint8_t *)pAddress + 817561600);
        setD = (uint8_t *)((uint8_t *)pAddress + 817627136);
        setE = (uint8_t *)((uint8_t *)pAddress + 817692672);
        setSR = (uint8_t *)((uint8_t *)pAddress + 817758208);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 817823744);
        setSP = (uint8_t *)((uint8_t *)pAddress + 817889280);
        setPC = (uint8_t *)((uint8_t *)pAddress + 817954816);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 818020352);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 818085888);
        JMP = (uint8_t *)((uint8_t *)pAddress + 818151424);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 818216960);
        setRR = (uint8_t *)((uint8_t *)pAddress + 818282496);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 818348032);
        offset = (uint32_t *)((uint8_t *)pAddress + 818413568);
        incStack = (int32_t *)((uint8_t *)pAddress + 818937856);
        incCode = (int32_t *)((uint8_t *)pAddress + 819200000);
        isStack = (uint8_t *)((uint8_t *)pAddress + 819462144);
        isCode = (uint8_t *)((uint8_t *)pAddress + 819527680);
        isMem = (uint8_t *)((uint8_t *)pAddress + 819593216);
        ind = (uint8_t *)((uint8_t *)pAddress + 819658752);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 819724288);
        mWR = (uint8_t *)((uint8_t *)pAddress + 819789824);
        mRD = (uint8_t *)((uint8_t *)pAddress + 819855360);
        sWR = (uint8_t *)((uint8_t *)pAddress + 819920896);
        sRD = (uint8_t *)((uint8_t *)pAddress + 819986432);
        arith = (uint8_t *)((uint8_t *)pAddress + 820051968);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 820117504);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 820183040);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 820248576);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 820314112);
        shl = (uint8_t *)((uint8_t *)pAddress + 820379648);
        shr = (uint8_t *)((uint8_t *)pAddress + 820445184);
        hashK = (uint8_t *)((uint8_t *)pAddress + 820510720);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 820576256);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 820641792);
        hashP = (uint8_t *)((uint8_t *)pAddress + 820707328);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 820772864);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 820838400);
        ecRecover = (uint8_t *)((uint8_t *)pAddress + 820903936);
        comparator = (uint8_t *)((uint8_t *)pAddress + 820969472);
        bin = (uint8_t *)((uint8_t *)pAddress + 821035008);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 821100544);
        assert = (uint8_t *)((uint8_t *)pAddress + 821166080);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 821231616);
        isNeg = (uint8_t *)((uint8_t *)pAddress + 821297152);
        isMaxMem = (uint8_t *)((uint8_t *)pAddress + 821362688);
    }

    static uint64_t degree (void) { return 65536; }
};

class CommitPols
{
public:
    Byte4CommitPols Byte4;
    ArithCommitPols Arith;
    BinaryCommitPols Binary;
    RamCommitPols Ram;
    PoseidonGCommitPols PoseidonG;
    StorageCommitPols Storage;
    MainCommitPols Main;

    CommitPols (void * pAddress) : Byte4(pAddress), Arith(pAddress), Binary(pAddress), Ram(pAddress), PoseidonG(pAddress), Storage(pAddress), Main(pAddress) {}

    static uint64_t size (void) { return 821428224; }
};

#endif // COMMIT_POLS_HPP
