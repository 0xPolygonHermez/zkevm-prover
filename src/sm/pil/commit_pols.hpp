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
        x1[0] = (FieldElement *)((uint8_t *)pAddress + 41943040);
        x1[1] = (FieldElement *)((uint8_t *)pAddress + 75497472);
        x1[2] = (FieldElement *)((uint8_t *)pAddress + 109051904);
        x1[3] = (FieldElement *)((uint8_t *)pAddress + 142606336);
        x1[4] = (FieldElement *)((uint8_t *)pAddress + 176160768);
        x1[5] = (FieldElement *)((uint8_t *)pAddress + 209715200);
        x1[6] = (FieldElement *)((uint8_t *)pAddress + 243269632);
        x1[7] = (FieldElement *)((uint8_t *)pAddress + 276824064);
        x1[8] = (FieldElement *)((uint8_t *)pAddress + 310378496);
        x1[9] = (FieldElement *)((uint8_t *)pAddress + 343932928);
        x1[10] = (FieldElement *)((uint8_t *)pAddress + 377487360);
        x1[11] = (FieldElement *)((uint8_t *)pAddress + 411041792);
        x1[12] = (FieldElement *)((uint8_t *)pAddress + 444596224);
        x1[13] = (FieldElement *)((uint8_t *)pAddress + 478150656);
        x1[14] = (FieldElement *)((uint8_t *)pAddress + 511705088);
        x1[15] = (FieldElement *)((uint8_t *)pAddress + 545259520);
        y1[0] = (FieldElement *)((uint8_t *)pAddress + 578813952);
        y1[1] = (FieldElement *)((uint8_t *)pAddress + 612368384);
        y1[2] = (FieldElement *)((uint8_t *)pAddress + 645922816);
        y1[3] = (FieldElement *)((uint8_t *)pAddress + 679477248);
        y1[4] = (FieldElement *)((uint8_t *)pAddress + 713031680);
        y1[5] = (FieldElement *)((uint8_t *)pAddress + 746586112);
        y1[6] = (FieldElement *)((uint8_t *)pAddress + 780140544);
        y1[7] = (FieldElement *)((uint8_t *)pAddress + 813694976);
        y1[8] = (FieldElement *)((uint8_t *)pAddress + 847249408);
        y1[9] = (FieldElement *)((uint8_t *)pAddress + 880803840);
        y1[10] = (FieldElement *)((uint8_t *)pAddress + 914358272);
        y1[11] = (FieldElement *)((uint8_t *)pAddress + 947912704);
        y1[12] = (FieldElement *)((uint8_t *)pAddress + 981467136);
        y1[13] = (FieldElement *)((uint8_t *)pAddress + 1015021568);
        y1[14] = (FieldElement *)((uint8_t *)pAddress + 1048576000);
        y1[15] = (FieldElement *)((uint8_t *)pAddress + 1082130432);
        x2[0] = (FieldElement *)((uint8_t *)pAddress + 1115684864);
        x2[1] = (FieldElement *)((uint8_t *)pAddress + 1149239296);
        x2[2] = (FieldElement *)((uint8_t *)pAddress + 1182793728);
        x2[3] = (FieldElement *)((uint8_t *)pAddress + 1216348160);
        x2[4] = (FieldElement *)((uint8_t *)pAddress + 1249902592);
        x2[5] = (FieldElement *)((uint8_t *)pAddress + 1283457024);
        x2[6] = (FieldElement *)((uint8_t *)pAddress + 1317011456);
        x2[7] = (FieldElement *)((uint8_t *)pAddress + 1350565888);
        x2[8] = (FieldElement *)((uint8_t *)pAddress + 1384120320);
        x2[9] = (FieldElement *)((uint8_t *)pAddress + 1417674752);
        x2[10] = (FieldElement *)((uint8_t *)pAddress + 1451229184);
        x2[11] = (FieldElement *)((uint8_t *)pAddress + 1484783616);
        x2[12] = (FieldElement *)((uint8_t *)pAddress + 1518338048);
        x2[13] = (FieldElement *)((uint8_t *)pAddress + 1551892480);
        x2[14] = (FieldElement *)((uint8_t *)pAddress + 1585446912);
        x2[15] = (FieldElement *)((uint8_t *)pAddress + 1619001344);
        y2[0] = (FieldElement *)((uint8_t *)pAddress + 1652555776);
        y2[1] = (FieldElement *)((uint8_t *)pAddress + 1686110208);
        y2[2] = (FieldElement *)((uint8_t *)pAddress + 1719664640);
        y2[3] = (FieldElement *)((uint8_t *)pAddress + 1753219072);
        y2[4] = (FieldElement *)((uint8_t *)pAddress + 1786773504);
        y2[5] = (FieldElement *)((uint8_t *)pAddress + 1820327936);
        y2[6] = (FieldElement *)((uint8_t *)pAddress + 1853882368);
        y2[7] = (FieldElement *)((uint8_t *)pAddress + 1887436800);
        y2[8] = (FieldElement *)((uint8_t *)pAddress + 1920991232);
        y2[9] = (FieldElement *)((uint8_t *)pAddress + 1954545664);
        y2[10] = (FieldElement *)((uint8_t *)pAddress + 1988100096);
        y2[11] = (FieldElement *)((uint8_t *)pAddress + 2021654528);
        y2[12] = (FieldElement *)((uint8_t *)pAddress + 2055208960);
        y2[13] = (FieldElement *)((uint8_t *)pAddress + 2088763392);
        y2[14] = (FieldElement *)((uint8_t *)pAddress + 2122317824);
        y2[15] = (FieldElement *)((uint8_t *)pAddress + 2155872256);
        x3[0] = (FieldElement *)((uint8_t *)pAddress + 2189426688);
        x3[1] = (FieldElement *)((uint8_t *)pAddress + 2222981120);
        x3[2] = (FieldElement *)((uint8_t *)pAddress + 2256535552);
        x3[3] = (FieldElement *)((uint8_t *)pAddress + 2290089984);
        x3[4] = (FieldElement *)((uint8_t *)pAddress + 2323644416);
        x3[5] = (FieldElement *)((uint8_t *)pAddress + 2357198848);
        x3[6] = (FieldElement *)((uint8_t *)pAddress + 2390753280);
        x3[7] = (FieldElement *)((uint8_t *)pAddress + 2424307712);
        x3[8] = (FieldElement *)((uint8_t *)pAddress + 2457862144);
        x3[9] = (FieldElement *)((uint8_t *)pAddress + 2491416576);
        x3[10] = (FieldElement *)((uint8_t *)pAddress + 2524971008);
        x3[11] = (FieldElement *)((uint8_t *)pAddress + 2558525440);
        x3[12] = (FieldElement *)((uint8_t *)pAddress + 2592079872);
        x3[13] = (FieldElement *)((uint8_t *)pAddress + 2625634304);
        x3[14] = (FieldElement *)((uint8_t *)pAddress + 2659188736);
        x3[15] = (FieldElement *)((uint8_t *)pAddress + 2692743168);
        y3[0] = (FieldElement *)((uint8_t *)pAddress + 2726297600);
        y3[1] = (FieldElement *)((uint8_t *)pAddress + 2759852032);
        y3[2] = (FieldElement *)((uint8_t *)pAddress + 2793406464);
        y3[3] = (FieldElement *)((uint8_t *)pAddress + 2826960896);
        y3[4] = (FieldElement *)((uint8_t *)pAddress + 2860515328);
        y3[5] = (FieldElement *)((uint8_t *)pAddress + 2894069760);
        y3[6] = (FieldElement *)((uint8_t *)pAddress + 2927624192);
        y3[7] = (FieldElement *)((uint8_t *)pAddress + 2961178624);
        y3[8] = (FieldElement *)((uint8_t *)pAddress + 2994733056);
        y3[9] = (FieldElement *)((uint8_t *)pAddress + 3028287488);
        y3[10] = (FieldElement *)((uint8_t *)pAddress + 3061841920);
        y3[11] = (FieldElement *)((uint8_t *)pAddress + 3095396352);
        y3[12] = (FieldElement *)((uint8_t *)pAddress + 3128950784);
        y3[13] = (FieldElement *)((uint8_t *)pAddress + 3162505216);
        y3[14] = (FieldElement *)((uint8_t *)pAddress + 3196059648);
        y3[15] = (FieldElement *)((uint8_t *)pAddress + 3229614080);
        s[0] = (FieldElement *)((uint8_t *)pAddress + 3263168512);
        s[1] = (FieldElement *)((uint8_t *)pAddress + 3296722944);
        s[2] = (FieldElement *)((uint8_t *)pAddress + 3330277376);
        s[3] = (FieldElement *)((uint8_t *)pAddress + 3363831808);
        s[4] = (FieldElement *)((uint8_t *)pAddress + 3397386240);
        s[5] = (FieldElement *)((uint8_t *)pAddress + 3430940672);
        s[6] = (FieldElement *)((uint8_t *)pAddress + 3464495104);
        s[7] = (FieldElement *)((uint8_t *)pAddress + 3498049536);
        s[8] = (FieldElement *)((uint8_t *)pAddress + 3531603968);
        s[9] = (FieldElement *)((uint8_t *)pAddress + 3565158400);
        s[10] = (FieldElement *)((uint8_t *)pAddress + 3598712832);
        s[11] = (FieldElement *)((uint8_t *)pAddress + 3632267264);
        s[12] = (FieldElement *)((uint8_t *)pAddress + 3665821696);
        s[13] = (FieldElement *)((uint8_t *)pAddress + 3699376128);
        s[14] = (FieldElement *)((uint8_t *)pAddress + 3732930560);
        s[15] = (FieldElement *)((uint8_t *)pAddress + 3766484992);
        q0[0] = (FieldElement *)((uint8_t *)pAddress + 3800039424);
        q0[1] = (FieldElement *)((uint8_t *)pAddress + 3833593856);
        q0[2] = (FieldElement *)((uint8_t *)pAddress + 3867148288);
        q0[3] = (FieldElement *)((uint8_t *)pAddress + 3900702720);
        q0[4] = (FieldElement *)((uint8_t *)pAddress + 3934257152);
        q0[5] = (FieldElement *)((uint8_t *)pAddress + 3967811584);
        q0[6] = (FieldElement *)((uint8_t *)pAddress + 4001366016);
        q0[7] = (FieldElement *)((uint8_t *)pAddress + 4034920448);
        q0[8] = (FieldElement *)((uint8_t *)pAddress + 4068474880);
        q0[9] = (FieldElement *)((uint8_t *)pAddress + 4102029312);
        q0[10] = (FieldElement *)((uint8_t *)pAddress + 4135583744);
        q0[11] = (FieldElement *)((uint8_t *)pAddress + 4169138176);
        q0[12] = (FieldElement *)((uint8_t *)pAddress + 4202692608);
        q0[13] = (FieldElement *)((uint8_t *)pAddress + 4236247040);
        q0[14] = (FieldElement *)((uint8_t *)pAddress + 4269801472);
        q0[15] = (FieldElement *)((uint8_t *)pAddress + 4303355904);
        q1[0] = (FieldElement *)((uint8_t *)pAddress + 4336910336);
        q1[1] = (FieldElement *)((uint8_t *)pAddress + 4370464768);
        q1[2] = (FieldElement *)((uint8_t *)pAddress + 4404019200);
        q1[3] = (FieldElement *)((uint8_t *)pAddress + 4437573632);
        q1[4] = (FieldElement *)((uint8_t *)pAddress + 4471128064);
        q1[5] = (FieldElement *)((uint8_t *)pAddress + 4504682496);
        q1[6] = (FieldElement *)((uint8_t *)pAddress + 4538236928);
        q1[7] = (FieldElement *)((uint8_t *)pAddress + 4571791360);
        q1[8] = (FieldElement *)((uint8_t *)pAddress + 4605345792);
        q1[9] = (FieldElement *)((uint8_t *)pAddress + 4638900224);
        q1[10] = (FieldElement *)((uint8_t *)pAddress + 4672454656);
        q1[11] = (FieldElement *)((uint8_t *)pAddress + 4706009088);
        q1[12] = (FieldElement *)((uint8_t *)pAddress + 4739563520);
        q1[13] = (FieldElement *)((uint8_t *)pAddress + 4773117952);
        q1[14] = (FieldElement *)((uint8_t *)pAddress + 4806672384);
        q1[15] = (FieldElement *)((uint8_t *)pAddress + 4840226816);
        q2[0] = (FieldElement *)((uint8_t *)pAddress + 4873781248);
        q2[1] = (FieldElement *)((uint8_t *)pAddress + 4907335680);
        q2[2] = (FieldElement *)((uint8_t *)pAddress + 4940890112);
        q2[3] = (FieldElement *)((uint8_t *)pAddress + 4974444544);
        q2[4] = (FieldElement *)((uint8_t *)pAddress + 5007998976);
        q2[5] = (FieldElement *)((uint8_t *)pAddress + 5041553408);
        q2[6] = (FieldElement *)((uint8_t *)pAddress + 5075107840);
        q2[7] = (FieldElement *)((uint8_t *)pAddress + 5108662272);
        q2[8] = (FieldElement *)((uint8_t *)pAddress + 5142216704);
        q2[9] = (FieldElement *)((uint8_t *)pAddress + 5175771136);
        q2[10] = (FieldElement *)((uint8_t *)pAddress + 5209325568);
        q2[11] = (FieldElement *)((uint8_t *)pAddress + 5242880000);
        q2[12] = (FieldElement *)((uint8_t *)pAddress + 5276434432);
        q2[13] = (FieldElement *)((uint8_t *)pAddress + 5309988864);
        q2[14] = (FieldElement *)((uint8_t *)pAddress + 5343543296);
        q2[15] = (FieldElement *)((uint8_t *)pAddress + 5377097728);
        selEq[0] = (FieldElement *)((uint8_t *)pAddress + 5410652160);
        selEq[1] = (FieldElement *)((uint8_t *)pAddress + 5444206592);
        selEq[2] = (FieldElement *)((uint8_t *)pAddress + 5477761024);
        selEq[3] = (FieldElement *)((uint8_t *)pAddress + 5511315456);
        carryL[0] = (FieldElement *)((uint8_t *)pAddress + 5544869888);
        carryL[1] = (FieldElement *)((uint8_t *)pAddress + 5578424320);
        carryL[2] = (FieldElement *)((uint8_t *)pAddress + 5611978752);
        carryH[0] = (FieldElement *)((uint8_t *)pAddress + 5645533184);
        carryH[1] = (FieldElement *)((uint8_t *)pAddress + 5679087616);
        carryH[2] = (FieldElement *)((uint8_t *)pAddress + 5712642048);
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
        freeInA = (uint8_t *)((uint8_t *)pAddress + 5746196480);
        freeInB = (uint8_t *)((uint8_t *)pAddress + 5750390784);
        freeInC = (uint8_t *)((uint8_t *)pAddress + 5754585088);
        a0 = (uint32_t *)((uint8_t *)pAddress + 5758779392);
        a1 = (uint32_t *)((uint8_t *)pAddress + 5792333824);
        a2 = (uint32_t *)((uint8_t *)pAddress + 5825888256);
        a3 = (uint32_t *)((uint8_t *)pAddress + 5859442688);
        a4 = (uint32_t *)((uint8_t *)pAddress + 5892997120);
        a5 = (uint32_t *)((uint8_t *)pAddress + 5926551552);
        a6 = (uint32_t *)((uint8_t *)pAddress + 5960105984);
        a7 = (uint32_t *)((uint8_t *)pAddress + 5993660416);
        b0 = (uint32_t *)((uint8_t *)pAddress + 6027214848);
        b1 = (uint32_t *)((uint8_t *)pAddress + 6060769280);
        b2 = (uint32_t *)((uint8_t *)pAddress + 6094323712);
        b3 = (uint32_t *)((uint8_t *)pAddress + 6127878144);
        b4 = (uint32_t *)((uint8_t *)pAddress + 6161432576);
        b5 = (uint32_t *)((uint8_t *)pAddress + 6194987008);
        b6 = (uint32_t *)((uint8_t *)pAddress + 6228541440);
        b7 = (uint32_t *)((uint8_t *)pAddress + 6262095872);
        c0 = (uint32_t *)((uint8_t *)pAddress + 6295650304);
        c1 = (uint32_t *)((uint8_t *)pAddress + 6329204736);
        c2 = (uint32_t *)((uint8_t *)pAddress + 6362759168);
        c3 = (uint32_t *)((uint8_t *)pAddress + 6396313600);
        c4 = (uint32_t *)((uint8_t *)pAddress + 6429868032);
        c5 = (uint32_t *)((uint8_t *)pAddress + 6463422464);
        c6 = (uint32_t *)((uint8_t *)pAddress + 6496976896);
        c7 = (uint32_t *)((uint8_t *)pAddress + 6530531328);
        c0Temp = (uint32_t *)((uint8_t *)pAddress + 6564085760);
        opcode = (uint8_t *)((uint8_t *)pAddress + 6597640192);
        cIn = (uint8_t *)((uint8_t *)pAddress + 6601834496);
        cOut = (uint8_t *)((uint8_t *)pAddress + 6606028800);
        last = (uint8_t *)((uint8_t *)pAddress + 6610223104);
        useCarry = (uint8_t *)((uint8_t *)pAddress + 6614417408);
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
        addr = (FieldElement *)((uint8_t *)pAddress + 6618611712);
        step = (FieldElement *)((uint8_t *)pAddress + 6652166144);
        mOp = (FieldElement *)((uint8_t *)pAddress + 6685720576);
        mWr = (FieldElement *)((uint8_t *)pAddress + 6719275008);
        val[0] = (FieldElement *)((uint8_t *)pAddress + 6752829440);
        val[1] = (FieldElement *)((uint8_t *)pAddress + 6786383872);
        val[2] = (FieldElement *)((uint8_t *)pAddress + 6819938304);
        val[3] = (FieldElement *)((uint8_t *)pAddress + 6853492736);
        val[4] = (FieldElement *)((uint8_t *)pAddress + 6887047168);
        val[5] = (FieldElement *)((uint8_t *)pAddress + 6920601600);
        val[6] = (FieldElement *)((uint8_t *)pAddress + 6954156032);
        val[7] = (FieldElement *)((uint8_t *)pAddress + 6987710464);
        lastAccess = (FieldElement *)((uint8_t *)pAddress + 7021264896);
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
        in0 = (FieldElement *)((uint8_t *)pAddress + 7054819328);
        in1 = (FieldElement *)((uint8_t *)pAddress + 7088373760);
        in2 = (FieldElement *)((uint8_t *)pAddress + 7121928192);
        in3 = (FieldElement *)((uint8_t *)pAddress + 7155482624);
        in4 = (FieldElement *)((uint8_t *)pAddress + 7189037056);
        in5 = (FieldElement *)((uint8_t *)pAddress + 7222591488);
        in6 = (FieldElement *)((uint8_t *)pAddress + 7256145920);
        in7 = (FieldElement *)((uint8_t *)pAddress + 7289700352);
        hashType = (FieldElement *)((uint8_t *)pAddress + 7323254784);
        cap1 = (FieldElement *)((uint8_t *)pAddress + 7356809216);
        cap2 = (FieldElement *)((uint8_t *)pAddress + 7390363648);
        cap3 = (FieldElement *)((uint8_t *)pAddress + 7423918080);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 7457472512);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 7491026944);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 7524581376);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 7558135808);
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
        free0 = (uint64_t *)((uint8_t *)pAddress + 7591690240);
        free1 = (uint64_t *)((uint8_t *)pAddress + 7625244672);
        free2 = (uint64_t *)((uint8_t *)pAddress + 7658799104);
        free3 = (uint64_t *)((uint8_t *)pAddress + 7692353536);
        hashLeft0 = (uint64_t *)((uint8_t *)pAddress + 7725907968);
        hashLeft1 = (uint64_t *)((uint8_t *)pAddress + 7759462400);
        hashLeft2 = (uint64_t *)((uint8_t *)pAddress + 7793016832);
        hashLeft3 = (uint64_t *)((uint8_t *)pAddress + 7826571264);
        hashRight0 = (uint64_t *)((uint8_t *)pAddress + 7860125696);
        hashRight1 = (uint64_t *)((uint8_t *)pAddress + 7893680128);
        hashRight2 = (uint64_t *)((uint8_t *)pAddress + 7927234560);
        hashRight3 = (uint64_t *)((uint8_t *)pAddress + 7960788992);
        oldRoot0 = (uint64_t *)((uint8_t *)pAddress + 7994343424);
        oldRoot1 = (uint64_t *)((uint8_t *)pAddress + 8027897856);
        oldRoot2 = (uint64_t *)((uint8_t *)pAddress + 8061452288);
        oldRoot3 = (uint64_t *)((uint8_t *)pAddress + 8095006720);
        newRoot0 = (uint64_t *)((uint8_t *)pAddress + 8128561152);
        newRoot1 = (uint64_t *)((uint8_t *)pAddress + 8162115584);
        newRoot2 = (uint64_t *)((uint8_t *)pAddress + 8195670016);
        newRoot3 = (uint64_t *)((uint8_t *)pAddress + 8229224448);
        valueLow0 = (uint64_t *)((uint8_t *)pAddress + 8262778880);
        valueLow1 = (uint64_t *)((uint8_t *)pAddress + 8296333312);
        valueLow2 = (uint64_t *)((uint8_t *)pAddress + 8329887744);
        valueLow3 = (uint64_t *)((uint8_t *)pAddress + 8363442176);
        valueHigh0 = (uint64_t *)((uint8_t *)pAddress + 8396996608);
        valueHigh1 = (uint64_t *)((uint8_t *)pAddress + 8430551040);
        valueHigh2 = (uint64_t *)((uint8_t *)pAddress + 8464105472);
        valueHigh3 = (uint64_t *)((uint8_t *)pAddress + 8497659904);
        siblingValueHash0 = (uint64_t *)((uint8_t *)pAddress + 8531214336);
        siblingValueHash1 = (uint64_t *)((uint8_t *)pAddress + 8564768768);
        siblingValueHash2 = (uint64_t *)((uint8_t *)pAddress + 8598323200);
        siblingValueHash3 = (uint64_t *)((uint8_t *)pAddress + 8631877632);
        rkey0 = (uint64_t *)((uint8_t *)pAddress + 8665432064);
        rkey1 = (uint64_t *)((uint8_t *)pAddress + 8698986496);
        rkey2 = (uint64_t *)((uint8_t *)pAddress + 8732540928);
        rkey3 = (uint64_t *)((uint8_t *)pAddress + 8766095360);
        siblingRkey0 = (uint64_t *)((uint8_t *)pAddress + 8799649792);
        siblingRkey1 = (uint64_t *)((uint8_t *)pAddress + 8833204224);
        siblingRkey2 = (uint64_t *)((uint8_t *)pAddress + 8866758656);
        siblingRkey3 = (uint64_t *)((uint8_t *)pAddress + 8900313088);
        rkeyBit = (uint64_t *)((uint8_t *)pAddress + 8933867520);
        level0 = (uint64_t *)((uint8_t *)pAddress + 8967421952);
        level1 = (uint64_t *)((uint8_t *)pAddress + 9000976384);
        level2 = (uint64_t *)((uint8_t *)pAddress + 9034530816);
        level3 = (uint64_t *)((uint8_t *)pAddress + 9068085248);
        pc = (uint64_t *)((uint8_t *)pAddress + 9101639680);
        selOldRoot = (uint8_t *)((uint8_t *)pAddress + 9135194112);
        selNewRoot = (uint8_t *)((uint8_t *)pAddress + 9139388416);
        selValueLow = (uint8_t *)((uint8_t *)pAddress + 9143582720);
        selValueHigh = (uint8_t *)((uint8_t *)pAddress + 9147777024);
        selSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 9151971328);
        selRkey = (uint8_t *)((uint8_t *)pAddress + 9156165632);
        selRkeyBit = (uint8_t *)((uint8_t *)pAddress + 9160359936);
        selSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 9164554240);
        selFree = (uint8_t *)((uint8_t *)pAddress + 9168748544);
        setHashLeft = (uint8_t *)((uint8_t *)pAddress + 9172942848);
        setHashRight = (uint8_t *)((uint8_t *)pAddress + 9177137152);
        setOldRoot = (uint8_t *)((uint8_t *)pAddress + 9181331456);
        setNewRoot = (uint8_t *)((uint8_t *)pAddress + 9185525760);
        setValueLow = (uint8_t *)((uint8_t *)pAddress + 9189720064);
        setValueHigh = (uint8_t *)((uint8_t *)pAddress + 9193914368);
        setSiblingValueHash = (uint8_t *)((uint8_t *)pAddress + 9198108672);
        setRkey = (uint8_t *)((uint8_t *)pAddress + 9202302976);
        setSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 9206497280);
        setRkeyBit = (uint8_t *)((uint8_t *)pAddress + 9210691584);
        setLevel = (uint8_t *)((uint8_t *)pAddress + 9214885888);
        iHash = (uint8_t *)((uint8_t *)pAddress + 9219080192);
        iHashType = (uint8_t *)((uint8_t *)pAddress + 9223274496);
        iLatchSet = (uint8_t *)((uint8_t *)pAddress + 9227468800);
        iLatchGet = (uint8_t *)((uint8_t *)pAddress + 9231663104);
        iClimbRkey = (uint8_t *)((uint8_t *)pAddress + 9235857408);
        iClimbSiblingRkey = (uint8_t *)((uint8_t *)pAddress + 9240051712);
        iRotateLevel = (uint8_t *)((uint8_t *)pAddress + 9244246016);
        iJmpz = (uint8_t *)((uint8_t *)pAddress + 9248440320);
        iJmp = (uint8_t *)((uint8_t *)pAddress + 9252634624);
        iConst0 = (uint64_t *)((uint8_t *)pAddress + 9256828928);
        iConst1 = (uint64_t *)((uint8_t *)pAddress + 9290383360);
        iConst2 = (uint64_t *)((uint8_t *)pAddress + 9323937792);
        iConst3 = (uint64_t *)((uint8_t *)pAddress + 9357492224);
        iAddress = (uint64_t *)((uint8_t *)pAddress + 9391046656);
        op0Inv = (FieldElement *)((uint8_t *)pAddress + 9424601088);
        op0inv = (FieldElement *)((uint8_t *)pAddress + 9458155520);
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
        freeA = (FieldElement *)((uint8_t *)pAddress + 9491709952);
        freeB = (FieldElement *)((uint8_t *)pAddress + 9525264384);
        gateType = (FieldElement *)((uint8_t *)pAddress + 9558818816);
        freeANorm = (FieldElement *)((uint8_t *)pAddress + 9592373248);
        freeBNorm = (FieldElement *)((uint8_t *)pAddress + 9625927680);
        freeCNorm = (FieldElement *)((uint8_t *)pAddress + 9659482112);
        a = (FieldElement *)((uint8_t *)pAddress + 9693036544);
        b = (FieldElement *)((uint8_t *)pAddress + 9726590976);
        c = (FieldElement *)((uint8_t *)pAddress + 9760145408);
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
        a = (FieldElement *)((uint8_t *)pAddress + 9793699840);
        b = (FieldElement *)((uint8_t *)pAddress + 9827254272);
        c = (FieldElement *)((uint8_t *)pAddress + 9860808704);
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
        bit = (FieldElement *)((uint8_t *)pAddress + 9894363136);
        field9 = (FieldElement *)((uint8_t *)pAddress + 9927917568);
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
        rBit = (FieldElement *)((uint8_t *)pAddress + 9961472000);
        sOutBit = (FieldElement *)((uint8_t *)pAddress + 9995026432);
        r8 = (FieldElement *)((uint8_t *)pAddress + 10028580864);
        connected = (FieldElement *)((uint8_t *)pAddress + 10062135296);
        sOut0 = (FieldElement *)((uint8_t *)pAddress + 10095689728);
        sOut1 = (FieldElement *)((uint8_t *)pAddress + 10129244160);
        sOut2 = (FieldElement *)((uint8_t *)pAddress + 10162798592);
        sOut3 = (FieldElement *)((uint8_t *)pAddress + 10196353024);
        sOut4 = (FieldElement *)((uint8_t *)pAddress + 10229907456);
        sOut5 = (FieldElement *)((uint8_t *)pAddress + 10263461888);
        sOut6 = (FieldElement *)((uint8_t *)pAddress + 10297016320);
        sOut7 = (FieldElement *)((uint8_t *)pAddress + 10330570752);
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
        freeIn = (FieldElement *)((uint8_t *)pAddress + 10364125184);
        connected = (FieldElement *)((uint8_t *)pAddress + 10397679616);
        addr = (FieldElement *)((uint8_t *)pAddress + 10431234048);
        rem = (FieldElement *)((uint8_t *)pAddress + 10464788480);
        remInv = (FieldElement *)((uint8_t *)pAddress + 10498342912);
        spare = (FieldElement *)((uint8_t *)pAddress + 10531897344);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 10565451776);
        len = (FieldElement *)((uint8_t *)pAddress + 10599006208);
        hash0 = (FieldElement *)((uint8_t *)pAddress + 10632560640);
        hash1 = (FieldElement *)((uint8_t *)pAddress + 10666115072);
        hash2 = (FieldElement *)((uint8_t *)pAddress + 10699669504);
        hash3 = (FieldElement *)((uint8_t *)pAddress + 10733223936);
        hash4 = (FieldElement *)((uint8_t *)pAddress + 10766778368);
        hash5 = (FieldElement *)((uint8_t *)pAddress + 10800332800);
        hash6 = (FieldElement *)((uint8_t *)pAddress + 10833887232);
        hash7 = (FieldElement *)((uint8_t *)pAddress + 10867441664);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 10900996096);
        crLen = (FieldElement *)((uint8_t *)pAddress + 10934550528);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 10968104960);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 11001659392);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 11035213824);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 11068768256);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 11102322688);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 11135877120);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 11169431552);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 11202985984);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 11236540416);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 11270094848);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 11303649280);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 11337203712);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 11370758144);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 11404312576);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 11437867008);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 11471421440);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 11504975872);
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
        acc[0] = (FieldElement *)((uint8_t *)pAddress + 11538530304);
        acc[1] = (FieldElement *)((uint8_t *)pAddress + 11572084736);
        acc[2] = (FieldElement *)((uint8_t *)pAddress + 11605639168);
        acc[3] = (FieldElement *)((uint8_t *)pAddress + 11639193600);
        acc[4] = (FieldElement *)((uint8_t *)pAddress + 11672748032);
        acc[5] = (FieldElement *)((uint8_t *)pAddress + 11706302464);
        acc[6] = (FieldElement *)((uint8_t *)pAddress + 11739856896);
        acc[7] = (FieldElement *)((uint8_t *)pAddress + 11773411328);
        freeIn = (FieldElement *)((uint8_t *)pAddress + 11806965760);
        addr = (FieldElement *)((uint8_t *)pAddress + 11840520192);
        rem = (FieldElement *)((uint8_t *)pAddress + 11874074624);
        remInv = (FieldElement *)((uint8_t *)pAddress + 11907629056);
        spare = (FieldElement *)((uint8_t *)pAddress + 11941183488);
        firstHash = (FieldElement *)((uint8_t *)pAddress + 11974737920);
        curHash0 = (FieldElement *)((uint8_t *)pAddress + 12008292352);
        curHash1 = (FieldElement *)((uint8_t *)pAddress + 12041846784);
        curHash2 = (FieldElement *)((uint8_t *)pAddress + 12075401216);
        curHash3 = (FieldElement *)((uint8_t *)pAddress + 12108955648);
        prevHash0 = (FieldElement *)((uint8_t *)pAddress + 12142510080);
        prevHash1 = (FieldElement *)((uint8_t *)pAddress + 12176064512);
        prevHash2 = (FieldElement *)((uint8_t *)pAddress + 12209618944);
        prevHash3 = (FieldElement *)((uint8_t *)pAddress + 12243173376);
        len = (FieldElement *)((uint8_t *)pAddress + 12276727808);
        crOffset = (FieldElement *)((uint8_t *)pAddress + 12310282240);
        crLen = (FieldElement *)((uint8_t *)pAddress + 12343836672);
        crOffsetInv = (FieldElement *)((uint8_t *)pAddress + 12377391104);
        crF0 = (FieldElement *)((uint8_t *)pAddress + 12410945536);
        crF1 = (FieldElement *)((uint8_t *)pAddress + 12444499968);
        crF2 = (FieldElement *)((uint8_t *)pAddress + 12478054400);
        crF3 = (FieldElement *)((uint8_t *)pAddress + 12511608832);
        crF4 = (FieldElement *)((uint8_t *)pAddress + 12545163264);
        crF5 = (FieldElement *)((uint8_t *)pAddress + 12578717696);
        crF6 = (FieldElement *)((uint8_t *)pAddress + 12612272128);
        crF7 = (FieldElement *)((uint8_t *)pAddress + 12645826560);
        crV0 = (FieldElement *)((uint8_t *)pAddress + 12679380992);
        crV1 = (FieldElement *)((uint8_t *)pAddress + 12712935424);
        crV2 = (FieldElement *)((uint8_t *)pAddress + 12746489856);
        crV3 = (FieldElement *)((uint8_t *)pAddress + 12780044288);
        crV4 = (FieldElement *)((uint8_t *)pAddress + 12813598720);
        crV5 = (FieldElement *)((uint8_t *)pAddress + 12847153152);
        crV6 = (FieldElement *)((uint8_t *)pAddress + 12880707584);
        crV7 = (FieldElement *)((uint8_t *)pAddress + 12914262016);
    }

    static uint64_t degree (void) { return 4194304; }
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
        inM = (uint8_t *)((uint8_t *)pAddress + 12947816448);
        inV = (uint8_t *)((uint8_t *)pAddress + 12952010752);
        wr = (uint8_t *)((uint8_t *)pAddress + 12956205056);
        m0[0] = (uint32_t *)((uint8_t *)pAddress + 12960399360);
        m0[1] = (uint32_t *)((uint8_t *)pAddress + 12993953792);
        m0[2] = (uint32_t *)((uint8_t *)pAddress + 13027508224);
        m0[3] = (uint32_t *)((uint8_t *)pAddress + 13061062656);
        m0[4] = (uint32_t *)((uint8_t *)pAddress + 13094617088);
        m0[5] = (uint32_t *)((uint8_t *)pAddress + 13128171520);
        m0[6] = (uint32_t *)((uint8_t *)pAddress + 13161725952);
        m0[7] = (uint32_t *)((uint8_t *)pAddress + 13195280384);
        m1[0] = (uint32_t *)((uint8_t *)pAddress + 13228834816);
        m1[1] = (uint32_t *)((uint8_t *)pAddress + 13262389248);
        m1[2] = (uint32_t *)((uint8_t *)pAddress + 13295943680);
        m1[3] = (uint32_t *)((uint8_t *)pAddress + 13329498112);
        m1[4] = (uint32_t *)((uint8_t *)pAddress + 13363052544);
        m1[5] = (uint32_t *)((uint8_t *)pAddress + 13396606976);
        m1[6] = (uint32_t *)((uint8_t *)pAddress + 13430161408);
        m1[7] = (uint32_t *)((uint8_t *)pAddress + 13463715840);
        w0[0] = (uint32_t *)((uint8_t *)pAddress + 13497270272);
        w0[1] = (uint32_t *)((uint8_t *)pAddress + 13530824704);
        w0[2] = (uint32_t *)((uint8_t *)pAddress + 13564379136);
        w0[3] = (uint32_t *)((uint8_t *)pAddress + 13597933568);
        w0[4] = (uint32_t *)((uint8_t *)pAddress + 13631488000);
        w0[5] = (uint32_t *)((uint8_t *)pAddress + 13665042432);
        w0[6] = (uint32_t *)((uint8_t *)pAddress + 13698596864);
        w0[7] = (uint32_t *)((uint8_t *)pAddress + 13732151296);
        w1[0] = (uint32_t *)((uint8_t *)pAddress + 13765705728);
        w1[1] = (uint32_t *)((uint8_t *)pAddress + 13799260160);
        w1[2] = (uint32_t *)((uint8_t *)pAddress + 13832814592);
        w1[3] = (uint32_t *)((uint8_t *)pAddress + 13866369024);
        w1[4] = (uint32_t *)((uint8_t *)pAddress + 13899923456);
        w1[5] = (uint32_t *)((uint8_t *)pAddress + 13933477888);
        w1[6] = (uint32_t *)((uint8_t *)pAddress + 13967032320);
        w1[7] = (uint32_t *)((uint8_t *)pAddress + 14000586752);
        v[0] = (uint32_t *)((uint8_t *)pAddress + 14034141184);
        v[1] = (uint32_t *)((uint8_t *)pAddress + 14067695616);
        v[2] = (uint32_t *)((uint8_t *)pAddress + 14101250048);
        v[3] = (uint32_t *)((uint8_t *)pAddress + 14134804480);
        v[4] = (uint32_t *)((uint8_t *)pAddress + 14168358912);
        v[5] = (uint32_t *)((uint8_t *)pAddress + 14201913344);
        v[6] = (uint32_t *)((uint8_t *)pAddress + 14235467776);
        v[7] = (uint32_t *)((uint8_t *)pAddress + 14269022208);
        offset = (uint8_t *)((uint8_t *)pAddress + 14302576640);
        selW = (uint8_t *)((uint8_t *)pAddress + 14306770944);
        factorV[0] = (FieldElement *)((uint8_t *)pAddress + 14310965248);
        factorV[1] = (FieldElement *)((uint8_t *)pAddress + 14344519680);
        factorV[2] = (FieldElement *)((uint8_t *)pAddress + 14378074112);
        factorV[3] = (FieldElement *)((uint8_t *)pAddress + 14411628544);
        factorV[4] = (FieldElement *)((uint8_t *)pAddress + 14445182976);
        factorV[5] = (FieldElement *)((uint8_t *)pAddress + 14478737408);
        factorV[6] = (FieldElement *)((uint8_t *)pAddress + 14512291840);
        factorV[7] = (FieldElement *)((uint8_t *)pAddress + 14545846272);
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
        A7 = (uint32_t *)((uint8_t *)pAddress + 14579400704);
        A6 = (uint32_t *)((uint8_t *)pAddress + 14612955136);
        A5 = (uint32_t *)((uint8_t *)pAddress + 14646509568);
        A4 = (uint32_t *)((uint8_t *)pAddress + 14680064000);
        A3 = (uint32_t *)((uint8_t *)pAddress + 14713618432);
        A2 = (uint32_t *)((uint8_t *)pAddress + 14747172864);
        A1 = (uint32_t *)((uint8_t *)pAddress + 14780727296);
        A0 = (FieldElement *)((uint8_t *)pAddress + 14814281728);
        B7 = (uint32_t *)((uint8_t *)pAddress + 14847836160);
        B6 = (uint32_t *)((uint8_t *)pAddress + 14881390592);
        B5 = (uint32_t *)((uint8_t *)pAddress + 14914945024);
        B4 = (uint32_t *)((uint8_t *)pAddress + 14948499456);
        B3 = (uint32_t *)((uint8_t *)pAddress + 14982053888);
        B2 = (uint32_t *)((uint8_t *)pAddress + 15015608320);
        B1 = (uint32_t *)((uint8_t *)pAddress + 15049162752);
        B0 = (FieldElement *)((uint8_t *)pAddress + 15082717184);
        C7 = (uint32_t *)((uint8_t *)pAddress + 15116271616);
        C6 = (uint32_t *)((uint8_t *)pAddress + 15149826048);
        C5 = (uint32_t *)((uint8_t *)pAddress + 15183380480);
        C4 = (uint32_t *)((uint8_t *)pAddress + 15216934912);
        C3 = (uint32_t *)((uint8_t *)pAddress + 15250489344);
        C2 = (uint32_t *)((uint8_t *)pAddress + 15284043776);
        C1 = (uint32_t *)((uint8_t *)pAddress + 15317598208);
        C0 = (FieldElement *)((uint8_t *)pAddress + 15351152640);
        D7 = (uint32_t *)((uint8_t *)pAddress + 15384707072);
        D6 = (uint32_t *)((uint8_t *)pAddress + 15418261504);
        D5 = (uint32_t *)((uint8_t *)pAddress + 15451815936);
        D4 = (uint32_t *)((uint8_t *)pAddress + 15485370368);
        D3 = (uint32_t *)((uint8_t *)pAddress + 15518924800);
        D2 = (uint32_t *)((uint8_t *)pAddress + 15552479232);
        D1 = (uint32_t *)((uint8_t *)pAddress + 15586033664);
        D0 = (FieldElement *)((uint8_t *)pAddress + 15619588096);
        E7 = (uint32_t *)((uint8_t *)pAddress + 15653142528);
        E6 = (uint32_t *)((uint8_t *)pAddress + 15686696960);
        E5 = (uint32_t *)((uint8_t *)pAddress + 15720251392);
        E4 = (uint32_t *)((uint8_t *)pAddress + 15753805824);
        E3 = (uint32_t *)((uint8_t *)pAddress + 15787360256);
        E2 = (uint32_t *)((uint8_t *)pAddress + 15820914688);
        E1 = (uint32_t *)((uint8_t *)pAddress + 15854469120);
        E0 = (FieldElement *)((uint8_t *)pAddress + 15888023552);
        SR7 = (uint32_t *)((uint8_t *)pAddress + 15921577984);
        SR6 = (uint32_t *)((uint8_t *)pAddress + 15955132416);
        SR5 = (uint32_t *)((uint8_t *)pAddress + 15988686848);
        SR4 = (uint32_t *)((uint8_t *)pAddress + 16022241280);
        SR3 = (uint32_t *)((uint8_t *)pAddress + 16055795712);
        SR2 = (uint32_t *)((uint8_t *)pAddress + 16089350144);
        SR1 = (uint32_t *)((uint8_t *)pAddress + 16122904576);
        SR0 = (uint32_t *)((uint8_t *)pAddress + 16156459008);
        CTX = (uint32_t *)((uint8_t *)pAddress + 16190013440);
        SP = (uint16_t *)((uint8_t *)pAddress + 16223567872);
        PC = (uint32_t *)((uint8_t *)pAddress + 16231956480);
        GAS = (uint64_t *)((uint8_t *)pAddress + 16265510912);
        MAXMEM = (uint32_t *)((uint8_t *)pAddress + 16299065344);
        zkPC = (uint32_t *)((uint8_t *)pAddress + 16332619776);
        RR = (uint32_t *)((uint8_t *)pAddress + 16366174208);
        HASHPOS = (uint32_t *)((uint8_t *)pAddress + 16399728640);
        CONST7 = (FieldElement *)((uint8_t *)pAddress + 16433283072);
        CONST6 = (FieldElement *)((uint8_t *)pAddress + 16466837504);
        CONST5 = (FieldElement *)((uint8_t *)pAddress + 16500391936);
        CONST4 = (FieldElement *)((uint8_t *)pAddress + 16533946368);
        CONST3 = (FieldElement *)((uint8_t *)pAddress + 16567500800);
        CONST2 = (FieldElement *)((uint8_t *)pAddress + 16601055232);
        CONST1 = (FieldElement *)((uint8_t *)pAddress + 16634609664);
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 16668164096);
        FREE7 = (FieldElement *)((uint8_t *)pAddress + 16701718528);
        FREE6 = (FieldElement *)((uint8_t *)pAddress + 16735272960);
        FREE5 = (FieldElement *)((uint8_t *)pAddress + 16768827392);
        FREE4 = (FieldElement *)((uint8_t *)pAddress + 16802381824);
        FREE3 = (FieldElement *)((uint8_t *)pAddress + 16835936256);
        FREE2 = (FieldElement *)((uint8_t *)pAddress + 16869490688);
        FREE1 = (FieldElement *)((uint8_t *)pAddress + 16903045120);
        FREE0 = (FieldElement *)((uint8_t *)pAddress + 16936599552);
        inA = (FieldElement *)((uint8_t *)pAddress + 16970153984);
        inB = (FieldElement *)((uint8_t *)pAddress + 17003708416);
        inC = (FieldElement *)((uint8_t *)pAddress + 17037262848);
        inD = (FieldElement *)((uint8_t *)pAddress + 17070817280);
        inE = (FieldElement *)((uint8_t *)pAddress + 17104371712);
        inSR = (FieldElement *)((uint8_t *)pAddress + 17137926144);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 17171480576);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 17205035008);
        inSP = (FieldElement *)((uint8_t *)pAddress + 17238589440);
        inPC = (FieldElement *)((uint8_t *)pAddress + 17272143872);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 17305698304);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 17339252736);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 17372807168);
        inRR = (FieldElement *)((uint8_t *)pAddress + 17406361600);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 17439916032);
        setA = (uint8_t *)((uint8_t *)pAddress + 17473470464);
        setB = (uint8_t *)((uint8_t *)pAddress + 17477664768);
        setC = (uint8_t *)((uint8_t *)pAddress + 17481859072);
        setD = (uint8_t *)((uint8_t *)pAddress + 17486053376);
        setE = (uint8_t *)((uint8_t *)pAddress + 17490247680);
        setSR = (uint8_t *)((uint8_t *)pAddress + 17494441984);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 17498636288);
        setSP = (uint8_t *)((uint8_t *)pAddress + 17502830592);
        setPC = (uint8_t *)((uint8_t *)pAddress + 17507024896);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 17511219200);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 17515413504);
        JMP = (uint8_t *)((uint8_t *)pAddress + 17519607808);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 17523802112);
        setRR = (uint8_t *)((uint8_t *)pAddress + 17527996416);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 17532190720);
        offset = (uint32_t *)((uint8_t *)pAddress + 17536385024);
        incStack = (int32_t *)((uint8_t *)pAddress + 17569939456);
        incCode = (int32_t *)((uint8_t *)pAddress + 17586716672);
        isStack = (uint8_t *)((uint8_t *)pAddress + 17603493888);
        isCode = (uint8_t *)((uint8_t *)pAddress + 17607688192);
        isMem = (uint8_t *)((uint8_t *)pAddress + 17611882496);
        ind = (uint8_t *)((uint8_t *)pAddress + 17616076800);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 17620271104);
        mOp = (uint8_t *)((uint8_t *)pAddress + 17624465408);
        mWR = (uint8_t *)((uint8_t *)pAddress + 17628659712);
        sWR = (uint8_t *)((uint8_t *)pAddress + 17632854016);
        sRD = (uint8_t *)((uint8_t *)pAddress + 17637048320);
        arith = (uint8_t *)((uint8_t *)pAddress + 17641242624);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 17645436928);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 17649631232);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 17653825536);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 17658019840);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 17662214144);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 17666408448);
        hashK = (uint8_t *)((uint8_t *)pAddress + 17670602752);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 17674797056);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 17678991360);
        hashP = (uint8_t *)((uint8_t *)pAddress + 17683185664);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 17687379968);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 17691574272);
        bin = (uint8_t *)((uint8_t *)pAddress + 17695768576);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 17699962880);
        assert = (uint8_t *)((uint8_t *)pAddress + 17704157184);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 17708351488);
        isNeg = (uint8_t *)((uint8_t *)pAddress + 17712545792);
        isMaxMem = (uint8_t *)((uint8_t *)pAddress + 17716740096);
        sKeyI[0] = (FieldElement *)((uint8_t *)pAddress + 17720934400);
        sKeyI[1] = (FieldElement *)((uint8_t *)pAddress + 17754488832);
        sKeyI[2] = (FieldElement *)((uint8_t *)pAddress + 17788043264);
        sKeyI[3] = (FieldElement *)((uint8_t *)pAddress + 17821597696);
        sKey[0] = (FieldElement *)((uint8_t *)pAddress + 17855152128);
        sKey[1] = (FieldElement *)((uint8_t *)pAddress + 17888706560);
        sKey[2] = (FieldElement *)((uint8_t *)pAddress + 17922260992);
        sKey[3] = (FieldElement *)((uint8_t *)pAddress + 17955815424);
    }

    static uint64_t degree (void) { return 4194304; }
};

class CommitPols
{
public:
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
    MemAlignCommitPols MemAlign;
    MainCommitPols Main;

    CommitPols (void * pAddress) : Byte4(pAddress), Arith(pAddress), Binary(pAddress), Mem(pAddress), PoseidonG(pAddress), Storage(pAddress), NormGate9(pAddress), KeccakF(pAddress), Nine2One(pAddress), PaddingKKBit(pAddress), PaddingKK(pAddress), PaddingPG(pAddress), MemAlign(pAddress), Main(pAddress) {}

    static uint64_t size (void) { return 17989369856; }
};

#endif // COMMIT_POLS_HPP
