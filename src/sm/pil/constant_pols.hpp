#ifndef CONSTANT_POLS_HPP
#define CONSTANT_POLS_HPP

#include <cstdint>
#include "ff/ff.hpp"

class GlobalConstantPols
{
public:
    FieldElement * ZH;
    FieldElement * ZHINV;
    uint8_t * L1;
    FieldElement * BYTE;
    FieldElement * BYTE2;

    GlobalConstantPols (void * pAddress)
    {
        ZH = (FieldElement *)((uint8_t *)pAddress + 0);
        ZHINV = (FieldElement *)((uint8_t *)pAddress + 33554432);
        L1 = (uint8_t *)((uint8_t *)pAddress + 67108864);
        BYTE = (FieldElement *)((uint8_t *)pAddress + 71303168);
        BYTE2 = (FieldElement *)((uint8_t *)pAddress + 104857600);
    }

    GlobalConstantPols (void * pAddress, uint64_t degree)
    {
        ZH = (FieldElement *)((uint8_t *)pAddress + 0);
        ZHINV = (FieldElement *)((uint8_t *)pAddress + 16);
        L1 = (uint8_t *)((uint8_t *)pAddress + 32);
        BYTE = (FieldElement *)((uint8_t *)pAddress + 34);
        BYTE2 = (FieldElement *)((uint8_t *)pAddress + 50);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 66; }
};

class RomConstantPols
{
public:
    FieldElement * CONST0;
    uint32_t * CONST1;
    uint32_t * CONST2;
    uint32_t * CONST3;
    uint32_t * CONST4;
    uint32_t * CONST5;
    uint32_t * CONST6;
    uint32_t * CONST7;
    uint32_t * offset;
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
    FieldElement * inHASHPOS;
    FieldElement * inSTEP;
    FieldElement * inRR;
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
    uint8_t * setHASHPOS;
    uint8_t * JMP;
    uint8_t * JMPC;
    uint8_t * setRR;
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
    uint32_t * line;
    uint8_t * opCodeNum;
    uint32_t * opCodeAddr;

    RomConstantPols (void * pAddress)
    {
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 138412032);
        CONST1 = (uint32_t *)((uint8_t *)pAddress + 171966464);
        CONST2 = (uint32_t *)((uint8_t *)pAddress + 205520896);
        CONST3 = (uint32_t *)((uint8_t *)pAddress + 239075328);
        CONST4 = (uint32_t *)((uint8_t *)pAddress + 272629760);
        CONST5 = (uint32_t *)((uint8_t *)pAddress + 306184192);
        CONST6 = (uint32_t *)((uint8_t *)pAddress + 339738624);
        CONST7 = (uint32_t *)((uint8_t *)pAddress + 373293056);
        offset = (uint32_t *)((uint8_t *)pAddress + 406847488);
        inA = (FieldElement *)((uint8_t *)pAddress + 440401920);
        inB = (FieldElement *)((uint8_t *)pAddress + 473956352);
        inC = (FieldElement *)((uint8_t *)pAddress + 507510784);
        inD = (FieldElement *)((uint8_t *)pAddress + 541065216);
        inE = (FieldElement *)((uint8_t *)pAddress + 574619648);
        inSR = (FieldElement *)((uint8_t *)pAddress + 608174080);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 641728512);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 675282944);
        inSP = (FieldElement *)((uint8_t *)pAddress + 708837376);
        inPC = (FieldElement *)((uint8_t *)pAddress + 742391808);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 775946240);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 809500672);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 843055104);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 876609536);
        inRR = (FieldElement *)((uint8_t *)pAddress + 910163968);
        setA = (uint8_t *)((uint8_t *)pAddress + 943718400);
        setB = (uint8_t *)((uint8_t *)pAddress + 947912704);
        setC = (uint8_t *)((uint8_t *)pAddress + 952107008);
        setD = (uint8_t *)((uint8_t *)pAddress + 956301312);
        setE = (uint8_t *)((uint8_t *)pAddress + 960495616);
        setSR = (uint8_t *)((uint8_t *)pAddress + 964689920);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 968884224);
        setSP = (uint8_t *)((uint8_t *)pAddress + 973078528);
        setPC = (uint8_t *)((uint8_t *)pAddress + 977272832);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 981467136);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 985661440);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 989855744);
        JMP = (uint8_t *)((uint8_t *)pAddress + 994050048);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 998244352);
        setRR = (uint8_t *)((uint8_t *)pAddress + 1002438656);
        incStack = (int32_t *)((uint8_t *)pAddress + 1006632960);
        incCode = (int32_t *)((uint8_t *)pAddress + 1023410176);
        isStack = (uint8_t *)((uint8_t *)pAddress + 1040187392);
        isCode = (uint8_t *)((uint8_t *)pAddress + 1044381696);
        isMem = (uint8_t *)((uint8_t *)pAddress + 1048576000);
        ind = (uint8_t *)((uint8_t *)pAddress + 1052770304);
        indRR = (uint8_t *)((uint8_t *)pAddress + 1056964608);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 1061158912);
        mOp = (uint8_t *)((uint8_t *)pAddress + 1065353216);
        mWR = (uint8_t *)((uint8_t *)pAddress + 1069547520);
        sWR = (uint8_t *)((uint8_t *)pAddress + 1073741824);
        sRD = (uint8_t *)((uint8_t *)pAddress + 1077936128);
        arith = (uint8_t *)((uint8_t *)pAddress + 1082130432);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 1086324736);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 1090519040);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 1094713344);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 1098907648);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 1103101952);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 1107296256);
        hashK = (uint8_t *)((uint8_t *)pAddress + 1111490560);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 1115684864);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 1119879168);
        hashP = (uint8_t *)((uint8_t *)pAddress + 1124073472);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 1128267776);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 1132462080);
        bin = (uint8_t *)((uint8_t *)pAddress + 1136656384);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 1140850688);
        assert = (uint8_t *)((uint8_t *)pAddress + 1145044992);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 1149239296);
        line = (uint32_t *)((uint8_t *)pAddress + 1153433600);
        opCodeNum = (uint8_t *)((uint8_t *)pAddress + 1186988032);
        opCodeAddr = (uint32_t *)((uint8_t *)pAddress + 1191182336);
    }

    RomConstantPols (void * pAddress, uint64_t degree)
    {
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 0);
        CONST1 = (uint32_t *)((uint8_t *)pAddress + 16);
        CONST2 = (uint32_t *)((uint8_t *)pAddress + 32);
        CONST3 = (uint32_t *)((uint8_t *)pAddress + 48);
        CONST4 = (uint32_t *)((uint8_t *)pAddress + 64);
        CONST5 = (uint32_t *)((uint8_t *)pAddress + 80);
        CONST6 = (uint32_t *)((uint8_t *)pAddress + 96);
        CONST7 = (uint32_t *)((uint8_t *)pAddress + 112);
        offset = (uint32_t *)((uint8_t *)pAddress + 128);
        inA = (FieldElement *)((uint8_t *)pAddress + 144);
        inB = (FieldElement *)((uint8_t *)pAddress + 160);
        inC = (FieldElement *)((uint8_t *)pAddress + 176);
        inD = (FieldElement *)((uint8_t *)pAddress + 192);
        inE = (FieldElement *)((uint8_t *)pAddress + 208);
        inSR = (FieldElement *)((uint8_t *)pAddress + 224);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 240);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 256);
        inSP = (FieldElement *)((uint8_t *)pAddress + 272);
        inPC = (FieldElement *)((uint8_t *)pAddress + 288);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 304);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 320);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 336);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 352);
        inRR = (FieldElement *)((uint8_t *)pAddress + 368);
        setA = (uint8_t *)((uint8_t *)pAddress + 384);
        setB = (uint8_t *)((uint8_t *)pAddress + 386);
        setC = (uint8_t *)((uint8_t *)pAddress + 388);
        setD = (uint8_t *)((uint8_t *)pAddress + 390);
        setE = (uint8_t *)((uint8_t *)pAddress + 392);
        setSR = (uint8_t *)((uint8_t *)pAddress + 394);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 396);
        setSP = (uint8_t *)((uint8_t *)pAddress + 398);
        setPC = (uint8_t *)((uint8_t *)pAddress + 400);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 402);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 404);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 406);
        JMP = (uint8_t *)((uint8_t *)pAddress + 408);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 410);
        setRR = (uint8_t *)((uint8_t *)pAddress + 412);
        incStack = (int32_t *)((uint8_t *)pAddress + 414);
        incCode = (int32_t *)((uint8_t *)pAddress + 422);
        isStack = (uint8_t *)((uint8_t *)pAddress + 430);
        isCode = (uint8_t *)((uint8_t *)pAddress + 432);
        isMem = (uint8_t *)((uint8_t *)pAddress + 434);
        ind = (uint8_t *)((uint8_t *)pAddress + 436);
        indRR = (uint8_t *)((uint8_t *)pAddress + 438);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 440);
        mOp = (uint8_t *)((uint8_t *)pAddress + 442);
        mWR = (uint8_t *)((uint8_t *)pAddress + 444);
        sWR = (uint8_t *)((uint8_t *)pAddress + 446);
        sRD = (uint8_t *)((uint8_t *)pAddress + 448);
        arith = (uint8_t *)((uint8_t *)pAddress + 450);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 452);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 454);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 456);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 458);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 460);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 462);
        hashK = (uint8_t *)((uint8_t *)pAddress + 464);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 466);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 468);
        hashP = (uint8_t *)((uint8_t *)pAddress + 470);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 472);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 474);
        bin = (uint8_t *)((uint8_t *)pAddress + 476);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 478);
        assert = (uint8_t *)((uint8_t *)pAddress + 480);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 482);
        line = (uint32_t *)((uint8_t *)pAddress + 484);
        opCodeNum = (uint8_t *)((uint8_t *)pAddress + 500);
        opCodeAddr = (uint32_t *)((uint8_t *)pAddress + 502);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 518; }
};

class Byte4ConstantPols
{
public:
    uint8_t * SET;

    Byte4ConstantPols (void * pAddress)
    {
        SET = (uint8_t *)((uint8_t *)pAddress + 1224736768);
    }

    Byte4ConstantPols (void * pAddress, uint64_t degree)
    {
        SET = (uint8_t *)((uint8_t *)pAddress + 0);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 2; }
};

class MemAlignConstantPols
{
public:
    uint8_t * BYTE2A;
    uint8_t * BYTE2B;
    FieldElement * FACTOR0[8];
    FieldElement * FACTOR1[8];
    FieldElement * FACTORV[8];
    FieldElement * STEP;
    uint8_t * WR;
    uint8_t * OFFSET;
    uint8_t * RESET;
    uint8_t * RESETL;
    uint8_t * SELW;

    MemAlignConstantPols (void * pAddress)
    {
        BYTE2A = (uint8_t *)((uint8_t *)pAddress + 1228931072);
        BYTE2B = (uint8_t *)((uint8_t *)pAddress + 1233125376);
        FACTOR0[0] = (FieldElement *)((uint8_t *)pAddress + 1237319680);
        FACTOR0[1] = (FieldElement *)((uint8_t *)pAddress + 1270874112);
        FACTOR0[2] = (FieldElement *)((uint8_t *)pAddress + 1304428544);
        FACTOR0[3] = (FieldElement *)((uint8_t *)pAddress + 1337982976);
        FACTOR0[4] = (FieldElement *)((uint8_t *)pAddress + 1371537408);
        FACTOR0[5] = (FieldElement *)((uint8_t *)pAddress + 1405091840);
        FACTOR0[6] = (FieldElement *)((uint8_t *)pAddress + 1438646272);
        FACTOR0[7] = (FieldElement *)((uint8_t *)pAddress + 1472200704);
        FACTOR1[0] = (FieldElement *)((uint8_t *)pAddress + 1505755136);
        FACTOR1[1] = (FieldElement *)((uint8_t *)pAddress + 1539309568);
        FACTOR1[2] = (FieldElement *)((uint8_t *)pAddress + 1572864000);
        FACTOR1[3] = (FieldElement *)((uint8_t *)pAddress + 1606418432);
        FACTOR1[4] = (FieldElement *)((uint8_t *)pAddress + 1639972864);
        FACTOR1[5] = (FieldElement *)((uint8_t *)pAddress + 1673527296);
        FACTOR1[6] = (FieldElement *)((uint8_t *)pAddress + 1707081728);
        FACTOR1[7] = (FieldElement *)((uint8_t *)pAddress + 1740636160);
        FACTORV[0] = (FieldElement *)((uint8_t *)pAddress + 1774190592);
        FACTORV[1] = (FieldElement *)((uint8_t *)pAddress + 1807745024);
        FACTORV[2] = (FieldElement *)((uint8_t *)pAddress + 1841299456);
        FACTORV[3] = (FieldElement *)((uint8_t *)pAddress + 1874853888);
        FACTORV[4] = (FieldElement *)((uint8_t *)pAddress + 1908408320);
        FACTORV[5] = (FieldElement *)((uint8_t *)pAddress + 1941962752);
        FACTORV[6] = (FieldElement *)((uint8_t *)pAddress + 1975517184);
        FACTORV[7] = (FieldElement *)((uint8_t *)pAddress + 2009071616);
        STEP = (FieldElement *)((uint8_t *)pAddress + 2042626048);
        WR = (uint8_t *)((uint8_t *)pAddress + 2076180480);
        OFFSET = (uint8_t *)((uint8_t *)pAddress + 2080374784);
        RESET = (uint8_t *)((uint8_t *)pAddress + 2084569088);
        RESETL = (uint8_t *)((uint8_t *)pAddress + 2088763392);
        SELW = (uint8_t *)((uint8_t *)pAddress + 2092957696);
    }

    MemAlignConstantPols (void * pAddress, uint64_t degree)
    {
        BYTE2A = (uint8_t *)((uint8_t *)pAddress + 0);
        BYTE2B = (uint8_t *)((uint8_t *)pAddress + 2);
        FACTOR0[0] = (FieldElement *)((uint8_t *)pAddress + 4);
        FACTOR0[1] = (FieldElement *)((uint8_t *)pAddress + 20);
        FACTOR0[2] = (FieldElement *)((uint8_t *)pAddress + 36);
        FACTOR0[3] = (FieldElement *)((uint8_t *)pAddress + 52);
        FACTOR0[4] = (FieldElement *)((uint8_t *)pAddress + 68);
        FACTOR0[5] = (FieldElement *)((uint8_t *)pAddress + 84);
        FACTOR0[6] = (FieldElement *)((uint8_t *)pAddress + 100);
        FACTOR0[7] = (FieldElement *)((uint8_t *)pAddress + 116);
        FACTOR1[0] = (FieldElement *)((uint8_t *)pAddress + 132);
        FACTOR1[1] = (FieldElement *)((uint8_t *)pAddress + 148);
        FACTOR1[2] = (FieldElement *)((uint8_t *)pAddress + 164);
        FACTOR1[3] = (FieldElement *)((uint8_t *)pAddress + 180);
        FACTOR1[4] = (FieldElement *)((uint8_t *)pAddress + 196);
        FACTOR1[5] = (FieldElement *)((uint8_t *)pAddress + 212);
        FACTOR1[6] = (FieldElement *)((uint8_t *)pAddress + 228);
        FACTOR1[7] = (FieldElement *)((uint8_t *)pAddress + 244);
        FACTORV[0] = (FieldElement *)((uint8_t *)pAddress + 260);
        FACTORV[1] = (FieldElement *)((uint8_t *)pAddress + 276);
        FACTORV[2] = (FieldElement *)((uint8_t *)pAddress + 292);
        FACTORV[3] = (FieldElement *)((uint8_t *)pAddress + 308);
        FACTORV[4] = (FieldElement *)((uint8_t *)pAddress + 324);
        FACTORV[5] = (FieldElement *)((uint8_t *)pAddress + 340);
        FACTORV[6] = (FieldElement *)((uint8_t *)pAddress + 356);
        FACTORV[7] = (FieldElement *)((uint8_t *)pAddress + 372);
        STEP = (FieldElement *)((uint8_t *)pAddress + 388);
        WR = (uint8_t *)((uint8_t *)pAddress + 404);
        OFFSET = (uint8_t *)((uint8_t *)pAddress + 406);
        RESET = (uint8_t *)((uint8_t *)pAddress + 408);
        RESETL = (uint8_t *)((uint8_t *)pAddress + 410);
        SELW = (uint8_t *)((uint8_t *)pAddress + 412);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 414; }
};

class ArithConstantPols
{
public:
    FieldElement * BIT19;
    FieldElement * GL_SIGNED_4BITS;
    FieldElement * GL_SIGNED_18BITS;
    uint8_t * ck[32];

    ArithConstantPols (void * pAddress)
    {
        BIT19 = (FieldElement *)((uint8_t *)pAddress + 2097152000);
        GL_SIGNED_4BITS = (FieldElement *)((uint8_t *)pAddress + 2130706432);
        GL_SIGNED_18BITS = (FieldElement *)((uint8_t *)pAddress + 2164260864);
        ck[0] = (uint8_t *)((uint8_t *)pAddress + 2197815296);
        ck[1] = (uint8_t *)((uint8_t *)pAddress + 2202009600);
        ck[2] = (uint8_t *)((uint8_t *)pAddress + 2206203904);
        ck[3] = (uint8_t *)((uint8_t *)pAddress + 2210398208);
        ck[4] = (uint8_t *)((uint8_t *)pAddress + 2214592512);
        ck[5] = (uint8_t *)((uint8_t *)pAddress + 2218786816);
        ck[6] = (uint8_t *)((uint8_t *)pAddress + 2222981120);
        ck[7] = (uint8_t *)((uint8_t *)pAddress + 2227175424);
        ck[8] = (uint8_t *)((uint8_t *)pAddress + 2231369728);
        ck[9] = (uint8_t *)((uint8_t *)pAddress + 2235564032);
        ck[10] = (uint8_t *)((uint8_t *)pAddress + 2239758336);
        ck[11] = (uint8_t *)((uint8_t *)pAddress + 2243952640);
        ck[12] = (uint8_t *)((uint8_t *)pAddress + 2248146944);
        ck[13] = (uint8_t *)((uint8_t *)pAddress + 2252341248);
        ck[14] = (uint8_t *)((uint8_t *)pAddress + 2256535552);
        ck[15] = (uint8_t *)((uint8_t *)pAddress + 2260729856);
        ck[16] = (uint8_t *)((uint8_t *)pAddress + 2264924160);
        ck[17] = (uint8_t *)((uint8_t *)pAddress + 2269118464);
        ck[18] = (uint8_t *)((uint8_t *)pAddress + 2273312768);
        ck[19] = (uint8_t *)((uint8_t *)pAddress + 2277507072);
        ck[20] = (uint8_t *)((uint8_t *)pAddress + 2281701376);
        ck[21] = (uint8_t *)((uint8_t *)pAddress + 2285895680);
        ck[22] = (uint8_t *)((uint8_t *)pAddress + 2290089984);
        ck[23] = (uint8_t *)((uint8_t *)pAddress + 2294284288);
        ck[24] = (uint8_t *)((uint8_t *)pAddress + 2298478592);
        ck[25] = (uint8_t *)((uint8_t *)pAddress + 2302672896);
        ck[26] = (uint8_t *)((uint8_t *)pAddress + 2306867200);
        ck[27] = (uint8_t *)((uint8_t *)pAddress + 2311061504);
        ck[28] = (uint8_t *)((uint8_t *)pAddress + 2315255808);
        ck[29] = (uint8_t *)((uint8_t *)pAddress + 2319450112);
        ck[30] = (uint8_t *)((uint8_t *)pAddress + 2323644416);
        ck[31] = (uint8_t *)((uint8_t *)pAddress + 2327838720);
    }

    ArithConstantPols (void * pAddress, uint64_t degree)
    {
        BIT19 = (FieldElement *)((uint8_t *)pAddress + 0);
        GL_SIGNED_4BITS = (FieldElement *)((uint8_t *)pAddress + 16);
        GL_SIGNED_18BITS = (FieldElement *)((uint8_t *)pAddress + 32);
        ck[0] = (uint8_t *)((uint8_t *)pAddress + 48);
        ck[1] = (uint8_t *)((uint8_t *)pAddress + 50);
        ck[2] = (uint8_t *)((uint8_t *)pAddress + 52);
        ck[3] = (uint8_t *)((uint8_t *)pAddress + 54);
        ck[4] = (uint8_t *)((uint8_t *)pAddress + 56);
        ck[5] = (uint8_t *)((uint8_t *)pAddress + 58);
        ck[6] = (uint8_t *)((uint8_t *)pAddress + 60);
        ck[7] = (uint8_t *)((uint8_t *)pAddress + 62);
        ck[8] = (uint8_t *)((uint8_t *)pAddress + 64);
        ck[9] = (uint8_t *)((uint8_t *)pAddress + 66);
        ck[10] = (uint8_t *)((uint8_t *)pAddress + 68);
        ck[11] = (uint8_t *)((uint8_t *)pAddress + 70);
        ck[12] = (uint8_t *)((uint8_t *)pAddress + 72);
        ck[13] = (uint8_t *)((uint8_t *)pAddress + 74);
        ck[14] = (uint8_t *)((uint8_t *)pAddress + 76);
        ck[15] = (uint8_t *)((uint8_t *)pAddress + 78);
        ck[16] = (uint8_t *)((uint8_t *)pAddress + 80);
        ck[17] = (uint8_t *)((uint8_t *)pAddress + 82);
        ck[18] = (uint8_t *)((uint8_t *)pAddress + 84);
        ck[19] = (uint8_t *)((uint8_t *)pAddress + 86);
        ck[20] = (uint8_t *)((uint8_t *)pAddress + 88);
        ck[21] = (uint8_t *)((uint8_t *)pAddress + 90);
        ck[22] = (uint8_t *)((uint8_t *)pAddress + 92);
        ck[23] = (uint8_t *)((uint8_t *)pAddress + 94);
        ck[24] = (uint8_t *)((uint8_t *)pAddress + 96);
        ck[25] = (uint8_t *)((uint8_t *)pAddress + 98);
        ck[26] = (uint8_t *)((uint8_t *)pAddress + 100);
        ck[27] = (uint8_t *)((uint8_t *)pAddress + 102);
        ck[28] = (uint8_t *)((uint8_t *)pAddress + 104);
        ck[29] = (uint8_t *)((uint8_t *)pAddress + 106);
        ck[30] = (uint8_t *)((uint8_t *)pAddress + 108);
        ck[31] = (uint8_t *)((uint8_t *)pAddress + 110);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 112; }
};

class BinaryConstantPols
{
public:
    uint8_t * P_OPCODE;
    uint8_t * P_A;
    uint8_t * P_B;
    uint8_t * P_CIN;
    uint8_t * P_LAST;
    uint8_t * P_C;
    uint8_t * P_COUT;
    uint8_t * RESET;
    uint32_t * FACTOR[8];

    BinaryConstantPols (void * pAddress)
    {
        P_OPCODE = (uint8_t *)((uint8_t *)pAddress + 2332033024);
        P_A = (uint8_t *)((uint8_t *)pAddress + 2336227328);
        P_B = (uint8_t *)((uint8_t *)pAddress + 2340421632);
        P_CIN = (uint8_t *)((uint8_t *)pAddress + 2344615936);
        P_LAST = (uint8_t *)((uint8_t *)pAddress + 2348810240);
        P_C = (uint8_t *)((uint8_t *)pAddress + 2353004544);
        P_COUT = (uint8_t *)((uint8_t *)pAddress + 2357198848);
        RESET = (uint8_t *)((uint8_t *)pAddress + 2361393152);
        FACTOR[0] = (uint32_t *)((uint8_t *)pAddress + 2365587456);
        FACTOR[1] = (uint32_t *)((uint8_t *)pAddress + 2399141888);
        FACTOR[2] = (uint32_t *)((uint8_t *)pAddress + 2432696320);
        FACTOR[3] = (uint32_t *)((uint8_t *)pAddress + 2466250752);
        FACTOR[4] = (uint32_t *)((uint8_t *)pAddress + 2499805184);
        FACTOR[5] = (uint32_t *)((uint8_t *)pAddress + 2533359616);
        FACTOR[6] = (uint32_t *)((uint8_t *)pAddress + 2566914048);
        FACTOR[7] = (uint32_t *)((uint8_t *)pAddress + 2600468480);
    }

    BinaryConstantPols (void * pAddress, uint64_t degree)
    {
        P_OPCODE = (uint8_t *)((uint8_t *)pAddress + 0);
        P_A = (uint8_t *)((uint8_t *)pAddress + 2);
        P_B = (uint8_t *)((uint8_t *)pAddress + 4);
        P_CIN = (uint8_t *)((uint8_t *)pAddress + 6);
        P_LAST = (uint8_t *)((uint8_t *)pAddress + 8);
        P_C = (uint8_t *)((uint8_t *)pAddress + 10);
        P_COUT = (uint8_t *)((uint8_t *)pAddress + 12);
        RESET = (uint8_t *)((uint8_t *)pAddress + 14);
        FACTOR[0] = (uint32_t *)((uint8_t *)pAddress + 16);
        FACTOR[1] = (uint32_t *)((uint8_t *)pAddress + 32);
        FACTOR[2] = (uint32_t *)((uint8_t *)pAddress + 48);
        FACTOR[3] = (uint32_t *)((uint8_t *)pAddress + 64);
        FACTOR[4] = (uint32_t *)((uint8_t *)pAddress + 80);
        FACTOR[5] = (uint32_t *)((uint8_t *)pAddress + 96);
        FACTOR[6] = (uint32_t *)((uint8_t *)pAddress + 112);
        FACTOR[7] = (uint32_t *)((uint8_t *)pAddress + 128);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 144; }
};

class PoseidonGConstantPols
{
public:
    FieldElement * LAST;
    FieldElement * LATCH;
    FieldElement * LASTBLOCK;
    FieldElement * PARTIAL;
    FieldElement * C[12];

    PoseidonGConstantPols (void * pAddress)
    {
        LAST = (FieldElement *)((uint8_t *)pAddress + 2634022912);
        LATCH = (FieldElement *)((uint8_t *)pAddress + 2667577344);
        LASTBLOCK = (FieldElement *)((uint8_t *)pAddress + 2701131776);
        PARTIAL = (FieldElement *)((uint8_t *)pAddress + 2734686208);
        C[0] = (FieldElement *)((uint8_t *)pAddress + 2768240640);
        C[1] = (FieldElement *)((uint8_t *)pAddress + 2801795072);
        C[2] = (FieldElement *)((uint8_t *)pAddress + 2835349504);
        C[3] = (FieldElement *)((uint8_t *)pAddress + 2868903936);
        C[4] = (FieldElement *)((uint8_t *)pAddress + 2902458368);
        C[5] = (FieldElement *)((uint8_t *)pAddress + 2936012800);
        C[6] = (FieldElement *)((uint8_t *)pAddress + 2969567232);
        C[7] = (FieldElement *)((uint8_t *)pAddress + 3003121664);
        C[8] = (FieldElement *)((uint8_t *)pAddress + 3036676096);
        C[9] = (FieldElement *)((uint8_t *)pAddress + 3070230528);
        C[10] = (FieldElement *)((uint8_t *)pAddress + 3103784960);
        C[11] = (FieldElement *)((uint8_t *)pAddress + 3137339392);
    }

    PoseidonGConstantPols (void * pAddress, uint64_t degree)
    {
        LAST = (FieldElement *)((uint8_t *)pAddress + 0);
        LATCH = (FieldElement *)((uint8_t *)pAddress + 16);
        LASTBLOCK = (FieldElement *)((uint8_t *)pAddress + 32);
        PARTIAL = (FieldElement *)((uint8_t *)pAddress + 48);
        C[0] = (FieldElement *)((uint8_t *)pAddress + 64);
        C[1] = (FieldElement *)((uint8_t *)pAddress + 80);
        C[2] = (FieldElement *)((uint8_t *)pAddress + 96);
        C[3] = (FieldElement *)((uint8_t *)pAddress + 112);
        C[4] = (FieldElement *)((uint8_t *)pAddress + 128);
        C[5] = (FieldElement *)((uint8_t *)pAddress + 144);
        C[6] = (FieldElement *)((uint8_t *)pAddress + 160);
        C[7] = (FieldElement *)((uint8_t *)pAddress + 176);
        C[8] = (FieldElement *)((uint8_t *)pAddress + 192);
        C[9] = (FieldElement *)((uint8_t *)pAddress + 208);
        C[10] = (FieldElement *)((uint8_t *)pAddress + 224);
        C[11] = (FieldElement *)((uint8_t *)pAddress + 240);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 256; }
};

class PaddingPGConstantPols
{
public:
    FieldElement * F[8];
    FieldElement * lastBlock;
    FieldElement * k_crOffset;
    FieldElement * k_crF0;
    FieldElement * k_crF1;
    FieldElement * k_crF2;
    FieldElement * k_crF3;
    FieldElement * k_crF4;
    FieldElement * k_crF5;
    FieldElement * k_crF6;
    FieldElement * k_crF7;

    PaddingPGConstantPols (void * pAddress)
    {
        F[0] = (FieldElement *)((uint8_t *)pAddress + 3170893824);
        F[1] = (FieldElement *)((uint8_t *)pAddress + 3204448256);
        F[2] = (FieldElement *)((uint8_t *)pAddress + 3238002688);
        F[3] = (FieldElement *)((uint8_t *)pAddress + 3271557120);
        F[4] = (FieldElement *)((uint8_t *)pAddress + 3305111552);
        F[5] = (FieldElement *)((uint8_t *)pAddress + 3338665984);
        F[6] = (FieldElement *)((uint8_t *)pAddress + 3372220416);
        F[7] = (FieldElement *)((uint8_t *)pAddress + 3405774848);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 3439329280);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 3472883712);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 3506438144);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 3539992576);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 3573547008);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 3607101440);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 3640655872);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 3674210304);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 3707764736);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 3741319168);
    }

    PaddingPGConstantPols (void * pAddress, uint64_t degree)
    {
        F[0] = (FieldElement *)((uint8_t *)pAddress + 0);
        F[1] = (FieldElement *)((uint8_t *)pAddress + 16);
        F[2] = (FieldElement *)((uint8_t *)pAddress + 32);
        F[3] = (FieldElement *)((uint8_t *)pAddress + 48);
        F[4] = (FieldElement *)((uint8_t *)pAddress + 64);
        F[5] = (FieldElement *)((uint8_t *)pAddress + 80);
        F[6] = (FieldElement *)((uint8_t *)pAddress + 96);
        F[7] = (FieldElement *)((uint8_t *)pAddress + 112);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 128);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 144);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 160);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 176);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 192);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 208);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 224);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 240);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 256);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 272);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 288; }
};

class StorageConstantPols
{
public:
    FieldElement * rHash;
    FieldElement * rHashType;
    FieldElement * rLatchGet;
    FieldElement * rLatchSet;
    FieldElement * rClimbRkey;
    FieldElement * rClimbSiblingRkey;
    FieldElement * rClimbSiblingRkeyN;
    FieldElement * rRotateLevel;
    FieldElement * rJmpz;
    FieldElement * rJmp;
    FieldElement * rConst0;
    FieldElement * rConst1;
    FieldElement * rConst2;
    FieldElement * rConst3;
    FieldElement * rAddress;
    FieldElement * rLine;

    StorageConstantPols (void * pAddress)
    {
        rHash = (FieldElement *)((uint8_t *)pAddress + 3774873600);
        rHashType = (FieldElement *)((uint8_t *)pAddress + 3808428032);
        rLatchGet = (FieldElement *)((uint8_t *)pAddress + 3841982464);
        rLatchSet = (FieldElement *)((uint8_t *)pAddress + 3875536896);
        rClimbRkey = (FieldElement *)((uint8_t *)pAddress + 3909091328);
        rClimbSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 3942645760);
        rClimbSiblingRkeyN = (FieldElement *)((uint8_t *)pAddress + 3976200192);
        rRotateLevel = (FieldElement *)((uint8_t *)pAddress + 4009754624);
        rJmpz = (FieldElement *)((uint8_t *)pAddress + 4043309056);
        rJmp = (FieldElement *)((uint8_t *)pAddress + 4076863488);
        rConst0 = (FieldElement *)((uint8_t *)pAddress + 4110417920);
        rConst1 = (FieldElement *)((uint8_t *)pAddress + 4143972352);
        rConst2 = (FieldElement *)((uint8_t *)pAddress + 4177526784);
        rConst3 = (FieldElement *)((uint8_t *)pAddress + 4211081216);
        rAddress = (FieldElement *)((uint8_t *)pAddress + 4244635648);
        rLine = (FieldElement *)((uint8_t *)pAddress + 4278190080);
    }

    StorageConstantPols (void * pAddress, uint64_t degree)
    {
        rHash = (FieldElement *)((uint8_t *)pAddress + 0);
        rHashType = (FieldElement *)((uint8_t *)pAddress + 16);
        rLatchGet = (FieldElement *)((uint8_t *)pAddress + 32);
        rLatchSet = (FieldElement *)((uint8_t *)pAddress + 48);
        rClimbRkey = (FieldElement *)((uint8_t *)pAddress + 64);
        rClimbSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 80);
        rClimbSiblingRkeyN = (FieldElement *)((uint8_t *)pAddress + 96);
        rRotateLevel = (FieldElement *)((uint8_t *)pAddress + 112);
        rJmpz = (FieldElement *)((uint8_t *)pAddress + 128);
        rJmp = (FieldElement *)((uint8_t *)pAddress + 144);
        rConst0 = (FieldElement *)((uint8_t *)pAddress + 160);
        rConst1 = (FieldElement *)((uint8_t *)pAddress + 176);
        rConst2 = (FieldElement *)((uint8_t *)pAddress + 192);
        rConst3 = (FieldElement *)((uint8_t *)pAddress + 208);
        rAddress = (FieldElement *)((uint8_t *)pAddress + 224);
        rLine = (FieldElement *)((uint8_t *)pAddress + 240);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 256; }
};

class NormGate9ConstantPols
{
public:
    FieldElement * Value3;
    FieldElement * Value3Norm;
    FieldElement * Gate9Type;
    FieldElement * Gate9A;
    FieldElement * Gate9B;
    FieldElement * Gate9C;
    FieldElement * Latch;
    FieldElement * Factor;

    NormGate9ConstantPols (void * pAddress)
    {
        Value3 = (FieldElement *)((uint8_t *)pAddress + 4311744512);
        Value3Norm = (FieldElement *)((uint8_t *)pAddress + 4345298944);
        Gate9Type = (FieldElement *)((uint8_t *)pAddress + 4378853376);
        Gate9A = (FieldElement *)((uint8_t *)pAddress + 4412407808);
        Gate9B = (FieldElement *)((uint8_t *)pAddress + 4445962240);
        Gate9C = (FieldElement *)((uint8_t *)pAddress + 4479516672);
        Latch = (FieldElement *)((uint8_t *)pAddress + 4513071104);
        Factor = (FieldElement *)((uint8_t *)pAddress + 4546625536);
    }

    NormGate9ConstantPols (void * pAddress, uint64_t degree)
    {
        Value3 = (FieldElement *)((uint8_t *)pAddress + 0);
        Value3Norm = (FieldElement *)((uint8_t *)pAddress + 16);
        Gate9Type = (FieldElement *)((uint8_t *)pAddress + 32);
        Gate9A = (FieldElement *)((uint8_t *)pAddress + 48);
        Gate9B = (FieldElement *)((uint8_t *)pAddress + 64);
        Gate9C = (FieldElement *)((uint8_t *)pAddress + 80);
        Latch = (FieldElement *)((uint8_t *)pAddress + 96);
        Factor = (FieldElement *)((uint8_t *)pAddress + 112);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 128; }
};

class KeccakFConstantPols
{
public:
    FieldElement * ConnA;
    FieldElement * ConnB;
    FieldElement * ConnC;
    FieldElement * NormalizedGate;
    FieldElement * GateType;

    KeccakFConstantPols (void * pAddress)
    {
        ConnA = (FieldElement *)((uint8_t *)pAddress + 4580179968);
        ConnB = (FieldElement *)((uint8_t *)pAddress + 4613734400);
        ConnC = (FieldElement *)((uint8_t *)pAddress + 4647288832);
        NormalizedGate = (FieldElement *)((uint8_t *)pAddress + 4680843264);
        GateType = (FieldElement *)((uint8_t *)pAddress + 4714397696);
    }

    KeccakFConstantPols (void * pAddress, uint64_t degree)
    {
        ConnA = (FieldElement *)((uint8_t *)pAddress + 0);
        ConnB = (FieldElement *)((uint8_t *)pAddress + 16);
        ConnC = (FieldElement *)((uint8_t *)pAddress + 32);
        NormalizedGate = (FieldElement *)((uint8_t *)pAddress + 48);
        GateType = (FieldElement *)((uint8_t *)pAddress + 64);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 80; }
};

class Nine2OneConstantPols
{
public:
    FieldElement * Field9latch;
    FieldElement * Factor;

    Nine2OneConstantPols (void * pAddress)
    {
        Field9latch = (FieldElement *)((uint8_t *)pAddress + 4747952128);
        Factor = (FieldElement *)((uint8_t *)pAddress + 4781506560);
    }

    Nine2OneConstantPols (void * pAddress, uint64_t degree)
    {
        Field9latch = (FieldElement *)((uint8_t *)pAddress + 0);
        Factor = (FieldElement *)((uint8_t *)pAddress + 16);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 32; }
};

class PaddingKKBitConstantPols
{
public:
    FieldElement * r8Id;
    FieldElement * sOutId;
    FieldElement * latchR8;
    FieldElement * Fr8;
    FieldElement * rBitValid;
    FieldElement * latchSOut;
    FieldElement * FSOut0;
    FieldElement * FSOut1;
    FieldElement * FSOut2;
    FieldElement * FSOut3;
    FieldElement * FSOut4;
    FieldElement * FSOut5;
    FieldElement * FSOut6;
    FieldElement * FSOut7;
    FieldElement * ConnSOutBit;
    FieldElement * ConnSInBit;
    FieldElement * ConnNine2OneBit;

    PaddingKKBitConstantPols (void * pAddress)
    {
        r8Id = (FieldElement *)((uint8_t *)pAddress + 4815060992);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 4848615424);
        latchR8 = (FieldElement *)((uint8_t *)pAddress + 4882169856);
        Fr8 = (FieldElement *)((uint8_t *)pAddress + 4915724288);
        rBitValid = (FieldElement *)((uint8_t *)pAddress + 4949278720);
        latchSOut = (FieldElement *)((uint8_t *)pAddress + 4982833152);
        FSOut0 = (FieldElement *)((uint8_t *)pAddress + 5016387584);
        FSOut1 = (FieldElement *)((uint8_t *)pAddress + 5049942016);
        FSOut2 = (FieldElement *)((uint8_t *)pAddress + 5083496448);
        FSOut3 = (FieldElement *)((uint8_t *)pAddress + 5117050880);
        FSOut4 = (FieldElement *)((uint8_t *)pAddress + 5150605312);
        FSOut5 = (FieldElement *)((uint8_t *)pAddress + 5184159744);
        FSOut6 = (FieldElement *)((uint8_t *)pAddress + 5217714176);
        FSOut7 = (FieldElement *)((uint8_t *)pAddress + 5251268608);
        ConnSOutBit = (FieldElement *)((uint8_t *)pAddress + 5284823040);
        ConnSInBit = (FieldElement *)((uint8_t *)pAddress + 5318377472);
        ConnNine2OneBit = (FieldElement *)((uint8_t *)pAddress + 5351931904);
    }

    PaddingKKBitConstantPols (void * pAddress, uint64_t degree)
    {
        r8Id = (FieldElement *)((uint8_t *)pAddress + 0);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 16);
        latchR8 = (FieldElement *)((uint8_t *)pAddress + 32);
        Fr8 = (FieldElement *)((uint8_t *)pAddress + 48);
        rBitValid = (FieldElement *)((uint8_t *)pAddress + 64);
        latchSOut = (FieldElement *)((uint8_t *)pAddress + 80);
        FSOut0 = (FieldElement *)((uint8_t *)pAddress + 96);
        FSOut1 = (FieldElement *)((uint8_t *)pAddress + 112);
        FSOut2 = (FieldElement *)((uint8_t *)pAddress + 128);
        FSOut3 = (FieldElement *)((uint8_t *)pAddress + 144);
        FSOut4 = (FieldElement *)((uint8_t *)pAddress + 160);
        FSOut5 = (FieldElement *)((uint8_t *)pAddress + 176);
        FSOut6 = (FieldElement *)((uint8_t *)pAddress + 192);
        FSOut7 = (FieldElement *)((uint8_t *)pAddress + 208);
        ConnSOutBit = (FieldElement *)((uint8_t *)pAddress + 224);
        ConnSInBit = (FieldElement *)((uint8_t *)pAddress + 240);
        ConnNine2OneBit = (FieldElement *)((uint8_t *)pAddress + 256);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 272; }
};

class PaddingKKConstantPols
{
public:
    FieldElement * r8Id;
    FieldElement * lastBlock;
    FieldElement * lastBlockLatch;
    FieldElement * r8valid;
    FieldElement * sOutId;
    FieldElement * forceLastHash;
    FieldElement * k_crOffset;
    FieldElement * k_crF0;
    FieldElement * k_crF1;
    FieldElement * k_crF2;
    FieldElement * k_crF3;
    FieldElement * k_crF4;
    FieldElement * k_crF5;
    FieldElement * k_crF6;
    FieldElement * k_crF7;
    FieldElement * crValid;

    PaddingKKConstantPols (void * pAddress)
    {
        r8Id = (FieldElement *)((uint8_t *)pAddress + 5385486336);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 5419040768);
        lastBlockLatch = (FieldElement *)((uint8_t *)pAddress + 5452595200);
        r8valid = (FieldElement *)((uint8_t *)pAddress + 5486149632);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 5519704064);
        forceLastHash = (FieldElement *)((uint8_t *)pAddress + 5553258496);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 5586812928);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 5620367360);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 5653921792);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 5687476224);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 5721030656);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 5754585088);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 5788139520);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 5821693952);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 5855248384);
        crValid = (FieldElement *)((uint8_t *)pAddress + 5888802816);
    }

    PaddingKKConstantPols (void * pAddress, uint64_t degree)
    {
        r8Id = (FieldElement *)((uint8_t *)pAddress + 0);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 16);
        lastBlockLatch = (FieldElement *)((uint8_t *)pAddress + 32);
        r8valid = (FieldElement *)((uint8_t *)pAddress + 48);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 64);
        forceLastHash = (FieldElement *)((uint8_t *)pAddress + 80);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 96);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 112);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 128);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 144);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 160);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 176);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 192);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 208);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 224);
        crValid = (FieldElement *)((uint8_t *)pAddress + 240);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 256; }
};

class MemConstantPols
{
public:
    FieldElement * INCS;
    FieldElement * ISNOTLAST;

    MemConstantPols (void * pAddress)
    {
        INCS = (FieldElement *)((uint8_t *)pAddress + 5922357248);
        ISNOTLAST = (FieldElement *)((uint8_t *)pAddress + 5955911680);
    }

    MemConstantPols (void * pAddress, uint64_t degree)
    {
        INCS = (FieldElement *)((uint8_t *)pAddress + 0);
        ISNOTLAST = (FieldElement *)((uint8_t *)pAddress + 16);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 32; }
};

class MainConstantPols
{
public:
    uint32_t * STEP;

    MainConstantPols (void * pAddress)
    {
        STEP = (uint32_t *)((uint8_t *)pAddress + 5989466112);
    }

    MainConstantPols (void * pAddress, uint64_t degree)
    {
        STEP = (uint32_t *)((uint8_t *)pAddress + 0);
    }

    static uint64_t degree (void) { return 4194304; }
    static uint64_t size (void) { return 16; }
};

class ConstantPols
{
public:
    GlobalConstantPols Global;
    RomConstantPols Rom;
    Byte4ConstantPols Byte4;
    MemAlignConstantPols MemAlign;
    ArithConstantPols Arith;
    BinaryConstantPols Binary;
    PoseidonGConstantPols PoseidonG;
    PaddingPGConstantPols PaddingPG;
    StorageConstantPols Storage;
    NormGate9ConstantPols NormGate9;
    KeccakFConstantPols KeccakF;
    Nine2OneConstantPols Nine2One;
    PaddingKKBitConstantPols PaddingKKBit;
    PaddingKKConstantPols PaddingKK;
    MemConstantPols Mem;
    MainConstantPols Main;

    ConstantPols (void * pAddress) : Global(pAddress), Rom(pAddress), Byte4(pAddress), MemAlign(pAddress), Arith(pAddress), Binary(pAddress), PoseidonG(pAddress), PaddingPG(pAddress), Storage(pAddress), NormGate9(pAddress), KeccakF(pAddress), Nine2One(pAddress), PaddingKKBit(pAddress), PaddingKK(pAddress), Mem(pAddress), Main(pAddress) {}

    static uint64_t size (void) { return 6023020544; }
};

#endif // CONSTANT_POLS_HPP
