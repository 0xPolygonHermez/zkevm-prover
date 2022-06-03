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
        ZHINV = (FieldElement *)((uint8_t *)pAddress + 16777216);
        L1 = (uint8_t *)((uint8_t *)pAddress + 33554432);
        BYTE = (FieldElement *)((uint8_t *)pAddress + 35651584);
        BYTE2 = (FieldElement *)((uint8_t *)pAddress + 52428800);
    }

    GlobalConstantPols (void * pAddress, uint64_t degree)
    {
        ZH = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        ZHINV = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        L1 = (uint8_t *)((uint8_t *)pAddress + 16*degree);
        BYTE = (FieldElement *)((uint8_t *)pAddress + 17*degree);
        BYTE2 = (FieldElement *)((uint8_t *)pAddress + 25*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 33; }
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
    uint8_t * JMPN;
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
    uint32_t * line;
    uint8_t * opCodeNum;
    uint32_t * opCodeAddr;

    RomConstantPols (void * pAddress)
    {
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 69206016);
        CONST1 = (uint32_t *)((uint8_t *)pAddress + 85983232);
        CONST2 = (uint32_t *)((uint8_t *)pAddress + 94371840);
        CONST3 = (uint32_t *)((uint8_t *)pAddress + 102760448);
        CONST4 = (uint32_t *)((uint8_t *)pAddress + 111149056);
        CONST5 = (uint32_t *)((uint8_t *)pAddress + 119537664);
        CONST6 = (uint32_t *)((uint8_t *)pAddress + 127926272);
        CONST7 = (uint32_t *)((uint8_t *)pAddress + 136314880);
        offset = (uint32_t *)((uint8_t *)pAddress + 144703488);
        inA = (FieldElement *)((uint8_t *)pAddress + 153092096);
        inB = (FieldElement *)((uint8_t *)pAddress + 169869312);
        inC = (FieldElement *)((uint8_t *)pAddress + 186646528);
        inD = (FieldElement *)((uint8_t *)pAddress + 203423744);
        inE = (FieldElement *)((uint8_t *)pAddress + 220200960);
        inSR = (FieldElement *)((uint8_t *)pAddress + 236978176);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 253755392);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 270532608);
        inSP = (FieldElement *)((uint8_t *)pAddress + 287309824);
        inPC = (FieldElement *)((uint8_t *)pAddress + 304087040);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 320864256);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 337641472);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 354418688);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 371195904);
        inRR = (FieldElement *)((uint8_t *)pAddress + 387973120);
        setA = (uint8_t *)((uint8_t *)pAddress + 404750336);
        setB = (uint8_t *)((uint8_t *)pAddress + 406847488);
        setC = (uint8_t *)((uint8_t *)pAddress + 408944640);
        setD = (uint8_t *)((uint8_t *)pAddress + 411041792);
        setE = (uint8_t *)((uint8_t *)pAddress + 413138944);
        setSR = (uint8_t *)((uint8_t *)pAddress + 415236096);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 417333248);
        setSP = (uint8_t *)((uint8_t *)pAddress + 419430400);
        setPC = (uint8_t *)((uint8_t *)pAddress + 421527552);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 423624704);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 425721856);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 427819008);
        JMP = (uint8_t *)((uint8_t *)pAddress + 429916160);
        JMPN = (uint8_t *)((uint8_t *)pAddress + 432013312);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 434110464);
        setRR = (uint8_t *)((uint8_t *)pAddress + 436207616);
        incStack = (int32_t *)((uint8_t *)pAddress + 438304768);
        incCode = (int32_t *)((uint8_t *)pAddress + 446693376);
        isStack = (uint8_t *)((uint8_t *)pAddress + 455081984);
        isCode = (uint8_t *)((uint8_t *)pAddress + 457179136);
        isMem = (uint8_t *)((uint8_t *)pAddress + 459276288);
        ind = (uint8_t *)((uint8_t *)pAddress + 461373440);
        indRR = (uint8_t *)((uint8_t *)pAddress + 463470592);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 465567744);
        mOp = (uint8_t *)((uint8_t *)pAddress + 467664896);
        mWR = (uint8_t *)((uint8_t *)pAddress + 469762048);
        sWR = (uint8_t *)((uint8_t *)pAddress + 471859200);
        sRD = (uint8_t *)((uint8_t *)pAddress + 473956352);
        arith = (uint8_t *)((uint8_t *)pAddress + 476053504);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 478150656);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 480247808);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 482344960);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 484442112);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 486539264);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 488636416);
        memAlignWR8 = (uint8_t *)((uint8_t *)pAddress + 490733568);
        hashK = (uint8_t *)((uint8_t *)pAddress + 492830720);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 494927872);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 497025024);
        hashP = (uint8_t *)((uint8_t *)pAddress + 499122176);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 501219328);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 503316480);
        bin = (uint8_t *)((uint8_t *)pAddress + 505413632);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 507510784);
        assert = (uint8_t *)((uint8_t *)pAddress + 509607936);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 511705088);
        line = (uint32_t *)((uint8_t *)pAddress + 513802240);
        opCodeNum = (uint8_t *)((uint8_t *)pAddress + 522190848);
        opCodeAddr = (uint32_t *)((uint8_t *)pAddress + 524288000);
    }

    RomConstantPols (void * pAddress, uint64_t degree)
    {
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        CONST1 = (uint32_t *)((uint8_t *)pAddress + 8*degree);
        CONST2 = (uint32_t *)((uint8_t *)pAddress + 12*degree);
        CONST3 = (uint32_t *)((uint8_t *)pAddress + 16*degree);
        CONST4 = (uint32_t *)((uint8_t *)pAddress + 20*degree);
        CONST5 = (uint32_t *)((uint8_t *)pAddress + 24*degree);
        CONST6 = (uint32_t *)((uint8_t *)pAddress + 28*degree);
        CONST7 = (uint32_t *)((uint8_t *)pAddress + 32*degree);
        offset = (uint32_t *)((uint8_t *)pAddress + 36*degree);
        inA = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        inB = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        inC = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        inD = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        inE = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        inSR = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        inSP = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        inPC = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 120*degree);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 128*degree);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 136*degree);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 144*degree);
        inRR = (FieldElement *)((uint8_t *)pAddress + 152*degree);
        setA = (uint8_t *)((uint8_t *)pAddress + 160*degree);
        setB = (uint8_t *)((uint8_t *)pAddress + 161*degree);
        setC = (uint8_t *)((uint8_t *)pAddress + 162*degree);
        setD = (uint8_t *)((uint8_t *)pAddress + 163*degree);
        setE = (uint8_t *)((uint8_t *)pAddress + 164*degree);
        setSR = (uint8_t *)((uint8_t *)pAddress + 165*degree);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 166*degree);
        setSP = (uint8_t *)((uint8_t *)pAddress + 167*degree);
        setPC = (uint8_t *)((uint8_t *)pAddress + 168*degree);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 169*degree);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 170*degree);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 171*degree);
        JMP = (uint8_t *)((uint8_t *)pAddress + 172*degree);
        JMPN = (uint8_t *)((uint8_t *)pAddress + 173*degree);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 174*degree);
        setRR = (uint8_t *)((uint8_t *)pAddress + 175*degree);
        incStack = (int32_t *)((uint8_t *)pAddress + 176*degree);
        incCode = (int32_t *)((uint8_t *)pAddress + 180*degree);
        isStack = (uint8_t *)((uint8_t *)pAddress + 184*degree);
        isCode = (uint8_t *)((uint8_t *)pAddress + 185*degree);
        isMem = (uint8_t *)((uint8_t *)pAddress + 186*degree);
        ind = (uint8_t *)((uint8_t *)pAddress + 187*degree);
        indRR = (uint8_t *)((uint8_t *)pAddress + 188*degree);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 189*degree);
        mOp = (uint8_t *)((uint8_t *)pAddress + 190*degree);
        mWR = (uint8_t *)((uint8_t *)pAddress + 191*degree);
        sWR = (uint8_t *)((uint8_t *)pAddress + 192*degree);
        sRD = (uint8_t *)((uint8_t *)pAddress + 193*degree);
        arith = (uint8_t *)((uint8_t *)pAddress + 194*degree);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 195*degree);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 196*degree);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 197*degree);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 198*degree);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 199*degree);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 200*degree);
        memAlignWR8 = (uint8_t *)((uint8_t *)pAddress + 201*degree);
        hashK = (uint8_t *)((uint8_t *)pAddress + 202*degree);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 203*degree);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 204*degree);
        hashP = (uint8_t *)((uint8_t *)pAddress + 205*degree);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 206*degree);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 207*degree);
        bin = (uint8_t *)((uint8_t *)pAddress + 208*degree);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 209*degree);
        assert = (uint8_t *)((uint8_t *)pAddress + 210*degree);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 211*degree);
        line = (uint32_t *)((uint8_t *)pAddress + 212*degree);
        opCodeNum = (uint8_t *)((uint8_t *)pAddress + 216*degree);
        opCodeAddr = (uint32_t *)((uint8_t *)pAddress + 217*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 221; }
};

class Byte4ConstantPols
{
public:
    uint8_t * SET;

    Byte4ConstantPols (void * pAddress)
    {
        SET = (uint8_t *)((uint8_t *)pAddress + 532676608);
    }

    Byte4ConstantPols (void * pAddress, uint64_t degree)
    {
        SET = (uint8_t *)((uint8_t *)pAddress + 0*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 1; }
};

class MemAlignConstantPols
{
public:
    FieldElement * BYTE2A;
    FieldElement * BYTE2B;
    uint8_t * BYTE_C3072;
    FieldElement * FACTOR[8];
    FieldElement * FACTORV[8];
    FieldElement * STEP;
    FieldElement * WR256;
    FieldElement * WR8;
    FieldElement * OFFSET;
    FieldElement * RESET;
    FieldElement * SELM1;

    MemAlignConstantPols (void * pAddress)
    {
        BYTE2A = (FieldElement *)((uint8_t *)pAddress + 534773760);
        BYTE2B = (FieldElement *)((uint8_t *)pAddress + 551550976);
        BYTE_C3072 = (uint8_t *)((uint8_t *)pAddress + 568328192);
        FACTOR[0] = (FieldElement *)((uint8_t *)pAddress + 570425344);
        FACTOR[1] = (FieldElement *)((uint8_t *)pAddress + 587202560);
        FACTOR[2] = (FieldElement *)((uint8_t *)pAddress + 603979776);
        FACTOR[3] = (FieldElement *)((uint8_t *)pAddress + 620756992);
        FACTOR[4] = (FieldElement *)((uint8_t *)pAddress + 637534208);
        FACTOR[5] = (FieldElement *)((uint8_t *)pAddress + 654311424);
        FACTOR[6] = (FieldElement *)((uint8_t *)pAddress + 671088640);
        FACTOR[7] = (FieldElement *)((uint8_t *)pAddress + 687865856);
        FACTORV[0] = (FieldElement *)((uint8_t *)pAddress + 704643072);
        FACTORV[1] = (FieldElement *)((uint8_t *)pAddress + 721420288);
        FACTORV[2] = (FieldElement *)((uint8_t *)pAddress + 738197504);
        FACTORV[3] = (FieldElement *)((uint8_t *)pAddress + 754974720);
        FACTORV[4] = (FieldElement *)((uint8_t *)pAddress + 771751936);
        FACTORV[5] = (FieldElement *)((uint8_t *)pAddress + 788529152);
        FACTORV[6] = (FieldElement *)((uint8_t *)pAddress + 805306368);
        FACTORV[7] = (FieldElement *)((uint8_t *)pAddress + 822083584);
        STEP = (FieldElement *)((uint8_t *)pAddress + 838860800);
        WR256 = (FieldElement *)((uint8_t *)pAddress + 855638016);
        WR8 = (FieldElement *)((uint8_t *)pAddress + 872415232);
        OFFSET = (FieldElement *)((uint8_t *)pAddress + 889192448);
        RESET = (FieldElement *)((uint8_t *)pAddress + 905969664);
        SELM1 = (FieldElement *)((uint8_t *)pAddress + 922746880);
    }

    MemAlignConstantPols (void * pAddress, uint64_t degree)
    {
        BYTE2A = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        BYTE2B = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        BYTE_C3072 = (uint8_t *)((uint8_t *)pAddress + 16*degree);
        FACTOR[0] = (FieldElement *)((uint8_t *)pAddress + 17*degree);
        FACTOR[1] = (FieldElement *)((uint8_t *)pAddress + 25*degree);
        FACTOR[2] = (FieldElement *)((uint8_t *)pAddress + 33*degree);
        FACTOR[3] = (FieldElement *)((uint8_t *)pAddress + 41*degree);
        FACTOR[4] = (FieldElement *)((uint8_t *)pAddress + 49*degree);
        FACTOR[5] = (FieldElement *)((uint8_t *)pAddress + 57*degree);
        FACTOR[6] = (FieldElement *)((uint8_t *)pAddress + 65*degree);
        FACTOR[7] = (FieldElement *)((uint8_t *)pAddress + 73*degree);
        FACTORV[0] = (FieldElement *)((uint8_t *)pAddress + 81*degree);
        FACTORV[1] = (FieldElement *)((uint8_t *)pAddress + 89*degree);
        FACTORV[2] = (FieldElement *)((uint8_t *)pAddress + 97*degree);
        FACTORV[3] = (FieldElement *)((uint8_t *)pAddress + 105*degree);
        FACTORV[4] = (FieldElement *)((uint8_t *)pAddress + 113*degree);
        FACTORV[5] = (FieldElement *)((uint8_t *)pAddress + 121*degree);
        FACTORV[6] = (FieldElement *)((uint8_t *)pAddress + 129*degree);
        FACTORV[7] = (FieldElement *)((uint8_t *)pAddress + 137*degree);
        STEP = (FieldElement *)((uint8_t *)pAddress + 145*degree);
        WR256 = (FieldElement *)((uint8_t *)pAddress + 153*degree);
        WR8 = (FieldElement *)((uint8_t *)pAddress + 161*degree);
        OFFSET = (FieldElement *)((uint8_t *)pAddress + 169*degree);
        RESET = (FieldElement *)((uint8_t *)pAddress + 177*degree);
        SELM1 = (FieldElement *)((uint8_t *)pAddress + 185*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 193; }
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
        BIT19 = (FieldElement *)((uint8_t *)pAddress + 939524096);
        GL_SIGNED_4BITS = (FieldElement *)((uint8_t *)pAddress + 956301312);
        GL_SIGNED_18BITS = (FieldElement *)((uint8_t *)pAddress + 973078528);
        ck[0] = (uint8_t *)((uint8_t *)pAddress + 989855744);
        ck[1] = (uint8_t *)((uint8_t *)pAddress + 991952896);
        ck[2] = (uint8_t *)((uint8_t *)pAddress + 994050048);
        ck[3] = (uint8_t *)((uint8_t *)pAddress + 996147200);
        ck[4] = (uint8_t *)((uint8_t *)pAddress + 998244352);
        ck[5] = (uint8_t *)((uint8_t *)pAddress + 1000341504);
        ck[6] = (uint8_t *)((uint8_t *)pAddress + 1002438656);
        ck[7] = (uint8_t *)((uint8_t *)pAddress + 1004535808);
        ck[8] = (uint8_t *)((uint8_t *)pAddress + 1006632960);
        ck[9] = (uint8_t *)((uint8_t *)pAddress + 1008730112);
        ck[10] = (uint8_t *)((uint8_t *)pAddress + 1010827264);
        ck[11] = (uint8_t *)((uint8_t *)pAddress + 1012924416);
        ck[12] = (uint8_t *)((uint8_t *)pAddress + 1015021568);
        ck[13] = (uint8_t *)((uint8_t *)pAddress + 1017118720);
        ck[14] = (uint8_t *)((uint8_t *)pAddress + 1019215872);
        ck[15] = (uint8_t *)((uint8_t *)pAddress + 1021313024);
        ck[16] = (uint8_t *)((uint8_t *)pAddress + 1023410176);
        ck[17] = (uint8_t *)((uint8_t *)pAddress + 1025507328);
        ck[18] = (uint8_t *)((uint8_t *)pAddress + 1027604480);
        ck[19] = (uint8_t *)((uint8_t *)pAddress + 1029701632);
        ck[20] = (uint8_t *)((uint8_t *)pAddress + 1031798784);
        ck[21] = (uint8_t *)((uint8_t *)pAddress + 1033895936);
        ck[22] = (uint8_t *)((uint8_t *)pAddress + 1035993088);
        ck[23] = (uint8_t *)((uint8_t *)pAddress + 1038090240);
        ck[24] = (uint8_t *)((uint8_t *)pAddress + 1040187392);
        ck[25] = (uint8_t *)((uint8_t *)pAddress + 1042284544);
        ck[26] = (uint8_t *)((uint8_t *)pAddress + 1044381696);
        ck[27] = (uint8_t *)((uint8_t *)pAddress + 1046478848);
        ck[28] = (uint8_t *)((uint8_t *)pAddress + 1048576000);
        ck[29] = (uint8_t *)((uint8_t *)pAddress + 1050673152);
        ck[30] = (uint8_t *)((uint8_t *)pAddress + 1052770304);
        ck[31] = (uint8_t *)((uint8_t *)pAddress + 1054867456);
    }

    ArithConstantPols (void * pAddress, uint64_t degree)
    {
        BIT19 = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        GL_SIGNED_4BITS = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        GL_SIGNED_18BITS = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        ck[0] = (uint8_t *)((uint8_t *)pAddress + 24*degree);
        ck[1] = (uint8_t *)((uint8_t *)pAddress + 25*degree);
        ck[2] = (uint8_t *)((uint8_t *)pAddress + 26*degree);
        ck[3] = (uint8_t *)((uint8_t *)pAddress + 27*degree);
        ck[4] = (uint8_t *)((uint8_t *)pAddress + 28*degree);
        ck[5] = (uint8_t *)((uint8_t *)pAddress + 29*degree);
        ck[6] = (uint8_t *)((uint8_t *)pAddress + 30*degree);
        ck[7] = (uint8_t *)((uint8_t *)pAddress + 31*degree);
        ck[8] = (uint8_t *)((uint8_t *)pAddress + 32*degree);
        ck[9] = (uint8_t *)((uint8_t *)pAddress + 33*degree);
        ck[10] = (uint8_t *)((uint8_t *)pAddress + 34*degree);
        ck[11] = (uint8_t *)((uint8_t *)pAddress + 35*degree);
        ck[12] = (uint8_t *)((uint8_t *)pAddress + 36*degree);
        ck[13] = (uint8_t *)((uint8_t *)pAddress + 37*degree);
        ck[14] = (uint8_t *)((uint8_t *)pAddress + 38*degree);
        ck[15] = (uint8_t *)((uint8_t *)pAddress + 39*degree);
        ck[16] = (uint8_t *)((uint8_t *)pAddress + 40*degree);
        ck[17] = (uint8_t *)((uint8_t *)pAddress + 41*degree);
        ck[18] = (uint8_t *)((uint8_t *)pAddress + 42*degree);
        ck[19] = (uint8_t *)((uint8_t *)pAddress + 43*degree);
        ck[20] = (uint8_t *)((uint8_t *)pAddress + 44*degree);
        ck[21] = (uint8_t *)((uint8_t *)pAddress + 45*degree);
        ck[22] = (uint8_t *)((uint8_t *)pAddress + 46*degree);
        ck[23] = (uint8_t *)((uint8_t *)pAddress + 47*degree);
        ck[24] = (uint8_t *)((uint8_t *)pAddress + 48*degree);
        ck[25] = (uint8_t *)((uint8_t *)pAddress + 49*degree);
        ck[26] = (uint8_t *)((uint8_t *)pAddress + 50*degree);
        ck[27] = (uint8_t *)((uint8_t *)pAddress + 51*degree);
        ck[28] = (uint8_t *)((uint8_t *)pAddress + 52*degree);
        ck[29] = (uint8_t *)((uint8_t *)pAddress + 53*degree);
        ck[30] = (uint8_t *)((uint8_t *)pAddress + 54*degree);
        ck[31] = (uint8_t *)((uint8_t *)pAddress + 55*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 56; }
};

class BinaryConstantPols
{
public:
    uint8_t * P_OPCODE;
    uint8_t * P_A;
    uint8_t * P_B;
    uint8_t * P_CIN;
    uint8_t * P_LAST;
    uint8_t * P_USE_CARRY;
    uint8_t * P_C;
    uint8_t * P_COUT;
    uint8_t * RESET;
    uint32_t * FACTOR[8];

    BinaryConstantPols (void * pAddress)
    {
        P_OPCODE = (uint8_t *)((uint8_t *)pAddress + 1056964608);
        P_A = (uint8_t *)((uint8_t *)pAddress + 1059061760);
        P_B = (uint8_t *)((uint8_t *)pAddress + 1061158912);
        P_CIN = (uint8_t *)((uint8_t *)pAddress + 1063256064);
        P_LAST = (uint8_t *)((uint8_t *)pAddress + 1065353216);
        P_USE_CARRY = (uint8_t *)((uint8_t *)pAddress + 1067450368);
        P_C = (uint8_t *)((uint8_t *)pAddress + 1069547520);
        P_COUT = (uint8_t *)((uint8_t *)pAddress + 1071644672);
        RESET = (uint8_t *)((uint8_t *)pAddress + 1073741824);
        FACTOR[0] = (uint32_t *)((uint8_t *)pAddress + 1075838976);
        FACTOR[1] = (uint32_t *)((uint8_t *)pAddress + 1084227584);
        FACTOR[2] = (uint32_t *)((uint8_t *)pAddress + 1092616192);
        FACTOR[3] = (uint32_t *)((uint8_t *)pAddress + 1101004800);
        FACTOR[4] = (uint32_t *)((uint8_t *)pAddress + 1109393408);
        FACTOR[5] = (uint32_t *)((uint8_t *)pAddress + 1117782016);
        FACTOR[6] = (uint32_t *)((uint8_t *)pAddress + 1126170624);
        FACTOR[7] = (uint32_t *)((uint8_t *)pAddress + 1134559232);
    }

    BinaryConstantPols (void * pAddress, uint64_t degree)
    {
        P_OPCODE = (uint8_t *)((uint8_t *)pAddress + 0*degree);
        P_A = (uint8_t *)((uint8_t *)pAddress + 1*degree);
        P_B = (uint8_t *)((uint8_t *)pAddress + 2*degree);
        P_CIN = (uint8_t *)((uint8_t *)pAddress + 3*degree);
        P_LAST = (uint8_t *)((uint8_t *)pAddress + 4*degree);
        P_USE_CARRY = (uint8_t *)((uint8_t *)pAddress + 5*degree);
        P_C = (uint8_t *)((uint8_t *)pAddress + 6*degree);
        P_COUT = (uint8_t *)((uint8_t *)pAddress + 7*degree);
        RESET = (uint8_t *)((uint8_t *)pAddress + 8*degree);
        FACTOR[0] = (uint32_t *)((uint8_t *)pAddress + 9*degree);
        FACTOR[1] = (uint32_t *)((uint8_t *)pAddress + 13*degree);
        FACTOR[2] = (uint32_t *)((uint8_t *)pAddress + 17*degree);
        FACTOR[3] = (uint32_t *)((uint8_t *)pAddress + 21*degree);
        FACTOR[4] = (uint32_t *)((uint8_t *)pAddress + 25*degree);
        FACTOR[5] = (uint32_t *)((uint8_t *)pAddress + 29*degree);
        FACTOR[6] = (uint32_t *)((uint8_t *)pAddress + 33*degree);
        FACTOR[7] = (uint32_t *)((uint8_t *)pAddress + 37*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 41; }
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
        LAST = (FieldElement *)((uint8_t *)pAddress + 1142947840);
        LATCH = (FieldElement *)((uint8_t *)pAddress + 1159725056);
        LASTBLOCK = (FieldElement *)((uint8_t *)pAddress + 1176502272);
        PARTIAL = (FieldElement *)((uint8_t *)pAddress + 1193279488);
        C[0] = (FieldElement *)((uint8_t *)pAddress + 1210056704);
        C[1] = (FieldElement *)((uint8_t *)pAddress + 1226833920);
        C[2] = (FieldElement *)((uint8_t *)pAddress + 1243611136);
        C[3] = (FieldElement *)((uint8_t *)pAddress + 1260388352);
        C[4] = (FieldElement *)((uint8_t *)pAddress + 1277165568);
        C[5] = (FieldElement *)((uint8_t *)pAddress + 1293942784);
        C[6] = (FieldElement *)((uint8_t *)pAddress + 1310720000);
        C[7] = (FieldElement *)((uint8_t *)pAddress + 1327497216);
        C[8] = (FieldElement *)((uint8_t *)pAddress + 1344274432);
        C[9] = (FieldElement *)((uint8_t *)pAddress + 1361051648);
        C[10] = (FieldElement *)((uint8_t *)pAddress + 1377828864);
        C[11] = (FieldElement *)((uint8_t *)pAddress + 1394606080);
    }

    PoseidonGConstantPols (void * pAddress, uint64_t degree)
    {
        LAST = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        LATCH = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        LASTBLOCK = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        PARTIAL = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        C[0] = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        C[1] = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        C[2] = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        C[3] = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        C[4] = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        C[5] = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        C[6] = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        C[7] = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        C[8] = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        C[9] = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        C[10] = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        C[11] = (FieldElement *)((uint8_t *)pAddress + 120*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 128; }
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
        F[0] = (FieldElement *)((uint8_t *)pAddress + 1411383296);
        F[1] = (FieldElement *)((uint8_t *)pAddress + 1428160512);
        F[2] = (FieldElement *)((uint8_t *)pAddress + 1444937728);
        F[3] = (FieldElement *)((uint8_t *)pAddress + 1461714944);
        F[4] = (FieldElement *)((uint8_t *)pAddress + 1478492160);
        F[5] = (FieldElement *)((uint8_t *)pAddress + 1495269376);
        F[6] = (FieldElement *)((uint8_t *)pAddress + 1512046592);
        F[7] = (FieldElement *)((uint8_t *)pAddress + 1528823808);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 1545601024);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 1562378240);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 1579155456);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 1595932672);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 1612709888);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 1629487104);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 1646264320);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 1663041536);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 1679818752);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 1696595968);
    }

    PaddingPGConstantPols (void * pAddress, uint64_t degree)
    {
        F[0] = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        F[1] = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        F[2] = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        F[3] = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        F[4] = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        F[5] = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        F[6] = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        F[7] = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 120*degree);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 128*degree);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 136*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 144; }
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
        rHash = (FieldElement *)((uint8_t *)pAddress + 1713373184);
        rHashType = (FieldElement *)((uint8_t *)pAddress + 1730150400);
        rLatchGet = (FieldElement *)((uint8_t *)pAddress + 1746927616);
        rLatchSet = (FieldElement *)((uint8_t *)pAddress + 1763704832);
        rClimbRkey = (FieldElement *)((uint8_t *)pAddress + 1780482048);
        rClimbSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 1797259264);
        rClimbSiblingRkeyN = (FieldElement *)((uint8_t *)pAddress + 1814036480);
        rRotateLevel = (FieldElement *)((uint8_t *)pAddress + 1830813696);
        rJmpz = (FieldElement *)((uint8_t *)pAddress + 1847590912);
        rJmp = (FieldElement *)((uint8_t *)pAddress + 1864368128);
        rConst0 = (FieldElement *)((uint8_t *)pAddress + 1881145344);
        rConst1 = (FieldElement *)((uint8_t *)pAddress + 1897922560);
        rConst2 = (FieldElement *)((uint8_t *)pAddress + 1914699776);
        rConst3 = (FieldElement *)((uint8_t *)pAddress + 1931476992);
        rAddress = (FieldElement *)((uint8_t *)pAddress + 1948254208);
        rLine = (FieldElement *)((uint8_t *)pAddress + 1965031424);
    }

    StorageConstantPols (void * pAddress, uint64_t degree)
    {
        rHash = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        rHashType = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        rLatchGet = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        rLatchSet = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        rClimbRkey = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        rClimbSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        rClimbSiblingRkeyN = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        rRotateLevel = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        rJmpz = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        rJmp = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        rConst0 = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        rConst1 = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        rConst2 = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        rConst3 = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        rAddress = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        rLine = (FieldElement *)((uint8_t *)pAddress + 120*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 128; }
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
        Value3 = (FieldElement *)((uint8_t *)pAddress + 1981808640);
        Value3Norm = (FieldElement *)((uint8_t *)pAddress + 1998585856);
        Gate9Type = (FieldElement *)((uint8_t *)pAddress + 2015363072);
        Gate9A = (FieldElement *)((uint8_t *)pAddress + 2032140288);
        Gate9B = (FieldElement *)((uint8_t *)pAddress + 2048917504);
        Gate9C = (FieldElement *)((uint8_t *)pAddress + 2065694720);
        Latch = (FieldElement *)((uint8_t *)pAddress + 2082471936);
        Factor = (FieldElement *)((uint8_t *)pAddress + 2099249152);
    }

    NormGate9ConstantPols (void * pAddress, uint64_t degree)
    {
        Value3 = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        Value3Norm = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        Gate9Type = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        Gate9A = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        Gate9B = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        Gate9C = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        Latch = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        Factor = (FieldElement *)((uint8_t *)pAddress + 56*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 64; }
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
        ConnA = (FieldElement *)((uint8_t *)pAddress + 2116026368);
        ConnB = (FieldElement *)((uint8_t *)pAddress + 2132803584);
        ConnC = (FieldElement *)((uint8_t *)pAddress + 2149580800);
        NormalizedGate = (FieldElement *)((uint8_t *)pAddress + 2166358016);
        GateType = (FieldElement *)((uint8_t *)pAddress + 2183135232);
    }

    KeccakFConstantPols (void * pAddress, uint64_t degree)
    {
        ConnA = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        ConnB = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        ConnC = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        NormalizedGate = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        GateType = (FieldElement *)((uint8_t *)pAddress + 32*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 40; }
};

class Nine2OneConstantPols
{
public:
    FieldElement * Field9latch;
    FieldElement * Factor;

    Nine2OneConstantPols (void * pAddress)
    {
        Field9latch = (FieldElement *)((uint8_t *)pAddress + 2199912448);
        Factor = (FieldElement *)((uint8_t *)pAddress + 2216689664);
    }

    Nine2OneConstantPols (void * pAddress, uint64_t degree)
    {
        Field9latch = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        Factor = (FieldElement *)((uint8_t *)pAddress + 8*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 16; }
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
        r8Id = (FieldElement *)((uint8_t *)pAddress + 2233466880);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 2250244096);
        latchR8 = (FieldElement *)((uint8_t *)pAddress + 2267021312);
        Fr8 = (FieldElement *)((uint8_t *)pAddress + 2283798528);
        rBitValid = (FieldElement *)((uint8_t *)pAddress + 2300575744);
        latchSOut = (FieldElement *)((uint8_t *)pAddress + 2317352960);
        FSOut0 = (FieldElement *)((uint8_t *)pAddress + 2334130176);
        FSOut1 = (FieldElement *)((uint8_t *)pAddress + 2350907392);
        FSOut2 = (FieldElement *)((uint8_t *)pAddress + 2367684608);
        FSOut3 = (FieldElement *)((uint8_t *)pAddress + 2384461824);
        FSOut4 = (FieldElement *)((uint8_t *)pAddress + 2401239040);
        FSOut5 = (FieldElement *)((uint8_t *)pAddress + 2418016256);
        FSOut6 = (FieldElement *)((uint8_t *)pAddress + 2434793472);
        FSOut7 = (FieldElement *)((uint8_t *)pAddress + 2451570688);
        ConnSOutBit = (FieldElement *)((uint8_t *)pAddress + 2468347904);
        ConnSInBit = (FieldElement *)((uint8_t *)pAddress + 2485125120);
        ConnNine2OneBit = (FieldElement *)((uint8_t *)pAddress + 2501902336);
    }

    PaddingKKBitConstantPols (void * pAddress, uint64_t degree)
    {
        r8Id = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        latchR8 = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        Fr8 = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        rBitValid = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        latchSOut = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        FSOut0 = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        FSOut1 = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        FSOut2 = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        FSOut3 = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        FSOut4 = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        FSOut5 = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        FSOut6 = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        FSOut7 = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        ConnSOutBit = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        ConnSInBit = (FieldElement *)((uint8_t *)pAddress + 120*degree);
        ConnNine2OneBit = (FieldElement *)((uint8_t *)pAddress + 128*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 136; }
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
        r8Id = (FieldElement *)((uint8_t *)pAddress + 2518679552);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 2535456768);
        lastBlockLatch = (FieldElement *)((uint8_t *)pAddress + 2552233984);
        r8valid = (FieldElement *)((uint8_t *)pAddress + 2569011200);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 2585788416);
        forceLastHash = (FieldElement *)((uint8_t *)pAddress + 2602565632);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 2619342848);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 2636120064);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 2652897280);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 2669674496);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 2686451712);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 2703228928);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 2720006144);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 2736783360);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 2753560576);
        crValid = (FieldElement *)((uint8_t *)pAddress + 2770337792);
    }

    PaddingKKConstantPols (void * pAddress, uint64_t degree)
    {
        r8Id = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 8*degree);
        lastBlockLatch = (FieldElement *)((uint8_t *)pAddress + 16*degree);
        r8valid = (FieldElement *)((uint8_t *)pAddress + 24*degree);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 32*degree);
        forceLastHash = (FieldElement *)((uint8_t *)pAddress + 40*degree);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 48*degree);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 56*degree);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 64*degree);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        crValid = (FieldElement *)((uint8_t *)pAddress + 120*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 128; }
};

class MemConstantPols
{
public:
    FieldElement * INCS;
    FieldElement * ISNOTLAST;

    MemConstantPols (void * pAddress)
    {
        INCS = (FieldElement *)((uint8_t *)pAddress + 2787115008);
        ISNOTLAST = (FieldElement *)((uint8_t *)pAddress + 2803892224);
    }

    MemConstantPols (void * pAddress, uint64_t degree)
    {
        INCS = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        ISNOTLAST = (FieldElement *)((uint8_t *)pAddress + 8*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 16; }
};

class MainConstantPols
{
public:
    uint32_t * STEP;

    MainConstantPols (void * pAddress)
    {
        STEP = (uint32_t *)((uint8_t *)pAddress + 2820669440);
    }

    MainConstantPols (void * pAddress, uint64_t degree)
    {
        STEP = (uint32_t *)((uint8_t *)pAddress + 0*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 4; }
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

    static uint64_t size (void) { return 2829058048; }
};

#endif // CONSTANT_POLS_HPP
