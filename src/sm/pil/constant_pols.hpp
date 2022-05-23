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
        CONST2 = (uint32_t *)((uint8_t *)pAddress + 102760448);
        CONST3 = (uint32_t *)((uint8_t *)pAddress + 119537664);
        CONST4 = (uint32_t *)((uint8_t *)pAddress + 136314880);
        CONST5 = (uint32_t *)((uint8_t *)pAddress + 153092096);
        CONST6 = (uint32_t *)((uint8_t *)pAddress + 169869312);
        CONST7 = (uint32_t *)((uint8_t *)pAddress + 186646528);
        offset = (uint32_t *)((uint8_t *)pAddress + 203423744);
        inA = (FieldElement *)((uint8_t *)pAddress + 220200960);
        inB = (FieldElement *)((uint8_t *)pAddress + 236978176);
        inC = (FieldElement *)((uint8_t *)pAddress + 253755392);
        inD = (FieldElement *)((uint8_t *)pAddress + 270532608);
        inE = (FieldElement *)((uint8_t *)pAddress + 287309824);
        inSR = (FieldElement *)((uint8_t *)pAddress + 304087040);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 320864256);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 337641472);
        inSP = (FieldElement *)((uint8_t *)pAddress + 354418688);
        inPC = (FieldElement *)((uint8_t *)pAddress + 371195904);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 387973120);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 404750336);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 421527552);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 438304768);
        inRR = (FieldElement *)((uint8_t *)pAddress + 455081984);
        setA = (uint8_t *)((uint8_t *)pAddress + 471859200);
        setB = (uint8_t *)((uint8_t *)pAddress + 473956352);
        setC = (uint8_t *)((uint8_t *)pAddress + 476053504);
        setD = (uint8_t *)((uint8_t *)pAddress + 478150656);
        setE = (uint8_t *)((uint8_t *)pAddress + 480247808);
        setSR = (uint8_t *)((uint8_t *)pAddress + 482344960);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 484442112);
        setSP = (uint8_t *)((uint8_t *)pAddress + 486539264);
        setPC = (uint8_t *)((uint8_t *)pAddress + 488636416);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 490733568);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 492830720);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 494927872);
        JMP = (uint8_t *)((uint8_t *)pAddress + 497025024);
        JMPN = (uint8_t *)((uint8_t *)pAddress + 499122176);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 501219328);
        setRR = (uint8_t *)((uint8_t *)pAddress + 503316480);
        incStack = (int32_t *)((uint8_t *)pAddress + 505413632);
        incCode = (int32_t *)((uint8_t *)pAddress + 513802240);
        isStack = (uint8_t *)((uint8_t *)pAddress + 522190848);
        isCode = (uint8_t *)((uint8_t *)pAddress + 524288000);
        isMem = (uint8_t *)((uint8_t *)pAddress + 526385152);
        ind = (uint8_t *)((uint8_t *)pAddress + 528482304);
        indRR = (uint8_t *)((uint8_t *)pAddress + 530579456);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 532676608);
        mOp = (uint8_t *)((uint8_t *)pAddress + 534773760);
        mWR = (uint8_t *)((uint8_t *)pAddress + 536870912);
        sWR = (uint8_t *)((uint8_t *)pAddress + 538968064);
        sRD = (uint8_t *)((uint8_t *)pAddress + 541065216);
        arith = (uint8_t *)((uint8_t *)pAddress + 543162368);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 545259520);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 547356672);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 549453824);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 551550976);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 553648128);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 555745280);
        hashK = (uint8_t *)((uint8_t *)pAddress + 557842432);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 559939584);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 562036736);
        hashP = (uint8_t *)((uint8_t *)pAddress + 564133888);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 566231040);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 568328192);
        bin = (uint8_t *)((uint8_t *)pAddress + 570425344);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 572522496);
        assert = (uint8_t *)((uint8_t *)pAddress + 574619648);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 576716800);
        line = (uint32_t *)((uint8_t *)pAddress + 578813952);
        opCodeNum = (uint8_t *)((uint8_t *)pAddress + 595591168);
        opCodeAddr = (uint32_t *)((uint8_t *)pAddress + 597688320);
    }

    RomConstantPols (void * pAddress, uint64_t degree)
    {
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 0*degree);
        CONST1 = (uint32_t *)((uint8_t *)pAddress + 8*degree);
        CONST2 = (uint32_t *)((uint8_t *)pAddress + 16*degree);
        CONST3 = (uint32_t *)((uint8_t *)pAddress + 24*degree);
        CONST4 = (uint32_t *)((uint8_t *)pAddress + 32*degree);
        CONST5 = (uint32_t *)((uint8_t *)pAddress + 40*degree);
        CONST6 = (uint32_t *)((uint8_t *)pAddress + 48*degree);
        CONST7 = (uint32_t *)((uint8_t *)pAddress + 56*degree);
        offset = (uint32_t *)((uint8_t *)pAddress + 64*degree);
        inA = (FieldElement *)((uint8_t *)pAddress + 72*degree);
        inB = (FieldElement *)((uint8_t *)pAddress + 80*degree);
        inC = (FieldElement *)((uint8_t *)pAddress + 88*degree);
        inD = (FieldElement *)((uint8_t *)pAddress + 96*degree);
        inE = (FieldElement *)((uint8_t *)pAddress + 104*degree);
        inSR = (FieldElement *)((uint8_t *)pAddress + 112*degree);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 120*degree);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 128*degree);
        inSP = (FieldElement *)((uint8_t *)pAddress + 136*degree);
        inPC = (FieldElement *)((uint8_t *)pAddress + 144*degree);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 152*degree);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 160*degree);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 168*degree);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 176*degree);
        inRR = (FieldElement *)((uint8_t *)pAddress + 184*degree);
        setA = (uint8_t *)((uint8_t *)pAddress + 192*degree);
        setB = (uint8_t *)((uint8_t *)pAddress + 193*degree);
        setC = (uint8_t *)((uint8_t *)pAddress + 194*degree);
        setD = (uint8_t *)((uint8_t *)pAddress + 195*degree);
        setE = (uint8_t *)((uint8_t *)pAddress + 196*degree);
        setSR = (uint8_t *)((uint8_t *)pAddress + 197*degree);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 198*degree);
        setSP = (uint8_t *)((uint8_t *)pAddress + 199*degree);
        setPC = (uint8_t *)((uint8_t *)pAddress + 200*degree);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 201*degree);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 202*degree);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 203*degree);
        JMP = (uint8_t *)((uint8_t *)pAddress + 204*degree);
        JMPN = (uint8_t *)((uint8_t *)pAddress + 205*degree);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 206*degree);
        setRR = (uint8_t *)((uint8_t *)pAddress + 207*degree);
        incStack = (int32_t *)((uint8_t *)pAddress + 208*degree);
        incCode = (int32_t *)((uint8_t *)pAddress + 212*degree);
        isStack = (uint8_t *)((uint8_t *)pAddress + 216*degree);
        isCode = (uint8_t *)((uint8_t *)pAddress + 217*degree);
        isMem = (uint8_t *)((uint8_t *)pAddress + 218*degree);
        ind = (uint8_t *)((uint8_t *)pAddress + 219*degree);
        indRR = (uint8_t *)((uint8_t *)pAddress + 220*degree);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 221*degree);
        mOp = (uint8_t *)((uint8_t *)pAddress + 222*degree);
        mWR = (uint8_t *)((uint8_t *)pAddress + 223*degree);
        sWR = (uint8_t *)((uint8_t *)pAddress + 224*degree);
        sRD = (uint8_t *)((uint8_t *)pAddress + 225*degree);
        arith = (uint8_t *)((uint8_t *)pAddress + 226*degree);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 227*degree);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 228*degree);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 229*degree);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 230*degree);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 231*degree);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 232*degree);
        hashK = (uint8_t *)((uint8_t *)pAddress + 233*degree);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 234*degree);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 235*degree);
        hashP = (uint8_t *)((uint8_t *)pAddress + 236*degree);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 237*degree);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 238*degree);
        bin = (uint8_t *)((uint8_t *)pAddress + 239*degree);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 240*degree);
        assert = (uint8_t *)((uint8_t *)pAddress + 241*degree);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 242*degree);
        line = (uint32_t *)((uint8_t *)pAddress + 243*degree);
        opCodeNum = (uint8_t *)((uint8_t *)pAddress + 251*degree);
        opCodeAddr = (uint32_t *)((uint8_t *)pAddress + 252*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 260; }
};

class Byte4ConstantPols
{
public:
    uint8_t * SET;

    Byte4ConstantPols (void * pAddress)
    {
        SET = (uint8_t *)((uint8_t *)pAddress + 614465536);
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
        BYTE2A = (uint8_t *)((uint8_t *)pAddress + 616562688);
        BYTE2B = (uint8_t *)((uint8_t *)pAddress + 618659840);
        FACTOR0[0] = (FieldElement *)((uint8_t *)pAddress + 620756992);
        FACTOR0[1] = (FieldElement *)((uint8_t *)pAddress + 637534208);
        FACTOR0[2] = (FieldElement *)((uint8_t *)pAddress + 654311424);
        FACTOR0[3] = (FieldElement *)((uint8_t *)pAddress + 671088640);
        FACTOR0[4] = (FieldElement *)((uint8_t *)pAddress + 687865856);
        FACTOR0[5] = (FieldElement *)((uint8_t *)pAddress + 704643072);
        FACTOR0[6] = (FieldElement *)((uint8_t *)pAddress + 721420288);
        FACTOR0[7] = (FieldElement *)((uint8_t *)pAddress + 738197504);
        FACTOR1[0] = (FieldElement *)((uint8_t *)pAddress + 754974720);
        FACTOR1[1] = (FieldElement *)((uint8_t *)pAddress + 771751936);
        FACTOR1[2] = (FieldElement *)((uint8_t *)pAddress + 788529152);
        FACTOR1[3] = (FieldElement *)((uint8_t *)pAddress + 805306368);
        FACTOR1[4] = (FieldElement *)((uint8_t *)pAddress + 822083584);
        FACTOR1[5] = (FieldElement *)((uint8_t *)pAddress + 838860800);
        FACTOR1[6] = (FieldElement *)((uint8_t *)pAddress + 855638016);
        FACTOR1[7] = (FieldElement *)((uint8_t *)pAddress + 872415232);
        FACTORV[0] = (FieldElement *)((uint8_t *)pAddress + 889192448);
        FACTORV[1] = (FieldElement *)((uint8_t *)pAddress + 905969664);
        FACTORV[2] = (FieldElement *)((uint8_t *)pAddress + 922746880);
        FACTORV[3] = (FieldElement *)((uint8_t *)pAddress + 939524096);
        FACTORV[4] = (FieldElement *)((uint8_t *)pAddress + 956301312);
        FACTORV[5] = (FieldElement *)((uint8_t *)pAddress + 973078528);
        FACTORV[6] = (FieldElement *)((uint8_t *)pAddress + 989855744);
        FACTORV[7] = (FieldElement *)((uint8_t *)pAddress + 1006632960);
        STEP = (FieldElement *)((uint8_t *)pAddress + 1023410176);
        WR = (uint8_t *)((uint8_t *)pAddress + 1040187392);
        OFFSET = (uint8_t *)((uint8_t *)pAddress + 1042284544);
        RESET = (uint8_t *)((uint8_t *)pAddress + 1044381696);
        RESETL = (uint8_t *)((uint8_t *)pAddress + 1046478848);
        SELW = (uint8_t *)((uint8_t *)pAddress + 1048576000);
    }

    MemAlignConstantPols (void * pAddress, uint64_t degree)
    {
        BYTE2A = (uint8_t *)((uint8_t *)pAddress + 0*degree);
        BYTE2B = (uint8_t *)((uint8_t *)pAddress + 1*degree);
        FACTOR0[0] = (FieldElement *)((uint8_t *)pAddress + 2*degree);
        FACTOR0[1] = (FieldElement *)((uint8_t *)pAddress + 10*degree);
        FACTOR0[2] = (FieldElement *)((uint8_t *)pAddress + 18*degree);
        FACTOR0[3] = (FieldElement *)((uint8_t *)pAddress + 26*degree);
        FACTOR0[4] = (FieldElement *)((uint8_t *)pAddress + 34*degree);
        FACTOR0[5] = (FieldElement *)((uint8_t *)pAddress + 42*degree);
        FACTOR0[6] = (FieldElement *)((uint8_t *)pAddress + 50*degree);
        FACTOR0[7] = (FieldElement *)((uint8_t *)pAddress + 58*degree);
        FACTOR1[0] = (FieldElement *)((uint8_t *)pAddress + 66*degree);
        FACTOR1[1] = (FieldElement *)((uint8_t *)pAddress + 74*degree);
        FACTOR1[2] = (FieldElement *)((uint8_t *)pAddress + 82*degree);
        FACTOR1[3] = (FieldElement *)((uint8_t *)pAddress + 90*degree);
        FACTOR1[4] = (FieldElement *)((uint8_t *)pAddress + 98*degree);
        FACTOR1[5] = (FieldElement *)((uint8_t *)pAddress + 106*degree);
        FACTOR1[6] = (FieldElement *)((uint8_t *)pAddress + 114*degree);
        FACTOR1[7] = (FieldElement *)((uint8_t *)pAddress + 122*degree);
        FACTORV[0] = (FieldElement *)((uint8_t *)pAddress + 130*degree);
        FACTORV[1] = (FieldElement *)((uint8_t *)pAddress + 138*degree);
        FACTORV[2] = (FieldElement *)((uint8_t *)pAddress + 146*degree);
        FACTORV[3] = (FieldElement *)((uint8_t *)pAddress + 154*degree);
        FACTORV[4] = (FieldElement *)((uint8_t *)pAddress + 162*degree);
        FACTORV[5] = (FieldElement *)((uint8_t *)pAddress + 170*degree);
        FACTORV[6] = (FieldElement *)((uint8_t *)pAddress + 178*degree);
        FACTORV[7] = (FieldElement *)((uint8_t *)pAddress + 186*degree);
        STEP = (FieldElement *)((uint8_t *)pAddress + 194*degree);
        WR = (uint8_t *)((uint8_t *)pAddress + 202*degree);
        OFFSET = (uint8_t *)((uint8_t *)pAddress + 203*degree);
        RESET = (uint8_t *)((uint8_t *)pAddress + 204*degree);
        RESETL = (uint8_t *)((uint8_t *)pAddress + 205*degree);
        SELW = (uint8_t *)((uint8_t *)pAddress + 206*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 207; }
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
        BIT19 = (FieldElement *)((uint8_t *)pAddress + 1050673152);
        GL_SIGNED_4BITS = (FieldElement *)((uint8_t *)pAddress + 1067450368);
        GL_SIGNED_18BITS = (FieldElement *)((uint8_t *)pAddress + 1084227584);
        ck[0] = (uint8_t *)((uint8_t *)pAddress + 1101004800);
        ck[1] = (uint8_t *)((uint8_t *)pAddress + 1103101952);
        ck[2] = (uint8_t *)((uint8_t *)pAddress + 1105199104);
        ck[3] = (uint8_t *)((uint8_t *)pAddress + 1107296256);
        ck[4] = (uint8_t *)((uint8_t *)pAddress + 1109393408);
        ck[5] = (uint8_t *)((uint8_t *)pAddress + 1111490560);
        ck[6] = (uint8_t *)((uint8_t *)pAddress + 1113587712);
        ck[7] = (uint8_t *)((uint8_t *)pAddress + 1115684864);
        ck[8] = (uint8_t *)((uint8_t *)pAddress + 1117782016);
        ck[9] = (uint8_t *)((uint8_t *)pAddress + 1119879168);
        ck[10] = (uint8_t *)((uint8_t *)pAddress + 1121976320);
        ck[11] = (uint8_t *)((uint8_t *)pAddress + 1124073472);
        ck[12] = (uint8_t *)((uint8_t *)pAddress + 1126170624);
        ck[13] = (uint8_t *)((uint8_t *)pAddress + 1128267776);
        ck[14] = (uint8_t *)((uint8_t *)pAddress + 1130364928);
        ck[15] = (uint8_t *)((uint8_t *)pAddress + 1132462080);
        ck[16] = (uint8_t *)((uint8_t *)pAddress + 1134559232);
        ck[17] = (uint8_t *)((uint8_t *)pAddress + 1136656384);
        ck[18] = (uint8_t *)((uint8_t *)pAddress + 1138753536);
        ck[19] = (uint8_t *)((uint8_t *)pAddress + 1140850688);
        ck[20] = (uint8_t *)((uint8_t *)pAddress + 1142947840);
        ck[21] = (uint8_t *)((uint8_t *)pAddress + 1145044992);
        ck[22] = (uint8_t *)((uint8_t *)pAddress + 1147142144);
        ck[23] = (uint8_t *)((uint8_t *)pAddress + 1149239296);
        ck[24] = (uint8_t *)((uint8_t *)pAddress + 1151336448);
        ck[25] = (uint8_t *)((uint8_t *)pAddress + 1153433600);
        ck[26] = (uint8_t *)((uint8_t *)pAddress + 1155530752);
        ck[27] = (uint8_t *)((uint8_t *)pAddress + 1157627904);
        ck[28] = (uint8_t *)((uint8_t *)pAddress + 1159725056);
        ck[29] = (uint8_t *)((uint8_t *)pAddress + 1161822208);
        ck[30] = (uint8_t *)((uint8_t *)pAddress + 1163919360);
        ck[31] = (uint8_t *)((uint8_t *)pAddress + 1166016512);
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
        P_OPCODE = (uint8_t *)((uint8_t *)pAddress + 1168113664);
        P_A = (uint8_t *)((uint8_t *)pAddress + 1170210816);
        P_B = (uint8_t *)((uint8_t *)pAddress + 1172307968);
        P_CIN = (uint8_t *)((uint8_t *)pAddress + 1174405120);
        P_LAST = (uint8_t *)((uint8_t *)pAddress + 1176502272);
        P_USE_CARRY = (uint8_t *)((uint8_t *)pAddress + 1178599424);
        P_C = (uint8_t *)((uint8_t *)pAddress + 1180696576);
        P_COUT = (uint8_t *)((uint8_t *)pAddress + 1182793728);
        RESET = (uint8_t *)((uint8_t *)pAddress + 1184890880);
        FACTOR[0] = (uint32_t *)((uint8_t *)pAddress + 1186988032);
        FACTOR[1] = (uint32_t *)((uint8_t *)pAddress + 1203765248);
        FACTOR[2] = (uint32_t *)((uint8_t *)pAddress + 1220542464);
        FACTOR[3] = (uint32_t *)((uint8_t *)pAddress + 1237319680);
        FACTOR[4] = (uint32_t *)((uint8_t *)pAddress + 1254096896);
        FACTOR[5] = (uint32_t *)((uint8_t *)pAddress + 1270874112);
        FACTOR[6] = (uint32_t *)((uint8_t *)pAddress + 1287651328);
        FACTOR[7] = (uint32_t *)((uint8_t *)pAddress + 1304428544);
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
        FACTOR[1] = (uint32_t *)((uint8_t *)pAddress + 17*degree);
        FACTOR[2] = (uint32_t *)((uint8_t *)pAddress + 25*degree);
        FACTOR[3] = (uint32_t *)((uint8_t *)pAddress + 33*degree);
        FACTOR[4] = (uint32_t *)((uint8_t *)pAddress + 41*degree);
        FACTOR[5] = (uint32_t *)((uint8_t *)pAddress + 49*degree);
        FACTOR[6] = (uint32_t *)((uint8_t *)pAddress + 57*degree);
        FACTOR[7] = (uint32_t *)((uint8_t *)pAddress + 65*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 73; }
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
        LAST = (FieldElement *)((uint8_t *)pAddress + 1321205760);
        LATCH = (FieldElement *)((uint8_t *)pAddress + 1337982976);
        LASTBLOCK = (FieldElement *)((uint8_t *)pAddress + 1354760192);
        PARTIAL = (FieldElement *)((uint8_t *)pAddress + 1371537408);
        C[0] = (FieldElement *)((uint8_t *)pAddress + 1388314624);
        C[1] = (FieldElement *)((uint8_t *)pAddress + 1405091840);
        C[2] = (FieldElement *)((uint8_t *)pAddress + 1421869056);
        C[3] = (FieldElement *)((uint8_t *)pAddress + 1438646272);
        C[4] = (FieldElement *)((uint8_t *)pAddress + 1455423488);
        C[5] = (FieldElement *)((uint8_t *)pAddress + 1472200704);
        C[6] = (FieldElement *)((uint8_t *)pAddress + 1488977920);
        C[7] = (FieldElement *)((uint8_t *)pAddress + 1505755136);
        C[8] = (FieldElement *)((uint8_t *)pAddress + 1522532352);
        C[9] = (FieldElement *)((uint8_t *)pAddress + 1539309568);
        C[10] = (FieldElement *)((uint8_t *)pAddress + 1556086784);
        C[11] = (FieldElement *)((uint8_t *)pAddress + 1572864000);
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
        F[0] = (FieldElement *)((uint8_t *)pAddress + 1589641216);
        F[1] = (FieldElement *)((uint8_t *)pAddress + 1606418432);
        F[2] = (FieldElement *)((uint8_t *)pAddress + 1623195648);
        F[3] = (FieldElement *)((uint8_t *)pAddress + 1639972864);
        F[4] = (FieldElement *)((uint8_t *)pAddress + 1656750080);
        F[5] = (FieldElement *)((uint8_t *)pAddress + 1673527296);
        F[6] = (FieldElement *)((uint8_t *)pAddress + 1690304512);
        F[7] = (FieldElement *)((uint8_t *)pAddress + 1707081728);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 1723858944);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 1740636160);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 1757413376);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 1774190592);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 1790967808);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 1807745024);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 1824522240);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 1841299456);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 1858076672);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 1874853888);
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
        rHash = (FieldElement *)((uint8_t *)pAddress + 1891631104);
        rHashType = (FieldElement *)((uint8_t *)pAddress + 1908408320);
        rLatchGet = (FieldElement *)((uint8_t *)pAddress + 1925185536);
        rLatchSet = (FieldElement *)((uint8_t *)pAddress + 1941962752);
        rClimbRkey = (FieldElement *)((uint8_t *)pAddress + 1958739968);
        rClimbSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 1975517184);
        rClimbSiblingRkeyN = (FieldElement *)((uint8_t *)pAddress + 1992294400);
        rRotateLevel = (FieldElement *)((uint8_t *)pAddress + 2009071616);
        rJmpz = (FieldElement *)((uint8_t *)pAddress + 2025848832);
        rJmp = (FieldElement *)((uint8_t *)pAddress + 2042626048);
        rConst0 = (FieldElement *)((uint8_t *)pAddress + 2059403264);
        rConst1 = (FieldElement *)((uint8_t *)pAddress + 2076180480);
        rConst2 = (FieldElement *)((uint8_t *)pAddress + 2092957696);
        rConst3 = (FieldElement *)((uint8_t *)pAddress + 2109734912);
        rAddress = (FieldElement *)((uint8_t *)pAddress + 2126512128);
        rLine = (FieldElement *)((uint8_t *)pAddress + 2143289344);
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
        Value3 = (FieldElement *)((uint8_t *)pAddress + 2160066560);
        Value3Norm = (FieldElement *)((uint8_t *)pAddress + 2176843776);
        Gate9Type = (FieldElement *)((uint8_t *)pAddress + 2193620992);
        Gate9A = (FieldElement *)((uint8_t *)pAddress + 2210398208);
        Gate9B = (FieldElement *)((uint8_t *)pAddress + 2227175424);
        Gate9C = (FieldElement *)((uint8_t *)pAddress + 2243952640);
        Latch = (FieldElement *)((uint8_t *)pAddress + 2260729856);
        Factor = (FieldElement *)((uint8_t *)pAddress + 2277507072);
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
        ConnA = (FieldElement *)((uint8_t *)pAddress + 2294284288);
        ConnB = (FieldElement *)((uint8_t *)pAddress + 2311061504);
        ConnC = (FieldElement *)((uint8_t *)pAddress + 2327838720);
        NormalizedGate = (FieldElement *)((uint8_t *)pAddress + 2344615936);
        GateType = (FieldElement *)((uint8_t *)pAddress + 2361393152);
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
        Field9latch = (FieldElement *)((uint8_t *)pAddress + 2378170368);
        Factor = (FieldElement *)((uint8_t *)pAddress + 2394947584);
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
        r8Id = (FieldElement *)((uint8_t *)pAddress + 2411724800);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 2428502016);
        latchR8 = (FieldElement *)((uint8_t *)pAddress + 2445279232);
        Fr8 = (FieldElement *)((uint8_t *)pAddress + 2462056448);
        rBitValid = (FieldElement *)((uint8_t *)pAddress + 2478833664);
        latchSOut = (FieldElement *)((uint8_t *)pAddress + 2495610880);
        FSOut0 = (FieldElement *)((uint8_t *)pAddress + 2512388096);
        FSOut1 = (FieldElement *)((uint8_t *)pAddress + 2529165312);
        FSOut2 = (FieldElement *)((uint8_t *)pAddress + 2545942528);
        FSOut3 = (FieldElement *)((uint8_t *)pAddress + 2562719744);
        FSOut4 = (FieldElement *)((uint8_t *)pAddress + 2579496960);
        FSOut5 = (FieldElement *)((uint8_t *)pAddress + 2596274176);
        FSOut6 = (FieldElement *)((uint8_t *)pAddress + 2613051392);
        FSOut7 = (FieldElement *)((uint8_t *)pAddress + 2629828608);
        ConnSOutBit = (FieldElement *)((uint8_t *)pAddress + 2646605824);
        ConnSInBit = (FieldElement *)((uint8_t *)pAddress + 2663383040);
        ConnNine2OneBit = (FieldElement *)((uint8_t *)pAddress + 2680160256);
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
        r8Id = (FieldElement *)((uint8_t *)pAddress + 2696937472);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 2713714688);
        lastBlockLatch = (FieldElement *)((uint8_t *)pAddress + 2730491904);
        r8valid = (FieldElement *)((uint8_t *)pAddress + 2747269120);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 2764046336);
        forceLastHash = (FieldElement *)((uint8_t *)pAddress + 2780823552);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 2797600768);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 2814377984);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 2831155200);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 2847932416);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 2864709632);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 2881486848);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 2898264064);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 2915041280);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 2931818496);
        crValid = (FieldElement *)((uint8_t *)pAddress + 2948595712);
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
        INCS = (FieldElement *)((uint8_t *)pAddress + 2965372928);
        ISNOTLAST = (FieldElement *)((uint8_t *)pAddress + 2982150144);
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
        STEP = (uint32_t *)((uint8_t *)pAddress + 2998927360);
    }

    MainConstantPols (void * pAddress, uint64_t degree)
    {
        STEP = (uint32_t *)((uint8_t *)pAddress + 0*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 8; }
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

    static uint64_t size (void) { return 3015704576; }
};

#endif // CONSTANT_POLS_HPP
