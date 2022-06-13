#ifndef CONSTANT_POLS_HPP
#define CONSTANT_POLS_HPP

#include <cstdint>
#include "goldilocks/goldilocks_base_field.hpp"

class GeneratedPol
{
private:
    Goldilocks::Element * pData;
public:
    GeneratedPol() : pData(NULL) {};
    Goldilocks::Element & operator[](int i) { return pData[i*252]; };
    Goldilocks::Element * operator=(Goldilocks::Element * pAddress) { pData = pAddress; return pData; };
};

class GlobalConstantPols
{
public:
    GeneratedPol L1;
    GeneratedPol BYTE;
    GeneratedPol BYTE2;

    GlobalConstantPols (void * pAddress)
    {
        L1 = (Goldilocks::Element *)((uint8_t *)pAddress + 0);
        BYTE = (Goldilocks::Element *)((uint8_t *)pAddress + 8);
        BYTE2 = (Goldilocks::Element *)((uint8_t *)pAddress + 16);
    }

    GlobalConstantPols (void * pAddress, uint64_t degree)
    {
        L1 = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        BYTE = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        BYTE2 = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 24; }
};

class RomConstantPols
{
public:
    GeneratedPol CONST0;
    GeneratedPol CONST1;
    GeneratedPol CONST2;
    GeneratedPol CONST3;
    GeneratedPol CONST4;
    GeneratedPol CONST5;
    GeneratedPol CONST6;
    GeneratedPol CONST7;
    GeneratedPol offset;
    GeneratedPol inA;
    GeneratedPol inB;
    GeneratedPol inC;
    GeneratedPol inD;
    GeneratedPol inE;
    GeneratedPol inSR;
    GeneratedPol inFREE;
    GeneratedPol inCTX;
    GeneratedPol inSP;
    GeneratedPol inPC;
    GeneratedPol inGAS;
    GeneratedPol inMAXMEM;
    GeneratedPol inHASHPOS;
    GeneratedPol inSTEP;
    GeneratedPol inRR;
    GeneratedPol setA;
    GeneratedPol setB;
    GeneratedPol setC;
    GeneratedPol setD;
    GeneratedPol setE;
    GeneratedPol setSR;
    GeneratedPol setCTX;
    GeneratedPol setSP;
    GeneratedPol setPC;
    GeneratedPol setGAS;
    GeneratedPol setMAXMEM;
    GeneratedPol setHASHPOS;
    GeneratedPol JMP;
    GeneratedPol JMPN;
    GeneratedPol JMPC;
    GeneratedPol setRR;
    GeneratedPol incStack;
    GeneratedPol incCode;
    GeneratedPol isStack;
    GeneratedPol isCode;
    GeneratedPol isMem;
    GeneratedPol ind;
    GeneratedPol indRR;
    GeneratedPol useCTX;
    GeneratedPol mOp;
    GeneratedPol mWR;
    GeneratedPol sWR;
    GeneratedPol sRD;
    GeneratedPol arith;
    GeneratedPol arithEq0;
    GeneratedPol arithEq1;
    GeneratedPol arithEq2;
    GeneratedPol arithEq3;
    GeneratedPol memAlign;
    GeneratedPol memAlignWR;
    GeneratedPol memAlignWR8;
    GeneratedPol hashK;
    GeneratedPol hashKLen;
    GeneratedPol hashKDigest;
    GeneratedPol hashP;
    GeneratedPol hashPLen;
    GeneratedPol hashPDigest;
    GeneratedPol bin;
    GeneratedPol binOpcode;
    GeneratedPol assert;
    GeneratedPol line;

    RomConstantPols (void * pAddress)
    {
        CONST0 = (Goldilocks::Element *)((uint8_t *)pAddress + 24);
        CONST1 = (Goldilocks::Element *)((uint8_t *)pAddress + 32);
        CONST2 = (Goldilocks::Element *)((uint8_t *)pAddress + 40);
        CONST3 = (Goldilocks::Element *)((uint8_t *)pAddress + 48);
        CONST4 = (Goldilocks::Element *)((uint8_t *)pAddress + 56);
        CONST5 = (Goldilocks::Element *)((uint8_t *)pAddress + 64);
        CONST6 = (Goldilocks::Element *)((uint8_t *)pAddress + 72);
        CONST7 = (Goldilocks::Element *)((uint8_t *)pAddress + 80);
        offset = (Goldilocks::Element *)((uint8_t *)pAddress + 88);
        inA = (Goldilocks::Element *)((uint8_t *)pAddress + 96);
        inB = (Goldilocks::Element *)((uint8_t *)pAddress + 104);
        inC = (Goldilocks::Element *)((uint8_t *)pAddress + 112);
        inD = (Goldilocks::Element *)((uint8_t *)pAddress + 120);
        inE = (Goldilocks::Element *)((uint8_t *)pAddress + 128);
        inSR = (Goldilocks::Element *)((uint8_t *)pAddress + 136);
        inFREE = (Goldilocks::Element *)((uint8_t *)pAddress + 144);
        inCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 152);
        inSP = (Goldilocks::Element *)((uint8_t *)pAddress + 160);
        inPC = (Goldilocks::Element *)((uint8_t *)pAddress + 168);
        inGAS = (Goldilocks::Element *)((uint8_t *)pAddress + 176);
        inMAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 184);
        inHASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 192);
        inSTEP = (Goldilocks::Element *)((uint8_t *)pAddress + 200);
        inRR = (Goldilocks::Element *)((uint8_t *)pAddress + 208);
        setA = (Goldilocks::Element *)((uint8_t *)pAddress + 216);
        setB = (Goldilocks::Element *)((uint8_t *)pAddress + 224);
        setC = (Goldilocks::Element *)((uint8_t *)pAddress + 232);
        setD = (Goldilocks::Element *)((uint8_t *)pAddress + 240);
        setE = (Goldilocks::Element *)((uint8_t *)pAddress + 248);
        setSR = (Goldilocks::Element *)((uint8_t *)pAddress + 256);
        setCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 264);
        setSP = (Goldilocks::Element *)((uint8_t *)pAddress + 272);
        setPC = (Goldilocks::Element *)((uint8_t *)pAddress + 280);
        setGAS = (Goldilocks::Element *)((uint8_t *)pAddress + 288);
        setMAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 296);
        setHASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 304);
        JMP = (Goldilocks::Element *)((uint8_t *)pAddress + 312);
        JMPN = (Goldilocks::Element *)((uint8_t *)pAddress + 320);
        JMPC = (Goldilocks::Element *)((uint8_t *)pAddress + 328);
        setRR = (Goldilocks::Element *)((uint8_t *)pAddress + 336);
        incStack = (Goldilocks::Element *)((uint8_t *)pAddress + 344);
        incCode = (Goldilocks::Element *)((uint8_t *)pAddress + 352);
        isStack = (Goldilocks::Element *)((uint8_t *)pAddress + 360);
        isCode = (Goldilocks::Element *)((uint8_t *)pAddress + 368);
        isMem = (Goldilocks::Element *)((uint8_t *)pAddress + 376);
        ind = (Goldilocks::Element *)((uint8_t *)pAddress + 384);
        indRR = (Goldilocks::Element *)((uint8_t *)pAddress + 392);
        useCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 400);
        mOp = (Goldilocks::Element *)((uint8_t *)pAddress + 408);
        mWR = (Goldilocks::Element *)((uint8_t *)pAddress + 416);
        sWR = (Goldilocks::Element *)((uint8_t *)pAddress + 424);
        sRD = (Goldilocks::Element *)((uint8_t *)pAddress + 432);
        arith = (Goldilocks::Element *)((uint8_t *)pAddress + 440);
        arithEq0 = (Goldilocks::Element *)((uint8_t *)pAddress + 448);
        arithEq1 = (Goldilocks::Element *)((uint8_t *)pAddress + 456);
        arithEq2 = (Goldilocks::Element *)((uint8_t *)pAddress + 464);
        arithEq3 = (Goldilocks::Element *)((uint8_t *)pAddress + 472);
        memAlign = (Goldilocks::Element *)((uint8_t *)pAddress + 480);
        memAlignWR = (Goldilocks::Element *)((uint8_t *)pAddress + 488);
        memAlignWR8 = (Goldilocks::Element *)((uint8_t *)pAddress + 496);
        hashK = (Goldilocks::Element *)((uint8_t *)pAddress + 504);
        hashKLen = (Goldilocks::Element *)((uint8_t *)pAddress + 512);
        hashKDigest = (Goldilocks::Element *)((uint8_t *)pAddress + 520);
        hashP = (Goldilocks::Element *)((uint8_t *)pAddress + 528);
        hashPLen = (Goldilocks::Element *)((uint8_t *)pAddress + 536);
        hashPDigest = (Goldilocks::Element *)((uint8_t *)pAddress + 544);
        bin = (Goldilocks::Element *)((uint8_t *)pAddress + 552);
        binOpcode = (Goldilocks::Element *)((uint8_t *)pAddress + 560);
        assert = (Goldilocks::Element *)((uint8_t *)pAddress + 568);
        line = (Goldilocks::Element *)((uint8_t *)pAddress + 576);
    }

    RomConstantPols (void * pAddress, uint64_t degree)
    {
        CONST0 = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        CONST1 = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        CONST2 = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        CONST3 = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        CONST4 = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        CONST5 = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        CONST6 = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        CONST7 = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        offset = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        inA = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        inB = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        inC = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        inD = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        inE = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        inSR = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        inFREE = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        inCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
        inSP = (Goldilocks::Element *)((uint8_t *)pAddress + 136*degree);
        inPC = (Goldilocks::Element *)((uint8_t *)pAddress + 144*degree);
        inGAS = (Goldilocks::Element *)((uint8_t *)pAddress + 152*degree);
        inMAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 160*degree);
        inHASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 168*degree);
        inSTEP = (Goldilocks::Element *)((uint8_t *)pAddress + 176*degree);
        inRR = (Goldilocks::Element *)((uint8_t *)pAddress + 184*degree);
        setA = (Goldilocks::Element *)((uint8_t *)pAddress + 192*degree);
        setB = (Goldilocks::Element *)((uint8_t *)pAddress + 200*degree);
        setC = (Goldilocks::Element *)((uint8_t *)pAddress + 208*degree);
        setD = (Goldilocks::Element *)((uint8_t *)pAddress + 216*degree);
        setE = (Goldilocks::Element *)((uint8_t *)pAddress + 224*degree);
        setSR = (Goldilocks::Element *)((uint8_t *)pAddress + 232*degree);
        setCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 240*degree);
        setSP = (Goldilocks::Element *)((uint8_t *)pAddress + 248*degree);
        setPC = (Goldilocks::Element *)((uint8_t *)pAddress + 256*degree);
        setGAS = (Goldilocks::Element *)((uint8_t *)pAddress + 264*degree);
        setMAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 272*degree);
        setHASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 280*degree);
        JMP = (Goldilocks::Element *)((uint8_t *)pAddress + 288*degree);
        JMPN = (Goldilocks::Element *)((uint8_t *)pAddress + 296*degree);
        JMPC = (Goldilocks::Element *)((uint8_t *)pAddress + 304*degree);
        setRR = (Goldilocks::Element *)((uint8_t *)pAddress + 312*degree);
        incStack = (Goldilocks::Element *)((uint8_t *)pAddress + 320*degree);
        incCode = (Goldilocks::Element *)((uint8_t *)pAddress + 328*degree);
        isStack = (Goldilocks::Element *)((uint8_t *)pAddress + 336*degree);
        isCode = (Goldilocks::Element *)((uint8_t *)pAddress + 344*degree);
        isMem = (Goldilocks::Element *)((uint8_t *)pAddress + 352*degree);
        ind = (Goldilocks::Element *)((uint8_t *)pAddress + 360*degree);
        indRR = (Goldilocks::Element *)((uint8_t *)pAddress + 368*degree);
        useCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 376*degree);
        mOp = (Goldilocks::Element *)((uint8_t *)pAddress + 384*degree);
        mWR = (Goldilocks::Element *)((uint8_t *)pAddress + 392*degree);
        sWR = (Goldilocks::Element *)((uint8_t *)pAddress + 400*degree);
        sRD = (Goldilocks::Element *)((uint8_t *)pAddress + 408*degree);
        arith = (Goldilocks::Element *)((uint8_t *)pAddress + 416*degree);
        arithEq0 = (Goldilocks::Element *)((uint8_t *)pAddress + 424*degree);
        arithEq1 = (Goldilocks::Element *)((uint8_t *)pAddress + 432*degree);
        arithEq2 = (Goldilocks::Element *)((uint8_t *)pAddress + 440*degree);
        arithEq3 = (Goldilocks::Element *)((uint8_t *)pAddress + 448*degree);
        memAlign = (Goldilocks::Element *)((uint8_t *)pAddress + 456*degree);
        memAlignWR = (Goldilocks::Element *)((uint8_t *)pAddress + 464*degree);
        memAlignWR8 = (Goldilocks::Element *)((uint8_t *)pAddress + 472*degree);
        hashK = (Goldilocks::Element *)((uint8_t *)pAddress + 480*degree);
        hashKLen = (Goldilocks::Element *)((uint8_t *)pAddress + 488*degree);
        hashKDigest = (Goldilocks::Element *)((uint8_t *)pAddress + 496*degree);
        hashP = (Goldilocks::Element *)((uint8_t *)pAddress + 504*degree);
        hashPLen = (Goldilocks::Element *)((uint8_t *)pAddress + 512*degree);
        hashPDigest = (Goldilocks::Element *)((uint8_t *)pAddress + 520*degree);
        bin = (Goldilocks::Element *)((uint8_t *)pAddress + 528*degree);
        binOpcode = (Goldilocks::Element *)((uint8_t *)pAddress + 536*degree);
        assert = (Goldilocks::Element *)((uint8_t *)pAddress + 544*degree);
        line = (Goldilocks::Element *)((uint8_t *)pAddress + 552*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 560; }
};

class Byte4ConstantPols
{
public:
    GeneratedPol SET;

    Byte4ConstantPols (void * pAddress)
    {
        SET = (Goldilocks::Element *)((uint8_t *)pAddress + 584);
    }

    Byte4ConstantPols (void * pAddress, uint64_t degree)
    {
        SET = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 8; }
};

class MemAlignConstantPols
{
public:
    GeneratedPol BYTE2A;
    GeneratedPol BYTE2B;
    GeneratedPol BYTE_C3072;
    GeneratedPol FACTOR[8];
    GeneratedPol FACTORV[8];
    GeneratedPol STEP;
    GeneratedPol WR256;
    GeneratedPol WR8;
    GeneratedPol OFFSET;
    GeneratedPol RESET;
    GeneratedPol SELM1;

    MemAlignConstantPols (void * pAddress)
    {
        BYTE2A = (Goldilocks::Element *)((uint8_t *)pAddress + 592);
        BYTE2B = (Goldilocks::Element *)((uint8_t *)pAddress + 600);
        BYTE_C3072 = (Goldilocks::Element *)((uint8_t *)pAddress + 608);
        FACTOR[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 616);
        FACTOR[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 624);
        FACTOR[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 632);
        FACTOR[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 640);
        FACTOR[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 648);
        FACTOR[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 656);
        FACTOR[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 664);
        FACTOR[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 672);
        FACTORV[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 680);
        FACTORV[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 688);
        FACTORV[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 696);
        FACTORV[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 704);
        FACTORV[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 712);
        FACTORV[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 720);
        FACTORV[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 728);
        FACTORV[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 736);
        STEP = (Goldilocks::Element *)((uint8_t *)pAddress + 744);
        WR256 = (Goldilocks::Element *)((uint8_t *)pAddress + 752);
        WR8 = (Goldilocks::Element *)((uint8_t *)pAddress + 760);
        OFFSET = (Goldilocks::Element *)((uint8_t *)pAddress + 768);
        RESET = (Goldilocks::Element *)((uint8_t *)pAddress + 776);
        SELM1 = (Goldilocks::Element *)((uint8_t *)pAddress + 784);
    }

    MemAlignConstantPols (void * pAddress, uint64_t degree)
    {
        BYTE2A = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        BYTE2B = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        BYTE_C3072 = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        FACTOR[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        FACTOR[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        FACTOR[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        FACTOR[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        FACTOR[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        FACTOR[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        FACTOR[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        FACTOR[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        FACTORV[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        FACTORV[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        FACTORV[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        FACTORV[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        FACTORV[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        FACTORV[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
        FACTORV[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 136*degree);
        FACTORV[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 144*degree);
        STEP = (Goldilocks::Element *)((uint8_t *)pAddress + 152*degree);
        WR256 = (Goldilocks::Element *)((uint8_t *)pAddress + 160*degree);
        WR8 = (Goldilocks::Element *)((uint8_t *)pAddress + 168*degree);
        OFFSET = (Goldilocks::Element *)((uint8_t *)pAddress + 176*degree);
        RESET = (Goldilocks::Element *)((uint8_t *)pAddress + 184*degree);
        SELM1 = (Goldilocks::Element *)((uint8_t *)pAddress + 192*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 200; }
};

class ArithConstantPols
{
public:
    GeneratedPol BIT19;
    GeneratedPol GL_SIGNED_4BITS;
    GeneratedPol GL_SIGNED_18BITS;
    GeneratedPol ck[32];

    ArithConstantPols (void * pAddress)
    {
        BIT19 = (Goldilocks::Element *)((uint8_t *)pAddress + 792);
        GL_SIGNED_4BITS = (Goldilocks::Element *)((uint8_t *)pAddress + 800);
        GL_SIGNED_18BITS = (Goldilocks::Element *)((uint8_t *)pAddress + 808);
        ck[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 816);
        ck[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 824);
        ck[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 832);
        ck[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 840);
        ck[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 848);
        ck[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 856);
        ck[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 864);
        ck[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 872);
        ck[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 880);
        ck[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 888);
        ck[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 896);
        ck[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 904);
        ck[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 912);
        ck[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 920);
        ck[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 928);
        ck[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 936);
        ck[16] = (Goldilocks::Element *)((uint8_t *)pAddress + 944);
        ck[17] = (Goldilocks::Element *)((uint8_t *)pAddress + 952);
        ck[18] = (Goldilocks::Element *)((uint8_t *)pAddress + 960);
        ck[19] = (Goldilocks::Element *)((uint8_t *)pAddress + 968);
        ck[20] = (Goldilocks::Element *)((uint8_t *)pAddress + 976);
        ck[21] = (Goldilocks::Element *)((uint8_t *)pAddress + 984);
        ck[22] = (Goldilocks::Element *)((uint8_t *)pAddress + 992);
        ck[23] = (Goldilocks::Element *)((uint8_t *)pAddress + 1000);
        ck[24] = (Goldilocks::Element *)((uint8_t *)pAddress + 1008);
        ck[25] = (Goldilocks::Element *)((uint8_t *)pAddress + 1016);
        ck[26] = (Goldilocks::Element *)((uint8_t *)pAddress + 1024);
        ck[27] = (Goldilocks::Element *)((uint8_t *)pAddress + 1032);
        ck[28] = (Goldilocks::Element *)((uint8_t *)pAddress + 1040);
        ck[29] = (Goldilocks::Element *)((uint8_t *)pAddress + 1048);
        ck[30] = (Goldilocks::Element *)((uint8_t *)pAddress + 1056);
        ck[31] = (Goldilocks::Element *)((uint8_t *)pAddress + 1064);
    }

    ArithConstantPols (void * pAddress, uint64_t degree)
    {
        BIT19 = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        GL_SIGNED_4BITS = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        GL_SIGNED_18BITS = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        ck[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        ck[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        ck[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        ck[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        ck[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        ck[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        ck[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        ck[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        ck[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        ck[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        ck[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        ck[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        ck[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        ck[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
        ck[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 136*degree);
        ck[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 144*degree);
        ck[16] = (Goldilocks::Element *)((uint8_t *)pAddress + 152*degree);
        ck[17] = (Goldilocks::Element *)((uint8_t *)pAddress + 160*degree);
        ck[18] = (Goldilocks::Element *)((uint8_t *)pAddress + 168*degree);
        ck[19] = (Goldilocks::Element *)((uint8_t *)pAddress + 176*degree);
        ck[20] = (Goldilocks::Element *)((uint8_t *)pAddress + 184*degree);
        ck[21] = (Goldilocks::Element *)((uint8_t *)pAddress + 192*degree);
        ck[22] = (Goldilocks::Element *)((uint8_t *)pAddress + 200*degree);
        ck[23] = (Goldilocks::Element *)((uint8_t *)pAddress + 208*degree);
        ck[24] = (Goldilocks::Element *)((uint8_t *)pAddress + 216*degree);
        ck[25] = (Goldilocks::Element *)((uint8_t *)pAddress + 224*degree);
        ck[26] = (Goldilocks::Element *)((uint8_t *)pAddress + 232*degree);
        ck[27] = (Goldilocks::Element *)((uint8_t *)pAddress + 240*degree);
        ck[28] = (Goldilocks::Element *)((uint8_t *)pAddress + 248*degree);
        ck[29] = (Goldilocks::Element *)((uint8_t *)pAddress + 256*degree);
        ck[30] = (Goldilocks::Element *)((uint8_t *)pAddress + 264*degree);
        ck[31] = (Goldilocks::Element *)((uint8_t *)pAddress + 272*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 280; }
};

class BinaryConstantPols
{
public:
    GeneratedPol P_OPCODE;
    GeneratedPol P_A;
    GeneratedPol P_B;
    GeneratedPol P_CIN;
    GeneratedPol P_LAST;
    GeneratedPol P_USE_CARRY;
    GeneratedPol P_C;
    GeneratedPol P_COUT;
    GeneratedPol RESET;
    GeneratedPol FACTOR[8];

    BinaryConstantPols (void * pAddress)
    {
        P_OPCODE = (Goldilocks::Element *)((uint8_t *)pAddress + 1072);
        P_A = (Goldilocks::Element *)((uint8_t *)pAddress + 1080);
        P_B = (Goldilocks::Element *)((uint8_t *)pAddress + 1088);
        P_CIN = (Goldilocks::Element *)((uint8_t *)pAddress + 1096);
        P_LAST = (Goldilocks::Element *)((uint8_t *)pAddress + 1104);
        P_USE_CARRY = (Goldilocks::Element *)((uint8_t *)pAddress + 1112);
        P_C = (Goldilocks::Element *)((uint8_t *)pAddress + 1120);
        P_COUT = (Goldilocks::Element *)((uint8_t *)pAddress + 1128);
        RESET = (Goldilocks::Element *)((uint8_t *)pAddress + 1136);
        FACTOR[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1144);
        FACTOR[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1152);
        FACTOR[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1160);
        FACTOR[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1168);
        FACTOR[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1176);
        FACTOR[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1184);
        FACTOR[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1192);
        FACTOR[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1200);
    }

    BinaryConstantPols (void * pAddress, uint64_t degree)
    {
        P_OPCODE = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        P_A = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        P_B = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        P_CIN = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        P_LAST = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        P_USE_CARRY = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        P_C = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        P_COUT = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        RESET = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        FACTOR[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        FACTOR[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        FACTOR[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        FACTOR[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        FACTOR[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        FACTOR[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        FACTOR[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        FACTOR[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 136; }
};

class PoseidonGConstantPols
{
public:
    GeneratedPol LAST;
    GeneratedPol LATCH;
    GeneratedPol LASTBLOCK;
    GeneratedPol PARTIAL;
    GeneratedPol C[12];

    PoseidonGConstantPols (void * pAddress)
    {
        LAST = (Goldilocks::Element *)((uint8_t *)pAddress + 1208);
        LATCH = (Goldilocks::Element *)((uint8_t *)pAddress + 1216);
        LASTBLOCK = (Goldilocks::Element *)((uint8_t *)pAddress + 1224);
        PARTIAL = (Goldilocks::Element *)((uint8_t *)pAddress + 1232);
        C[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1240);
        C[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1248);
        C[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1256);
        C[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1264);
        C[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1272);
        C[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1280);
        C[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1288);
        C[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1296);
        C[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 1304);
        C[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 1312);
        C[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 1320);
        C[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 1328);
    }

    PoseidonGConstantPols (void * pAddress, uint64_t degree)
    {
        LAST = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        LATCH = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        LASTBLOCK = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        PARTIAL = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        C[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        C[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        C[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        C[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        C[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        C[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        C[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        C[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        C[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        C[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        C[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        C[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 128; }
};

class PaddingPGConstantPols
{
public:
    GeneratedPol F[8];
    GeneratedPol lastBlock;
    GeneratedPol k_crOffset;
    GeneratedPol k_crF0;
    GeneratedPol k_crF1;
    GeneratedPol k_crF2;
    GeneratedPol k_crF3;
    GeneratedPol k_crF4;
    GeneratedPol k_crF5;
    GeneratedPol k_crF6;
    GeneratedPol k_crF7;

    PaddingPGConstantPols (void * pAddress)
    {
        F[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1336);
        F[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1344);
        F[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1352);
        F[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1360);
        F[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1368);
        F[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1376);
        F[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1384);
        F[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1392);
        lastBlock = (Goldilocks::Element *)((uint8_t *)pAddress + 1400);
        k_crOffset = (Goldilocks::Element *)((uint8_t *)pAddress + 1408);
        k_crF0 = (Goldilocks::Element *)((uint8_t *)pAddress + 1416);
        k_crF1 = (Goldilocks::Element *)((uint8_t *)pAddress + 1424);
        k_crF2 = (Goldilocks::Element *)((uint8_t *)pAddress + 1432);
        k_crF3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1440);
        k_crF4 = (Goldilocks::Element *)((uint8_t *)pAddress + 1448);
        k_crF5 = (Goldilocks::Element *)((uint8_t *)pAddress + 1456);
        k_crF6 = (Goldilocks::Element *)((uint8_t *)pAddress + 1464);
        k_crF7 = (Goldilocks::Element *)((uint8_t *)pAddress + 1472);
    }

    PaddingPGConstantPols (void * pAddress, uint64_t degree)
    {
        F[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        F[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        F[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        F[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        F[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        F[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        F[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        F[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        lastBlock = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        k_crOffset = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        k_crF0 = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        k_crF1 = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        k_crF2 = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        k_crF3 = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        k_crF4 = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        k_crF5 = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        k_crF6 = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
        k_crF7 = (Goldilocks::Element *)((uint8_t *)pAddress + 136*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 144; }
};

class StorageConstantPols
{
public:
    GeneratedPol rHash;
    GeneratedPol rHashType;
    GeneratedPol rLatchGet;
    GeneratedPol rLatchSet;
    GeneratedPol rClimbRkey;
    GeneratedPol rClimbSiblingRkey;
    GeneratedPol rClimbSiblingRkeyN;
    GeneratedPol rRotateLevel;
    GeneratedPol rJmpz;
    GeneratedPol rJmp;
    GeneratedPol rConst0;
    GeneratedPol rConst1;
    GeneratedPol rConst2;
    GeneratedPol rConst3;
    GeneratedPol rAddress;
    GeneratedPol rLine;

    StorageConstantPols (void * pAddress)
    {
        rHash = (Goldilocks::Element *)((uint8_t *)pAddress + 1480);
        rHashType = (Goldilocks::Element *)((uint8_t *)pAddress + 1488);
        rLatchGet = (Goldilocks::Element *)((uint8_t *)pAddress + 1496);
        rLatchSet = (Goldilocks::Element *)((uint8_t *)pAddress + 1504);
        rClimbRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 1512);
        rClimbSiblingRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 1520);
        rClimbSiblingRkeyN = (Goldilocks::Element *)((uint8_t *)pAddress + 1528);
        rRotateLevel = (Goldilocks::Element *)((uint8_t *)pAddress + 1536);
        rJmpz = (Goldilocks::Element *)((uint8_t *)pAddress + 1544);
        rJmp = (Goldilocks::Element *)((uint8_t *)pAddress + 1552);
        rConst0 = (Goldilocks::Element *)((uint8_t *)pAddress + 1560);
        rConst1 = (Goldilocks::Element *)((uint8_t *)pAddress + 1568);
        rConst2 = (Goldilocks::Element *)((uint8_t *)pAddress + 1576);
        rConst3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1584);
        rAddress = (Goldilocks::Element *)((uint8_t *)pAddress + 1592);
        rLine = (Goldilocks::Element *)((uint8_t *)pAddress + 1600);
    }

    StorageConstantPols (void * pAddress, uint64_t degree)
    {
        rHash = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        rHashType = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        rLatchGet = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        rLatchSet = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        rClimbRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        rClimbSiblingRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        rClimbSiblingRkeyN = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        rRotateLevel = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        rJmpz = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        rJmp = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        rConst0 = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        rConst1 = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        rConst2 = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        rConst3 = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        rAddress = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        rLine = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 128; }
};

class NormGate9ConstantPols
{
public:
    GeneratedPol Value3;
    GeneratedPol Value3Norm;
    GeneratedPol Gate9Type;
    GeneratedPol Gate9A;
    GeneratedPol Gate9B;
    GeneratedPol Gate9C;
    GeneratedPol Latch;
    GeneratedPol Factor;

    NormGate9ConstantPols (void * pAddress)
    {
        Value3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1608);
        Value3Norm = (Goldilocks::Element *)((uint8_t *)pAddress + 1616);
        Gate9Type = (Goldilocks::Element *)((uint8_t *)pAddress + 1624);
        Gate9A = (Goldilocks::Element *)((uint8_t *)pAddress + 1632);
        Gate9B = (Goldilocks::Element *)((uint8_t *)pAddress + 1640);
        Gate9C = (Goldilocks::Element *)((uint8_t *)pAddress + 1648);
        Latch = (Goldilocks::Element *)((uint8_t *)pAddress + 1656);
        Factor = (Goldilocks::Element *)((uint8_t *)pAddress + 1664);
    }

    NormGate9ConstantPols (void * pAddress, uint64_t degree)
    {
        Value3 = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        Value3Norm = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        Gate9Type = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        Gate9A = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        Gate9B = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        Gate9C = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        Latch = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        Factor = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 64; }
};

class KeccakFConstantPols
{
public:
    GeneratedPol ConnA;
    GeneratedPol ConnB;
    GeneratedPol ConnC;
    GeneratedPol NormalizedGate;
    GeneratedPol GateType;

    KeccakFConstantPols (void * pAddress)
    {
        ConnA = (Goldilocks::Element *)((uint8_t *)pAddress + 1672);
        ConnB = (Goldilocks::Element *)((uint8_t *)pAddress + 1680);
        ConnC = (Goldilocks::Element *)((uint8_t *)pAddress + 1688);
        NormalizedGate = (Goldilocks::Element *)((uint8_t *)pAddress + 1696);
        GateType = (Goldilocks::Element *)((uint8_t *)pAddress + 1704);
    }

    KeccakFConstantPols (void * pAddress, uint64_t degree)
    {
        ConnA = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        ConnB = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        ConnC = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        NormalizedGate = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        GateType = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 40; }
};

class Nine2OneConstantPols
{
public:
    GeneratedPol Field9latch;
    GeneratedPol Factor;

    Nine2OneConstantPols (void * pAddress)
    {
        Field9latch = (Goldilocks::Element *)((uint8_t *)pAddress + 1712);
        Factor = (Goldilocks::Element *)((uint8_t *)pAddress + 1720);
    }

    Nine2OneConstantPols (void * pAddress, uint64_t degree)
    {
        Field9latch = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        Factor = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 16; }
};

class PaddingKKBitConstantPols
{
public:
    GeneratedPol r8Id;
    GeneratedPol sOutId;
    GeneratedPol latchR8;
    GeneratedPol Fr8;
    GeneratedPol rBitValid;
    GeneratedPol latchSOut;
    GeneratedPol FSOut0;
    GeneratedPol FSOut1;
    GeneratedPol FSOut2;
    GeneratedPol FSOut3;
    GeneratedPol FSOut4;
    GeneratedPol FSOut5;
    GeneratedPol FSOut6;
    GeneratedPol FSOut7;
    GeneratedPol ConnSOutBit;
    GeneratedPol ConnSInBit;
    GeneratedPol ConnNine2OneBit;

    PaddingKKBitConstantPols (void * pAddress)
    {
        r8Id = (Goldilocks::Element *)((uint8_t *)pAddress + 1728);
        sOutId = (Goldilocks::Element *)((uint8_t *)pAddress + 1736);
        latchR8 = (Goldilocks::Element *)((uint8_t *)pAddress + 1744);
        Fr8 = (Goldilocks::Element *)((uint8_t *)pAddress + 1752);
        rBitValid = (Goldilocks::Element *)((uint8_t *)pAddress + 1760);
        latchSOut = (Goldilocks::Element *)((uint8_t *)pAddress + 1768);
        FSOut0 = (Goldilocks::Element *)((uint8_t *)pAddress + 1776);
        FSOut1 = (Goldilocks::Element *)((uint8_t *)pAddress + 1784);
        FSOut2 = (Goldilocks::Element *)((uint8_t *)pAddress + 1792);
        FSOut3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1800);
        FSOut4 = (Goldilocks::Element *)((uint8_t *)pAddress + 1808);
        FSOut5 = (Goldilocks::Element *)((uint8_t *)pAddress + 1816);
        FSOut6 = (Goldilocks::Element *)((uint8_t *)pAddress + 1824);
        FSOut7 = (Goldilocks::Element *)((uint8_t *)pAddress + 1832);
        ConnSOutBit = (Goldilocks::Element *)((uint8_t *)pAddress + 1840);
        ConnSInBit = (Goldilocks::Element *)((uint8_t *)pAddress + 1848);
        ConnNine2OneBit = (Goldilocks::Element *)((uint8_t *)pAddress + 1856);
    }

    PaddingKKBitConstantPols (void * pAddress, uint64_t degree)
    {
        r8Id = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        sOutId = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        latchR8 = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        Fr8 = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        rBitValid = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        latchSOut = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        FSOut0 = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        FSOut1 = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        FSOut2 = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        FSOut3 = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        FSOut4 = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        FSOut5 = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        FSOut6 = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        FSOut7 = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        ConnSOutBit = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        ConnSInBit = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        ConnNine2OneBit = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 136; }
};

class PaddingKKConstantPols
{
public:
    GeneratedPol r8Id;
    GeneratedPol lastBlock;
    GeneratedPol lastBlockLatch;
    GeneratedPol r8valid;
    GeneratedPol sOutId;
    GeneratedPol forceLastHash;
    GeneratedPol k_crOffset;
    GeneratedPol k_crF0;
    GeneratedPol k_crF1;
    GeneratedPol k_crF2;
    GeneratedPol k_crF3;
    GeneratedPol k_crF4;
    GeneratedPol k_crF5;
    GeneratedPol k_crF6;
    GeneratedPol k_crF7;
    GeneratedPol crValid;

    PaddingKKConstantPols (void * pAddress)
    {
        r8Id = (Goldilocks::Element *)((uint8_t *)pAddress + 1864);
        lastBlock = (Goldilocks::Element *)((uint8_t *)pAddress + 1872);
        lastBlockLatch = (Goldilocks::Element *)((uint8_t *)pAddress + 1880);
        r8valid = (Goldilocks::Element *)((uint8_t *)pAddress + 1888);
        sOutId = (Goldilocks::Element *)((uint8_t *)pAddress + 1896);
        forceLastHash = (Goldilocks::Element *)((uint8_t *)pAddress + 1904);
        k_crOffset = (Goldilocks::Element *)((uint8_t *)pAddress + 1912);
        k_crF0 = (Goldilocks::Element *)((uint8_t *)pAddress + 1920);
        k_crF1 = (Goldilocks::Element *)((uint8_t *)pAddress + 1928);
        k_crF2 = (Goldilocks::Element *)((uint8_t *)pAddress + 1936);
        k_crF3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1944);
        k_crF4 = (Goldilocks::Element *)((uint8_t *)pAddress + 1952);
        k_crF5 = (Goldilocks::Element *)((uint8_t *)pAddress + 1960);
        k_crF6 = (Goldilocks::Element *)((uint8_t *)pAddress + 1968);
        k_crF7 = (Goldilocks::Element *)((uint8_t *)pAddress + 1976);
        crValid = (Goldilocks::Element *)((uint8_t *)pAddress + 1984);
    }

    PaddingKKConstantPols (void * pAddress, uint64_t degree)
    {
        r8Id = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        lastBlock = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
        lastBlockLatch = (Goldilocks::Element *)((uint8_t *)pAddress + 16*degree);
        r8valid = (Goldilocks::Element *)((uint8_t *)pAddress + 24*degree);
        sOutId = (Goldilocks::Element *)((uint8_t *)pAddress + 32*degree);
        forceLastHash = (Goldilocks::Element *)((uint8_t *)pAddress + 40*degree);
        k_crOffset = (Goldilocks::Element *)((uint8_t *)pAddress + 48*degree);
        k_crF0 = (Goldilocks::Element *)((uint8_t *)pAddress + 56*degree);
        k_crF1 = (Goldilocks::Element *)((uint8_t *)pAddress + 64*degree);
        k_crF2 = (Goldilocks::Element *)((uint8_t *)pAddress + 72*degree);
        k_crF3 = (Goldilocks::Element *)((uint8_t *)pAddress + 80*degree);
        k_crF4 = (Goldilocks::Element *)((uint8_t *)pAddress + 88*degree);
        k_crF5 = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        k_crF6 = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        k_crF7 = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        crValid = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 128; }
};

class MemConstantPols
{
public:
    GeneratedPol INCS;
    GeneratedPol ISNOTLAST;

    MemConstantPols (void * pAddress)
    {
        INCS = (Goldilocks::Element *)((uint8_t *)pAddress + 1992);
        ISNOTLAST = (Goldilocks::Element *)((uint8_t *)pAddress + 2000);
    }

    MemConstantPols (void * pAddress, uint64_t degree)
    {
        INCS = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
        ISNOTLAST = (Goldilocks::Element *)((uint8_t *)pAddress + 8*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 16; }
};

class MainConstantPols
{
public:
    GeneratedPol STEP;

    MainConstantPols (void * pAddress)
    {
        STEP = (Goldilocks::Element *)((uint8_t *)pAddress + 2008);
    }

    MainConstantPols (void * pAddress, uint64_t degree)
    {
        STEP = (Goldilocks::Element *)((uint8_t *)pAddress + 0*degree);
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

    static uint64_t size (void) { return 4227858432; }
};

#endif // CONSTANT_POLS_HPP
