#ifndef CONSTANT_POLS_HPP
#define CONSTANT_POLS_HPP

#include <cstdint>
#include "goldilocks/goldilocks_base_field.hpp"

class ConstantGeneratedPol
{
private:
    Goldilocks::Element * pData;
public:
    ConstantGeneratedPol() : pData(NULL) {};
    Goldilocks::Element & operator[](int i) { return pData[i*253]; };
    Goldilocks::Element * operator=(Goldilocks::Element * pAddress) { pData = pAddress; return pData; };
};

class GlobalConstantPols
{
public:
    ConstantGeneratedPol L1;
    ConstantGeneratedPol BYTE;
    ConstantGeneratedPol BYTE2;

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
    ConstantGeneratedPol CONST0;
    ConstantGeneratedPol CONST1;
    ConstantGeneratedPol CONST2;
    ConstantGeneratedPol CONST3;
    ConstantGeneratedPol CONST4;
    ConstantGeneratedPol CONST5;
    ConstantGeneratedPol CONST6;
    ConstantGeneratedPol CONST7;
    ConstantGeneratedPol offset;
    ConstantGeneratedPol inA;
    ConstantGeneratedPol inB;
    ConstantGeneratedPol inC;
    ConstantGeneratedPol inROTL_C;
    ConstantGeneratedPol inD;
    ConstantGeneratedPol inE;
    ConstantGeneratedPol inSR;
    ConstantGeneratedPol inFREE;
    ConstantGeneratedPol inCTX;
    ConstantGeneratedPol inSP;
    ConstantGeneratedPol inPC;
    ConstantGeneratedPol inGAS;
    ConstantGeneratedPol inMAXMEM;
    ConstantGeneratedPol inHASHPOS;
    ConstantGeneratedPol inSTEP;
    ConstantGeneratedPol inRR;
    ConstantGeneratedPol setA;
    ConstantGeneratedPol setB;
    ConstantGeneratedPol setC;
    ConstantGeneratedPol setD;
    ConstantGeneratedPol setE;
    ConstantGeneratedPol setSR;
    ConstantGeneratedPol setCTX;
    ConstantGeneratedPol setSP;
    ConstantGeneratedPol setPC;
    ConstantGeneratedPol setGAS;
    ConstantGeneratedPol setMAXMEM;
    ConstantGeneratedPol setHASHPOS;
    ConstantGeneratedPol JMP;
    ConstantGeneratedPol JMPN;
    ConstantGeneratedPol JMPC;
    ConstantGeneratedPol setRR;
    ConstantGeneratedPol incStack;
    ConstantGeneratedPol incCode;
    ConstantGeneratedPol isStack;
    ConstantGeneratedPol isCode;
    ConstantGeneratedPol isMem;
    ConstantGeneratedPol ind;
    ConstantGeneratedPol indRR;
    ConstantGeneratedPol useCTX;
    ConstantGeneratedPol mOp;
    ConstantGeneratedPol mWR;
    ConstantGeneratedPol sWR;
    ConstantGeneratedPol sRD;
    ConstantGeneratedPol arith;
    ConstantGeneratedPol arithEq0;
    ConstantGeneratedPol arithEq1;
    ConstantGeneratedPol arithEq2;
    ConstantGeneratedPol arithEq3;
    ConstantGeneratedPol memAlign;
    ConstantGeneratedPol memAlignWR;
    ConstantGeneratedPol memAlignWR8;
    ConstantGeneratedPol hashK;
    ConstantGeneratedPol hashKLen;
    ConstantGeneratedPol hashKDigest;
    ConstantGeneratedPol hashP;
    ConstantGeneratedPol hashPLen;
    ConstantGeneratedPol hashPDigest;
    ConstantGeneratedPol bin;
    ConstantGeneratedPol binOpcode;
    ConstantGeneratedPol assert;
    ConstantGeneratedPol line;

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
        inROTL_C = (Goldilocks::Element *)((uint8_t *)pAddress + 120);
        inD = (Goldilocks::Element *)((uint8_t *)pAddress + 128);
        inE = (Goldilocks::Element *)((uint8_t *)pAddress + 136);
        inSR = (Goldilocks::Element *)((uint8_t *)pAddress + 144);
        inFREE = (Goldilocks::Element *)((uint8_t *)pAddress + 152);
        inCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 160);
        inSP = (Goldilocks::Element *)((uint8_t *)pAddress + 168);
        inPC = (Goldilocks::Element *)((uint8_t *)pAddress + 176);
        inGAS = (Goldilocks::Element *)((uint8_t *)pAddress + 184);
        inMAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 192);
        inHASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 200);
        inSTEP = (Goldilocks::Element *)((uint8_t *)pAddress + 208);
        inRR = (Goldilocks::Element *)((uint8_t *)pAddress + 216);
        setA = (Goldilocks::Element *)((uint8_t *)pAddress + 224);
        setB = (Goldilocks::Element *)((uint8_t *)pAddress + 232);
        setC = (Goldilocks::Element *)((uint8_t *)pAddress + 240);
        setD = (Goldilocks::Element *)((uint8_t *)pAddress + 248);
        setE = (Goldilocks::Element *)((uint8_t *)pAddress + 256);
        setSR = (Goldilocks::Element *)((uint8_t *)pAddress + 264);
        setCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 272);
        setSP = (Goldilocks::Element *)((uint8_t *)pAddress + 280);
        setPC = (Goldilocks::Element *)((uint8_t *)pAddress + 288);
        setGAS = (Goldilocks::Element *)((uint8_t *)pAddress + 296);
        setMAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 304);
        setHASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 312);
        JMP = (Goldilocks::Element *)((uint8_t *)pAddress + 320);
        JMPN = (Goldilocks::Element *)((uint8_t *)pAddress + 328);
        JMPC = (Goldilocks::Element *)((uint8_t *)pAddress + 336);
        setRR = (Goldilocks::Element *)((uint8_t *)pAddress + 344);
        incStack = (Goldilocks::Element *)((uint8_t *)pAddress + 352);
        incCode = (Goldilocks::Element *)((uint8_t *)pAddress + 360);
        isStack = (Goldilocks::Element *)((uint8_t *)pAddress + 368);
        isCode = (Goldilocks::Element *)((uint8_t *)pAddress + 376);
        isMem = (Goldilocks::Element *)((uint8_t *)pAddress + 384);
        ind = (Goldilocks::Element *)((uint8_t *)pAddress + 392);
        indRR = (Goldilocks::Element *)((uint8_t *)pAddress + 400);
        useCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 408);
        mOp = (Goldilocks::Element *)((uint8_t *)pAddress + 416);
        mWR = (Goldilocks::Element *)((uint8_t *)pAddress + 424);
        sWR = (Goldilocks::Element *)((uint8_t *)pAddress + 432);
        sRD = (Goldilocks::Element *)((uint8_t *)pAddress + 440);
        arith = (Goldilocks::Element *)((uint8_t *)pAddress + 448);
        arithEq0 = (Goldilocks::Element *)((uint8_t *)pAddress + 456);
        arithEq1 = (Goldilocks::Element *)((uint8_t *)pAddress + 464);
        arithEq2 = (Goldilocks::Element *)((uint8_t *)pAddress + 472);
        arithEq3 = (Goldilocks::Element *)((uint8_t *)pAddress + 480);
        memAlign = (Goldilocks::Element *)((uint8_t *)pAddress + 488);
        memAlignWR = (Goldilocks::Element *)((uint8_t *)pAddress + 496);
        memAlignWR8 = (Goldilocks::Element *)((uint8_t *)pAddress + 504);
        hashK = (Goldilocks::Element *)((uint8_t *)pAddress + 512);
        hashKLen = (Goldilocks::Element *)((uint8_t *)pAddress + 520);
        hashKDigest = (Goldilocks::Element *)((uint8_t *)pAddress + 528);
        hashP = (Goldilocks::Element *)((uint8_t *)pAddress + 536);
        hashPLen = (Goldilocks::Element *)((uint8_t *)pAddress + 544);
        hashPDigest = (Goldilocks::Element *)((uint8_t *)pAddress + 552);
        bin = (Goldilocks::Element *)((uint8_t *)pAddress + 560);
        binOpcode = (Goldilocks::Element *)((uint8_t *)pAddress + 568);
        assert = (Goldilocks::Element *)((uint8_t *)pAddress + 576);
        line = (Goldilocks::Element *)((uint8_t *)pAddress + 584);
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
        inROTL_C = (Goldilocks::Element *)((uint8_t *)pAddress + 96*degree);
        inD = (Goldilocks::Element *)((uint8_t *)pAddress + 104*degree);
        inE = (Goldilocks::Element *)((uint8_t *)pAddress + 112*degree);
        inSR = (Goldilocks::Element *)((uint8_t *)pAddress + 120*degree);
        inFREE = (Goldilocks::Element *)((uint8_t *)pAddress + 128*degree);
        inCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 136*degree);
        inSP = (Goldilocks::Element *)((uint8_t *)pAddress + 144*degree);
        inPC = (Goldilocks::Element *)((uint8_t *)pAddress + 152*degree);
        inGAS = (Goldilocks::Element *)((uint8_t *)pAddress + 160*degree);
        inMAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 168*degree);
        inHASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 176*degree);
        inSTEP = (Goldilocks::Element *)((uint8_t *)pAddress + 184*degree);
        inRR = (Goldilocks::Element *)((uint8_t *)pAddress + 192*degree);
        setA = (Goldilocks::Element *)((uint8_t *)pAddress + 200*degree);
        setB = (Goldilocks::Element *)((uint8_t *)pAddress + 208*degree);
        setC = (Goldilocks::Element *)((uint8_t *)pAddress + 216*degree);
        setD = (Goldilocks::Element *)((uint8_t *)pAddress + 224*degree);
        setE = (Goldilocks::Element *)((uint8_t *)pAddress + 232*degree);
        setSR = (Goldilocks::Element *)((uint8_t *)pAddress + 240*degree);
        setCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 248*degree);
        setSP = (Goldilocks::Element *)((uint8_t *)pAddress + 256*degree);
        setPC = (Goldilocks::Element *)((uint8_t *)pAddress + 264*degree);
        setGAS = (Goldilocks::Element *)((uint8_t *)pAddress + 272*degree);
        setMAXMEM = (Goldilocks::Element *)((uint8_t *)pAddress + 280*degree);
        setHASHPOS = (Goldilocks::Element *)((uint8_t *)pAddress + 288*degree);
        JMP = (Goldilocks::Element *)((uint8_t *)pAddress + 296*degree);
        JMPN = (Goldilocks::Element *)((uint8_t *)pAddress + 304*degree);
        JMPC = (Goldilocks::Element *)((uint8_t *)pAddress + 312*degree);
        setRR = (Goldilocks::Element *)((uint8_t *)pAddress + 320*degree);
        incStack = (Goldilocks::Element *)((uint8_t *)pAddress + 328*degree);
        incCode = (Goldilocks::Element *)((uint8_t *)pAddress + 336*degree);
        isStack = (Goldilocks::Element *)((uint8_t *)pAddress + 344*degree);
        isCode = (Goldilocks::Element *)((uint8_t *)pAddress + 352*degree);
        isMem = (Goldilocks::Element *)((uint8_t *)pAddress + 360*degree);
        ind = (Goldilocks::Element *)((uint8_t *)pAddress + 368*degree);
        indRR = (Goldilocks::Element *)((uint8_t *)pAddress + 376*degree);
        useCTX = (Goldilocks::Element *)((uint8_t *)pAddress + 384*degree);
        mOp = (Goldilocks::Element *)((uint8_t *)pAddress + 392*degree);
        mWR = (Goldilocks::Element *)((uint8_t *)pAddress + 400*degree);
        sWR = (Goldilocks::Element *)((uint8_t *)pAddress + 408*degree);
        sRD = (Goldilocks::Element *)((uint8_t *)pAddress + 416*degree);
        arith = (Goldilocks::Element *)((uint8_t *)pAddress + 424*degree);
        arithEq0 = (Goldilocks::Element *)((uint8_t *)pAddress + 432*degree);
        arithEq1 = (Goldilocks::Element *)((uint8_t *)pAddress + 440*degree);
        arithEq2 = (Goldilocks::Element *)((uint8_t *)pAddress + 448*degree);
        arithEq3 = (Goldilocks::Element *)((uint8_t *)pAddress + 456*degree);
        memAlign = (Goldilocks::Element *)((uint8_t *)pAddress + 464*degree);
        memAlignWR = (Goldilocks::Element *)((uint8_t *)pAddress + 472*degree);
        memAlignWR8 = (Goldilocks::Element *)((uint8_t *)pAddress + 480*degree);
        hashK = (Goldilocks::Element *)((uint8_t *)pAddress + 488*degree);
        hashKLen = (Goldilocks::Element *)((uint8_t *)pAddress + 496*degree);
        hashKDigest = (Goldilocks::Element *)((uint8_t *)pAddress + 504*degree);
        hashP = (Goldilocks::Element *)((uint8_t *)pAddress + 512*degree);
        hashPLen = (Goldilocks::Element *)((uint8_t *)pAddress + 520*degree);
        hashPDigest = (Goldilocks::Element *)((uint8_t *)pAddress + 528*degree);
        bin = (Goldilocks::Element *)((uint8_t *)pAddress + 536*degree);
        binOpcode = (Goldilocks::Element *)((uint8_t *)pAddress + 544*degree);
        assert = (Goldilocks::Element *)((uint8_t *)pAddress + 552*degree);
        line = (Goldilocks::Element *)((uint8_t *)pAddress + 560*degree);
    }

    static uint64_t degree (void) { return 2097152; }
    static uint64_t size (void) { return 568; }
};

class Byte4ConstantPols
{
public:
    ConstantGeneratedPol SET;

    Byte4ConstantPols (void * pAddress)
    {
        SET = (Goldilocks::Element *)((uint8_t *)pAddress + 592);
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
    ConstantGeneratedPol BYTE2A;
    ConstantGeneratedPol BYTE2B;
    ConstantGeneratedPol BYTE_C3072;
    ConstantGeneratedPol FACTOR[8];
    ConstantGeneratedPol FACTORV[8];
    ConstantGeneratedPol STEP;
    ConstantGeneratedPol WR256;
    ConstantGeneratedPol WR8;
    ConstantGeneratedPol OFFSET;
    ConstantGeneratedPol RESET;
    ConstantGeneratedPol SELM1;

    MemAlignConstantPols (void * pAddress)
    {
        BYTE2A = (Goldilocks::Element *)((uint8_t *)pAddress + 600);
        BYTE2B = (Goldilocks::Element *)((uint8_t *)pAddress + 608);
        BYTE_C3072 = (Goldilocks::Element *)((uint8_t *)pAddress + 616);
        FACTOR[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 624);
        FACTOR[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 632);
        FACTOR[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 640);
        FACTOR[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 648);
        FACTOR[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 656);
        FACTOR[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 664);
        FACTOR[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 672);
        FACTOR[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 680);
        FACTORV[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 688);
        FACTORV[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 696);
        FACTORV[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 704);
        FACTORV[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 712);
        FACTORV[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 720);
        FACTORV[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 728);
        FACTORV[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 736);
        FACTORV[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 744);
        STEP = (Goldilocks::Element *)((uint8_t *)pAddress + 752);
        WR256 = (Goldilocks::Element *)((uint8_t *)pAddress + 760);
        WR8 = (Goldilocks::Element *)((uint8_t *)pAddress + 768);
        OFFSET = (Goldilocks::Element *)((uint8_t *)pAddress + 776);
        RESET = (Goldilocks::Element *)((uint8_t *)pAddress + 784);
        SELM1 = (Goldilocks::Element *)((uint8_t *)pAddress + 792);
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
    ConstantGeneratedPol BIT19;
    ConstantGeneratedPol GL_SIGNED_4BITS;
    ConstantGeneratedPol GL_SIGNED_18BITS;
    ConstantGeneratedPol ck[32];

    ArithConstantPols (void * pAddress)
    {
        BIT19 = (Goldilocks::Element *)((uint8_t *)pAddress + 800);
        GL_SIGNED_4BITS = (Goldilocks::Element *)((uint8_t *)pAddress + 808);
        GL_SIGNED_18BITS = (Goldilocks::Element *)((uint8_t *)pAddress + 816);
        ck[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 824);
        ck[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 832);
        ck[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 840);
        ck[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 848);
        ck[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 856);
        ck[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 864);
        ck[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 872);
        ck[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 880);
        ck[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 888);
        ck[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 896);
        ck[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 904);
        ck[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 912);
        ck[12] = (Goldilocks::Element *)((uint8_t *)pAddress + 920);
        ck[13] = (Goldilocks::Element *)((uint8_t *)pAddress + 928);
        ck[14] = (Goldilocks::Element *)((uint8_t *)pAddress + 936);
        ck[15] = (Goldilocks::Element *)((uint8_t *)pAddress + 944);
        ck[16] = (Goldilocks::Element *)((uint8_t *)pAddress + 952);
        ck[17] = (Goldilocks::Element *)((uint8_t *)pAddress + 960);
        ck[18] = (Goldilocks::Element *)((uint8_t *)pAddress + 968);
        ck[19] = (Goldilocks::Element *)((uint8_t *)pAddress + 976);
        ck[20] = (Goldilocks::Element *)((uint8_t *)pAddress + 984);
        ck[21] = (Goldilocks::Element *)((uint8_t *)pAddress + 992);
        ck[22] = (Goldilocks::Element *)((uint8_t *)pAddress + 1000);
        ck[23] = (Goldilocks::Element *)((uint8_t *)pAddress + 1008);
        ck[24] = (Goldilocks::Element *)((uint8_t *)pAddress + 1016);
        ck[25] = (Goldilocks::Element *)((uint8_t *)pAddress + 1024);
        ck[26] = (Goldilocks::Element *)((uint8_t *)pAddress + 1032);
        ck[27] = (Goldilocks::Element *)((uint8_t *)pAddress + 1040);
        ck[28] = (Goldilocks::Element *)((uint8_t *)pAddress + 1048);
        ck[29] = (Goldilocks::Element *)((uint8_t *)pAddress + 1056);
        ck[30] = (Goldilocks::Element *)((uint8_t *)pAddress + 1064);
        ck[31] = (Goldilocks::Element *)((uint8_t *)pAddress + 1072);
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
    ConstantGeneratedPol P_OPCODE;
    ConstantGeneratedPol P_A;
    ConstantGeneratedPol P_B;
    ConstantGeneratedPol P_CIN;
    ConstantGeneratedPol P_LAST;
    ConstantGeneratedPol P_USE_CARRY;
    ConstantGeneratedPol P_C;
    ConstantGeneratedPol P_COUT;
    ConstantGeneratedPol RESET;
    ConstantGeneratedPol FACTOR[8];

    BinaryConstantPols (void * pAddress)
    {
        P_OPCODE = (Goldilocks::Element *)((uint8_t *)pAddress + 1080);
        P_A = (Goldilocks::Element *)((uint8_t *)pAddress + 1088);
        P_B = (Goldilocks::Element *)((uint8_t *)pAddress + 1096);
        P_CIN = (Goldilocks::Element *)((uint8_t *)pAddress + 1104);
        P_LAST = (Goldilocks::Element *)((uint8_t *)pAddress + 1112);
        P_USE_CARRY = (Goldilocks::Element *)((uint8_t *)pAddress + 1120);
        P_C = (Goldilocks::Element *)((uint8_t *)pAddress + 1128);
        P_COUT = (Goldilocks::Element *)((uint8_t *)pAddress + 1136);
        RESET = (Goldilocks::Element *)((uint8_t *)pAddress + 1144);
        FACTOR[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1152);
        FACTOR[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1160);
        FACTOR[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1168);
        FACTOR[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1176);
        FACTOR[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1184);
        FACTOR[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1192);
        FACTOR[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1200);
        FACTOR[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1208);
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
    ConstantGeneratedPol LAST;
    ConstantGeneratedPol LATCH;
    ConstantGeneratedPol LASTBLOCK;
    ConstantGeneratedPol PARTIAL;
    ConstantGeneratedPol C[12];

    PoseidonGConstantPols (void * pAddress)
    {
        LAST = (Goldilocks::Element *)((uint8_t *)pAddress + 1216);
        LATCH = (Goldilocks::Element *)((uint8_t *)pAddress + 1224);
        LASTBLOCK = (Goldilocks::Element *)((uint8_t *)pAddress + 1232);
        PARTIAL = (Goldilocks::Element *)((uint8_t *)pAddress + 1240);
        C[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1248);
        C[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1256);
        C[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1264);
        C[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1272);
        C[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1280);
        C[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1288);
        C[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1296);
        C[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1304);
        C[8] = (Goldilocks::Element *)((uint8_t *)pAddress + 1312);
        C[9] = (Goldilocks::Element *)((uint8_t *)pAddress + 1320);
        C[10] = (Goldilocks::Element *)((uint8_t *)pAddress + 1328);
        C[11] = (Goldilocks::Element *)((uint8_t *)pAddress + 1336);
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
    ConstantGeneratedPol F[8];
    ConstantGeneratedPol lastBlock;
    ConstantGeneratedPol k_crOffset;
    ConstantGeneratedPol k_crF0;
    ConstantGeneratedPol k_crF1;
    ConstantGeneratedPol k_crF2;
    ConstantGeneratedPol k_crF3;
    ConstantGeneratedPol k_crF4;
    ConstantGeneratedPol k_crF5;
    ConstantGeneratedPol k_crF6;
    ConstantGeneratedPol k_crF7;

    PaddingPGConstantPols (void * pAddress)
    {
        F[0] = (Goldilocks::Element *)((uint8_t *)pAddress + 1344);
        F[1] = (Goldilocks::Element *)((uint8_t *)pAddress + 1352);
        F[2] = (Goldilocks::Element *)((uint8_t *)pAddress + 1360);
        F[3] = (Goldilocks::Element *)((uint8_t *)pAddress + 1368);
        F[4] = (Goldilocks::Element *)((uint8_t *)pAddress + 1376);
        F[5] = (Goldilocks::Element *)((uint8_t *)pAddress + 1384);
        F[6] = (Goldilocks::Element *)((uint8_t *)pAddress + 1392);
        F[7] = (Goldilocks::Element *)((uint8_t *)pAddress + 1400);
        lastBlock = (Goldilocks::Element *)((uint8_t *)pAddress + 1408);
        k_crOffset = (Goldilocks::Element *)((uint8_t *)pAddress + 1416);
        k_crF0 = (Goldilocks::Element *)((uint8_t *)pAddress + 1424);
        k_crF1 = (Goldilocks::Element *)((uint8_t *)pAddress + 1432);
        k_crF2 = (Goldilocks::Element *)((uint8_t *)pAddress + 1440);
        k_crF3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1448);
        k_crF4 = (Goldilocks::Element *)((uint8_t *)pAddress + 1456);
        k_crF5 = (Goldilocks::Element *)((uint8_t *)pAddress + 1464);
        k_crF6 = (Goldilocks::Element *)((uint8_t *)pAddress + 1472);
        k_crF7 = (Goldilocks::Element *)((uint8_t *)pAddress + 1480);
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
    ConstantGeneratedPol rHash;
    ConstantGeneratedPol rHashType;
    ConstantGeneratedPol rLatchGet;
    ConstantGeneratedPol rLatchSet;
    ConstantGeneratedPol rClimbRkey;
    ConstantGeneratedPol rClimbSiblingRkey;
    ConstantGeneratedPol rClimbSiblingRkeyN;
    ConstantGeneratedPol rRotateLevel;
    ConstantGeneratedPol rJmpz;
    ConstantGeneratedPol rJmp;
    ConstantGeneratedPol rConst0;
    ConstantGeneratedPol rConst1;
    ConstantGeneratedPol rConst2;
    ConstantGeneratedPol rConst3;
    ConstantGeneratedPol rAddress;
    ConstantGeneratedPol rLine;

    StorageConstantPols (void * pAddress)
    {
        rHash = (Goldilocks::Element *)((uint8_t *)pAddress + 1488);
        rHashType = (Goldilocks::Element *)((uint8_t *)pAddress + 1496);
        rLatchGet = (Goldilocks::Element *)((uint8_t *)pAddress + 1504);
        rLatchSet = (Goldilocks::Element *)((uint8_t *)pAddress + 1512);
        rClimbRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 1520);
        rClimbSiblingRkey = (Goldilocks::Element *)((uint8_t *)pAddress + 1528);
        rClimbSiblingRkeyN = (Goldilocks::Element *)((uint8_t *)pAddress + 1536);
        rRotateLevel = (Goldilocks::Element *)((uint8_t *)pAddress + 1544);
        rJmpz = (Goldilocks::Element *)((uint8_t *)pAddress + 1552);
        rJmp = (Goldilocks::Element *)((uint8_t *)pAddress + 1560);
        rConst0 = (Goldilocks::Element *)((uint8_t *)pAddress + 1568);
        rConst1 = (Goldilocks::Element *)((uint8_t *)pAddress + 1576);
        rConst2 = (Goldilocks::Element *)((uint8_t *)pAddress + 1584);
        rConst3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1592);
        rAddress = (Goldilocks::Element *)((uint8_t *)pAddress + 1600);
        rLine = (Goldilocks::Element *)((uint8_t *)pAddress + 1608);
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
    ConstantGeneratedPol Value3;
    ConstantGeneratedPol Value3Norm;
    ConstantGeneratedPol Gate9Type;
    ConstantGeneratedPol Gate9A;
    ConstantGeneratedPol Gate9B;
    ConstantGeneratedPol Gate9C;
    ConstantGeneratedPol Latch;
    ConstantGeneratedPol Factor;

    NormGate9ConstantPols (void * pAddress)
    {
        Value3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1616);
        Value3Norm = (Goldilocks::Element *)((uint8_t *)pAddress + 1624);
        Gate9Type = (Goldilocks::Element *)((uint8_t *)pAddress + 1632);
        Gate9A = (Goldilocks::Element *)((uint8_t *)pAddress + 1640);
        Gate9B = (Goldilocks::Element *)((uint8_t *)pAddress + 1648);
        Gate9C = (Goldilocks::Element *)((uint8_t *)pAddress + 1656);
        Latch = (Goldilocks::Element *)((uint8_t *)pAddress + 1664);
        Factor = (Goldilocks::Element *)((uint8_t *)pAddress + 1672);
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
    ConstantGeneratedPol ConnA;
    ConstantGeneratedPol ConnB;
    ConstantGeneratedPol ConnC;
    ConstantGeneratedPol NormalizedGate;
    ConstantGeneratedPol GateType;

    KeccakFConstantPols (void * pAddress)
    {
        ConnA = (Goldilocks::Element *)((uint8_t *)pAddress + 1680);
        ConnB = (Goldilocks::Element *)((uint8_t *)pAddress + 1688);
        ConnC = (Goldilocks::Element *)((uint8_t *)pAddress + 1696);
        NormalizedGate = (Goldilocks::Element *)((uint8_t *)pAddress + 1704);
        GateType = (Goldilocks::Element *)((uint8_t *)pAddress + 1712);
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
    ConstantGeneratedPol Field9latch;
    ConstantGeneratedPol Factor;

    Nine2OneConstantPols (void * pAddress)
    {
        Field9latch = (Goldilocks::Element *)((uint8_t *)pAddress + 1720);
        Factor = (Goldilocks::Element *)((uint8_t *)pAddress + 1728);
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
    ConstantGeneratedPol r8Id;
    ConstantGeneratedPol sOutId;
    ConstantGeneratedPol latchR8;
    ConstantGeneratedPol Fr8;
    ConstantGeneratedPol rBitValid;
    ConstantGeneratedPol latchSOut;
    ConstantGeneratedPol FSOut0;
    ConstantGeneratedPol FSOut1;
    ConstantGeneratedPol FSOut2;
    ConstantGeneratedPol FSOut3;
    ConstantGeneratedPol FSOut4;
    ConstantGeneratedPol FSOut5;
    ConstantGeneratedPol FSOut6;
    ConstantGeneratedPol FSOut7;
    ConstantGeneratedPol ConnSOutBit;
    ConstantGeneratedPol ConnSInBit;
    ConstantGeneratedPol ConnNine2OneBit;

    PaddingKKBitConstantPols (void * pAddress)
    {
        r8Id = (Goldilocks::Element *)((uint8_t *)pAddress + 1736);
        sOutId = (Goldilocks::Element *)((uint8_t *)pAddress + 1744);
        latchR8 = (Goldilocks::Element *)((uint8_t *)pAddress + 1752);
        Fr8 = (Goldilocks::Element *)((uint8_t *)pAddress + 1760);
        rBitValid = (Goldilocks::Element *)((uint8_t *)pAddress + 1768);
        latchSOut = (Goldilocks::Element *)((uint8_t *)pAddress + 1776);
        FSOut0 = (Goldilocks::Element *)((uint8_t *)pAddress + 1784);
        FSOut1 = (Goldilocks::Element *)((uint8_t *)pAddress + 1792);
        FSOut2 = (Goldilocks::Element *)((uint8_t *)pAddress + 1800);
        FSOut3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1808);
        FSOut4 = (Goldilocks::Element *)((uint8_t *)pAddress + 1816);
        FSOut5 = (Goldilocks::Element *)((uint8_t *)pAddress + 1824);
        FSOut6 = (Goldilocks::Element *)((uint8_t *)pAddress + 1832);
        FSOut7 = (Goldilocks::Element *)((uint8_t *)pAddress + 1840);
        ConnSOutBit = (Goldilocks::Element *)((uint8_t *)pAddress + 1848);
        ConnSInBit = (Goldilocks::Element *)((uint8_t *)pAddress + 1856);
        ConnNine2OneBit = (Goldilocks::Element *)((uint8_t *)pAddress + 1864);
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
    ConstantGeneratedPol r8Id;
    ConstantGeneratedPol lastBlock;
    ConstantGeneratedPol lastBlockLatch;
    ConstantGeneratedPol r8valid;
    ConstantGeneratedPol sOutId;
    ConstantGeneratedPol forceLastHash;
    ConstantGeneratedPol k_crOffset;
    ConstantGeneratedPol k_crF0;
    ConstantGeneratedPol k_crF1;
    ConstantGeneratedPol k_crF2;
    ConstantGeneratedPol k_crF3;
    ConstantGeneratedPol k_crF4;
    ConstantGeneratedPol k_crF5;
    ConstantGeneratedPol k_crF6;
    ConstantGeneratedPol k_crF7;
    ConstantGeneratedPol crValid;

    PaddingKKConstantPols (void * pAddress)
    {
        r8Id = (Goldilocks::Element *)((uint8_t *)pAddress + 1872);
        lastBlock = (Goldilocks::Element *)((uint8_t *)pAddress + 1880);
        lastBlockLatch = (Goldilocks::Element *)((uint8_t *)pAddress + 1888);
        r8valid = (Goldilocks::Element *)((uint8_t *)pAddress + 1896);
        sOutId = (Goldilocks::Element *)((uint8_t *)pAddress + 1904);
        forceLastHash = (Goldilocks::Element *)((uint8_t *)pAddress + 1912);
        k_crOffset = (Goldilocks::Element *)((uint8_t *)pAddress + 1920);
        k_crF0 = (Goldilocks::Element *)((uint8_t *)pAddress + 1928);
        k_crF1 = (Goldilocks::Element *)((uint8_t *)pAddress + 1936);
        k_crF2 = (Goldilocks::Element *)((uint8_t *)pAddress + 1944);
        k_crF3 = (Goldilocks::Element *)((uint8_t *)pAddress + 1952);
        k_crF4 = (Goldilocks::Element *)((uint8_t *)pAddress + 1960);
        k_crF5 = (Goldilocks::Element *)((uint8_t *)pAddress + 1968);
        k_crF6 = (Goldilocks::Element *)((uint8_t *)pAddress + 1976);
        k_crF7 = (Goldilocks::Element *)((uint8_t *)pAddress + 1984);
        crValid = (Goldilocks::Element *)((uint8_t *)pAddress + 1992);
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
    ConstantGeneratedPol INCS;
    ConstantGeneratedPol ISNOTLAST;

    MemConstantPols (void * pAddress)
    {
        INCS = (Goldilocks::Element *)((uint8_t *)pAddress + 2000);
        ISNOTLAST = (Goldilocks::Element *)((uint8_t *)pAddress + 2008);
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
    ConstantGeneratedPol STEP;

    MainConstantPols (void * pAddress)
    {
        STEP = (Goldilocks::Element *)((uint8_t *)pAddress + 2016);
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

    static uint64_t size (void) { return 4244635648; }
};

#endif // CONSTANT_POLS_HPP
