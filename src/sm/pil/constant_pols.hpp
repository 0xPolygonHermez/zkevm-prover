#ifndef CONSTANT_POLS_HPP
#define CONSTANT_POLS_HPP

#include <cstdint>
#include "ff/ff.hpp"

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
        BYTE2A = (uint8_t *)((uint8_t *)pAddress + 0);
        BYTE2B = (uint8_t *)((uint8_t *)pAddress + 4194304);
        FACTOR0[0] = (FieldElement *)((uint8_t *)pAddress + 8388608);
        FACTOR0[1] = (FieldElement *)((uint8_t *)pAddress + 41943040);
        FACTOR0[2] = (FieldElement *)((uint8_t *)pAddress + 75497472);
        FACTOR0[3] = (FieldElement *)((uint8_t *)pAddress + 109051904);
        FACTOR0[4] = (FieldElement *)((uint8_t *)pAddress + 142606336);
        FACTOR0[5] = (FieldElement *)((uint8_t *)pAddress + 176160768);
        FACTOR0[6] = (FieldElement *)((uint8_t *)pAddress + 209715200);
        FACTOR0[7] = (FieldElement *)((uint8_t *)pAddress + 243269632);
        FACTOR1[0] = (FieldElement *)((uint8_t *)pAddress + 276824064);
        FACTOR1[1] = (FieldElement *)((uint8_t *)pAddress + 310378496);
        FACTOR1[2] = (FieldElement *)((uint8_t *)pAddress + 343932928);
        FACTOR1[3] = (FieldElement *)((uint8_t *)pAddress + 377487360);
        FACTOR1[4] = (FieldElement *)((uint8_t *)pAddress + 411041792);
        FACTOR1[5] = (FieldElement *)((uint8_t *)pAddress + 444596224);
        FACTOR1[6] = (FieldElement *)((uint8_t *)pAddress + 478150656);
        FACTOR1[7] = (FieldElement *)((uint8_t *)pAddress + 511705088);
        FACTORV[0] = (FieldElement *)((uint8_t *)pAddress + 545259520);
        FACTORV[1] = (FieldElement *)((uint8_t *)pAddress + 578813952);
        FACTORV[2] = (FieldElement *)((uint8_t *)pAddress + 612368384);
        FACTORV[3] = (FieldElement *)((uint8_t *)pAddress + 645922816);
        FACTORV[4] = (FieldElement *)((uint8_t *)pAddress + 679477248);
        FACTORV[5] = (FieldElement *)((uint8_t *)pAddress + 713031680);
        FACTORV[6] = (FieldElement *)((uint8_t *)pAddress + 746586112);
        FACTORV[7] = (FieldElement *)((uint8_t *)pAddress + 780140544);
        STEP = (FieldElement *)((uint8_t *)pAddress + 813694976);
        WR = (uint8_t *)((uint8_t *)pAddress + 847249408);
        OFFSET = (uint8_t *)((uint8_t *)pAddress + 851443712);
        RESET = (uint8_t *)((uint8_t *)pAddress + 855638016);
        RESETL = (uint8_t *)((uint8_t *)pAddress + 859832320);
        SELW = (uint8_t *)((uint8_t *)pAddress + 864026624);
    }

    static uint64_t degree (void) { return 4194304; }
};

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
        ZH = (FieldElement *)((uint8_t *)pAddress + 868220928);
        ZHINV = (FieldElement *)((uint8_t *)pAddress + 901775360);
        L1 = (uint8_t *)((uint8_t *)pAddress + 935329792);
        BYTE = (FieldElement *)((uint8_t *)pAddress + 939524096);
        BYTE2 = (FieldElement *)((uint8_t *)pAddress + 973078528);
    }

    static uint64_t degree (void) { return 4194304; }
};

class Byte4ConstantPols
{
public:
    uint8_t * SET;

    Byte4ConstantPols (void * pAddress)
    {
        SET = (uint8_t *)((uint8_t *)pAddress + 1006632960);
    }

    static uint64_t degree (void) { return 4194304; }
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
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 1010827264);
        CONST1 = (uint32_t *)((uint8_t *)pAddress + 1044381696);
        CONST2 = (uint32_t *)((uint8_t *)pAddress + 1077936128);
        CONST3 = (uint32_t *)((uint8_t *)pAddress + 1111490560);
        CONST4 = (uint32_t *)((uint8_t *)pAddress + 1145044992);
        CONST5 = (uint32_t *)((uint8_t *)pAddress + 1178599424);
        CONST6 = (uint32_t *)((uint8_t *)pAddress + 1212153856);
        CONST7 = (uint32_t *)((uint8_t *)pAddress + 1245708288);
        offset = (uint32_t *)((uint8_t *)pAddress + 1279262720);
        inA = (FieldElement *)((uint8_t *)pAddress + 1312817152);
        inB = (FieldElement *)((uint8_t *)pAddress + 1346371584);
        inC = (FieldElement *)((uint8_t *)pAddress + 1379926016);
        inD = (FieldElement *)((uint8_t *)pAddress + 1413480448);
        inE = (FieldElement *)((uint8_t *)pAddress + 1447034880);
        inSR = (FieldElement *)((uint8_t *)pAddress + 1480589312);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 1514143744);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 1547698176);
        inSP = (FieldElement *)((uint8_t *)pAddress + 1581252608);
        inPC = (FieldElement *)((uint8_t *)pAddress + 1614807040);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 1648361472);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 1681915904);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 1715470336);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 1749024768);
        inRR = (FieldElement *)((uint8_t *)pAddress + 1782579200);
        setA = (uint8_t *)((uint8_t *)pAddress + 1816133632);
        setB = (uint8_t *)((uint8_t *)pAddress + 1820327936);
        setC = (uint8_t *)((uint8_t *)pAddress + 1824522240);
        setD = (uint8_t *)((uint8_t *)pAddress + 1828716544);
        setE = (uint8_t *)((uint8_t *)pAddress + 1832910848);
        setSR = (uint8_t *)((uint8_t *)pAddress + 1837105152);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 1841299456);
        setSP = (uint8_t *)((uint8_t *)pAddress + 1845493760);
        setPC = (uint8_t *)((uint8_t *)pAddress + 1849688064);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 1853882368);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 1858076672);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 1862270976);
        JMP = (uint8_t *)((uint8_t *)pAddress + 1866465280);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 1870659584);
        setRR = (uint8_t *)((uint8_t *)pAddress + 1874853888);
        incStack = (int32_t *)((uint8_t *)pAddress + 1879048192);
        incCode = (int32_t *)((uint8_t *)pAddress + 1895825408);
        isStack = (uint8_t *)((uint8_t *)pAddress + 1912602624);
        isCode = (uint8_t *)((uint8_t *)pAddress + 1916796928);
        isMem = (uint8_t *)((uint8_t *)pAddress + 1920991232);
        ind = (uint8_t *)((uint8_t *)pAddress + 1925185536);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 1929379840);
        mOp = (uint8_t *)((uint8_t *)pAddress + 1933574144);
        mWR = (uint8_t *)((uint8_t *)pAddress + 1937768448);
        sWR = (uint8_t *)((uint8_t *)pAddress + 1941962752);
        sRD = (uint8_t *)((uint8_t *)pAddress + 1946157056);
        arith = (uint8_t *)((uint8_t *)pAddress + 1950351360);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 1954545664);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 1958739968);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 1962934272);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 1967128576);
        memAlign = (uint8_t *)((uint8_t *)pAddress + 1971322880);
        memAlignWR = (uint8_t *)((uint8_t *)pAddress + 1975517184);
        hashK = (uint8_t *)((uint8_t *)pAddress + 1979711488);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 1983905792);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 1988100096);
        hashP = (uint8_t *)((uint8_t *)pAddress + 1992294400);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 1996488704);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 2000683008);
        bin = (uint8_t *)((uint8_t *)pAddress + 2004877312);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 2009071616);
        assert = (uint8_t *)((uint8_t *)pAddress + 2013265920);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 2017460224);
        line = (uint32_t *)((uint8_t *)pAddress + 2021654528);
        opCodeNum = (uint8_t *)((uint8_t *)pAddress + 2055208960);
        opCodeAddr = (uint32_t *)((uint8_t *)pAddress + 2059403264);
    }

    static uint64_t degree (void) { return 4194304; }
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
        BIT19 = (FieldElement *)((uint8_t *)pAddress + 2092957696);
        GL_SIGNED_4BITS = (FieldElement *)((uint8_t *)pAddress + 2126512128);
        GL_SIGNED_18BITS = (FieldElement *)((uint8_t *)pAddress + 2160066560);
        ck[0] = (uint8_t *)((uint8_t *)pAddress + 2193620992);
        ck[1] = (uint8_t *)((uint8_t *)pAddress + 2197815296);
        ck[2] = (uint8_t *)((uint8_t *)pAddress + 2202009600);
        ck[3] = (uint8_t *)((uint8_t *)pAddress + 2206203904);
        ck[4] = (uint8_t *)((uint8_t *)pAddress + 2210398208);
        ck[5] = (uint8_t *)((uint8_t *)pAddress + 2214592512);
        ck[6] = (uint8_t *)((uint8_t *)pAddress + 2218786816);
        ck[7] = (uint8_t *)((uint8_t *)pAddress + 2222981120);
        ck[8] = (uint8_t *)((uint8_t *)pAddress + 2227175424);
        ck[9] = (uint8_t *)((uint8_t *)pAddress + 2231369728);
        ck[10] = (uint8_t *)((uint8_t *)pAddress + 2235564032);
        ck[11] = (uint8_t *)((uint8_t *)pAddress + 2239758336);
        ck[12] = (uint8_t *)((uint8_t *)pAddress + 2243952640);
        ck[13] = (uint8_t *)((uint8_t *)pAddress + 2248146944);
        ck[14] = (uint8_t *)((uint8_t *)pAddress + 2252341248);
        ck[15] = (uint8_t *)((uint8_t *)pAddress + 2256535552);
        ck[16] = (uint8_t *)((uint8_t *)pAddress + 2260729856);
        ck[17] = (uint8_t *)((uint8_t *)pAddress + 2264924160);
        ck[18] = (uint8_t *)((uint8_t *)pAddress + 2269118464);
        ck[19] = (uint8_t *)((uint8_t *)pAddress + 2273312768);
        ck[20] = (uint8_t *)((uint8_t *)pAddress + 2277507072);
        ck[21] = (uint8_t *)((uint8_t *)pAddress + 2281701376);
        ck[22] = (uint8_t *)((uint8_t *)pAddress + 2285895680);
        ck[23] = (uint8_t *)((uint8_t *)pAddress + 2290089984);
        ck[24] = (uint8_t *)((uint8_t *)pAddress + 2294284288);
        ck[25] = (uint8_t *)((uint8_t *)pAddress + 2298478592);
        ck[26] = (uint8_t *)((uint8_t *)pAddress + 2302672896);
        ck[27] = (uint8_t *)((uint8_t *)pAddress + 2306867200);
        ck[28] = (uint8_t *)((uint8_t *)pAddress + 2311061504);
        ck[29] = (uint8_t *)((uint8_t *)pAddress + 2315255808);
        ck[30] = (uint8_t *)((uint8_t *)pAddress + 2319450112);
        ck[31] = (uint8_t *)((uint8_t *)pAddress + 2323644416);
    }

    static uint64_t degree (void) { return 4194304; }
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
        P_OPCODE = (uint8_t *)((uint8_t *)pAddress + 2327838720);
        P_A = (uint8_t *)((uint8_t *)pAddress + 2332033024);
        P_B = (uint8_t *)((uint8_t *)pAddress + 2336227328);
        P_CIN = (uint8_t *)((uint8_t *)pAddress + 2340421632);
        P_LAST = (uint8_t *)((uint8_t *)pAddress + 2344615936);
        P_C = (uint8_t *)((uint8_t *)pAddress + 2348810240);
        P_COUT = (uint8_t *)((uint8_t *)pAddress + 2353004544);
        RESET = (uint8_t *)((uint8_t *)pAddress + 2357198848);
        FACTOR[0] = (uint32_t *)((uint8_t *)pAddress + 2361393152);
        FACTOR[1] = (uint32_t *)((uint8_t *)pAddress + 2394947584);
        FACTOR[2] = (uint32_t *)((uint8_t *)pAddress + 2428502016);
        FACTOR[3] = (uint32_t *)((uint8_t *)pAddress + 2462056448);
        FACTOR[4] = (uint32_t *)((uint8_t *)pAddress + 2495610880);
        FACTOR[5] = (uint32_t *)((uint8_t *)pAddress + 2529165312);
        FACTOR[6] = (uint32_t *)((uint8_t *)pAddress + 2562719744);
        FACTOR[7] = (uint32_t *)((uint8_t *)pAddress + 2596274176);
    }

    static uint64_t degree (void) { return 4194304; }
};

class MemConstantPols
{
public:
    FieldElement * INCS;
    FieldElement * ISNOTLAST;

    MemConstantPols (void * pAddress)
    {
        INCS = (FieldElement *)((uint8_t *)pAddress + 2629828608);
        ISNOTLAST = (FieldElement *)((uint8_t *)pAddress + 2663383040);
    }

    static uint64_t degree (void) { return 4194304; }
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
        LAST = (FieldElement *)((uint8_t *)pAddress + 2696937472);
        LATCH = (FieldElement *)((uint8_t *)pAddress + 2730491904);
        LASTBLOCK = (FieldElement *)((uint8_t *)pAddress + 2764046336);
        PARTIAL = (FieldElement *)((uint8_t *)pAddress + 2797600768);
        C[0] = (FieldElement *)((uint8_t *)pAddress + 2831155200);
        C[1] = (FieldElement *)((uint8_t *)pAddress + 2864709632);
        C[2] = (FieldElement *)((uint8_t *)pAddress + 2898264064);
        C[3] = (FieldElement *)((uint8_t *)pAddress + 2931818496);
        C[4] = (FieldElement *)((uint8_t *)pAddress + 2965372928);
        C[5] = (FieldElement *)((uint8_t *)pAddress + 2998927360);
        C[6] = (FieldElement *)((uint8_t *)pAddress + 3032481792);
        C[7] = (FieldElement *)((uint8_t *)pAddress + 3066036224);
        C[8] = (FieldElement *)((uint8_t *)pAddress + 3099590656);
        C[9] = (FieldElement *)((uint8_t *)pAddress + 3133145088);
        C[10] = (FieldElement *)((uint8_t *)pAddress + 3166699520);
        C[11] = (FieldElement *)((uint8_t *)pAddress + 3200253952);
    }

    static uint64_t degree (void) { return 4194304; }
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
        rHash = (FieldElement *)((uint8_t *)pAddress + 3233808384);
        rHashType = (FieldElement *)((uint8_t *)pAddress + 3267362816);
        rLatchGet = (FieldElement *)((uint8_t *)pAddress + 3300917248);
        rLatchSet = (FieldElement *)((uint8_t *)pAddress + 3334471680);
        rClimbRkey = (FieldElement *)((uint8_t *)pAddress + 3368026112);
        rClimbSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 3401580544);
        rClimbSiblingRkeyN = (FieldElement *)((uint8_t *)pAddress + 3435134976);
        rRotateLevel = (FieldElement *)((uint8_t *)pAddress + 3468689408);
        rJmpz = (FieldElement *)((uint8_t *)pAddress + 3502243840);
        rJmp = (FieldElement *)((uint8_t *)pAddress + 3535798272);
        rConst0 = (FieldElement *)((uint8_t *)pAddress + 3569352704);
        rConst1 = (FieldElement *)((uint8_t *)pAddress + 3602907136);
        rConst2 = (FieldElement *)((uint8_t *)pAddress + 3636461568);
        rConst3 = (FieldElement *)((uint8_t *)pAddress + 3670016000);
        rAddress = (FieldElement *)((uint8_t *)pAddress + 3703570432);
        rLine = (FieldElement *)((uint8_t *)pAddress + 3737124864);
    }

    static uint64_t degree (void) { return 4194304; }
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
        Value3 = (FieldElement *)((uint8_t *)pAddress + 3770679296);
        Value3Norm = (FieldElement *)((uint8_t *)pAddress + 3804233728);
        Gate9Type = (FieldElement *)((uint8_t *)pAddress + 3837788160);
        Gate9A = (FieldElement *)((uint8_t *)pAddress + 3871342592);
        Gate9B = (FieldElement *)((uint8_t *)pAddress + 3904897024);
        Gate9C = (FieldElement *)((uint8_t *)pAddress + 3938451456);
        Latch = (FieldElement *)((uint8_t *)pAddress + 3972005888);
        Factor = (FieldElement *)((uint8_t *)pAddress + 4005560320);
    }

    static uint64_t degree (void) { return 4194304; }
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
        ConnA = (FieldElement *)((uint8_t *)pAddress + 4039114752);
        ConnB = (FieldElement *)((uint8_t *)pAddress + 4072669184);
        ConnC = (FieldElement *)((uint8_t *)pAddress + 4106223616);
        NormalizedGate = (FieldElement *)((uint8_t *)pAddress + 4139778048);
        GateType = (FieldElement *)((uint8_t *)pAddress + 4173332480);
    }

    static uint64_t degree (void) { return 4194304; }
};

class Nine2OneConstantPols
{
public:
    FieldElement * Field9latch;
    FieldElement * Factor;

    Nine2OneConstantPols (void * pAddress)
    {
        Field9latch = (FieldElement *)((uint8_t *)pAddress + 4206886912);
        Factor = (FieldElement *)((uint8_t *)pAddress + 4240441344);
    }

    static uint64_t degree (void) { return 4194304; }
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
        r8Id = (FieldElement *)((uint8_t *)pAddress + 4273995776);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 4307550208);
        latchR8 = (FieldElement *)((uint8_t *)pAddress + 4341104640);
        Fr8 = (FieldElement *)((uint8_t *)pAddress + 4374659072);
        rBitValid = (FieldElement *)((uint8_t *)pAddress + 4408213504);
        latchSOut = (FieldElement *)((uint8_t *)pAddress + 4441767936);
        FSOut0 = (FieldElement *)((uint8_t *)pAddress + 4475322368);
        FSOut1 = (FieldElement *)((uint8_t *)pAddress + 4508876800);
        FSOut2 = (FieldElement *)((uint8_t *)pAddress + 4542431232);
        FSOut3 = (FieldElement *)((uint8_t *)pAddress + 4575985664);
        FSOut4 = (FieldElement *)((uint8_t *)pAddress + 4609540096);
        FSOut5 = (FieldElement *)((uint8_t *)pAddress + 4643094528);
        FSOut6 = (FieldElement *)((uint8_t *)pAddress + 4676648960);
        FSOut7 = (FieldElement *)((uint8_t *)pAddress + 4710203392);
        ConnSOutBit = (FieldElement *)((uint8_t *)pAddress + 4743757824);
        ConnSInBit = (FieldElement *)((uint8_t *)pAddress + 4777312256);
        ConnNine2OneBit = (FieldElement *)((uint8_t *)pAddress + 4810866688);
    }

    static uint64_t degree (void) { return 4194304; }
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
        r8Id = (FieldElement *)((uint8_t *)pAddress + 4844421120);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 4877975552);
        lastBlockLatch = (FieldElement *)((uint8_t *)pAddress + 4911529984);
        r8valid = (FieldElement *)((uint8_t *)pAddress + 4945084416);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 4978638848);
        forceLastHash = (FieldElement *)((uint8_t *)pAddress + 5012193280);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 5045747712);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 5079302144);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 5112856576);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 5146411008);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 5179965440);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 5213519872);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 5247074304);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 5280628736);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 5314183168);
        crValid = (FieldElement *)((uint8_t *)pAddress + 5347737600);
    }

    static uint64_t degree (void) { return 4194304; }
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
        F[0] = (FieldElement *)((uint8_t *)pAddress + 5381292032);
        F[1] = (FieldElement *)((uint8_t *)pAddress + 5414846464);
        F[2] = (FieldElement *)((uint8_t *)pAddress + 5448400896);
        F[3] = (FieldElement *)((uint8_t *)pAddress + 5481955328);
        F[4] = (FieldElement *)((uint8_t *)pAddress + 5515509760);
        F[5] = (FieldElement *)((uint8_t *)pAddress + 5549064192);
        F[6] = (FieldElement *)((uint8_t *)pAddress + 5582618624);
        F[7] = (FieldElement *)((uint8_t *)pAddress + 5616173056);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 5649727488);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 5683281920);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 5716836352);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 5750390784);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 5783945216);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 5817499648);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 5851054080);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 5884608512);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 5918162944);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 5951717376);
    }

    static uint64_t degree (void) { return 4194304; }
};

class MainConstantPols
{
public:
    uint32_t * STEP;

    MainConstantPols (void * pAddress)
    {
        STEP = (uint32_t *)((uint8_t *)pAddress + 5985271808);
    }

    static uint64_t degree (void) { return 4194304; }
};

class ConstantPols
{
public:
    MemAlignConstantPols MemAlign;
    GlobalConstantPols Global;
    Byte4ConstantPols Byte4;
    RomConstantPols Rom;
    ArithConstantPols Arith;
    BinaryConstantPols Binary;
    MemConstantPols Mem;
    PoseidonGConstantPols PoseidonG;
    StorageConstantPols Storage;
    NormGate9ConstantPols NormGate9;
    KeccakFConstantPols KeccakF;
    Nine2OneConstantPols Nine2One;
    PaddingKKBitConstantPols PaddingKKBit;
    PaddingKKConstantPols PaddingKK;
    PaddingPGConstantPols PaddingPG;
    MainConstantPols Main;

    ConstantPols (void * pAddress) : MemAlign(pAddress), Global(pAddress), Byte4(pAddress), Rom(pAddress), Arith(pAddress), Binary(pAddress), Mem(pAddress), PoseidonG(pAddress), Storage(pAddress), NormGate9(pAddress), KeccakF(pAddress), Nine2One(pAddress), PaddingKKBit(pAddress), PaddingKK(pAddress), PaddingPG(pAddress), Main(pAddress) {}

    static uint64_t size (void) { return 6018826240; }
};

#endif // CONSTANT_POLS_HPP
