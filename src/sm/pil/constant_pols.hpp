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

    static uint64_t degree (void) { return 4194304; }
};

class Byte4ConstantPols
{
public:
    uint8_t * SET;

    Byte4ConstantPols (void * pAddress)
    {
        SET = (uint8_t *)((uint8_t *)pAddress + 138412032);
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
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 142606336);
        CONST1 = (uint32_t *)((uint8_t *)pAddress + 176160768);
        CONST2 = (uint32_t *)((uint8_t *)pAddress + 209715200);
        CONST3 = (uint32_t *)((uint8_t *)pAddress + 243269632);
        CONST4 = (uint32_t *)((uint8_t *)pAddress + 276824064);
        CONST5 = (uint32_t *)((uint8_t *)pAddress + 310378496);
        CONST6 = (uint32_t *)((uint8_t *)pAddress + 343932928);
        CONST7 = (uint32_t *)((uint8_t *)pAddress + 377487360);
        offset = (uint32_t *)((uint8_t *)pAddress + 411041792);
        inA = (FieldElement *)((uint8_t *)pAddress + 444596224);
        inB = (FieldElement *)((uint8_t *)pAddress + 478150656);
        inC = (FieldElement *)((uint8_t *)pAddress + 511705088);
        inD = (FieldElement *)((uint8_t *)pAddress + 545259520);
        inE = (FieldElement *)((uint8_t *)pAddress + 578813952);
        inSR = (FieldElement *)((uint8_t *)pAddress + 612368384);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 645922816);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 679477248);
        inSP = (FieldElement *)((uint8_t *)pAddress + 713031680);
        inPC = (FieldElement *)((uint8_t *)pAddress + 746586112);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 780140544);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 813694976);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 847249408);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 880803840);
        inRR = (FieldElement *)((uint8_t *)pAddress + 914358272);
        setA = (uint8_t *)((uint8_t *)pAddress + 947912704);
        setB = (uint8_t *)((uint8_t *)pAddress + 952107008);
        setC = (uint8_t *)((uint8_t *)pAddress + 956301312);
        setD = (uint8_t *)((uint8_t *)pAddress + 960495616);
        setE = (uint8_t *)((uint8_t *)pAddress + 964689920);
        setSR = (uint8_t *)((uint8_t *)pAddress + 968884224);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 973078528);
        setSP = (uint8_t *)((uint8_t *)pAddress + 977272832);
        setPC = (uint8_t *)((uint8_t *)pAddress + 981467136);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 985661440);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 989855744);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 994050048);
        JMP = (uint8_t *)((uint8_t *)pAddress + 998244352);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 1002438656);
        setRR = (uint8_t *)((uint8_t *)pAddress + 1006632960);
        incStack = (int32_t *)((uint8_t *)pAddress + 1010827264);
        incCode = (int32_t *)((uint8_t *)pAddress + 1027604480);
        isStack = (uint8_t *)((uint8_t *)pAddress + 1044381696);
        isCode = (uint8_t *)((uint8_t *)pAddress + 1048576000);
        isMem = (uint8_t *)((uint8_t *)pAddress + 1052770304);
        ind = (uint8_t *)((uint8_t *)pAddress + 1056964608);
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
        BIT19 = (FieldElement *)((uint8_t *)pAddress + 1224736768);
        GL_SIGNED_4BITS = (FieldElement *)((uint8_t *)pAddress + 1258291200);
        GL_SIGNED_18BITS = (FieldElement *)((uint8_t *)pAddress + 1291845632);
        ck[0] = (uint8_t *)((uint8_t *)pAddress + 1325400064);
        ck[1] = (uint8_t *)((uint8_t *)pAddress + 1329594368);
        ck[2] = (uint8_t *)((uint8_t *)pAddress + 1333788672);
        ck[3] = (uint8_t *)((uint8_t *)pAddress + 1337982976);
        ck[4] = (uint8_t *)((uint8_t *)pAddress + 1342177280);
        ck[5] = (uint8_t *)((uint8_t *)pAddress + 1346371584);
        ck[6] = (uint8_t *)((uint8_t *)pAddress + 1350565888);
        ck[7] = (uint8_t *)((uint8_t *)pAddress + 1354760192);
        ck[8] = (uint8_t *)((uint8_t *)pAddress + 1358954496);
        ck[9] = (uint8_t *)((uint8_t *)pAddress + 1363148800);
        ck[10] = (uint8_t *)((uint8_t *)pAddress + 1367343104);
        ck[11] = (uint8_t *)((uint8_t *)pAddress + 1371537408);
        ck[12] = (uint8_t *)((uint8_t *)pAddress + 1375731712);
        ck[13] = (uint8_t *)((uint8_t *)pAddress + 1379926016);
        ck[14] = (uint8_t *)((uint8_t *)pAddress + 1384120320);
        ck[15] = (uint8_t *)((uint8_t *)pAddress + 1388314624);
        ck[16] = (uint8_t *)((uint8_t *)pAddress + 1392508928);
        ck[17] = (uint8_t *)((uint8_t *)pAddress + 1396703232);
        ck[18] = (uint8_t *)((uint8_t *)pAddress + 1400897536);
        ck[19] = (uint8_t *)((uint8_t *)pAddress + 1405091840);
        ck[20] = (uint8_t *)((uint8_t *)pAddress + 1409286144);
        ck[21] = (uint8_t *)((uint8_t *)pAddress + 1413480448);
        ck[22] = (uint8_t *)((uint8_t *)pAddress + 1417674752);
        ck[23] = (uint8_t *)((uint8_t *)pAddress + 1421869056);
        ck[24] = (uint8_t *)((uint8_t *)pAddress + 1426063360);
        ck[25] = (uint8_t *)((uint8_t *)pAddress + 1430257664);
        ck[26] = (uint8_t *)((uint8_t *)pAddress + 1434451968);
        ck[27] = (uint8_t *)((uint8_t *)pAddress + 1438646272);
        ck[28] = (uint8_t *)((uint8_t *)pAddress + 1442840576);
        ck[29] = (uint8_t *)((uint8_t *)pAddress + 1447034880);
        ck[30] = (uint8_t *)((uint8_t *)pAddress + 1451229184);
        ck[31] = (uint8_t *)((uint8_t *)pAddress + 1455423488);
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
        P_OPCODE = (uint8_t *)((uint8_t *)pAddress + 1459617792);
        P_A = (uint8_t *)((uint8_t *)pAddress + 1463812096);
        P_B = (uint8_t *)((uint8_t *)pAddress + 1468006400);
        P_CIN = (uint8_t *)((uint8_t *)pAddress + 1472200704);
        P_LAST = (uint8_t *)((uint8_t *)pAddress + 1476395008);
        P_C = (uint8_t *)((uint8_t *)pAddress + 1480589312);
        P_COUT = (uint8_t *)((uint8_t *)pAddress + 1484783616);
        RESET = (uint8_t *)((uint8_t *)pAddress + 1488977920);
        FACTOR[0] = (uint32_t *)((uint8_t *)pAddress + 1493172224);
        FACTOR[1] = (uint32_t *)((uint8_t *)pAddress + 1526726656);
        FACTOR[2] = (uint32_t *)((uint8_t *)pAddress + 1560281088);
        FACTOR[3] = (uint32_t *)((uint8_t *)pAddress + 1593835520);
        FACTOR[4] = (uint32_t *)((uint8_t *)pAddress + 1627389952);
        FACTOR[5] = (uint32_t *)((uint8_t *)pAddress + 1660944384);
        FACTOR[6] = (uint32_t *)((uint8_t *)pAddress + 1694498816);
        FACTOR[7] = (uint32_t *)((uint8_t *)pAddress + 1728053248);
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
        INCS = (FieldElement *)((uint8_t *)pAddress + 1761607680);
        ISNOTLAST = (FieldElement *)((uint8_t *)pAddress + 1795162112);
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
        LAST = (FieldElement *)((uint8_t *)pAddress + 1828716544);
        LATCH = (FieldElement *)((uint8_t *)pAddress + 1862270976);
        LASTBLOCK = (FieldElement *)((uint8_t *)pAddress + 1895825408);
        PARTIAL = (FieldElement *)((uint8_t *)pAddress + 1929379840);
        C[0] = (FieldElement *)((uint8_t *)pAddress + 1962934272);
        C[1] = (FieldElement *)((uint8_t *)pAddress + 1996488704);
        C[2] = (FieldElement *)((uint8_t *)pAddress + 2030043136);
        C[3] = (FieldElement *)((uint8_t *)pAddress + 2063597568);
        C[4] = (FieldElement *)((uint8_t *)pAddress + 2097152000);
        C[5] = (FieldElement *)((uint8_t *)pAddress + 2130706432);
        C[6] = (FieldElement *)((uint8_t *)pAddress + 2164260864);
        C[7] = (FieldElement *)((uint8_t *)pAddress + 2197815296);
        C[8] = (FieldElement *)((uint8_t *)pAddress + 2231369728);
        C[9] = (FieldElement *)((uint8_t *)pAddress + 2264924160);
        C[10] = (FieldElement *)((uint8_t *)pAddress + 2298478592);
        C[11] = (FieldElement *)((uint8_t *)pAddress + 2332033024);
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
        rHash = (FieldElement *)((uint8_t *)pAddress + 2365587456);
        rHashType = (FieldElement *)((uint8_t *)pAddress + 2399141888);
        rLatchGet = (FieldElement *)((uint8_t *)pAddress + 2432696320);
        rLatchSet = (FieldElement *)((uint8_t *)pAddress + 2466250752);
        rClimbRkey = (FieldElement *)((uint8_t *)pAddress + 2499805184);
        rClimbSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 2533359616);
        rRotateLevel = (FieldElement *)((uint8_t *)pAddress + 2566914048);
        rJmpz = (FieldElement *)((uint8_t *)pAddress + 2600468480);
        rJmp = (FieldElement *)((uint8_t *)pAddress + 2634022912);
        rConst0 = (FieldElement *)((uint8_t *)pAddress + 2667577344);
        rConst1 = (FieldElement *)((uint8_t *)pAddress + 2701131776);
        rConst2 = (FieldElement *)((uint8_t *)pAddress + 2734686208);
        rConst3 = (FieldElement *)((uint8_t *)pAddress + 2768240640);
        rAddress = (FieldElement *)((uint8_t *)pAddress + 2801795072);
        rLine = (FieldElement *)((uint8_t *)pAddress + 2835349504);
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
        Value3 = (FieldElement *)((uint8_t *)pAddress + 2868903936);
        Value3Norm = (FieldElement *)((uint8_t *)pAddress + 2902458368);
        Gate9Type = (FieldElement *)((uint8_t *)pAddress + 2936012800);
        Gate9A = (FieldElement *)((uint8_t *)pAddress + 2969567232);
        Gate9B = (FieldElement *)((uint8_t *)pAddress + 3003121664);
        Gate9C = (FieldElement *)((uint8_t *)pAddress + 3036676096);
        Latch = (FieldElement *)((uint8_t *)pAddress + 3070230528);
        Factor = (FieldElement *)((uint8_t *)pAddress + 3103784960);
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
        ConnA = (FieldElement *)((uint8_t *)pAddress + 3137339392);
        ConnB = (FieldElement *)((uint8_t *)pAddress + 3170893824);
        ConnC = (FieldElement *)((uint8_t *)pAddress + 3204448256);
        NormalizedGate = (FieldElement *)((uint8_t *)pAddress + 3238002688);
        GateType = (FieldElement *)((uint8_t *)pAddress + 3271557120);
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
        Field9latch = (FieldElement *)((uint8_t *)pAddress + 3305111552);
        Factor = (FieldElement *)((uint8_t *)pAddress + 3338665984);
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
        r8Id = (FieldElement *)((uint8_t *)pAddress + 3372220416);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 3405774848);
        latchR8 = (FieldElement *)((uint8_t *)pAddress + 3439329280);
        Fr8 = (FieldElement *)((uint8_t *)pAddress + 3472883712);
        rBitValid = (FieldElement *)((uint8_t *)pAddress + 3506438144);
        latchSOut = (FieldElement *)((uint8_t *)pAddress + 3539992576);
        FSOut0 = (FieldElement *)((uint8_t *)pAddress + 3573547008);
        FSOut1 = (FieldElement *)((uint8_t *)pAddress + 3607101440);
        FSOut2 = (FieldElement *)((uint8_t *)pAddress + 3640655872);
        FSOut3 = (FieldElement *)((uint8_t *)pAddress + 3674210304);
        FSOut4 = (FieldElement *)((uint8_t *)pAddress + 3707764736);
        FSOut5 = (FieldElement *)((uint8_t *)pAddress + 3741319168);
        FSOut6 = (FieldElement *)((uint8_t *)pAddress + 3774873600);
        FSOut7 = (FieldElement *)((uint8_t *)pAddress + 3808428032);
        ConnSOutBit = (FieldElement *)((uint8_t *)pAddress + 3841982464);
        ConnSInBit = (FieldElement *)((uint8_t *)pAddress + 3875536896);
        ConnNine2OneBit = (FieldElement *)((uint8_t *)pAddress + 3909091328);
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
        r8Id = (FieldElement *)((uint8_t *)pAddress + 3942645760);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 3976200192);
        lastBlockLatch = (FieldElement *)((uint8_t *)pAddress + 4009754624);
        r8valid = (FieldElement *)((uint8_t *)pAddress + 4043309056);
        sOutId = (FieldElement *)((uint8_t *)pAddress + 4076863488);
        forceLastHash = (FieldElement *)((uint8_t *)pAddress + 4110417920);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 4143972352);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 4177526784);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 4211081216);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 4244635648);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 4278190080);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 4311744512);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 4345298944);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 4378853376);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 4412407808);
        crValid = (FieldElement *)((uint8_t *)pAddress + 4445962240);
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
        F[0] = (FieldElement *)((uint8_t *)pAddress + 4479516672);
        F[1] = (FieldElement *)((uint8_t *)pAddress + 4513071104);
        F[2] = (FieldElement *)((uint8_t *)pAddress + 4546625536);
        F[3] = (FieldElement *)((uint8_t *)pAddress + 4580179968);
        F[4] = (FieldElement *)((uint8_t *)pAddress + 4613734400);
        F[5] = (FieldElement *)((uint8_t *)pAddress + 4647288832);
        F[6] = (FieldElement *)((uint8_t *)pAddress + 4680843264);
        F[7] = (FieldElement *)((uint8_t *)pAddress + 4714397696);
        lastBlock = (FieldElement *)((uint8_t *)pAddress + 4747952128);
        k_crOffset = (FieldElement *)((uint8_t *)pAddress + 4781506560);
        k_crF0 = (FieldElement *)((uint8_t *)pAddress + 4815060992);
        k_crF1 = (FieldElement *)((uint8_t *)pAddress + 4848615424);
        k_crF2 = (FieldElement *)((uint8_t *)pAddress + 4882169856);
        k_crF3 = (FieldElement *)((uint8_t *)pAddress + 4915724288);
        k_crF4 = (FieldElement *)((uint8_t *)pAddress + 4949278720);
        k_crF5 = (FieldElement *)((uint8_t *)pAddress + 4982833152);
        k_crF6 = (FieldElement *)((uint8_t *)pAddress + 5016387584);
        k_crF7 = (FieldElement *)((uint8_t *)pAddress + 5049942016);
    }

    static uint64_t degree (void) { return 4194304; }
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
        BYTE2A = (uint8_t *)((uint8_t *)pAddress + 5083496448);
        BYTE2B = (uint8_t *)((uint8_t *)pAddress + 5087690752);
        FACTOR0[0] = (FieldElement *)((uint8_t *)pAddress + 5091885056);
        FACTOR0[1] = (FieldElement *)((uint8_t *)pAddress + 5125439488);
        FACTOR0[2] = (FieldElement *)((uint8_t *)pAddress + 5158993920);
        FACTOR0[3] = (FieldElement *)((uint8_t *)pAddress + 5192548352);
        FACTOR0[4] = (FieldElement *)((uint8_t *)pAddress + 5226102784);
        FACTOR0[5] = (FieldElement *)((uint8_t *)pAddress + 5259657216);
        FACTOR0[6] = (FieldElement *)((uint8_t *)pAddress + 5293211648);
        FACTOR0[7] = (FieldElement *)((uint8_t *)pAddress + 5326766080);
        FACTOR1[0] = (FieldElement *)((uint8_t *)pAddress + 5360320512);
        FACTOR1[1] = (FieldElement *)((uint8_t *)pAddress + 5393874944);
        FACTOR1[2] = (FieldElement *)((uint8_t *)pAddress + 5427429376);
        FACTOR1[3] = (FieldElement *)((uint8_t *)pAddress + 5460983808);
        FACTOR1[4] = (FieldElement *)((uint8_t *)pAddress + 5494538240);
        FACTOR1[5] = (FieldElement *)((uint8_t *)pAddress + 5528092672);
        FACTOR1[6] = (FieldElement *)((uint8_t *)pAddress + 5561647104);
        FACTOR1[7] = (FieldElement *)((uint8_t *)pAddress + 5595201536);
        FACTORV[0] = (FieldElement *)((uint8_t *)pAddress + 5628755968);
        FACTORV[1] = (FieldElement *)((uint8_t *)pAddress + 5662310400);
        FACTORV[2] = (FieldElement *)((uint8_t *)pAddress + 5695864832);
        FACTORV[3] = (FieldElement *)((uint8_t *)pAddress + 5729419264);
        FACTORV[4] = (FieldElement *)((uint8_t *)pAddress + 5762973696);
        FACTORV[5] = (FieldElement *)((uint8_t *)pAddress + 5796528128);
        FACTORV[6] = (FieldElement *)((uint8_t *)pAddress + 5830082560);
        FACTORV[7] = (FieldElement *)((uint8_t *)pAddress + 5863636992);
        STEP = (FieldElement *)((uint8_t *)pAddress + 5897191424);
        WR = (uint8_t *)((uint8_t *)pAddress + 5930745856);
        OFFSET = (uint8_t *)((uint8_t *)pAddress + 5934940160);
        RESET = (uint8_t *)((uint8_t *)pAddress + 5939134464);
        RESETL = (uint8_t *)((uint8_t *)pAddress + 5943328768);
        SELW = (uint8_t *)((uint8_t *)pAddress + 5947523072);
    }

    static uint64_t degree (void) { return 4194304; }
};

class MainConstantPols
{
public:
    uint32_t * STEP;

    MainConstantPols (void * pAddress)
    {
        STEP = (uint32_t *)((uint8_t *)pAddress + 5951717376);
    }

    static uint64_t degree (void) { return 4194304; }
};

class ConstantPols
{
public:
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
    MemAlignConstantPols MemAlign;
    MainConstantPols Main;

    ConstantPols (void * pAddress) : Global(pAddress), Byte4(pAddress), Rom(pAddress), Arith(pAddress), Binary(pAddress), Mem(pAddress), PoseidonG(pAddress), Storage(pAddress), NormGate9(pAddress), KeccakF(pAddress), Nine2One(pAddress), PaddingKKBit(pAddress), PaddingKK(pAddress), PaddingPG(pAddress), MemAlign(pAddress), Main(pAddress) {}

    static uint64_t size (void) { return 5985271808; }
};

#endif // CONSTANT_POLS_HPP
