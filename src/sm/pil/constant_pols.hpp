#ifndef CONSTANT_POLS_HPP
#define CONSTANT_POLS_HPP

#include <cstdint>
#include "ff/ff.hpp"

class GLOBALConstantPols
{
public:
    FieldElement * ZH;
    FieldElement * ZHINV;
    uint8_t * L1;
    FieldElement * BYTE;
    uint16_t * BYTE2;

    GLOBALConstantPols (void * pAddress)
    {
        ZH = (FieldElement *)((uint8_t *)pAddress + 0);
        ZHINV = (FieldElement *)((uint8_t *)pAddress + 524288);
        L1 = (uint8_t *)((uint8_t *)pAddress + 1048576);
        BYTE = (FieldElement *)((uint8_t *)pAddress + 1114112);
        BYTE2 = (uint16_t *)((uint8_t *)pAddress + 1638400);
    }

    static uint64_t degree (void) { return 65536; }
};

class Byte4ConstantPols
{
public:
    uint8_t * SET;

    Byte4ConstantPols (void * pAddress)
    {
        SET = (uint8_t *)((uint8_t *)pAddress + 1769472);
    }

    static uint64_t degree (void) { return 65536; }
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
    uint32_t * line;
    uint8_t * opCodeNum;
    uint32_t * opCodeAddr;

    RomConstantPols (void * pAddress)
    {
        CONST0 = (FieldElement *)((uint8_t *)pAddress + 1835008);
        CONST1 = (uint32_t *)((uint8_t *)pAddress + 2359296);
        CONST2 = (uint32_t *)((uint8_t *)pAddress + 2883584);
        CONST3 = (uint32_t *)((uint8_t *)pAddress + 3407872);
        CONST4 = (uint32_t *)((uint8_t *)pAddress + 3932160);
        CONST5 = (uint32_t *)((uint8_t *)pAddress + 4456448);
        CONST6 = (uint32_t *)((uint8_t *)pAddress + 4980736);
        CONST7 = (uint32_t *)((uint8_t *)pAddress + 5505024);
        offset = (uint32_t *)((uint8_t *)pAddress + 6029312);
        inA = (FieldElement *)((uint8_t *)pAddress + 6553600);
        inB = (FieldElement *)((uint8_t *)pAddress + 7077888);
        inC = (FieldElement *)((uint8_t *)pAddress + 7602176);
        inD = (FieldElement *)((uint8_t *)pAddress + 8126464);
        inE = (FieldElement *)((uint8_t *)pAddress + 8650752);
        inSR = (FieldElement *)((uint8_t *)pAddress + 9175040);
        inFREE = (FieldElement *)((uint8_t *)pAddress + 9699328);
        inCTX = (FieldElement *)((uint8_t *)pAddress + 10223616);
        inSP = (FieldElement *)((uint8_t *)pAddress + 10747904);
        inPC = (FieldElement *)((uint8_t *)pAddress + 11272192);
        inGAS = (FieldElement *)((uint8_t *)pAddress + 11796480);
        inMAXMEM = (FieldElement *)((uint8_t *)pAddress + 12320768);
        inHASHPOS = (FieldElement *)((uint8_t *)pAddress + 12845056);
        inSTEP = (FieldElement *)((uint8_t *)pAddress + 13369344);
        inRR = (FieldElement *)((uint8_t *)pAddress + 13893632);
        setA = (uint8_t *)((uint8_t *)pAddress + 14417920);
        setB = (uint8_t *)((uint8_t *)pAddress + 14483456);
        setC = (uint8_t *)((uint8_t *)pAddress + 14548992);
        setD = (uint8_t *)((uint8_t *)pAddress + 14614528);
        setE = (uint8_t *)((uint8_t *)pAddress + 14680064);
        setSR = (uint8_t *)((uint8_t *)pAddress + 14745600);
        setCTX = (uint8_t *)((uint8_t *)pAddress + 14811136);
        setSP = (uint8_t *)((uint8_t *)pAddress + 14876672);
        setPC = (uint8_t *)((uint8_t *)pAddress + 14942208);
        setGAS = (uint8_t *)((uint8_t *)pAddress + 15007744);
        setMAXMEM = (uint8_t *)((uint8_t *)pAddress + 15073280);
        setHASHPOS = (uint8_t *)((uint8_t *)pAddress + 15138816);
        JMP = (uint8_t *)((uint8_t *)pAddress + 15204352);
        JMPC = (uint8_t *)((uint8_t *)pAddress + 15269888);
        setRR = (uint8_t *)((uint8_t *)pAddress + 15335424);
        incStack = (int32_t *)((uint8_t *)pAddress + 15400960);
        incCode = (int32_t *)((uint8_t *)pAddress + 15663104);
        isStack = (uint8_t *)((uint8_t *)pAddress + 15925248);
        isCode = (uint8_t *)((uint8_t *)pAddress + 15990784);
        isMem = (uint8_t *)((uint8_t *)pAddress + 16056320);
        ind = (uint8_t *)((uint8_t *)pAddress + 16121856);
        useCTX = (uint8_t *)((uint8_t *)pAddress + 16187392);
        mWR = (uint8_t *)((uint8_t *)pAddress + 16252928);
        mRD = (uint8_t *)((uint8_t *)pAddress + 16318464);
        sWR = (uint8_t *)((uint8_t *)pAddress + 16384000);
        sRD = (uint8_t *)((uint8_t *)pAddress + 16449536);
        arith = (uint8_t *)((uint8_t *)pAddress + 16515072);
        arithEq0 = (uint8_t *)((uint8_t *)pAddress + 16580608);
        arithEq1 = (uint8_t *)((uint8_t *)pAddress + 16646144);
        arithEq2 = (uint8_t *)((uint8_t *)pAddress + 16711680);
        arithEq3 = (uint8_t *)((uint8_t *)pAddress + 16777216);
        shl = (uint8_t *)((uint8_t *)pAddress + 16842752);
        shr = (uint8_t *)((uint8_t *)pAddress + 16908288);
        hashK = (uint8_t *)((uint8_t *)pAddress + 16973824);
        hashKLen = (uint8_t *)((uint8_t *)pAddress + 17039360);
        hashKDigest = (uint8_t *)((uint8_t *)pAddress + 17104896);
        hashP = (uint8_t *)((uint8_t *)pAddress + 17170432);
        hashPLen = (uint8_t *)((uint8_t *)pAddress + 17235968);
        hashPDigest = (uint8_t *)((uint8_t *)pAddress + 17301504);
        ecRecover = (uint8_t *)((uint8_t *)pAddress + 17367040);
        comparator = (uint8_t *)((uint8_t *)pAddress + 17432576);
        bin = (uint8_t *)((uint8_t *)pAddress + 17498112);
        binOpcode = (uint8_t *)((uint8_t *)pAddress + 17563648);
        assert = (uint8_t *)((uint8_t *)pAddress + 17629184);
        opcodeRomMap = (uint8_t *)((uint8_t *)pAddress + 17694720);
        line = (uint32_t *)((uint8_t *)pAddress + 17760256);
        opCodeNum = (uint8_t *)((uint8_t *)pAddress + 18284544);
        opCodeAddr = (uint32_t *)((uint8_t *)pAddress + 18350080);
    }

    static uint64_t degree (void) { return 65536; }
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
        BIT19 = (FieldElement *)((uint8_t *)pAddress + 18874368);
        GL_SIGNED_4BITS = (FieldElement *)((uint8_t *)pAddress + 23068672);
        GL_SIGNED_18BITS = (FieldElement *)((uint8_t *)pAddress + 27262976);
        ck[0] = (uint8_t *)((uint8_t *)pAddress + 31457280);
        ck[1] = (uint8_t *)((uint8_t *)pAddress + 31981568);
        ck[2] = (uint8_t *)((uint8_t *)pAddress + 32505856);
        ck[3] = (uint8_t *)((uint8_t *)pAddress + 33030144);
        ck[4] = (uint8_t *)((uint8_t *)pAddress + 33554432);
        ck[5] = (uint8_t *)((uint8_t *)pAddress + 34078720);
        ck[6] = (uint8_t *)((uint8_t *)pAddress + 34603008);
        ck[7] = (uint8_t *)((uint8_t *)pAddress + 35127296);
        ck[8] = (uint8_t *)((uint8_t *)pAddress + 35651584);
        ck[9] = (uint8_t *)((uint8_t *)pAddress + 36175872);
        ck[10] = (uint8_t *)((uint8_t *)pAddress + 36700160);
        ck[11] = (uint8_t *)((uint8_t *)pAddress + 37224448);
        ck[12] = (uint8_t *)((uint8_t *)pAddress + 37748736);
        ck[13] = (uint8_t *)((uint8_t *)pAddress + 38273024);
        ck[14] = (uint8_t *)((uint8_t *)pAddress + 38797312);
        ck[15] = (uint8_t *)((uint8_t *)pAddress + 39321600);
        ck[16] = (uint8_t *)((uint8_t *)pAddress + 39845888);
        ck[17] = (uint8_t *)((uint8_t *)pAddress + 40370176);
        ck[18] = (uint8_t *)((uint8_t *)pAddress + 40894464);
        ck[19] = (uint8_t *)((uint8_t *)pAddress + 41418752);
        ck[20] = (uint8_t *)((uint8_t *)pAddress + 41943040);
        ck[21] = (uint8_t *)((uint8_t *)pAddress + 42467328);
        ck[22] = (uint8_t *)((uint8_t *)pAddress + 42991616);
        ck[23] = (uint8_t *)((uint8_t *)pAddress + 43515904);
        ck[24] = (uint8_t *)((uint8_t *)pAddress + 44040192);
        ck[25] = (uint8_t *)((uint8_t *)pAddress + 44564480);
        ck[26] = (uint8_t *)((uint8_t *)pAddress + 45088768);
        ck[27] = (uint8_t *)((uint8_t *)pAddress + 45613056);
        ck[28] = (uint8_t *)((uint8_t *)pAddress + 46137344);
        ck[29] = (uint8_t *)((uint8_t *)pAddress + 46661632);
        ck[30] = (uint8_t *)((uint8_t *)pAddress + 47185920);
        ck[31] = (uint8_t *)((uint8_t *)pAddress + 47710208);
    }

    static uint64_t degree (void) { return 524288; }
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
        P_OPCODE = (uint8_t *)((uint8_t *)pAddress + 48234496);
        P_A = (uint8_t *)((uint8_t *)pAddress + 48300032);
        P_B = (uint8_t *)((uint8_t *)pAddress + 48365568);
        P_CIN = (uint8_t *)((uint8_t *)pAddress + 48431104);
        P_LAST = (uint8_t *)((uint8_t *)pAddress + 48496640);
        P_C = (uint8_t *)((uint8_t *)pAddress + 48562176);
        P_COUT = (uint8_t *)((uint8_t *)pAddress + 48627712);
        RESET = (uint8_t *)((uint8_t *)pAddress + 48693248);
        FACTOR[0] = (uint32_t *)((uint8_t *)pAddress + 48758784);
        FACTOR[1] = (uint32_t *)((uint8_t *)pAddress + 49283072);
        FACTOR[2] = (uint32_t *)((uint8_t *)pAddress + 49807360);
        FACTOR[3] = (uint32_t *)((uint8_t *)pAddress + 50331648);
        FACTOR[4] = (uint32_t *)((uint8_t *)pAddress + 50855936);
        FACTOR[5] = (uint32_t *)((uint8_t *)pAddress + 51380224);
        FACTOR[6] = (uint32_t *)((uint8_t *)pAddress + 51904512);
        FACTOR[7] = (uint32_t *)((uint8_t *)pAddress + 52428800);
    }

    static uint64_t degree (void) { return 65536; }
};

class RamConstantPols
{
public:
    FieldElement * INCS;
    FieldElement * ISNOTLAST;

    RamConstantPols (void * pAddress)
    {
        INCS = (FieldElement *)((uint8_t *)pAddress + 52953088);
        ISNOTLAST = (FieldElement *)((uint8_t *)pAddress + 53477376);
    }

    static uint64_t degree (void) { return 65536; }
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
        LAST = (FieldElement *)((uint8_t *)pAddress + 54001664);
        LATCH = (FieldElement *)((uint8_t *)pAddress + 54525952);
        LASTBLOCK = (FieldElement *)((uint8_t *)pAddress + 55050240);
        PARTIAL = (FieldElement *)((uint8_t *)pAddress + 55574528);
        C[0] = (FieldElement *)((uint8_t *)pAddress + 56098816);
        C[1] = (FieldElement *)((uint8_t *)pAddress + 56623104);
        C[2] = (FieldElement *)((uint8_t *)pAddress + 57147392);
        C[3] = (FieldElement *)((uint8_t *)pAddress + 57671680);
        C[4] = (FieldElement *)((uint8_t *)pAddress + 58195968);
        C[5] = (FieldElement *)((uint8_t *)pAddress + 58720256);
        C[6] = (FieldElement *)((uint8_t *)pAddress + 59244544);
        C[7] = (FieldElement *)((uint8_t *)pAddress + 59768832);
        C[8] = (FieldElement *)((uint8_t *)pAddress + 60293120);
        C[9] = (FieldElement *)((uint8_t *)pAddress + 60817408);
        C[10] = (FieldElement *)((uint8_t *)pAddress + 61341696);
        C[11] = (FieldElement *)((uint8_t *)pAddress + 61865984);
    }

    static uint64_t degree (void) { return 65536; }
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
        rHash = (FieldElement *)((uint8_t *)pAddress + 62390272);
        rHashType = (FieldElement *)((uint8_t *)pAddress + 62914560);
        rLatchGet = (FieldElement *)((uint8_t *)pAddress + 63438848);
        rLatchSet = (FieldElement *)((uint8_t *)pAddress + 63963136);
        rClimbRkey = (FieldElement *)((uint8_t *)pAddress + 64487424);
        rClimbSiblingRkey = (FieldElement *)((uint8_t *)pAddress + 65011712);
        rRotateLevel = (FieldElement *)((uint8_t *)pAddress + 65536000);
        rJmpz = (FieldElement *)((uint8_t *)pAddress + 66060288);
        rJmp = (FieldElement *)((uint8_t *)pAddress + 66584576);
        rConst0 = (FieldElement *)((uint8_t *)pAddress + 67108864);
        rConst1 = (FieldElement *)((uint8_t *)pAddress + 67633152);
        rConst2 = (FieldElement *)((uint8_t *)pAddress + 68157440);
        rConst3 = (FieldElement *)((uint8_t *)pAddress + 68681728);
        rAddress = (FieldElement *)((uint8_t *)pAddress + 69206016);
        rLine = (FieldElement *)((uint8_t *)pAddress + 69730304);
    }

    static uint64_t degree (void) { return 65536; }
};

class MainConstantPols
{
public:
    uint32_t * STEP;

    MainConstantPols (void * pAddress)
    {
        STEP = (uint32_t *)((uint8_t *)pAddress + 70254592);
    }

    static uint64_t degree (void) { return 65536; }
};

class ConstantPols
{
public:
    GLOBALConstantPols GLOBAL;
    Byte4ConstantPols Byte4;
    RomConstantPols Rom;
    ArithConstantPols Arith;
    BinaryConstantPols Binary;
    RamConstantPols Ram;
    PoseidonGConstantPols PoseidonG;
    StorageConstantPols Storage;
    MainConstantPols Main;

    ConstantPols (void * pAddress) : GLOBAL(pAddress), Byte4(pAddress), Rom(pAddress), Arith(pAddress), Binary(pAddress), Ram(pAddress), PoseidonG(pAddress), Storage(pAddress), Main(pAddress) {}

    static uint64_t size (void) { return 70778880; }
};

#endif // CONSTANT_POLS_HPP
