#ifndef ROM_LINE_HPP_fork_10_blob
#define ROM_LINE_HPP_fork_10_blob

#include <vector>
#include "gmpxx.h"
#include "goldilocks_base_field.hpp"
#include "main_sm/fork_10_blob/main/rom_command.hpp"

using namespace std;

namespace fork_10_blob
{

// This class defines each of the ctx.rom[i] memory structures that contains the corresponding ROM line data
class RomLine {
public:
    string fileName;
    uint64_t line;
    string lineStr;
    vector<RomCommand *> cmdBefore;
    Goldilocks::Element inA;
    Goldilocks::Element inB;
    Goldilocks::Element inC;
    Goldilocks::Element inD;
    Goldilocks::Element inE;
    Goldilocks::Element inSR;
    Goldilocks::Element inCTX;
    Goldilocks::Element inSP;
    Goldilocks::Element inPC;
    Goldilocks::Element inGAS;
    Goldilocks::Element inSTEP;
    Goldilocks::Element inFREE;
    Goldilocks::Element inFREE0;
    Goldilocks::Element inRR;
    Goldilocks::Element inHASHPOS;
    Goldilocks::Element inCntArith;
    Goldilocks::Element inCntBinary;
    Goldilocks::Element inCntMemAlign;
    Goldilocks::Element inCntKeccakF;
    Goldilocks::Element inCntSha256F;
    Goldilocks::Element inCntPoseidonG;
    Goldilocks::Element inCntPaddingPG;
    Goldilocks::Element inROTL_C;
    Goldilocks::Element inRCX;
    Goldilocks::Element inRID;
    bool bConstPresent;
    Goldilocks::Element CONST;
    bool bConstLPresent;
    mpz_class CONSTL;
    bool bCondConstPresent;
    Goldilocks::Element condConst;
    bool bJmpAddrPresent;
    Goldilocks::Element jmpAddr;
    bool bElseAddrPresent;
    Goldilocks::Element elseAddr;
    string elseAddrLabel;
    uint8_t elseUseAddrRel;
    uint8_t mOp;
    uint8_t mWR;
    uint8_t memUseAddrRel;
    uint8_t assumeFree;
    uint64_t hashBytes;
    uint8_t hashBytesInD;
    uint8_t hashK;
    uint8_t hashKLen;
    uint8_t hashKDigest;
    uint8_t hashP;
    uint8_t hashPLen;
    uint8_t hashPDigest;
    uint8_t hashS;
    uint8_t hashSLen;
    uint8_t hashSDigest;
    uint8_t JMP;
    uint8_t JMPC;
    uint8_t JMPN;
    uint8_t JMPZ;
    uint8_t call;
    uint8_t return_;
    uint8_t save;
    uint8_t restore;
    uint8_t jmpUseAddrRel;
    uint8_t useElseAddr;
    bool bOffsetPresent;
    int32_t offset;
    string offsetLabel;
    uint8_t useCTX;
    uint8_t isStack;
    uint8_t isMem;
    int32_t incStack;
    int32_t hashOffset;
    Goldilocks::Element ind;
    Goldilocks::Element indRR;
    RomCommand freeInTag;
    uint8_t assert;
    uint8_t setA;
    uint8_t setB;
    uint8_t setC;
    uint8_t setD;
    uint8_t setE;
    uint8_t setSR;
    uint8_t setCTX;
    uint8_t setSP;
    uint8_t setPC;
    uint8_t setGAS;
    uint8_t setRR;
    uint8_t setHASHPOS;
    uint8_t setRCX;
    uint8_t setRID;
    uint8_t sRD;
    uint8_t sWR;
    uint8_t arith;
    uint8_t arithEquation;
    uint8_t bin;
    uint8_t binOpcode;
    vector<RomCommand *> cmdAfter;
    uint8_t memAlignRD;
    uint8_t memAlignWR;
    uint8_t repeat;
    uint8_t free0IsByte;
    uint8_t mode384;

    string toString(Goldilocks &fr);
};

} // namespace

#endif
