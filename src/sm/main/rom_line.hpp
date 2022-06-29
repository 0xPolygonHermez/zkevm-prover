#ifndef ROM_LINE_HPP
#define ROM_LINE_HPP

#include <vector>
#include "gmpxx.h"
#include "goldilocks/goldilocks_base_field.hpp"
#include "rom_command.hpp"

using namespace std;

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
    Goldilocks::Element inMAXMEM;
    Goldilocks::Element inSTEP;
    Goldilocks::Element inFREE;
    Goldilocks::Element inRR;
    Goldilocks::Element inHASHPOS;
    Goldilocks::Element inCntArith;
    Goldilocks::Element inCntBinary;
    Goldilocks::Element inCntMemAlign;
    Goldilocks::Element inCntKeccakF;
    Goldilocks::Element inCntPoseidonG;
    Goldilocks::Element inCntPaddingPG;
    bool bConstPresent;
    Goldilocks::Element CONST;
    bool bConstLPresent;
    mpz_class CONSTL;
    uint8_t mOp;
    uint8_t mWR;
    uint8_t hashK;
    uint8_t hashKLen;
    uint8_t hashKDigest;
    uint8_t hashP;
    uint8_t hashPLen;
    uint8_t hashPDigest;
    uint8_t JMP;
    uint8_t JMPC;
    uint8_t JMPN;
    bool bOffsetPresent;
    uint32_t offset;
    string offsetLabel;
    uint8_t useCTX;
    uint8_t isCode;
    uint8_t isStack;
    uint8_t isMem;
    int32_t incCode;
    int32_t incStack;
    uint8_t ind;
    uint8_t indRR;
    RomCommand freeInTag;
    uint8_t ecRecover;
    uint8_t shl;
    uint8_t shr;
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
    uint8_t setMAXMEM;
    uint8_t setRR;
    uint8_t setHASHPOS;
    uint8_t sRD;
    uint8_t sWR;
    uint8_t arith;
    uint8_t arithEq0;
    uint8_t arithEq1;
    uint8_t arithEq2;
    uint8_t arithEq3;
    uint8_t bin;
    uint8_t binOpcode;
    uint8_t comparator;
    uint8_t opcodeRomMap;
    vector<RomCommand *> cmdAfter;
    uint8_t memAlign;
    uint8_t memAlignWR;
    uint8_t memAlignWR8;

    string toString(Goldilocks &fr);
};

#endif