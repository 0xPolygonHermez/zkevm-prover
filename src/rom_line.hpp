#ifndef ROM_LINE_HPP
#define ROM_LINE_HPP

#include <vector>
#include "rom_command.hpp"

using namespace std;

// This class defines each of the ctx.rom[i] memory structures that contains the corresponding ROM line data
class RomLine {
public:
    string fileName;
    uint64_t line;
    vector<RomCommand *> cmdBefore;
    int32_t inA;
    int32_t inB;
    int32_t inC;
    int32_t inD;
    int32_t inE;
    int32_t inSR;
    int32_t inCTX;
    int32_t inSP;
    int32_t inPC;
    int32_t inGAS;
    int32_t inMAXMEM;
    int32_t inSTEP;
    uint8_t bConstPresent;
    int32_t CONST;
    uint8_t mRD;
    uint8_t mWR;
    uint8_t hashRD;
    uint8_t hashWR;
    uint8_t hashE;
    uint8_t JMP;
    uint8_t JMPC;
    uint8_t bOffsetPresent;
    uint32_t offset;
    uint8_t useCTX;
    uint8_t isCode;
    uint8_t isStack;
    uint8_t isMem;
    int32_t incCode;
    int32_t incStack;
    uint8_t ind;
    int32_t inFREE;
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
    uint8_t sRD;
    uint8_t sWR;
    uint8_t arith;
    uint8_t bin;
    uint8_t comparator;
    uint8_t opcodeRomMap;
    vector<RomCommand *> cmdAfter;
};

#endif