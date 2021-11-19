#ifndef ROM_LINE_HPP
#define ROM_LINE_HPP

#include <vector>
#include "rom_command.hpp"

using namespace std;

class RomLine {
public:
    string fileName;
    uint64_t line;
    vector<RomCommand *> cmdBefore;
    uint8_t inA;
    uint8_t inB;
    uint8_t inC;
    uint8_t inD;
    uint8_t inE;
    uint8_t inSR;
    uint8_t inCTX;
    uint8_t inSP;
    uint8_t inPC;
    uint8_t inGAS;
    uint8_t inMAXMEM;
    uint8_t inSTEP;
    uint8_t bConstPresent;
    //uint64_t CONST; // TODO: Check type (signed)
    int32_t CONST;
    uint8_t mRD;
    uint8_t mWR;
    uint8_t hashRD;
    uint8_t hashWR;
    uint8_t hashE;
    uint8_t JMP;
    uint8_t JMPC;
    uint8_t bOffsetPresent;
    //int64_t offset; // TODO: Check type (signed)
    uint32_t offset;
    uint8_t useCTX; // TODO: Shouldn't it be isCTX or isContext?
    uint8_t isCode;
    uint8_t isStack;
    uint8_t isMem;
    uint8_t inc;
    uint8_t dec;
    uint8_t ind;
    uint8_t inFREE;
    RomCommand freeInTag;
    uint8_t ecRecover;
    uint8_t shl;
    uint8_t shr;
    uint8_t neg;
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