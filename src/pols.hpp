#ifndef POLS_HPP
#define POLS_HPP

#include <nlohmann/json.hpp>
#include <string>
#include "config.hpp"
#include "definitions.hpp"
#include "pol_types.hpp"
#include "pil.hpp"

using namespace std;
using json = nlohmann::json;

class Pols
{
public:
    // Committed (output file) polynomials
    PolFieldElement A0;
    PolU32 A1;
    PolU32 A2;
    PolU32 A3;
    PolU32 A4;
    PolU32 A5;
    PolU32 A6;
    PolU32 A7;
    PolFieldElement B0;
    PolU32 B1;
    PolU32 B2;
    PolU32 B3;
    PolU32 B4;
    PolU32 B5;
    PolU32 B6;
    PolU32 B7;
    PolFieldElement C0;
    PolU32 C1;
    PolU32 C2;
    PolU32 C3;
    PolU32 C4;
    PolU32 C5;
    PolU32 C6;
    PolU32 C7;
    PolFieldElement D0;
    PolU32 D1;
    PolU32 D2;
    PolU32 D3;
    PolU32 D4;
    PolU32 D5;
    PolU32 D6;
    PolU32 D7;
    PolFieldElement E0;
    PolU32 E1;
    PolU32 E2;
    PolU32 E3;
    PolU32 E4;
    PolU32 E5;
    PolU32 E6;
    PolU32 E7;
    PolFieldElement FREE0;
    PolFieldElement FREE1;
    PolFieldElement FREE2;
    PolFieldElement FREE3;
    PolFieldElement FREE4;
    PolFieldElement FREE5;
    PolFieldElement FREE6;
    PolFieldElement FREE7;
    PolFieldElement CONST;
    PolU32 CTX;
    PolU64 GAS;
    PolBool JMP;
    PolBool JMPC;
    PolU32 MAXMEM;
    PolU32 PC;
    PolU32 RR;
    PolU16 SP;
    PolU32 SR0;
    PolU32 SR1;
    PolU32 SR2;
    PolU32 SR3;
    PolU32 SR4;
    PolU32 SR5;
    PolU32 SR6;
    PolU32 SR7;
    PolBool arith;
    PolBool assert;
    PolBool bin;
    PolBool comparator;
    PolBool ecRecover;
    PolBool hashE;
    PolBool hashRD;
    PolBool hashWR;
    PolFieldElement inA;
    PolFieldElement inB;
    PolFieldElement inC;
    PolFieldElement inD;
    PolFieldElement inE;
    PolFieldElement inCTX;
    PolFieldElement inFREE;
    PolFieldElement inGAS;
    PolFieldElement inMAXMEM;
    PolFieldElement inPC;
    PolFieldElement inSP;
    PolFieldElement inSR;
    PolFieldElement inSTEP;
    PolFieldElement inRR;
    PolS32 incCode;
    PolS32 incStack;
    PolBool ind;
    PolBool isCode;
    PolBool isMaxMem;
    PolBool isMem;
    PolBool isNeg;
    PolBool isStack;
    PolBool mRD;
    PolBool mWR;
    PolU32 offset;
    PolBool opcodeRomMap;
    PolBool sRD;
    PolBool sWR;
    PolBool setA;
    PolBool setB;
    PolBool setC;
    PolBool setD;
    PolBool setE;
    PolBool setCTX;
    PolBool setGAS;
    PolBool setMAXMEM;
    PolBool setPC;
    PolBool setSP;
    PolBool setSR;
    PolBool setRR;
    PolBool shl;
    PolBool shr;
    PolBool useCTX;
    PolU32 zkPC;
    PolU16 byte4_freeIN;
    PolU32 byte4_out;

    // Constant (input file) polynomials
    PolU16 global_byte2;
    PolBool global_L1;
    PolFieldElement global_ZH;
    PolFieldElement global_ZHINV;
    PolBool byte4_SET;
    PolFieldElement ROM;
    PolU32 STEP;
    PolFieldElement rom_CONST;
    PolBool rom_JMP;
    PolBool rom_JMPC;
    PolBool rom_arith;
    PolBool rom_assert;
    PolBool rom_bin;
    PolBool rom_comparator;
    PolBool rom_ecRecover;
    PolBool rom_hashE;
    PolBool rom_hashRD;
    PolBool rom_hashWR;
    PolFieldElement rom_inA;
    PolFieldElement rom_inB;
    PolFieldElement rom_inC;
    PolFieldElement rom_inD;
    PolFieldElement rom_inE;
    PolFieldElement rom_inCTX;
    PolFieldElement rom_inFREE;
    PolFieldElement rom_inGAS;
    PolFieldElement rom_inMAXMEM;
    PolFieldElement rom_inPC;
    PolFieldElement rom_inSP;
    PolFieldElement rom_inSR;
    PolFieldElement rom_inSTEP;
    PolFieldElement rom_inRR;
    PolS32 rom_incCode;
    PolS32 rom_incStack;
    PolBool rom_ind;
    PolBool rom_isCode;
    PolBool rom_isMem;
    PolBool rom_isStack;
    PolU32 rom_line;
    PolBool rom_mRD;
    PolBool rom_mWR;
    PolU32 rom_offset;
    PolBool rom_opcodeRomMap;
    PolBool rom_sRD;
    PolBool rom_sWR;
    PolBool rom_setA;
    PolBool rom_setB;
    PolBool rom_setC;
    PolBool rom_setD;
    PolBool rom_setE;
    PolBool rom_setCTX;
    PolBool rom_setGAS;
    PolBool rom_setMAXMEM;
    PolBool rom_setPC;
    PolBool rom_setSP;
    PolBool rom_setSR;
    PolBool rom_setRR;
    PolBool rom_shl;
    PolBool rom_shr;
    PolBool rom_useCTX;
    PolU32 rom_opCodeAddr;
    PolU8 rom_opCodeNum;

    // List of id-ordered pols, and its size
    Pol * orderedPols[NPOLS];
    uint64_t size;

    void load (const vector<PolJsonData> &v);
    void mapToOutputFile (const string &outputFileName, bool bFastMode = false);
    void mapToInputFile (const string &inputFileName, bool bFastMode = false);
    void unmap (bool bFastMode = false);

private:
    uint64_t polsSize;
    uint8_t * pPolsMappedMemmory;
    string fileName;
    Pol * find (const string &name);
    void addPol (const string &name, const uint64_t id, const string &elementType);
    void mapToFile (const string &fileName, bool bOutput, bool bFastMode = false);
};



#endif