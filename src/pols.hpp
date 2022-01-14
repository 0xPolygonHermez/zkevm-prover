#ifndef POLS_HPP
#define POLS_HPP

#include <nlohmann/json.hpp>
#include <string>
#include "config.hpp"
#include "pol_types.hpp"
#include "pil.hpp"

using namespace std;
using json = nlohmann::json;

class Pols
{
public:
    // Committed (output file) polynomials
    PolFieldElement A0;
    PolU64 A1;
    PolU64 A2;
    PolU64 A3;
    PolFieldElement B0;
    PolU64 B1;
    PolU64 B2;
    PolU64 B3;
    PolFieldElement C0;
    PolU64 C1;
    PolU64 C2;
    PolU64 C3;
    PolFieldElement D0;
    PolU64 D1;
    PolU64 D2;
    PolU64 D3;
    PolFieldElement E0;
    PolU64 E1;
    PolU64 E2;
    PolU64 E3;
    PolFieldElement FREE0;
    PolFieldElement FREE1;
    PolFieldElement FREE2;
    PolFieldElement FREE3;
    PolS32 CONST;
    PolU32 CTX;
    PolU64 GAS;
    PolBool JMP;
    PolBool JMPC;
    PolU32 MAXMEM;
    PolU32 PC;
    PolU16 SP;
    PolFieldElement SR;
    PolBool arith;
    PolBool assert;
    PolBool bin;
    PolBool comparator;
    PolBool ecRecover;
    PolBool hashE;
    PolBool hashRD;
    PolBool hashWR;
    PolS32 inA;
    PolS32 inB;
    PolS32 inC;
    PolS32 inD;
    PolS32 inE;
    PolS32 inCTX;
    PolS32 inFREE;
    PolS32 inGAS;
    PolS32 inMAXMEM;
    PolS32 inPC;
    PolS32 inSP;
    PolS32 inSR;
    PolS32 inSTEP;
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
    PolS32 rom_CONST;
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
    PolS32 rom_inA;
    PolS32 rom_inB;
    PolS32 rom_inC;
    PolS32 rom_inD;
    PolS32 rom_inE;
    PolS32 rom_inCTX;
    PolS32 rom_inFREE;
    PolS32 rom_inGAS;
    PolS32 rom_inMAXMEM;
    PolS32 rom_inPC;
    PolS32 rom_inSP;
    PolS32 rom_inSR;
    PolS32 rom_inSTEP;
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
    PolBool rom_shl;
    PolBool rom_shr;
    PolBool rom_useCTX;

    // List of id-ordered pols, and its size
    Pol * orderedPols[NPOLS];
    uint64_t size;

    void load (const vector<PolJsonData> &v);
    void mapToOutputFile (const string &outputFileName);
    void mapToInputFile (const string &inputFileName);
    void unmap (void);

private:
    uint64_t polsSize;
    uint8_t * pPolsMappedMemmory;
    string fileName;
    Pol * find (const string &name);
    void addPol (const string &name, const uint64_t id, const string &elementType);
    void mapToFile (const string &fileName, bool bOutput);
};



#endif