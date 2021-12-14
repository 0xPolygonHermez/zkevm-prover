#ifndef POLS_HPP
#define POLS_HPP

#include <nlohmann/json.hpp>
#include <string>
#include "config.hpp"
#include "pol_types.hpp"

using namespace std;
using json = nlohmann::json;

class Pols
{
public:
    // Output file polynomials
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
    PolBool inA;
    PolBool inB;
    PolBool inC;
    PolBool inD;
    PolBool inE;
    PolBool inCTX;
    PolBool inFREE;
    PolBool inGAS;
    PolBool inMAXMEM;
    PolBool inPC;
    PolBool inSP;
    PolBool inSR;
    PolBool inSTEP;
    PolBool inc;
    PolBool dec;
    PolBool ind;
    PolBool isCode;
    PolBool isMaxMem;
    PolBool isMem;
    PolBool isNeg;
    PolBool isStack;
    PolBool mRD;
    PolBool mWR;
    PolBool neg;
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

    // Input file polynomials
    PolU16 global_byte2;
    PolBool global_L1;
    PolFieldElement global_ZH;
    PolFieldElement global_ZHINV;
    PolBool byte4_SET;
    PolFieldElement ROM;
    PolU32 STEP;

    void load (const vector<PolJsonData> &v);
    void mapToOutputFile (const string &outputFileName);
    void mapToInputFile (const string &inputFileName);
    void unmap (void);

    static void parse (const json &pil, vector<PolJsonData> &cmPols, vector<PolJsonData> &constPols);

private:
    Pol * orderedPols[NPOLS];
    uint64_t polsSize;
    uint8_t * pPolsMappedMemmory;
    string fileName;
    Pol * find (const string &name);
    void addPol (const string &name, const uint64_t id, const string &elementType);
    void mapToFile (const string &fileName, bool bOutput);
};



#endif