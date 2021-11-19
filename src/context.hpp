#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <vector>
#include "rom_line.hpp"
#include "rom_command.hpp"
#include "ffiasm/fr.hpp"
#include "pol_types.hpp"

using namespace std;
using json = nlohmann::json;

#define NEVALUATIONS 4096 //1<<23 // 8M
#define NPOLS 86 //512

class DbValue
{
public:
    string value[16];
};

class Context {
public:
    uint64_t ln; // Program Counter (PC)
    uint64_t step; // Interation, instruction execution loop counter, polynomial evaluation

    // Input JSON file data
    string oldStateRoot;
    string newStateRoot;
    string sequencerAddr;
    uint64_t chainId;
    vector<string> txs;
    map<string,string> keys; // TODO: This is in fact a map<fe,256b>.  Should we change the type?
    map<string,DbValue> db; // TODO: this is in fact a map<fe,fe[16]>.  Should we change the type? 

    // ROM JSON file data
    string fileName; // From ROM JSON file instruction
    uint64_t line; // From ROM JSON file instruction
    uint64_t romSize;
    RomLine *pRom;

    map<string,RawFr::Element> vars; 
    RawFr *pFr;
    string outputFile;
    
    //RawFr::Element mem[MEMORY_SIZE][4]; // TODO: Check with Jordi if this should be int64_t
    // TODO: we could use a mapping, instead.  Slow, but range of addresses would not be a problem
    // DO MAPPING
    // 4 pages 2^32 positions
    // if not created, return a 0
    map<uint64_t,RawFr::Element[4]> mem; // store everything here, with absolute addresses
    // stor is HDD, 2^253 positionsx4x64bits.  They do not start at 0.  
    // input JSON will include the initial values of the rellevant storage positions
    // if input does not contain that position, launch error
    map<RawFr::Element,uint64_t[4]> stor; // Will have to convert from fe to 64b scalars, check size

    map<uint32_t,bool> byte4;

     // Polynomials
    Pol * pols[NPOLS];
    uint64_t polsSize;
    uint8_t * pPolsMappedMemmory;
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
};

/* Declare Context ctx to use rom[i].A0 and pols(A0)[i] */
#define rom ctx.pRom
#define pols(name) ctx.name.pData

#endif