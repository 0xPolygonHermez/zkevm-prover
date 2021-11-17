#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <vector>
#include "rom_line.hpp"
#include "rom_command.hpp"
#include "ffiasm/fr.hpp"

using namespace std;
using json = nlohmann::json;

#define NEVALUATIONS 4096 //1<<23 // 8M
#define NPOLS 100 //512

typedef RawFr::Element tPolynomial[NEVALUATIONS]; // This one will be dynamic
typedef tPolynomial tExecutorOutput[NPOLS]; // This one could be static

typedef struct {
    string value[16];
} tDbValue;

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
    map<string,tDbValue> db; // TODO: this is in fact a map<fe,fe[16]>.  Should we change the type? 

    // ROM JSON file data
    string fileName; // From ROM JSON file instruction
    uint64_t line; // From ROM JSON file instruction
    uint64_t romSize;
    tRomLine *pRom;

    map<string,RawFr::Element> vars; 
    RawFr *pFr;
    tExecutorOutput * pPols;
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
};

/* Declare Context ctx to use pols[A0][i] and rom[i].A0 */
#define pols (*ctx.pPols)
#define rom ctx.pRom

#endif