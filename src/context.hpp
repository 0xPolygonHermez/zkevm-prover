#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <vector>
#include <gmpxx.h>
#include "rom_line.hpp"
#include "rom_command.hpp"
#include "ffiasm/fr.hpp"
#include "pol_types.hpp"
#include "smt.hpp"

using namespace std;
using json = nlohmann::json;

#define NEVALUATIONS 65536 //4096 //1<<23 // 8M
#define NPOLS 86 //512
#define ARITY 4

class HashValue
{
public:
    vector<uint8_t> data;
    string hash;
};

class LastSWrite
{
public:
    uint64_t step;
    RawFr::Element key;
    RawFr::Element newRoot;
};

class TxData
{
public:
    string originalTx;
    string signData;
    // signature = r + s + v
    mpz_class r;
    mpz_class s;
    uint16_t v;
};

class Fea
{
public:
    RawFr::Element fe0;
    RawFr::Element fe1;
    RawFr::Element fe2;
    RawFr::Element fe3;
};

class Context {
public:

    // Finite field data
    RawFr &fr; // Finite field reference
    mpz_class prime; // Prime number used to generate the finite field fr
    Context(RawFr &fr) : fr(fr) { ; }; // Constructor, setting finite field reference

    // Evaluations data
    uint64_t zkPC; // Zero-knowledge program counter
    uint64_t step; // Iteration, instruction execution loop counter, polynomial evaluation counter
#ifdef LOG_FILENAME
    string   fileName; // From ROM JSON file instruction
    uint64_t line; // From ROM JSON file instruction
#endif

    // ROM JSON file data:
    uint64_t romSize;
    RomLine *pRom;

    // PIL JSON file polynomials data:
    Pol * orderedPols[NPOLS];
    uint64_t polsSize;
    uint8_t * pPolsMappedMemmory;
    Pols polynomials;
   
    // Input JSON file data:
    // Global data
    string oldStateRoot;
    string newStateRoot;
    string sequencerAddr;
    uint64_t chainId;
    // Transactions and global hash
    vector<TxData> txs;
    mpz_class globalHash;
    // Storage
    map< RawFr::Element, mpz_class, CompareFe> sto; // Storage
    // Database
    map< RawFr::Element, vector<RawFr::Element>, CompareFe > db; // This is in fact a map<fe,fe[16]>.  In the future, we sill use an external database. 

    // Hash database, used in hashRD and hashWR
    map< uint64_t, HashValue> hash;

    // Variables database, used in evalCommand() declareVar/setVar/getVar
    map<string,RawFr::Element> vars; 



    string outputFile;
    
    //RawFr::Element mem[MEMORY_SIZE][4]; // TODO: Check with Jordi if this should be int64_t
    // TODO: we could use a mapping, instead.  Slow, but range of addresses would not be a problem
    // DO MAPPING
    // 4 pages 2^32 positions
    // if not created, return a 0
    map<uint64_t,Fea> mem; // store everything here, with absolute addresses
    // stor is HDD, 2^253 positionsx4x64bits.  They do not start at 0.  
    // input JSON will include the initial values of the rellevant storage positions
    // if input does not contain that position, launch error
    map<RawFr::Element,uint64_t[4]> stor; // Will have to convert from fe to 64b scalars, check size
    LastSWrite lastSWrite;

    map<uint32_t,bool> byte4;

};

/* Declare Context ctx to use rom[i].A0 and pols(A0)[i] */
#define rom ctx.pRom
#define pols(name) ctx.polynomials.name.pData

#endif