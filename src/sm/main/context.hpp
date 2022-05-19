#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <vector>
#include <gmpxx.h>
#include "config.hpp"
#include "rom.hpp"
#include "rom_command.hpp"
#include "ff/ff.hpp"
#include "smt.hpp"
#include "sm/pil/commit_pols.hpp"
#include "database.hpp"
#include "input.hpp"

using namespace std;
using json = nlohmann::json;

class HashRead
{
public:
    uint64_t pos;
    uint64_t len;
};

class HashValue
{
public:
    vector<uint8_t> data;
    vector<HashRead> reads;
    mpz_class digest;
    bool bDigested;
    HashValue() : bDigested(false) {};
};

class LastSWrite
{
public:
    uint64_t step;
    FieldElement key[4];
    FieldElement keyI[4];
    FieldElement newRoot[4];
    void reset (FiniteField &fr)
    {
        step = 0;
        key[0] = fr.zero();
        key[1] = fr.zero();
        key[2] = fr.zero();
        key[3] = fr.zero();
        keyI[0] = fr.zero();
        keyI[1] = fr.zero();
        keyI[2] = fr.zero();
        keyI[3] = fr.zero();
        newRoot[0] = fr.zero();
        newRoot[1] = fr.zero();
        newRoot[2] = fr.zero();
        newRoot[3] = fr.zero();
    }
};

class Fea
{
public:
    FieldElement fe0;
    FieldElement fe1;
    FieldElement fe2;
    FieldElement fe3;
    FieldElement fe4;
    FieldElement fe5;
    FieldElement fe6;
    FieldElement fe7;
};

class OutLog
{
public:
    vector<string> topics;
    vector<string> data;
};

class TouchedStorageSlot
{
public:
    uint32_t addr;
    uint32_t key;
};

class Context {
public:

    FiniteField &fr; // Finite field reference
    MainCommitPols &pols; // PIL JSON file polynomials data
    Byte4CommitPols &byte4Pols;
    const Input &input; // Input JSON file data
    Database &db; // Database reference
    Context(FiniteField &fr, MainCommitPols &pols, Byte4CommitPols &byte4Pols, const Input &input, Database &db) : fr(fr), pols(pols), byte4Pols(byte4Pols), input(input), db(db) { ; }; // Constructor, setting references

    // Evaluations data
    uint64_t zkPC; // Zero-knowledge program counter
    uint64_t step; // Iteration, instruction execution loop counter, polynomial evaluation counter
    uint64_t N; // Polynomials degree
#ifdef LOG_FILENAME
    string   fileName; // From ROM JSON file instruction
    uint64_t line; // From ROM JSON file instruction
#endif

    // Storage
#ifdef USE_LOCAL_STORAGE
    map< FieldElement, mpz_class, CompareFe> sto; // Input JSON will include the initial values of the rellevant storage positions
#endif
    LastSWrite lastSWrite; // Keep track of the last storage write

    // HashK database, used in hashK, hashKLen and hashKDigest
    map< uint64_t, HashValue > hashK;

    // HashP database, used in hashP, hashPLen and hashPDigest
    map< uint64_t, HashValue > hashP;

    // Variables database, used in evalCommand() declareVar/setVar/getVar
    map< string, FieldElement > vars; 
    
    // Memory map, using absolute address as key, and field element array as value
    map< uint64_t, Fea > mem;

    // Used to write byte4_freeIN and byte4_out polynomials after all evaluations have been done
    map< uint32_t, bool > byte4;

    map< uint32_t, OutLog> outLogs;

    map< uint32_t, vector<mpz_class> > touchedAddress;
    map< uint32_t, vector<TouchedStorageSlot> > touchedStorageSlots;
};

#endif