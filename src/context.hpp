#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <vector>
#include <gmpxx.h>
#include "config.hpp"
#include "rom.hpp"
#include "rom_command.hpp"
#include "ffiasm/fr.hpp"
#include "smt.hpp"
#include "pols.hpp"
#include "database.hpp"

using namespace std;
using json = nlohmann::json;

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

    RawFr &fr; // Finite field reference
    mpz_class prime; // Prime number used to generate the finite field fr
    Pols &pols; // PIL JSON file polynomials data:
    Context(RawFr &fr, Pols &pols) : fr(fr), pols(pols), db(fr) { ; }; // Constructor, setting finite field reference

    // Evaluations data
    uint64_t zkPC; // Zero-knowledge program counter
    uint64_t step; // Iteration, instruction execution loop counter, polynomial evaluation counter
#ifdef LOG_FILENAME
    string   fileName; // From ROM JSON file instruction
    uint64_t line; // From ROM JSON file instruction
#endif

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
    #ifdef USE_LOCAL_STORAGE
    map< RawFr::Element, mpz_class, CompareFe> sto; // Input JSON will include the initial values of the rellevant storage positions
    #endif
    LastSWrite lastSWrite; // Keep track of the last storage write

    // Database
    Database db;

    // Hash database, used in hashRD and hashWR
    map< uint64_t, HashValue> hash;

    // Variables database, used in evalCommand() declareVar/setVar/getVar
    map<string,RawFr::Element> vars; 
    
    // Memory map, using absolute address as key, and field element array as value
    map<uint64_t,Fea> mem;

    // Used to write byte4_freeIN and byte4_out polynomials after all evaluations have been done
    map<uint32_t,bool> byte4;

};

/* Declare Context ctx to use pols(A0)[i] */
#define pol(name) ctx.pols.name.pData

#endif