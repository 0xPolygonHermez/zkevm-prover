#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <vector>
#include <gmpxx.h>
#include "config.hpp"
#include "rom.hpp"
#include "rom_command.hpp"
#include "goldilocks_base_field.hpp"
#include "smt.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "database.hpp"
#include "input.hpp"
#include "ffiasm/fec.hpp"
#include "ffiasm/fnec.hpp"
#include "full_tracer.hpp"
#include "rom.hpp"
#include "prover_request.hpp"

using namespace std;
using json = nlohmann::json;

class HashValue
{
public:
    vector<uint8_t> data;
    map<uint64_t, uint64_t> reads;
    mpz_class digest;
    bool bDigested;
    HashValue() : bDigested(false) {};
};

class LastSWrite
{
public:
    Goldilocks &fr;
    uint64_t step;
    Goldilocks::Element key[4];
    Goldilocks::Element keyI[4];
    Goldilocks::Element newRoot[4];
    SmtSetResult res;
    void reset (void)
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
        res.mode = "";
    }
    LastSWrite(Goldilocks &fr) : fr(fr) { reset(); }
};

class Fea
{
public:
    Goldilocks::Element fe0;
    Goldilocks::Element fe1;
    Goldilocks::Element fe2;
    Goldilocks::Element fe3;
    Goldilocks::Element fe4;
    Goldilocks::Element fe5;
    Goldilocks::Element fe6;
    Goldilocks::Element fe7;
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

    Goldilocks &fr; // Finite field reference
    RawFec &fec; // Fec reference
    RawFnec &fnec; // Fnec reference
    MainCommitPols &pols; // PIL JSON file polynomials data
    const Rom &rom; // Rom reference
    LastSWrite lastSWrite; // Keep track of the last storage write
    ProverRequest &proverRequest;
    uint64_t lastStep;
    Context(Goldilocks &fr, RawFec &fec, RawFnec &fnec, MainCommitPols &pols, const Rom &rom, ProverRequest &proverRequest) : fr(fr), fec(fec), fnec(fnec), pols(pols), rom(rom), lastSWrite(fr), proverRequest(proverRequest), lastStep(0) { ; }; // Constructor, setting references

    // Evaluations data
    uint64_t * pZKPC; // Zero-knowledge program counter
    uint64_t * pStep; // Iteration, instruction execution loop counter, polynomial evaluation counter
    uint64_t N; // Polynomials degree
#ifdef LOG_FILENAME
    string   fileName; // From ROM JSON file instruction
    uint64_t line; // From ROM JSON file instruction
#endif

    // Storage
#ifdef USE_LOCAL_STORAGE
    map< Goldilocks::Element, mpz_class, CompareFe> sto; // Input JSON will include the initial values of the rellevant storage positions
#endif

    // HashK database, used in hashK, hashKLen and hashKDigest
    map< uint64_t, HashValue > hashK;

    // HashP database, used in hashP, hashPLen and hashPDigest
    map< uint64_t, HashValue > hashP;

    // Variables database, used in evalCommand() declareVar/setVar/getVar
    map< string, mpz_class > vars;
    
    // Memory map, using absolute address as key, and field element array as value
    map< uint64_t, Fea > mem; // TODO: Use array<Goldilocks::Element,8> instead of Fea, or declare Fea8, Fea4 at a higher level

    map< uint32_t, OutLog> outLogs;

    vector<mpz_class> touchedAddress;
    vector<TouchedStorageSlot> touchedStorageSlots;
    map<string, string> contractsBytecode;
};

#endif