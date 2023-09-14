#ifndef INPUT_HPP
#define INPUT_HPP

#include <nlohmann/json.hpp>
#include <gmpxx.h>
#include "config.hpp"
#include "public_inputs_extended.hpp"
#include "goldilocks_base_field.hpp"
#include "compare_fe.hpp"
#include "database.hpp"
#include "zkresult.hpp"
#include "trace_config.hpp"

using json = nlohmann::json;

// Max keccak SM capacity is: (2^23)÷158418=52,952366524=52, 52×136×9=63648
// We keep a security margin for other small keccaks, padding bytes, etc. = 60000
// This max length is checked in preprocessTxs()
#define MAX_BATCH_L2_DATA_SIZE (120000)

class Input
{
    Goldilocks &fr;
    void db2json (json &input, const DatabaseMap::MTMap &db, string name) const;
    void contractsBytecode2json (json &input, const DatabaseMap::ProgramMap &contractsBytecode, string name) const;

public:
    PublicInputsExtended publicInputsExtended;
    string from; // Used for unsigned transactions in process batch requests

    // These fields are only used if this is an executor process batch
    bool bUpdateMerkleTree; // if true, save DB writes to SQL database
    bool bNoCounters; // if true, do not increase counters nor limit evaluations
    bool bGetKeys; // if true, return the keys used to read or write storage data
    TraceConfig traceConfig; // FullTracer configuration

    // Constructor
    Input (Goldilocks &fr) :
        fr(fr),
        bUpdateMerkleTree(false),
        bNoCounters(false),
        bGetKeys(false) {};

    // Loads the input object data from a JSON object
    zkresult load (json &input);

    // Saves the input object data into a JSON object
    void save (json &input) const;
    void save (json &input, DatabaseMap &dbReadLog) const;

private:
    void loadGlobals      (json &input);
    void saveGlobals      (json &input) const;

public:
    DatabaseMap::MTMap db;
    DatabaseMap::ProgramMap contractsBytecode;

    void loadDatabase     (json &input);
    void saveDatabase     (json &input) const;
    void saveDatabase     (json &input, DatabaseMap &dbReadLog) const;

    bool operator==(Input &input)
    {
        return
            publicInputsExtended == input.publicInputsExtended &&
            from == input.from &&
            bUpdateMerkleTree == input.bUpdateMerkleTree &&
            bNoCounters == input.bNoCounters &&
            traceConfig == input.traceConfig &&
            db == input.db &&
            contractsBytecode == input.contractsBytecode;
    };

    bool operator!=(Input &input) { return !(*this == input); };
    
    Input & operator=(const Input &other)
    {
        publicInputsExtended = other.publicInputsExtended;
        from = other.from;
        bUpdateMerkleTree = other.bUpdateMerkleTree;
        bNoCounters = other.bNoCounters;
        traceConfig = other.traceConfig;
        db = other.db;
        contractsBytecode = other.contractsBytecode;
        return *this;
    }
};

#endif