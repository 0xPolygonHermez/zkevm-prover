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

class L1Data
{
public:
    bool bPresent;
    mpz_class globalExitRoot;
    mpz_class blockHashL1;
    uint64_t minTimestamp;
    vector<mpz_class> smtProof;
    L1Data() : bPresent(false), minTimestamp(0) {};
};

class OverrideEntry
{
public:
    bool bBalance; // Indicates whether balance is valid or not
    mpz_class balance; // Fake balance to set for the account before executing the call
    uint64_t nonce; // Fake nonce to set for the account before executing the call
    vector<uint8_t> code; // Fake EVM bytecode to inject into the account before executing the call (byte array, i.e. binary data)
    unordered_map<string, mpz_class> state; // Fake key-value mapping to override all slots in the account storage before executing the call (key string, value binary)
    unordered_map<string, mpz_class> stateDiff; // Fake key-value mapping to override individual slots in the account storage before executing the call (key string, value binary)
    OverrideEntry() : bBalance(false), nonce(0) {};
};

class InputDebug
{
public:
    uint64_t gasLimit;
    InputDebug() : gasLimit(0) {};
};

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
    bool bSkipVerifyL1InfoRoot; // If true, skip the check when l1Data is verified (fork ID >= 7)
    bool bSkipFirstChangeL2Block; // If true, skip the restriction to start a batch with a changeL2Block transaction (fork ID >= 7)
    bool bSkipWriteBlockInfoRoot; // If true, skip the block info root (fork ID >= 7)
    TraceConfig traceConfig; // FullTracer configuration
    unordered_map<uint64_t, L1Data> l1InfoTreeData;
    unordered_map<string, OverrideEntry> stateOverride;
    uint64_t stepsN;
    InputDebug debug;

    // Constructor
    Input (Goldilocks &fr) :
        fr(fr),
        bUpdateMerkleTree(false),
        bNoCounters(false),
        bGetKeys(false),
        bSkipVerifyL1InfoRoot(false),
        bSkipFirstChangeL2Block(false),
        bSkipWriteBlockInfoRoot(false),
        stepsN(0)
        {};

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