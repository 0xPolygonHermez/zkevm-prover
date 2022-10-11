#ifndef INPUT_HPP
#define INPUT_HPP

#include <nlohmann/json.hpp>
#include <gmpxx.h>
#include "config.hpp"
#include "public_inputs.hpp"
#include "goldilocks_base_field.hpp"
#include "compare_fe.hpp"
#include "database.hpp"
#include "zkresult.hpp"

using json = nlohmann::json;

// Max keccak SM capacity is: (2^23)÷158418=52,952366524=52, 52×136×9=63648
// We keep a security margin for other small keccaks, padding bytes, etc. = 60000
// This max length is checked in preprocessTxs()
#define MAX_BATCH_L2_DATA_SIZE (60000)

class Input
{
    Goldilocks &fr;
    void db2json (json &input, const std::map<string, vector<Goldilocks::Element>> &db, string name) const;
    void contractsBytecode2json (json &input, const std::map<string, vector<uint8_t>> &contractsBytecode, string name) const;

public:
    PublicInputs publicInputs;
    string globalExitRoot;
    string batchL2Data;
    uint64_t txsLen;
    mpz_class batchHashData;
    mpz_class globalHash; // Used by executor, not by gRPC server
    string from; // Used for unsigned transactions

    // These fields are only used if this is an executor process batch
    bool bUpdateMerkleTree; // if true, save DB writes to SQL database
    bool bNoCounters; // if true, do not increase counters nor limit evaluations
    string txHashToGenerateExecuteTrace; // return execute traces of this tx
    string txHashToGenerateCallTrace; // return call traces of this tx

    // Constructor
    Input (Goldilocks &fr) :
        fr(fr),
        txsLen(0),
        bUpdateMerkleTree(true),
        bNoCounters(false) {};

    // Loads the input object data from a JSON object
    zkresult load (json &input);

    // Saves the input object data into a JSON object
    void save (json &input) const;
    void save (json &input, Database &database) const;

private:
    void loadGlobals      (json &input);
    void saveGlobals      (json &input) const;

public:
    //map< Goldilocks::Element, vector<Goldilocks::Element>, CompareFe > db; // This is in fact a map<fe,fe[16]>
    map< string, vector<Goldilocks::Element> > db; // This is in fact a map<fe,fe[16]>
    map< string, vector<uint8_t> > contractsBytecode;
    void loadDatabase     (json &input);
    void saveDatabase     (json &input) const;
    void saveDatabase     (json &input, Database &database) const;
    zkresult preprocessTxs(void);
};

#endif