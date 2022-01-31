#ifndef INPUT_HPP
#define INPUT_HPP

#include <nlohmann/json.hpp>
#include <gmpxx.h>
#include "config.hpp"
#include "public_inputs.hpp"
#include "ffiasm/fr.hpp"
#include "compare_fe.hpp"
#include "database.hpp"

using json = nlohmann::json;

class Input
{
    RawFr &fr;
    void db2json (json &input, const std::map<RawFr::Element, vector<RawFr::Element>, CompareFe> &db, string name) const;
public:
    Input(RawFr &fr) : fr(fr) {};
    string message; // used in gRPC: "calculate", "cancel"
    PublicInputs publicInputs;
    string globalExitRoot;
    
    string batchL2Data;
    uint64_t txsLen;
    mpz_class batchHashData;

    vector<string> txs;
    map<string, string> keys;

    // Used by executor, not by gRPC server
    mpz_class globalHash;

    // Loads the input object data from a JSON object
    void load (json &input);

    // Saves the input object data into a JSON object
    void save (json &input) const;
    void save (json &input, const Database &database) const;

private:
    void loadGlobals      (json &input);
    void saveGlobals      (json &input) const;
    void loadTransactions (json &input);
    void saveTransactions (json &input) const;
#ifdef USE_LOCAL_STORAGE
public:
    map< RawFr::Element, mpz_class, CompareFe> sto; // Input JSON will include the initial values of the rellevant storage positions
    void loadStorage      (json &input);
    void saveStorage      (json &input) const;
#endif
public:
    map< RawFr::Element, vector<RawFr::Element>, CompareFe > db; // This is in fact a map<fe,fe[16]>
    void loadDatabase     (json &input);
    void saveDatabase     (json &input) const;
    void saveDatabase     (json &input, const Database &database) const;
    void preprocessTxs    (void);
};

#endif