#ifndef INPUT_HPP
#define INPUT_HPP

#include <nlohmann/json.hpp>
#include <gmpxx.h>
#include "config.hpp"
#include "public_inputs.hpp"
#include "ffiasm/fr.hpp"
#include "compare_fe.hpp"

using json = nlohmann::json;

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

class Input
{
    RawFr &fr;
public:
    Input(RawFr &fr) : fr(fr) {};
    string message; // used in gRPC: "calculate", "cancel"
    PublicInputs publicInputs;
    string globalExitRoot;
    vector<string> txStrings;
    vector<TxData> txs;
    map<string, string> keys;

    // Used by executor, not by gRPC server
    mpz_class globalHash;

    // Loads the input JSON file transactions into memory
    void load (json &input);

private:
    void loadGlobals      (json &input);
    void loadTransactions (json &input);
#ifdef USE_LOCAL_STORAGE
public:
    map< RawFr::Element, mpz_class, CompareFe> sto; // Input JSON will include the initial values of the rellevant storage positions
    void loadStorage      (json &input);
#endif
#ifdef DATABASE_INIT_WITH_INPUT_DB
public:
    map< RawFr::Element, vector<RawFr::Element>, CompareFe > db; // This is in fact a map<fe,fe[16]>
    void loadDatabase     (json &input);
#endif
};

#endif