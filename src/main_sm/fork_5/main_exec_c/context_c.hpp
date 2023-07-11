#ifndef CONTEXT_C_HPP_fork_5
#define CONTEXT_C_HPP_fork_5

#include "variables_c.hpp"
#include "batch_data.hpp"
#include "config.hpp"
#include "prover_request.hpp"
#include "hashdb_interface.hpp"

namespace fork_5
{

class ContextC
{
public:
    Goldilocks &fr; // Finite field reference
    const Config &config; // Configuration reference
    ProverRequest &proverRequest; // Prover request reference
    HashDBInterface *pHashDB; // Hash DB pointer

    // Global variables
    GlobalVariablesC globalVars;

    // Context variables
    unordered_map<uint64_t, ContextVariablesC> contextVars;

    // Dynamic state root, i.e. latest, new state root
    Goldilocks::Element root[4];

    // Decoded batch L2 data, plus derived data during execution
    BatchData batch;

    // Current tx counter, i.e. we are processing batch.tx[tx]
    uint64_t tx;

    ContextC(Goldilocks &fr, const Config &config, ProverRequest &proverRequest, HashDBInterface *pHashDB) : fr(fr), config(config), proverRequest(proverRequest), pHashDB(pHashDB), tx(0) {;};

};

}

#endif