#ifndef MAIN_EXECUTOR_HPP
#define MAIN_EXECUTOR_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "config.hpp"
#include "rom.hpp"
#include "scalar.hpp"
#include "statedb_factory.hpp"
#include "goldilocks/poseidon_goldilocks.hpp"
#include "context.hpp"
#include "counters.hpp"
#include "sm/storage/smt_action.hpp"
#include "goldilocks/goldilocks_base_field.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "main_exec_required.hpp"
#include "prover_request.hpp"

using namespace std;
using json = nlohmann::json;

class MainExecutor {
public:

    // Finite field data
    Goldilocks &fr; // Finite field reference

    // Number of evaluations, i.e. polynomials degree
    const uint64_t N;

    // Poseidon instance
    PoseidonGoldilocks &poseidon;
    
    // ROM JSON file data:
    Rom rom;

    // StateDB interface
    StateDBClient *pStateDB;

    // Database server configuration, if any
    const Config &config;

    // Constructor
    MainExecutor(Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config);
    
    // Destructor
    ~MainExecutor();

    void execute (ProverRequest &proverRequest, MainCommitPols &cmPols, MainExecRequired &required);

private:

    void initState(Context &ctx);
    void checkFinalState(Context &ctx);
};

#endif