#ifndef MAIN_EXECUTOR_HPP
#define MAIN_EXECUTOR_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "config.hpp"
#include "rom.hpp"
#include "scalar.hpp"
#include "smt.hpp"
#include "poseidon_opt/poseidon_goldilocks_old.hpp"
#include "context.hpp"
#include "counters.hpp"
#include "sm/storage/smt_action.hpp"
#include "goldilocks/goldilocks_base_field.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "main_exec_required.hpp"

using namespace std;
using json = nlohmann::json;

class MainExecutor {
public:

    // Finite field data
    Goldilocks &fr; // Finite field reference

    // Number of evaluations, i.e. polynomials degree
    const uint64_t N;

    // Poseidon instance
    Poseidon_goldilocks &poseidon;
    
    // ROM JSON file data:
    const Rom &rom;

    // SMT instance
    Smt smt;

    // Database server configuration, if any
    const Config &config;

    // Constructor requires a RawFR
    MainExecutor(Goldilocks &fr, Poseidon_goldilocks &poseidon, const Rom &rom, const Config &config) :
        fr(fr),
        N(MainCommitPols::degree()),
        poseidon(poseidon),
        rom(rom),
        smt(fr),
        config(config) {};

    void execute (const Input &input, MainCommitPols &cmPols, Database &db, Counters &counters, MainExecRequired &required, bool bFastMode = false);

private:

    void initState(Context &ctx);
    void checkFinalState(Context &ctx);
};

#endif