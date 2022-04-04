#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "config.hpp"
#include "rom.hpp"
#include "scalar.hpp"
#include "smt.hpp"
#include "poseidon_opt/poseidon_goldilocks.hpp"
#include "context.hpp"
#include "counters.hpp"
#include "smt_action.hpp"
#include "memory_access_list.hpp"
#include "ff/ff.hpp"

using namespace std;
using json = nlohmann::json;

class Executor {
public:

    // Finite field data
    FiniteField &fr; // Finite field reference

    // Poseidon instance
    Poseidon_goldilocks &poseidon;
    
    // ROM JSON file data:
    const Rom &rom;

    // SMT instance
    Smt smt;

    // Database server configuration, if any
    const Config &config;

    // Constructor requires a RawFR
    Executor(FiniteField &fr, Poseidon_goldilocks &poseidon, const Rom &rom, const Config &config) : fr(fr), poseidon(poseidon), rom(rom), smt(fr), config(config) {};

    void execute (const Input &input, Pols &cmPols, Database &db, Counters &counters, vector<SmtAction> &smtActionList, MemoryAccessList &memoryAccessList, bool bFastMode = false);

private:

    void initState(Context &ctx);
    void checkFinalState(Context &ctx);
};

#endif