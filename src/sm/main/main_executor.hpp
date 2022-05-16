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
#include "sm/storage/smt_action.hpp"
#include "sm/memory/memory_access_list.hpp"
#include "ff/ff.hpp"
#include "sm/pil/commit_pols.hpp"

using namespace std;
using json = nlohmann::json;

class MainExecRequired
{
public:
    vector<SmtAction> smtActionList;
    MemoryAccessList memoryAccessList;
};

class MainExecutor {
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
    MainExecutor(FiniteField &fr, Poseidon_goldilocks &poseidon, const Rom &rom, const Config &config) : fr(fr), poseidon(poseidon), rom(rom), smt(fr), config(config) {};

    void execute (const Input &input, MainCommitPols &cmPols, Byte4CommitPols &byte4Pols, Database &db, Counters &counters, MainExecRequired &mainExecRequired, bool bFastMode = false);

private:

    void initState(Context &ctx);
    void checkFinalState(Context &ctx);
};

#endif