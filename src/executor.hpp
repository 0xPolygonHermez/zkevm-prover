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
#include "smt_action_list.hpp"
#include "memory_access_list.hpp"
#include "ff/ff.hpp"

using namespace std;
using json = nlohmann::json;

class Executor {
public:

    // Finite field data
    FiniteField &fr; // Finite field reference
    mpz_class prime; // Prime number used to generate the finite field fr

    // ROM JSON file data:
    const Rom &rom;

    // Poseidon instance
    Poseidon_goldilocks poseidon;

    // SMT instance
    Smt smt;

    // Database server configuration, if any
    const Config &config;

    // Constructor requires a RawFR
    Executor(FiniteField &fr, const Rom &rom, const Config &config) : fr(fr), rom(rom), smt(ARITY), config(config) { prime = fr.prime(); }; // Constructor, setting finite field reference and prime

    void execute (const Input &input, Pols &cmPols, Database &db, Counters &counters, SmtActionList &smtActionList, MemoryAccessList &memoryAccessList, bool bFastMode = false);

private:

    void initState(Context &ctx);
    void checkFinalState(Context &ctx);
};

#endif