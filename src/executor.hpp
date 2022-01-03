#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "config.hpp"
#include "rom.hpp"
#include "ffiasm/fr.hpp"
#include "scalar.hpp"
#include "smt.hpp"
#include "poseidon_opt/poseidon_opt.hpp"
#include "context.hpp"

using namespace std;
using json = nlohmann::json;

class Executor {
public:

    // Finite field data
    RawFr &fr; // Finite field reference
    mpz_class prime; // Prime number used to generate the finite field fr

    // ROM JSON file data:
    Rom &romData;

    // Poseidon instance
    Poseidon_opt poseidon;

    // SMT instance
    Smt smt;

    // Constructor requires a RawFR
    Executor(RawFr &fr, Rom &romData) : fr(fr), romData(romData), smt(ARITY) { GetPrimeNumber(fr, prime); }; // Constructor, setting finite field reference and prime

    void execute (json &input, Pols &pols);

private:

    void initState(Context &ctx);
    void checkFinalState(Context &ctx);
};

/* Declare Context ctx to use rom[i].A0 */
#define rom romData.romData

#endif