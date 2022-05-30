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
#include "ff/ff.hpp"
#include "sm/pil/commit_pols.hpp"
#include "sm/binary/binary_action.hpp"
#include "sm/arith/arith_action.hpp"
#include "sm/padding_kk/padding_kk_executor.hpp"
#include "sm/padding_kkbit/padding_kkbit_executor.hpp"
#include "sm/nine2one/nine2one_executor.hpp"
#include "sm/norm_gate9/norm_gate9_executor.hpp"
#include "sm/memory/memory_executor.hpp"
#include "sm/padding_pg/padding_pg_executor.hpp"
#include "sm/mem_align/mem_align_executor.hpp"

using namespace std;
using json = nlohmann::json;

class MainExecRequired
{
public:
    vector<SmtAction> Storage;
    vector<MemoryAccess> Memory;
    vector<BinaryAction> Binary;
    vector<ArithAction> Arith;
    vector<PaddingKKExecutorInput> PaddingKK;
    vector<PaddingKKBitExecutorInput> PaddingKKBit;
    vector<Nine2OneExecutorInput> Nine2One;
    vector<vector<FieldElement>> KeccakF;
    vector<NormGate9ExecutorInput> NormGate9;
    map<uint32_t, bool> Byte4;
    vector<PaddingPGExecutorInput> PaddingPG;
    vector<array<FieldElement, 16>> PoseidonG;
    vector<MemAlignAction> MemAlign;
};

class MainExecutor {
public:

    // Finite field data
    FiniteField &fr; // Finite field reference

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
    MainExecutor(FiniteField &fr, Poseidon_goldilocks &poseidon, const Rom &rom, const Config &config) :
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