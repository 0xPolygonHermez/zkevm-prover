#ifndef MAIN_EXECUTOR_HPP_fork_0
#define MAIN_EXECUTOR_HPP_fork_0

#include <string>
#include <nlohmann/json.hpp>
#include "config.hpp"
#include "main_sm/fork_0/main/rom.hpp"
#include "main_sm/fork_0/main/context.hpp"
#include "main_sm/fork_0/pols_generated/commit_pols.hpp"
#include "main_sm/fork_0/main/main_exec_required.hpp"
#include "scalar.hpp"
#include "statedb_factory.hpp"
#include "poseidon_goldilocks.hpp"
#include "counters.hpp"
#include "sm/storage/smt_action.hpp"
#include "goldilocks_base_field.hpp"
#include "prover_request.hpp"

using namespace std;
using json = nlohmann::json;

namespace fork_0
{

class MainExecutor
{
public:

    // Finite field data
    Goldilocks &fr; // Finite field reference

    // RawFec instance
    RawFec fec;

    // RawFnec instance
    RawFnec fnec;

    // Number of evaluations, i.e. polynomials degree
    const uint64_t N;

    // Number of evaluations when counters are not used
    const uint64_t N_NoCounters;

    // Poseidon instance
    PoseidonGoldilocks &poseidon;

    // ROM JSON file data:
    Rom rom;

    // Database server configuration, if any
    const Config &config;

    // ROM labels
    uint64_t finalizeExecutionLabel;
    uint64_t assertNewStateRootLabel;
    uint64_t assertNewLocalExitRootLabel;
    uint64_t checkAndSaveFromLabel;

    // Max counter values
    const uint64_t MAX_CNT_ARITH;
    const uint64_t MAX_CNT_BINARY;
    const uint64_t MAX_CNT_MEM_ALIGN;
    const uint64_t MAX_CNT_KECCAK_F;
    const uint64_t MAX_CNT_PADDING_PG;
    const uint64_t MAX_CNT_POSEIDON_G;

    // Constructor
    MainExecutor(Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config);

    // Destructor
    ~MainExecutor();

    void execute (ProverRequest &proverRequest, fork_0::MainCommitPols &cmPols, MainExecRequired &required);

    // Initial and final evaluations/state management
    void initState(Context &ctx);
    void checkFinalState(Context &ctx);
    void assertOutputs(Context &ctx);
};

} // namespace

#endif