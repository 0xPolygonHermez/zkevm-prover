#ifndef MAIN_EXECUTOR_HPP_fork_9
#define MAIN_EXECUTOR_HPP_fork_9

#include <string>
#include <nlohmann/json.hpp>
#include <pthread.h>
#include <semaphore.h>
#include "config.hpp"
#include "main_sm/fork_9/main/rom.hpp"
#include "main_sm/fork_9/main/context.hpp"
#include "main_sm/fork_9/pols_generated/commit_pols.hpp"
#include "main_sm/fork_9/main/main_exec_required.hpp"
#include "scalar.hpp"
#include "hashdb_factory.hpp"
#include "poseidon_goldilocks.hpp"
#include "counters.hpp"
#include "goldilocks_base_field.hpp"
#include "prover_request.hpp"

using namespace std;
using json = nlohmann::json;

namespace fork_9
{

class MainExecutor {
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
#ifdef MULTI_ROM_TEST
    Rom rom_gas_limit_100000000;
    Rom rom_gas_limit_2147483647;
    Rom rom_gas_limit_89128960;
#endif

    // Database server configuration, if any
    const Config &config;

    // ROM labels
    uint64_t finalizeExecutionLabel;
    uint64_t checkAndSaveFromLabel;
    uint64_t ecrecoverStoreArgsLabel;
    uint64_t ecrecoverEndLabel;
    uint64_t checkFirstTxTypeLabel;
    uint64_t writeBlockInfoRootLabel;
    uint64_t verifyMerkleProofEndLabel;
    uint64_t outOfCountersStepLabel;
    uint64_t outOfCountersArithLabel;
    uint64_t outOfCountersBinaryLabel;
    uint64_t outOfCountersKeccakLabel;
    uint64_t outOfCountersSha256Label;
    uint64_t outOfCountersMemalignLabel;
    uint64_t outOfCountersPoseidonLabel;
    uint64_t outOfCountersPaddingLabel;

    // Labels lock
    pthread_mutex_t labelsMutex;    // Mutex to protect the labels vector

    // HashDB
    HashDBInterface *pHashDB;

    // When we reach this zkPC, state root (SR) will be consolidated (from virtual to real state root)
    const uint64_t consolidateStateRootZKPC = 4928;

    // Constructor
    MainExecutor(Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config);

    // Destructor
    ~MainExecutor();

    void execute (ProverRequest &proverRequest, MainCommitPols &cmPols, MainExecRequired &required);

    // Initial and final evaluations/state management
    void initState(Context &ctx);
    void checkFinalState(Context &ctx);
    void assertOutputs(Context &ctx);
    void logError(Context &ctx, const string &message = "");
    void linearPoseidon(Context &ctx, const vector<uint8_t> &data, Goldilocks::Element (&result)[4]);

    // Labels lock / unlock
    void labelsLock(void) { pthread_mutex_lock(&labelsMutex); };
    void labelsUnlock(void) { pthread_mutex_unlock(&labelsMutex); };
};

} // namespace

#endif