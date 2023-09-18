#ifndef SHA256_SM_EXECUTOR_HPP
#define SHA256_SM_EXECUTOR_HPP

#include <array>
#include "definitions.hpp"
#include "config.hpp"
#include "sha256_instruction.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "timer.hpp"
#include "gate_state.hpp"
#include "sha256_config.hpp"
#include "sha256.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class Sha256ExecuteInput
{
public:
    uint8_t Sin[54][9][1600];
    Sha256ExecuteInput ()
    {
        memset(Sin, 0, sizeof(Sin));
    }
};

class Sha256ExecuteOutput
{
public:
    uint64_t pol[3][1<<23];
};

class Sha256Executor
{
    Goldilocks &fr;
    const Config &config;
    const uint64_t N;
    const uint64_t numberOfSlots;
    vector<Sha256Instruction> program;
    bool bLoaded;
public:

    /* Constructor */
    Sha256Executor (Goldilocks &fr, const Config &config) :
        fr(fr),
        config(config),
        N(PROVER_FORK_NAMESPACE::Sha256CommitPols::pilDegree()),
        numberOfSlots((N-1)/SHA256GateConfig.slotSize)
    {
        bLoaded = false;

        // Avoid initialization if we are not going to generate any proof
        if (!config.generateProof() && !config.runFileExecute &&!config.runSHA256Test) return;

        TimerStart(SHA256_SM_EXECUTOR_LOAD);
        json j;
        file2json(config.sha256ScriptFile, j);
        loadScript(j);
        TimerStopAndLog(SHA256_SM_EXECUTOR_LOAD);
    }

    /* Loads evaluations and SoutRefs from a json object */
    void loadScript (json j);

    /* Input is a vector of numberOfSlots*1600 fe, output is Sha256Pols */
    void execute (const vector<vector<Goldilocks::Element>> &input, PROVER_FORK_NAMESPACE::Sha256CommitPols &pols);

    void setPol (PROVER_FORK_NAMESPACE::CommitPol (&pol)[4], uint64_t index, uint64_t value);
    uint64_t getPol (PROVER_FORK_NAMESPACE::CommitPol (&pol)[4], uint64_t index);
};

#endif