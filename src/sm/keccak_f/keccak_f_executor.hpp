#ifndef KECCAK_SM_EXECUTOR_HPP
#define KECCAK_SM_EXECUTOR_HPP

#include <array>
#include "definitions.hpp"
#include "config.hpp"
//#include "keccak2/keccak2.hpp"
#include "keccak_instruction.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "timer.hpp"
#include "gate_state.hpp"
#include "keccak.hpp"

USING_PROVER_FORK_NAMESPACE;

using namespace std;

class KeccakFExecuteInput
{
public:
    uint8_t Sin[54][9][1600];
    KeccakFExecuteInput ()
    {
        memset(Sin, 0, sizeof(Sin));
    }
};

class KeccakFExecuteOutput
{
public:
    uint64_t pol[3][1<<23];
};

class KeccakFExecutor
{
    Goldilocks &fr;
    const Config &config;
    const uint64_t N;
    const uint64_t numberOfSlots;
    vector<KeccakInstruction> program;
    bool bLoaded;
public:

    /* Constructor */
    KeccakFExecutor (Goldilocks &fr, const Config &config) :
        fr(fr),
        config(config),
        N(PROVER_FORK_NAMESPACE::KeccakFCommitPols::pilDegree()),
        numberOfSlots((N-1)/KeccakGateConfig.slotSize)
    {
        bLoaded = false;

        // Avoid initialization if we are not going to generate any proof
        if (!config.generateProof() && !config.runFileExecute &&!config.runKeccakTest) return;

        TimerStart(KECCAK_F_SM_EXECUTOR_LOAD);
        json j;
        file2json(config.keccakScriptFile, j);
        loadScript(j);
        TimerStopAndLog(KECCAK_F_SM_EXECUTOR_LOAD);
    }

    /* Loads evaluations and SoutRefs from a json object */
    void loadScript (json j);

    /* Executs Keccak-f() over the provided state */
    void execute (GateState &S);

    /* Input is a vector of numberOfSlots*1600 fe, output is KeccakPols */
    void execute (const vector<vector<Goldilocks::Element>> &input, PROVER_FORK_NAMESPACE::KeccakFCommitPols &pols);

    void setPol (PROVER_FORK_NAMESPACE::CommitPol (&pol)[4], uint64_t index, uint64_t value);
    uint64_t getPol (PROVER_FORK_NAMESPACE::CommitPol (&pol)[4], uint64_t index);

    /* Calculates keccak hash of input data.  Output must be 32-bytes long. */
    /* Internally, it calls execute(GateState) */
    //void Keccak (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput);
};

#endif