#ifndef KECCAK_SM_EXECUTOR_HPP
#define KECCAK_SM_EXECUTOR_HPP

#include <array>
#include "config.hpp"
#include "keccak_state.hpp"
#include "keccak2/keccak2.hpp"
#include "keccak_instruction.hpp"
#include "commit_pols.hpp"
#include "sm/norm_gate9/norm_gate9_executor.hpp"

using namespace std;

class KeccakExecuteInput
{
public:
    uint8_t Sin[Keccak_NumberOfSlots][9][1600];
    KeccakExecuteInput ()
    {
        memset(Sin, 0, sizeof(Sin));
    }
};

class KeccakExecuteOutput
{
public:
    uint64_t pol[3][Keccak_PolLength];
};

class KeccakExecutor
{
    const Config &config;
    vector<KeccakInstruction> program;
    bool bLoaded;
public:

    /* Constructor */
    KeccakExecutor (const Config &config) : config(config)
    {
        bLoaded = false;

        TimerStart(KECCAK_F_SM_EXECUTOR_LOAD);
        json j;
        file2json(config.keccakScriptFile, j);
        loadScript(j);
        TimerStopAndLog(KECCAK_F_SM_EXECUTOR_LOAD);
    }

    /* Loads evaluations and SoutRefs from a json object */
    void loadScript (json j);

    /* Executs Keccak-f() over the provided state */
    void execute (KeccakState &S);

    /* bit must be a 2^23 array, with 54 sequences of Sin[1600],Sout[1600] starting at position 1 */
    void execute (uint8_t * bit);

    /* Input is 54*9 Sin, Rin; output is the 3 field element polynomials: a, b, r */
    void execute (KeccakExecuteInput &input, KeccakExecuteOutput &output);

    /* Input is fe[numberOfSlots*1600], output is KeccakPols */
    void execute (const FieldElement *input, const uint64_t inputLength, KeccakFCommitPols &pols);

    /* Input is a vector of numberOfSlots*1600 fe, output is KeccakPols */
    void execute (const vector<vector<FieldElement>> &input, KeccakFCommitPols &pols, vector<NormGate9ExecutorInput> &required);

    /* Calculates keccak hash of input data.  Output must be 32-bytes long. */
    /* Internally, it calls execute(KeccakState) */
    void Keccak (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput);
};

#endif