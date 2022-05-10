#ifndef KECCAK_SM_EXECUTOR_HPP
#define KECCAK_SM_EXECUTOR_HPP

#include <array>
#include "config.hpp"
#include "keccak_state.hpp"
#include "keccak2/keccak2.hpp"
#include "keccak_instruction.hpp"

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
    //KeccakSMInstruction program[KeccakSM_NumberOfSlots]
    bool bLoaded;
public:

    /* Constructor */
    KeccakExecutor (const Config &config) : config(config)
    {
        bLoaded = false;
    }

    /* Loads evaluations and SoutRefs from a json object */
    void loadScript (json j);

    /* Executs Keccak-f() over the provided state */
    void execute (KeccakState &S);

    /* bit must be a 2^23 array, with 54 sequences of Sin[1600],Sout[1600] starting at position 1 */
    void execute (uint8_t * bit);

    /* Input is 54*9 Sin, Rin; output is the 3 field element polynomials: a, b, r */
    void execute (KeccakExecuteInput &input, KeccakExecuteOutput &output);

    /* Calculates keccak hash of input data.  Output must be 32-bytes long. */
    /* Internally, it calls execute(KeccakState) */
    void Keccak (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput);
};

#endif