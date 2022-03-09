#ifndef KECCAK_SM_EXECUTOR_HPP
#define KECCAK_SM_EXECUTOR_HPP

#include "config.hpp"
#include "keccak_sm_state.hpp"
#include "keccak2/keccak2.hpp"

class KeccakInstruction
{
public:
    GateOperation op;
    uint64_t refa;
    uint64_t refb;
    uint64_t refr;
    uint64_t pina;
    uint64_t pinb;
    KeccakInstruction () {
        op = gop_xor;
        refa = 0;
        refb = 0;
        refr = 0;
        pina = 0;
        pinb = 0;
    }
};

class KeccakSMExecuteInput
{
public:
    uint8_t Sin[54][9][1600];
    uint8_t Rin[54][9][1088];
    KeccakSMExecuteInput ()
    {
        memset(Sin, 0, sizeof(Sin));
        memset(Rin, 0, sizeof(Rin));
    }
};

class KeccakSMExecuteOutput
{
public:
    uint64_t pol[3][KeccakSM_PolLength];
};

class KeccakSMExecutor
{
    const Config &config;
    vector<KeccakInstruction> program;
    bool bLoaded;
    uint64_t slotSize;
public:

    /* Constructor */
    KeccakSMExecutor (const Config &config) : config(config)
    {
        bLoaded = false;
    }

    /* Loads evaluations and SoutRefs from a json object */
    void loadScript (json j);

    /* Executs Keccak-f() over the provided state */
    void execute (KeccakSMState &S);

    /* bit must be a 2^23 array, with 54 sequences of Sin[1600],Sout[1600] starting at position 1 */
    void execute (uint8_t * bit);

    /* Input is 54*9 Sin, Rin; output is the 3 field element polynomials: a, b, r */
    void execute (KeccakSMExecuteInput &input, KeccakSMExecuteOutput &output);

    /* Calculates keccak hash of input data.  Output must be 32-bytes long. */
    /* Internally, it calls execute(KeccakSMState) */
    void KeccakSM (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput);
};

void KeccakSMExecutorTest (const Config &config);

#endif