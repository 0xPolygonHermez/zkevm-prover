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

class KeccakSMExecutor
{
    const Config &config;
    vector<KeccakInstruction> program;
    bool bLoaded;
public:

    /* Constructor */
    KeccakSMExecutor (const Config &config) : config(config)
    {
        bLoaded = false;
    }

    /* Loads evaluations and SoutRefs from a json object */
    void loadScript (json j);

    /* bits must be an array of u8 long enough to store all references */
    void execute (KeccakSMState &S);

    /* Calculates keccak hash of input data.  Output must be 32-bytes long. */
    void KeccakSM (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput);
};

void KeccakSMExecutorTest (const Config &config);

#endif