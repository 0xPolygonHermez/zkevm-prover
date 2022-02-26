#ifndef KECCAK_SM_EXECUTOR_HPP
#define KECCAK_SM_EXECUTOR_HPP

#include "config.hpp"
#include "keccak_sm_state.hpp"
#include "keccak2/keccak2.hpp"

class KeccakSMExecutor
{
    const Config &config;
    vector<Gate> evals;
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