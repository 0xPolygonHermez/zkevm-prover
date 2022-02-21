#ifndef KECCAK_SM_HPP
#define KECCAK_SM_HPP

#include "keccak_sm/keccak_sm_state.hpp"
#include "keccak2_input.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "config.hpp"

/* Transformations */
void KeccakSMTheta (KeccakSMState &S);
void KeccakSMRho   (KeccakSMState &S);
void KeccakSMPi    (KeccakSMState &S);
void KeccakSMChi   (KeccakSMState &S, bool bLastRound = false);
void KeccakSMIota  (KeccakSMState &S, uint64_t ir, bool bLastRound = false);

/* Keccak F 1600 */
void KeccakSMF (KeccakSMState &S);

/*
    Input is a buffer of any length, including 0
    Output is 256 bits long buffer containing the 32 bytes keccak hash of the input
*/
void KeccakSM (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput, string scriptFile="");

/* Generate script */
void KeccakSMGenerateScript (const Config &config);

/* Unit test */
void KeccakSMTest (void);

#endif