#ifndef KECCAK_SM_HPP
#define KECCAK_SM_HPP

#include "sm/keccak_f/keccak_state.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "config.hpp"

/* Transformations */
void KeccakTheta (KeccakState &S, uint64_t ir);
void KeccakRho   (KeccakState &S);
void KeccakPi    (KeccakState &S);
void KeccakChi   (KeccakState &S, uint64_t ir);
void KeccakIota  (KeccakState &S, uint64_t ir);

/* Keccak F 1600 */
void KeccakF (KeccakState &S);

/*
    Input is a buffer of any length, including 0
    Output is 256 bits long buffer containing the 32 bytes keccak hash of the input
*/
void Keccak (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput, string scriptFile="", string polsFile="", string connectionsFile="");

/* Generate script */
void KeccakGenerateScript (const Config &config);

/* Unit test */
void KeccakSMTest (void);

#endif