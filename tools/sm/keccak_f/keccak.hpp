#ifndef KECCAK_SM_HPP
#define KECCAK_SM_HPP

#include "utils.hpp"
#include "scalar.hpp"
#include "config.hpp"
#include "gate_state.hpp"

/* Gets the 0...1599 position of the bit (x,y,z), as per Keccak spec */
#define Bit(x,y,z)   (64*(x) + 320*(y) + (z))

extern GateConfig KeccakGateConfig;

/* Transformations */
void KeccakTheta (GateState &S, uint64_t ir);
void KeccakRho   (GateState &S);
void KeccakPi    (GateState &S);
void KeccakChi   (GateState &S, uint64_t ir);
void KeccakIota  (GateState &S, uint64_t ir);

/* Keccak F 1600 */
void KeccakF (GateState &S);

/*
    Input is a buffer of any length, including 0
    Output is 256 bits long buffer containing the 32 bytes keccak hash of the input
*/
void Keccak (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput);

/* Generate script */
void KeccakGen (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput, string scriptFile="", string polsFile="", string connectionsFile="");
void KeccakGenerateScript (const Config &config);

/* Unit test */
void KeccakTest (void);

#endif