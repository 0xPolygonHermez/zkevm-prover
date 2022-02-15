#ifndef SMKECCAK_HPP
#define SMKECCAK_HPP

#include "smkeccak/smkeccak_state.hpp"
#include "keccak_input.hpp"
#include "utils.hpp"
#include "scalar.hpp"

/* Transformations */
void SMKeccakTheta (SMKeccakState &S);
void SMKeccakRho   (SMKeccakState &S);
void SMKeccakPi    (SMKeccakState &S);
void SMKeccakChi   (SMKeccakState &S);
void SMKeccakIota  (SMKeccakState &S, uint64_t ir);

/* Keccak F 1600 */
void SMKeccakF (SMKeccakState &S);

/*
    Input is a buffer of any length, including 0
    Output is 256 bits long buffer containing the 32 bytes keccak hash of the input
*/
void SMKeccak (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput);

/* Unit test */
void SMKeccakTest (void);

#endif