#ifndef KECCAK2_HPP
#define KECCAK2_HPP

#include "keccak_state.hpp"
#include "keccak_input.hpp"
#include "utils.hpp"
#include "scalar.hpp"

/* Transformations */
void KeccakTheta (const KeccakState &Sin, KeccakState &Sout);
void KeccakRho   (const KeccakState &Sin, KeccakState &Sout);
void KeccakPi    (const KeccakState &Sin, KeccakState &Sout);
void KeccakChi   (const KeccakState &Sin, KeccakState &Sout);
void KeccakIota  (const KeccakState &Sin, KeccakState &Sout, uint64_t ir);

/* Round */
void KeccakRound (const KeccakState &Sin, KeccakState &Sout, uint64_t ir);

/* Keccak F 1600 */
void KeccakF (const KeccakState &Sin, KeccakState &Sout);

/*
    Input is a buffer of any length, including 0
    Output is 256 bits long buffer containing the 32 bytes keccak hash of the input
*/
void Keccak2 (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput);

/* Unit test */
void KeccakTest (void);

#endif