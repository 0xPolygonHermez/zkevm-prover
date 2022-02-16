#ifndef KECCAK2_HPP
#define KECCAK2_HPP

#include "keccak2_state.hpp"
#include "keccak2_input.hpp"
#include "utils.hpp"
#include "scalar.hpp"

/* Transformations */
void Keccak2Theta (Keccak2State &Sin, Keccak2State &Sout);
void Keccak2Rho   (Keccak2State &Sin, Keccak2State &Sout);
void Keccak2Pi    (Keccak2State &Sin, Keccak2State &Sout);
void Keccak2Chi   (Keccak2State &Sin, Keccak2State &Sout);
void Keccak2Iota  (Keccak2State &Sin, Keccak2State &Sout, uint64_t ir);

/* Keccak F 1600 */
void Keccak2F (Keccak2State &Sin, Keccak2State &Sout);

/*
    Input is a buffer of any length, including 0
    Output is 256 bits long buffer containing the 32 bytes keccak hash of the input
*/
void Keccak2 (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput);

/* Unit test */
void Keccak2Test (void);

#endif