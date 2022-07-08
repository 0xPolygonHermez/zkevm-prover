#ifndef KECCAK_RC_HPP
#define KECCAK_RC_HPP

#include <cstdint>

// Initializes KeccakRC.
// It can be called multiple times; only the first time will do the job
void KeccakRCInit (void);

extern uint8_t KeccakRC[24][64];

#endif