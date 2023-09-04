#ifndef SHA256_CONFIG_HPP
#define SHA256_CONFIG_HPP

#include "gate_config.hpp"

extern GateConfig SHA256GateConfig;

/*

Sin:
    64x32 = 512 bits =   0..511 = data
    8x32  = 256 bits = 512..767 = hash state

Sout:
    8x32  = 256 bits =   0..255 = hash state

Counters:
    xors      = 60080 = 37.6177%
    ors       = 35520 = 22.24%
    andps     = 0 = 0%
    ands      = 64112 = 40.1423%
    nextRef-1 = 160480
*/

#endif