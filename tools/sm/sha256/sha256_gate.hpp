#ifndef SHA256_GATE_HPP
#define SHA256_GATE_HPP

#include <string>
#include <stdint.h>

#include "gate_state.hpp"

using namespace std;

void SHA256Gate (GateState S, const uint8_t * chunkBytes, string scriptFile="", string polsFile="", string connectionsFile="");

#endif