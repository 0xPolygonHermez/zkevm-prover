#ifndef SHA256_GATE_HPP
#define SHA256_GATE_HPP

#include <string>
#include <stdint.h>

using namespace std;

void SHA256Gate (const uint8_t * pData, uint64_t dataSize, string &hash, string scriptFile="", string polsFile="", string connectionsFile="");
void SHA256GateString (const string &s, string &hash);

#endif