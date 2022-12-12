#ifndef OPCODE_NAME_HPP
#define OPCODE_NAME_HPP

#include <unordered_map>

using namespace std;

typedef struct
{
    uint8_t      codeID;
    const char * pName;
} OpcodeInfo;

extern OpcodeInfo opcodeName[256];

#endif