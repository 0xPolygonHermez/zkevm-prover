#ifndef OPCODE_NAME_HPP_fork_1
#define OPCODE_NAME_HPP_fork_1

#include <unordered_map>

using namespace std;

namespace fork_1
{

typedef struct
{
    uint8_t      codeID;
    const char * pName;
} OpcodeInfo;

extern OpcodeInfo opcodeName[256];

} // namespace

#endif