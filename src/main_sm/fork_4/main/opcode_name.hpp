#ifndef OPCODE_NAME_HPP_fork_4
#define OPCODE_NAME_HPP_fork_4

#include <unordered_map>

using namespace std;

namespace fork_4
{

typedef struct
{
    uint8_t      codeID;
    const char * pName;
    uint64_t     gas;
} OpcodeInfo;

extern OpcodeInfo opcodeInfo[256];

} // namespace

#endif