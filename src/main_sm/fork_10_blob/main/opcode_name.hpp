#ifndef OPCODE_NAME_HPP_fork_10_blob
#define OPCODE_NAME_HPP_fork_10_blob

#include <unordered_map>

using namespace std;

namespace fork_10_blob
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