#include "main_sm/fork_4/main/opcode_address.hpp"
#include "main_sm/fork_4/main/eth_opcodes.hpp"

namespace fork_4
{

uint64_t opcodeAddress[256];

void opcodeAddressInit (json &labels)
{
    memset(&opcodeAddress, 0, sizeof(opcodeAddress));

    for (uint64_t i=0; i<256; i++)
    {
        if (labels.contains(ethOpcode[i]) &&
            labels[ethOpcode[i]].is_number() )
        {
            opcodeAddress[i] = labels[ethOpcode[i]];
        }
        else
        {
            opcodeAddress[i] = labels["opINVALID"];
        }
    }
}

} // namespace