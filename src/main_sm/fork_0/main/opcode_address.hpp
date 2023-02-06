#ifndef OPCODE_ADDRESS_HPP_fork_0
#define OPCODE_ADDRESS_HPP_fork_0

#include <string>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

namespace fork_0
{

extern uint64_t opcodeAddress[256];

void opcodeAddressInit (json &labels);

}

#endif