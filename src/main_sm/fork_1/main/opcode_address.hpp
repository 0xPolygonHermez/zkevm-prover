#ifndef OPCODE_ADDRESS_HPP_fork_1
#define OPCODE_ADDRESS_HPP_fork_1

#include <string>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

namespace fork_1
{

extern uint64_t opcodeAddress[256];

void opcodeAddressInit (json &labels);

} // namespace

#endif