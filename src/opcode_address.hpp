#ifndef OPCODE_ADDRESS_HPP
#define OPCODE_ADDRESS_HPP

#include <string>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

extern uint64_t opcodeAddress[256];

void opcodeAddressInit (json &labels);

#endif