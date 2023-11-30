#ifndef JOINZKIN_MULTICHAIN__STARK_HPP
#define JOINZKIN_MULTICHAIN__STARK_HPP

#include <nlohmann/json.hpp>

using ordered_json = nlohmann::ordered_json;

ordered_json joinzkinmultichain(ordered_json &zkin1, ordered_json &zkin2, ordered_json &verKey, uint64_t steps);

#endif