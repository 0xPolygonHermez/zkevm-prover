#ifndef PROOF2ZKIN_HPP
#define PROOF2ZKIN_HPP

#include <nlohmann/json.hpp>

using json = nlohmann::json;

void proof2zkin(const json &proof, json &zkin);

#endif