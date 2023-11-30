#ifndef CALCULATE_SHA_256_PUBLICS_HPP
#define CALCULATE_SHA_256_PUBLICS_HPP

#include <nlohmann/json.hpp>
#include "sha256.hpp"
#include "scalar.hpp"

using ordered_json = nlohmann::ordered_json;

ordered_json calculateSha256(ordered_json &publics, ordered_json &prevHashInfo);

#endif