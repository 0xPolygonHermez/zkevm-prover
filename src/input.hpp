#ifndef INPUT_HPP
#define INPUT_HPP

#include <nlohmann/json.hpp>
#include "context.hpp"

using json = nlohmann::json;

// Loads the input JSON file transactions into memory
void loadInput (Context &ctx, json &input);

#endif