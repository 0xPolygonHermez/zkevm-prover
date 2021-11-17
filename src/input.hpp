#ifndef INPUT_HPP
#define INPUT_HPP

#include <nlohmann/json.hpp>
#include "context.hpp"

using json = nlohmann::json;

void preprocessTxs (Context &ctx, json &input);
#endif