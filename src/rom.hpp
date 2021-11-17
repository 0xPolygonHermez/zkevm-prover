#ifndef ROM_HPP
#define ROM_HPP

#include <nlohmann/json.hpp>
#include "context.hpp"

using json = nlohmann::json;

void loadRom(Context &ctx, json &romJson);
void unloadRom(Context &ctx);

#endif