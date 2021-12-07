#ifndef ROM_HPP
#define ROM_HPP

#include <nlohmann/json.hpp>
#include "context.hpp"

using json = nlohmann::json;

// Parses the ROM JSON data and stores them in memory, in ctx.rom[i]
void loadRom(Context &ctx, json &romJson);

// Frees any memory allocated in loadRom()
void unloadRom(Context &ctx);

#endif