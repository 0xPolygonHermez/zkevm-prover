#ifndef ROM_HPP
#define ROM_HPP

#include <nlohmann/json.hpp>
#include <rom_line.hpp>

using json = nlohmann::json;

class Rom
{
public:
    uint64_t romSize;
    RomLine *romData;
    Rom() {romSize=0; romData=NULL; }

    // Parses the ROM JSON data and stores them in memory, in ctx.rom[i]
    void loadRom(json &romJson);

    // Frees any memory allocated in loadRom()
    void unloadRom(void);
};

#endif