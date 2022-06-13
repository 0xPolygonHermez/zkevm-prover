#ifndef ROM_HPP
#define ROM_HPP

#include <nlohmann/json.hpp>
#include <rom_line.hpp>

using json = nlohmann::json;

class Rom
{
public:
    uint64_t size;
    RomLine *line;
    Rom() { size=0; line=NULL; }
    ~Rom() { if (line!=NULL) unload(); }

    // Parses the ROM JSON data and stores them in memory, in ctx.rom[i]
    void load(Goldilocks &fr, json &romJson);

    // Frees any memory allocated in loadRom()
    void unload(void);
};

#endif