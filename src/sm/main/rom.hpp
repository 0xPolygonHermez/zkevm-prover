#ifndef ROM_HPP
#define ROM_HPP

#include <nlohmann/json.hpp>
#include <string>
#include "rom_line.hpp"

using json = nlohmann::json;
using namespace std;

class Rom
{
public:
    uint64_t size; // Size of the ROM program, i.e. number of ROM lines found in rom.json
    RomLine *line; // ROM program lines, parsed and stored in memory
    unordered_map<string, uint64_t> memoryMap; // Map of memory variables offsets
    unordered_map<string, uint64_t> labels; // ROM lines labels, i.e. names of the ROM lines
    Rom() { size=0; line=NULL; }
    ~Rom() { if (line!=NULL) unload(); }

    // Parses the ROM JSON data and stores them in memory, in ctx.rom[i]
    void load(Goldilocks &fr, json &romJson);

    uint64_t getLabel(const string &label) const;
    uint64_t getMemoryOffset(const string &label) const;

    // Frees any memory allocated in loadRom()
    void unload(void);
private:
    void loadProgram(Goldilocks &fr, json &romJson);
    void loadLabels(Goldilocks &fr, json &romJson);
};

#endif