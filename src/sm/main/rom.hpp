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
    uint64_t size;
    RomLine *line;
    map<string, uint64_t> memoryMap;
    Rom() { size=0; line=NULL; }
    ~Rom() { if (line!=NULL) unload(); }

    // Parses the ROM JSON data and stores them in memory, in ctx.rom[i]
    void load(Goldilocks &fr, json &romJson);

    uint64_t getMemoryOffset(string &label) const
    {
        map<string,uint64_t>::const_iterator it;
        it = memoryMap.find(label);
        if (it==memoryMap.end())
        {
            cerr << "Error: Rom::getMemoryOffset() could not find label=" << label << endl;
            //exit(-1);
            return 0;
        }
        return it->second;
    }

    // Frees any memory allocated in loadRom()
    void unload(void);
};

#endif