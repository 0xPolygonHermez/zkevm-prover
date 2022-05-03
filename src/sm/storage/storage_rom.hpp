#ifndef STORAGE_ROM_HPP
#define STORAGE_ROM_HPP

#include <vector>
#include <nlohmann/json.hpp>
#include "storage_rom_line.hpp"

using namespace std;
using json = nlohmann::json;

class StorageRom
{
public:
    vector<StorageRomLine> line;
    void load (json &j);
};

#endif