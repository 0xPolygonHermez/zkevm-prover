#include <vector>
#include <iostream>
#include "rom_command.hpp"

using namespace std;
using json = nlohmann::json;

void parseRomCommand (RomCommand &cmd, json tag)
{
    // Skipt if not present
    if (tag.is_null()) {
        cmd.isPresent = false;
        return;
    }
    cmd.isPresent = true;

    // This must be a ROM command, not an array of them
    if (tag.is_array()) {
        cerr << "Error: parseRomCommand() found tag is an array: " << tag << endl;
        exit(-1);
    }

    // op is a mandatory element
    cmd.op = tag["op"];

    // Parse optional elements
    if (tag.contains("varName")) cmd.varName = tag["varName"];
    if (tag.contains("regName")) cmd.regName = tag["regName"];
    if (tag.contains("funcName")) cmd.funcName = tag["funcName"];
    if (tag.contains("num")) cmd.num = tag["num"];
    if (tag.contains("values")) parseRomCommandArray(cmd.values, tag["values"]);
    if (tag.contains("params")) parseRomCommandArray(cmd.params, tag["params"]);
}

void parseRomCommandArray (vector<RomCommand *> &values, json tag)
{
    // Skip if not present
    if (tag.is_null()) return;

    // This must be a ROM command array, not one of them
    if (!tag.is_array()) {
        cerr << "Error: parseRomCommandArray() found tag is not an array: " << tag << endl;
        exit(-1);
    }

    // Parse every command in the array
    for (uint64_t i=0; i<tag.size(); i++) {
        RomCommand *pRomCommand = new RomCommand();
        parseRomCommand(*pRomCommand, tag[i]);
        values.push_back(pRomCommand);
    }
}

void freeRomCommand (RomCommand &cmd)
{
    // Fee the ROM command arrays content
    freeRomCommandArray(cmd.values);
    freeRomCommandArray(cmd.params);
}

void freeRomCommandArray (vector<RomCommand *> &array)
{
    // Free all ROM commands
    for (vector<class RomCommand *>::iterator it = array.begin(); it != array.end(); it++ ) {
        freeRomCommand(**it);
        delete(*it);
    }

    // Empty the array
    array.clear();
}