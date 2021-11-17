#include <vector>
#include <iostream>
#include "rom_command.hpp"

using namespace std;
using json = nlohmann::json;

void parseRomCommand (RomCommand &cmd, json tag)
{
    if (tag.is_null()) return;
    if (tag.is_array()) {
        cerr << "Error: parseRomCommand() found tag is an array: " << tag << endl;
        exit(-1);
    }
    cmd.op = tag["op"]; // op is a mandatory element
    if (tag.contains("varName")) cmd.varName = tag["varName"];
    if (tag.contains("regName")) cmd.regName = tag["regName"];
    if (tag.contains("funcName")) cmd.funcName = tag["funcName"];
    if (tag.contains("num")) cmd.num = tag["num"];
    if (tag.contains("values")) parseRomCommandArray(cmd.values, tag["values"]);
    if (tag.contains("params")) parseRomCommandArray(cmd.params, tag["params"]);
}

void parseRomCommandArray (vector<RomCommand *> &values, json tag)
{
    if (tag.is_null()) return;
    if (!tag.is_array()) {
        cerr << "Error: parseRomCommandArray() found tag is not an array: " << tag << endl;
        exit(-1);
    }
    for (uint64_t i=0; i<tag.size(); i++) {
        RomCommand *pRomCommand = new RomCommand();
        parseRomCommand(*pRomCommand, tag[i]);
        values.push_back(pRomCommand);
    }
}

void freeRomCommand (RomCommand &cmd)
{
    freeRomCommandArray(cmd.values);
    freeRomCommandArray(cmd.params);
}

void freeRomCommandArray (vector<RomCommand *> &array)
{
    vector<class RomCommand *>::iterator it;
    for (it = array.begin(); it != array.end(); it++ ) {
        freeRomCommand(**it);
        delete(*it);
    }
    array.clear();
}