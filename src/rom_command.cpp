#include <vector>
#include <iostream>
#include "rom_command.hpp"

using namespace std;
using json = nlohmann::json;

void parseRomCommand(json tag, romCommand &cmd)
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
    if (tag.contains("values")) parseRomCommandArray(tag["values"],cmd.values);
    if (tag.contains("params")) parseRomCommandArray(tag["params"],cmd.params);
}

void parseRomCommandArray(json tag, vector<romCommand *> &values)
{
    if (tag.is_null()) return;
    if (!tag.is_array()) {
        cerr << "Error: parseRomCommandArray() found tag is not an array: " << tag << endl;
        exit(-1);
    }
    for (uint64_t i=0; i<tag.size(); i++) {
        romCommand *pRomCommand = new romCommand();
        parseRomCommand(tag[i],*pRomCommand);
        values.push_back(pRomCommand);
    }
}

void freeRomCommand(romCommand &cmd)
{
    freeRomCommandArray(cmd.values);
    freeRomCommandArray(cmd.params);
}

void freeRomCommandArray(vector<romCommand *> &array)
{
    vector<class romCommand *>::iterator it;
    for(it = array.begin(); it != array.end(); it++ ) {
        freeRomCommand(**it);
        delete(*it);
    }
    array.clear();
}