#ifndef ROM_COMMAND_HPP
#define ROM_COMMAND_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

// Contains a ROM command data, and arrays possibly containing other ROM commands data
class RomCommand {
public:
    bool isPresent; // presence flag
    string op; // command
    string varName; // variable name
    string regName; // register name
    string funcName; // function name
    uint64_t num; //number
    vector<RomCommand *> values;
    vector<RomCommand *> params;
    uint64_t offset; // TODO: Parse offset in RomCommand
    RomCommand() : isPresent(false), num(0), offset(0) {};
};

// Functions to parse/free a ROM command, or an array of them
void parseRomCommandArray (vector<RomCommand *> &values, json tag);
void parseRomCommand      (RomCommand &cmd, json tag);
void freeRomCommandArray  (vector<RomCommand *> &array);
void freeRomCommand       (RomCommand &cmd);

#endif
