#ifndef ROM_COMMAND_HPP
#define ROM_COMMAND_HPP

#include <string>
#include <nlohmann/json.hpp>
#include <vector>

using namespace std;
using json = nlohmann::json;

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
};

void parseRomCommandArray (vector<RomCommand *> &values, json tag);
void parseRomCommand      (RomCommand &cmd, json tag);
void freeRomCommandArray  (vector<RomCommand *> &array);
void freeRomCommand       (RomCommand &cmd);

#endif
