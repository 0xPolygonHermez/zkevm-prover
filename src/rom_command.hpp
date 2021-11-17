#ifndef ROM_COMMAND_HPP
#define ROM_COMMAND_HPP

#include <string>
#include <nlohmann/json.hpp>
#include <vector>

using namespace std;
using json = nlohmann::json;

class romCommand {
public:
    string op; // command
    string varName; // variable name
    string regName; // register name
    string funcName; // function name
    uint64_t num; //number
    vector<romCommand *> values;
    vector<romCommand *> params;
};

void parseRomCommandArray(json tag, vector<romCommand *> &values);
void parseRomCommand(json tag, romCommand &cmd);
void freeRomCommandArray(vector<romCommand *> &array);
void freeRomCommand(romCommand &cmd);

#endif
