#ifndef ROM_COMMAND_HPP
#define ROM_COMMAND_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <gmpxx.h>

using namespace std;
using json = nlohmann::json;

typedef enum : int {
    f_empty = 0,
    f_beforeLast = 1,
    f_getGlobalHash = 2,
    f_getGlobalExitRoot = 3,
    f_getOldStateRoot = 4,
    f_getNewStateRoot = 5,
    f_getSequencerAddr = 6,
    f_getOldLocalExitRoot = 7,
    f_getNewLocalExitRoot = 8,
    f_getNumBatch = 9,
    f_getTimestamp = 10,
    f_getBatchHashData = 11,
    f_getTxs = 12,
    f_getTxsLen = 13,
    f_addrOp = 14,
    f_eventLog = 15,
    f_cond = 16,
    f_inverseFpEc = 17,
    f_inverseFnEc = 18,
    f_sqrtFpEc = 19,
    f_xAddPointEc = 20,
    f_yAddPointEc = 21,
    f_xDblPointEc = 22,
    f_yDblPointEc = 23,
    f_getBytecode = 24,
    f_touchedAddress = 25,
    f_touchedStorageSlots = 26,
    f_bitwise_ = 27,
    f_comp_ = 28,
    f_loadScalar = 29,
    f_getGlobalExitRootManagerAddr = 30,
    f_log = 31,
    f_resetTouchedAddress = 32,    
    f_resetStorageSlots = 33,
    f_exp = 34,
    f_storeLog = 35,
    f_memAlignWR_W0 = 36,
    f_memAlignWR_W1 = 37,
    f_memAlignWR8_W0 = 38,
    f_saveContractBytecode = 39,
    f_onOpcode = 40
} tFunction;

tFunction string2Function(string s);

// Contains a ROM command data, and arrays possibly containing other ROM commands data
class RomCommand {
public:
    bool isPresent; // presence flag
    string op; // command
    string varName; // variable name
    string regName; // register name
    string funcName; // function name
    tFunction function;
    mpz_class num; //number
    vector<RomCommand *> values;
    vector<RomCommand *> params;
    uint64_t offset;
    RomCommand() : isPresent(false), num(0), offset(0) {};
    string toString(void);
};

// Functions to parse/free a ROM command, or an array of them
void parseRomCommandArray (vector<RomCommand *> &values, json tag);
void parseRomCommand      (RomCommand &cmd, json tag);
void freeRomCommandArray  (vector<RomCommand *> &array);
void freeRomCommand       (RomCommand &cmd);

#endif