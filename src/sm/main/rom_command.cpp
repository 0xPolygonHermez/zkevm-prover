#include <vector>
#include <iostream>
#include <string>
#include "rom_command.hpp"

using namespace std;
using json = nlohmann::json;

tFunction string2Function(string s)
{
    if (s == "beforeLast") return f_beforeLast;
    else if (s == "getGlobalHash") return f_getGlobalHash;
    else if (s == "getGlobalExitRoot") return f_getGlobalExitRoot;
    else if (s == "getOldStateRoot") return f_getOldStateRoot;
    else if (s == "getNewStateRoot") return f_getNewStateRoot;
    else if (s == "getSequencerAddr") return f_getSequencerAddr;
    else if (s == "getOldLocalExitRoot") return f_getOldLocalExitRoot;
    else if (s == "getNewLocalExitRoot") return f_getNewLocalExitRoot;
    else if (s == "getNumBatch") return f_getNumBatch;
    else if (s == "getTimestamp") return f_getTimestamp;
    else if (s == "getBatchHashData") return f_getBatchHashData;
    else if (s == "getTxs") return f_getTxs;
    else if (s == "getTxsLen") return f_getTxsLen;
    else if (s == "addrOp") return f_addrOp;
    else if (s == "eventLog") return f_eventLog;
    else if (s == "cond") return f_cond;
    else if (s == "inverseFpEc") return f_inverseFpEc;
    else if (s == "inverseFnEc") return f_inverseFnEc;
    else if (s == "sqrtFpEc") return f_sqrtFpEc;
    else if (s == "xAddPointEc") return f_xAddPointEc;
    else if (s == "yAddPointEc") return f_yAddPointEc;
    else if (s == "xDblPointEc") return f_xDblPointEc;
    else if (s == "yDblPointEc") return f_yDblPointEc;
    else if (s == "getBytecode") return f_getBytecode;
    else if (s == "touchedAddress") return f_touchedAddress;
    else if (s == "touchedStorageSlots") return f_touchedStorageSlots;
    else if (s == "bitwise_") return f_bitwise_;
    else if (s == "comp_") return f_comp_;
    else if (s == "loadScalar") return f_loadScalar;
    else if (s == "getGlobalExitRootManagerAddr") return f_getGlobalExitRootManagerAddr;
    else if (s == "log") return f_log;
    else if (s == "resetTouchedAddress") return f_resetTouchedAddress;    
    else if (s == "resetStorageSlots") return f_resetStorageSlots;
    else if (s == "exp") return f_exp;
    else if (s == "storeLog") return f_storeLog;
    else if (s == "memAlignWR_W0") return f_memAlignWR_W0;
    else if (s == "memAlignWR_W1") return f_memAlignWR_W1;
    else if (s == "memAlignWR8_W0") return f_memAlignWR8_W0;
    else if (s == "saveContractBytecode") return f_saveContractBytecode;
    else if (s == "onOpcode") return f_onOpcode;
    else if (s.rfind("precompiled_", 0) == 0) {
        cerr << "Error: string2function() ignore string = " << s << endl;
        return f_empty;
    }
    else {
        cerr << "Error: string2function() invalid string = " << s << endl;
        exit(-1);
    }
}

string RomCommand::toString (void)
{
    string result;

    if (!isPresent) return "";

    if (op.size() != 0) result += " op=" + op;
    if (funcName.size() != 0) result += " funcName=" + funcName;
    if (varName.size() != 0) result += " varName=" + varName;
    if (regName.size() != 0) result += " regName=" + regName;
    if (num != mpz_class(0)) result += " num=" + num.get_str(16);
    if (offset != 0) result += " offset=" + to_string(offset);

    for (uint64_t i=0; i<values.size(); i++)
    {
        result += " values[" + to_string(i) +"]={" + values[i]->toString() + " }";
    }

    for (uint64_t i=0; i<params.size(); i++)
    {
        result += " params[" + to_string(i) +"]={" + params[i]->toString() + " }";
    }

    return result;
}

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
    if (tag.contains("funcName")) {
        cmd.funcName = tag["funcName"];
        cmd.function = string2Function (cmd.funcName);
    }
    if (tag.contains("num")) { string aux = tag["num"]; cmd.num.set_str(aux, 10); }
    if (tag.contains("offset") && tag["offset"].is_number()) { cmd.offset = tag["offset"]; } // TODO: Why some offsets are strings? "FNEC", "FPEC"
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