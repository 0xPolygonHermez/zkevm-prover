#include <iostream>
#include "storage_rom.hpp"
#include "utils.hpp"
#include "exit_process.hpp"
#include "zklog.hpp"

void StorageRom::load(json &j)
{
    // Parse the program and store them into the line vector
    if ( !j.contains("program") ||
         !j["program"].is_array() )
    {
        zklog.error("StorageRom::load() could not find a root program array");
        exitProcess();
    }
    for (uint64_t i=0; i<j["program"].size(); i++)
    {
        StorageRomLine romLine;

        // Mandatory fields
        romLine.line = j["program"][i]["line"];
        romLine.fileName = j["program"][i]["fileName"];
        romLine.lineStr = j["program"][i]["lineStr"];

        json l = j["program"][i];

        // Instructions
        if (l["jmpz"] == 1) romLine.jmpz = true;
        if (l["jmpnz"] == 1) romLine.jmpnz = true;
        if (l["jmp"] == 1) romLine.jmp = true;
        if (l["hash"] == 1) romLine.hash = true;
        if (l["hashType"].is_number()) romLine.hashType = l["hashType"].get<uint64_t>();
        if (l["climbRkey"] == 1) romLine.climbRkey = true;
        if (l["climbSiblingRkey"] == 1) romLine.climbSiblingRkey = true;
        if (l["climbBitN"] == 1) romLine.climbBitN = true;
        if (l["latchGet"] == 1) romLine.latchGet = true;
        if (l["latchSet"] == 1) romLine.latchSet = true;

        // Selectors
        if (l["inFREE"] == 1) romLine.inFREE = true;
        if (l["inOLD_ROOT"] == 1) romLine.inOLD_ROOT = true;
        if (l["inNEW_ROOT"] == 1) romLine.inNEW_ROOT = true;
        if (l["inVALUE_LOW"] == 1) romLine.inVALUE_LOW = true;
        if (l["inVALUE_HIGH"] == 1) romLine.inVALUE_HIGH = true;
        if (l["inRKEY"] == 1) romLine.inRKEY = true;
        if (l["inRKEY_BIT"] == 1) romLine.inRKEY_BIT = true;
        if (l["inSIBLING_RKEY"].is_number()) romLine.inSIBLING_RKEY = l["inSIBLING_RKEY"].get<int64_t>();
        if (l["inSIBLING_VALUE_HASH"] == 1) romLine.inSIBLING_VALUE_HASH = true;
        if (l["inROTL_VH"] == 1) romLine.inROTL_VH = true;
        if (l["inLEVEL"] == 1) romLine.inLEVEL = true;

        // Setters
        if (l["setRKEY"] == 1) romLine.setRKEY = true;
        if (l["setRKEY_BIT"] == 1) romLine.setRKEY_BIT = true;
        if (l["setVALUE_LOW"] == 1) romLine.setVALUE_LOW = true;
        if (l["setVALUE_HIGH"] == 1) romLine.setVALUE_HIGH = true;
        if (l["setLEVEL"] == 1) romLine.setLEVEL = true;
        if (l["setOLD_ROOT"] == 1) romLine.setOLD_ROOT = true;
        if (l["setNEW_ROOT"] == 1) romLine.setNEW_ROOT = true;
        if (l["setHASH_LEFT"] == 1) romLine.setHASH_LEFT = true;
        if (l["setHASH_RIGHT"] == 1) romLine.setHASH_RIGHT = true;
        if (l["setSIBLING_RKEY"] == 1) romLine.setSIBLING_RKEY = true;
        if (l["setSIBLING_VALUE_HASH"] == 1) romLine.setSIBLING_VALUE_HASH = true;

        // Jump parameters
        if (romLine.jmp || romLine.jmpz || romLine.jmpnz)
        {
            romLine.jmpAddressLabel = l["jmpAddressLabel"];
            romLine.jmpAddress = l["jmpAddress"];
        }

        // inFREE parameters
        if (romLine.inFREE)
        {
            romLine.op = l["freeInTag"]["op"];
            if (romLine.op=="functionCall")
            {
                romLine.funcName = l["freeInTag"]["funcName"];
                const uint64_t paramCount = l["freeInTag"]["params"].size();
                json params = l["freeInTag"]["params"];
                for (uint64_t iParam = 0; iParam < paramCount; iParam++)
                {
                    romLine.params.push_back(params[iParam]["num"]);
                }
            }
        }

        // Constant
        if (l["CONST"].is_number())
        {
            romLine.CONST = to_string(l["CONST"].get<int64_t>());
        }

        line.push_back(romLine);
    }
}