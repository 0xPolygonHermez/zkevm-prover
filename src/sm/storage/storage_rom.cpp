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

        // Instructions
        if (j["program"][i].contains("iJmpz")) romLine.iJmpz = true;
        if (j["program"][i].contains("iJmp")) romLine.iJmp = true;
        if (j["program"][i].contains("iRotateLevel")) romLine.iRotateLevel = true;
        if (j["program"][i].contains("iHash")) romLine.iHash = true;
        if (j["program"][i].contains("iHashType")) romLine.iHashType = j["program"][i]["iHashType"];
        if (j["program"][i].contains("iClimbRkey")) romLine.iClimbRkey = true;
        if (j["program"][i].contains("iClimbSiblingRkey")) romLine.iClimbSiblingRkey = true;
        if (j["program"][i].contains("iClimbSiblingRkeyN")) romLine.iClimbSiblingRkeyN = true;
        if (j["program"][i].contains("iLatchGet")) romLine.iLatchGet = true;
        if (j["program"][i].contains("iLatchSet")) romLine.iLatchSet = true;
        
        // Selectors
        if (j["program"][i].contains("inFREE")) romLine.inFREE = true;
        if (j["program"][i].contains("inOLD_ROOT")) romLine.inOLD_ROOT = true;
        if (j["program"][i].contains("inNEW_ROOT")) romLine.inNEW_ROOT = true;
        if (j["program"][i].contains("inVALUE_LOW")) romLine.inVALUE_LOW = true;
        if (j["program"][i].contains("inVALUE_HIGH")) romLine.inVALUE_HIGH = true;
        if (j["program"][i].contains("inRKEY")) romLine.inRKEY = true;
        if (j["program"][i].contains("inRKEY_BIT")) romLine.inRKEY_BIT = true;
        if (j["program"][i].contains("inSIBLING_RKEY")) romLine.inSIBLING_RKEY = true;
        if (j["program"][i].contains("inSIBLING_VALUE_HASH")) romLine.inSIBLING_VALUE_HASH = true;
        if (j["program"][i].contains("inROTL_VH")) romLine.inROTL_VH = true;

        // Setters
        if (j["program"][i].contains("setRKEY")) romLine.setRKEY = true;
        if (j["program"][i].contains("setRKEY_BIT")) romLine.setRKEY_BIT = true;
        if (j["program"][i].contains("setVALUE_LOW")) romLine.setVALUE_LOW = true;
        if (j["program"][i].contains("setVALUE_HIGH")) romLine.setVALUE_HIGH = true;
        if (j["program"][i].contains("setLEVEL")) romLine.setLEVEL = true;
        if (j["program"][i].contains("setOLD_ROOT")) romLine.setOLD_ROOT = true;
        if (j["program"][i].contains("setNEW_ROOT")) romLine.setNEW_ROOT = true;
        if (j["program"][i].contains("setHASH_LEFT")) romLine.setHASH_LEFT = true;
        if (j["program"][i].contains("setHASH_RIGHT")) romLine.setHASH_RIGHT = true;
        if (j["program"][i].contains("setSIBLING_RKEY")) romLine.setSIBLING_RKEY = true;
        if (j["program"][i].contains("setSIBLING_VALUE_HASH")) romLine.setSIBLING_VALUE_HASH = true;

        // Jump parameters
        if (romLine.iJmp || romLine.iJmpz)
        {
            romLine.addressLabel = j["program"][i]["addressLabel"];
            romLine.address = j["program"][i]["address"];
        }

        // inFREE parameters
        if (romLine.inFREE)
        {
            romLine.op = j["program"][i]["freeInTag"]["op"];
            if (romLine.op=="functionCall")
            {
                romLine.funcName = j["program"][i]["freeInTag"]["funcName"];
                for (uint64_t p=0; p<j["program"][i]["freeInTag"]["params"].size(); p++)
                {
                    romLine.params.push_back(j["program"][i]["freeInTag"]["params"][p]["num"]);
                }
            }
        }

        // Constant
        if (j["program"][i].contains("CONST"))
        {
            romLine.CONST = j["program"][i]["CONST"];
        }

        line.push_back(romLine);
    }
}