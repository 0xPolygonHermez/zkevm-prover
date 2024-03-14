#include <iostream>
#include "storage_rom_line.hpp"
#include "zklog.hpp"

void StorageRomLine::print (uint64_t l)
{
    size_t found = fileName.find_last_of("/\\");
    string path = fileName.substr(0,found);
    string file = fileName.substr(found+1);

    string s;

    // Mandatory fields
    s = "StorageRomLine l=" + to_string(l) + " line=" + to_string(line) + " file=" + file + " ";

     // Selectors
    if (inFREE) s += "inFREE ";
    if (op.size()>0) // inFREE parameters
    {
        s += "op=" + op;
        s += " funcName=" + funcName;
        s += " #params=" + to_string(params.size()) + " ";
        for (uint64_t i=0; i<params.size(); i++)
        {
            s += "params[" + to_string(i) + "]=" + to_string(params[i]) + " ";
        }
    }
    if (CONST.size()>0) s += "CONST=" + CONST + " "; // Constant
    if (inOLD_ROOT) s += "inOLD_ROOT ";
    if (inNEW_ROOT) s += "inNEW_ROOT ";
    if (inRKEY_BIT) s += "inRKEY_BIT ";
    if (inVALUE_LOW) s += "inVALUE_LOW ";
    if (inVALUE_HIGH) s += "inVALUE_HIGH ";
    if (inRKEY) s += "inRKEY ";
    if (inSIBLING_RKEY) s += "inSIBLING_RKEY ";
    if (inSIBLING_VALUE_HASH) s += "inSIBLING_VALUE_HASH ";
    if (inROTL_VH) s += "inROTL_VH ";

    // Instructions
    if (iJmpz) s += "iJmpz ";
    if (iJmp) s += "iJmp ";
    if (addressLabel.size()>0) s += "addressLabel=" + addressLabel + " "; // Jump parameter
    if (address>0) s += "address=" + to_string(address) + " "; // Jump parameter
    if (iRotateLevel) s += "iRotateLevel ";
    if (iHash) s += "iHash iHashType=" + to_string(iHashType) + " ";
    if (iClimbRkey) s += "iClimbRkey ";
    if (iClimbSiblingRkey) s += "iClimbSiblingRkey ";
    if (iClimbSiblingRkeyN) s += "iClimbSiblingRkeyN ";
    if (iLatchGet) s += "iLatchGet ";
    if (iLatchSet) s += "iLatchSet ";

    // Setters
    if (setRKEY) s += "setRKEY ";
    if (setRKEY_BIT) s += "setRKEY_BIT ";
    if (setVALUE_LOW) s += "setVALUE_LOW ";
    if (setVALUE_HIGH) s += "setVALUE_HIGH ";
    if (setLEVEL) s += "setLEVEL ";
    if (setOLD_ROOT) s += "setOLD_ROOT ";
    if (setNEW_ROOT) s += "setNEW_ROOT ";
    if (setHASH_LEFT) s += "setHASH_LEFT ";
    if (setHASH_RIGHT) s += "setHASH_RIGHT ";
    if (setSIBLING_RKEY) s += "setSIBLING_RKEY ";
    if (setSIBLING_VALUE_HASH) s += "setSIBLING_VALUE_HASH ";

    zklog.info(s);
}