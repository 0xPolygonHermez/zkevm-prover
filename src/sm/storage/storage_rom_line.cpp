#include <iostream>
#include "storage_rom_line.hpp"

void StorageRomLine::print (uint64_t l)
{
    size_t found = fileName.find_last_of("/\\");
    string path = fileName.substr(0,found);
    string file = fileName.substr(found+1);

    // Mandatory fields
    cout << "StorageRomLine l=" << l << " line=" << line << " file=" << file << " ";

     // Selectors
    if (inFREE) cout << "inFREE ";
    if (op.size()>0) // inFREE parameters
    {
        cout << "op=" << op;
        cout << " funcName=" << funcName;
        cout << " #params=" << params.size() << " ";
        for (uint64_t i=0; i<params.size(); i++)
        {
            cout << "params[" << i << "]=" << params[i] << " ";
        }
    }
    if (CONST.size()>0) cout << "CONST=" << CONST << " "; // Constant
    if (inOLD_ROOT) cout << "inOLD_ROOT ";
    if (inNEW_ROOT) cout << "inNEW_ROOT ";
    if (inRKEY_BIT) cout << "inRKEY_BIT ";
    if (inVALUE_LOW) cout << "inVALUE_LOW ";
    if (inVALUE_HIGH) cout << "inVALUE_HIGH ";
    if (inRKEY) cout << "inRKEY ";
    if (inSIBLING_RKEY) cout << "inSIBLING_RKEY ";
    if (inSIBLING_VALUE_HASH) cout << "inSIBLING_VALUE_HASH ";
    if (inROTL_VH) cout << "inROTL_VH ";

    // Instructions
    if (iJmpz) cout << "iJmpz ";
    if (iJmp) cout << "iJmp ";
    if (addressLabel.size()>0) cout << "addressLabel=" << addressLabel << " "; // Jump parameter
    if (address>0) cout << "address=" << address << " "; // Jump parameter
    if (iRotateLevel) cout << "iRotateLevel ";
    if (iHash) cout << "iHash " << "iHashType=" << iHashType << " ";
    if (iClimbRkey) cout << "iClimbRkey ";
    if (iClimbSiblingRkey) cout << "iClimbSiblingRkey ";
    if (iClimbSiblingRkeyN) cout << "iClimbSiblingRkeyN ";
    if (iLatchGet) cout << "iLatchGet ";
    if (iLatchSet) cout << "iLatchSet ";

    // Setters
    if (setRKEY) cout << "setRKEY ";
    if (setRKEY_BIT) cout << "setRKEY_BIT ";
    if (setVALUE_LOW) cout << "setVALUE_LOW ";
    if (setVALUE_HIGH) cout << "setVALUE_HIGH ";
    if (setLEVEL) cout << "setLEVEL ";
    if (setOLD_ROOT) cout << "setOLD_ROOT ";
    if (setNEW_ROOT) cout << "setNEW_ROOT ";
    if (setHASH_LEFT) cout << "setHASH_LEFT ";
    if (setHASH_RIGHT) cout << "setHASH_RIGHT ";
    if (setSIBLING_RKEY) cout << "setSIBLING_RKEY ";
    if (setSIBLING_VALUE_HASH) cout << "setSIBLING_VALUE_HASH ";

    cout << endl;
}