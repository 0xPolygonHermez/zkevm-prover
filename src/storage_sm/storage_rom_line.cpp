#include <iostream>
#include "storage_rom_line.hpp"

void StorageRomLine::print (void)
{
    // Mandatory fields
    cout << "StorageRomLine line=" << line << " fileName=" << fileName << endl;

    // Instructions
    if (iJmpz) cout << "iJmpz=true" << endl;
    if (iJmp) cout << "iJmp=true" << endl;
    if (iRotateLevel) cout << "iRotateLevel=true" << endl;
    if (iHash) cout << "iHash=true" << endl;
    if (iClimbRkey) cout << "iClimbRkey=true" << endl;
    if (iClimbSiblingRkey) cout << "iClimbSiblingRkey=true" << endl;
    if (iLatchGet) cout << "iLatchGet=true" << endl;
    if (iLatchSet) cout << "iLatchSet=true" << endl;

    // Selectors
    if (inFREE) cout << "inFREE=true" << endl;
    if (inOLD_ROOT) cout << "inOLD_ROOT=true" << endl;
    if (inNEW_ROOT) cout << "inNEW_ROOT=true" << endl;
    if (inRKEY_BIT) cout << "inRKEY_BIT=true" << endl;
    if (inVALUE_LOW) cout << "inVALUE_LOW=true" << endl;
    if (inVALUE_HIGH) cout << "inVALUE_HIGH=true" << endl;
    if (inRKEY) cout << "inRKEY=true" << endl;
    if (inSIBLING_RKEY) cout << "inSIBLING_RKEY=true" << endl;
    if (inSIBLING_VALUE_HASH) cout << "inSIBLING_VALUE_HASH=true" << endl;

    // Setters
    if (setRKEY) cout << "setRKEY=true" << endl;
    if (setRKEY_BIT) cout << "setRKEY_BIT=true" << endl;
    if (setVALUE_LOW) cout << "setVALUE_LOW=true" << endl;
    if (setVALUE_HIGH) cout << "setVALUE_HIGH=true" << endl;
    if (setLEVEL) cout << "setLEVEL=true" << endl;
    if (setOLD_ROOT) cout << "setOLD_ROOT=true" << endl;
    if (setNEW_ROOT) cout << "setNEW_ROOT=true" << endl;
    if (setHASH_LEFT) cout << "setHASH_LEFT=true" << endl;
    if (setHASH_RIGHT) cout << "setHASH_RIGHT=true" << endl;
    if (setSIBLING_RKEY) cout << "setSIBLING_RKEY=true" << endl;
    if (setSIBLING_VALUE_HASH) cout << "setSIBLING_VALUE_HASH=true" << endl;

    // Jump parameters
    if (addressLabel.size()>0) cout << "addressLabel=" << addressLabel << endl;
    if (address>0) cout << "address=" << address << endl;

    // inFREE parameters
    if (op.size()>0)
    {
        cout << "op=" << op << endl;
        cout << "  funcName=" << funcName << endl;
        cout << "  #params=" << params.size() << endl;
        for (uint64_t i=0; i<params.size(); i++)
        {
            cout << "    params[" << i << "]=" << params[i] << endl;
        }
    }

    // Constant
    if (CONST.size()>0) cout << "CONST=" << CONST << endl;
}