#ifndef STORAGE_ROM_LINE_HPP
#define STORAGE_ROM_LINE_HPP

#include <stdint.h>
#include <string>
#include <vector>

using namespace std;

class StorageRomLine
{
public:
    // Mandatory fields
    uint64_t line;
    string fileName;

    // Instructions
    bool iJmpz;
    bool iJmp;
    bool iRotateLevel;
    bool iHash;
    uint64_t iHashType;
    bool iClimbRkey;
    bool iClimbSiblingRkey;
    bool iClimbSiblingRkeyN;
    bool iLatchGet;
    bool iLatchSet;

    // Selectors
    bool inFREE;
    bool inOLD_ROOT;
    bool inNEW_ROOT;
    bool inRKEY_BIT;
    bool inVALUE_LOW;
    bool inVALUE_HIGH;
    bool inRKEY;
    bool inSIBLING_RKEY;
    bool inSIBLING_VALUE_HASH;
    bool inROTL_VH;

    // Setters
    bool setRKEY;
    bool setRKEY_BIT;
    bool setVALUE_LOW;
    bool setVALUE_HIGH;
    bool setLEVEL;
    bool setOLD_ROOT;
    bool setNEW_ROOT;
    bool setHASH_LEFT;
    bool setHASH_RIGHT;
    bool setSIBLING_RKEY;
    bool setSIBLING_VALUE_HASH;

    // Jump parameters
    string addressLabel;
    uint64_t address;

    // inFREE parameters
    string op;
    string funcName;
    vector<uint64_t> params;

    // Constant
    string CONST;

    StorageRomLine ()
    {
        line = 0;
        inRKEY_BIT = false;
        iJmpz = false;
        iJmp = false;
        iRotateLevel = false;
        iHash = false;
        iHashType = 0;
        iClimbRkey = false;
        iClimbSiblingRkey = false;
        iClimbSiblingRkeyN = false;
        iLatchGet = false;
        iLatchSet = false;
        inFREE = false;
        inOLD_ROOT = false;
        inNEW_ROOT = false;
        inRKEY_BIT = false;
        inVALUE_LOW = false;
        inVALUE_HIGH = false;
        inRKEY = false;
        inSIBLING_RKEY = false;
        inSIBLING_VALUE_HASH = false;
        inROTL_VH = false;
        setRKEY = false;
        setRKEY_BIT = false;
        setVALUE_LOW = false;
        setVALUE_HIGH = false;
        setLEVEL = false;
        setOLD_ROOT = false;
        setNEW_ROOT = false;
        setHASH_LEFT = false;
        setHASH_RIGHT = false;
        setSIBLING_RKEY = false;
        setSIBLING_VALUE_HASH = false;
        address = 0;
    }
    void print (uint64_t l);
};

#endif