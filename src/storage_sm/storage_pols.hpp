#ifndef STORAGE_POLS_HPP
#define STORAGE_POLS_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include "config.hpp"
#include "ff/ff.hpp"

class StoragePols
{
private:
    const Config &config;
public:
    uint64_t * PC;
    uint64_t * RKEY_BIT;

    uint64_t * HASH_LEFT0;
    uint64_t * HASH_LEFT1;
    uint64_t * HASH_LEFT2;
    uint64_t * HASH_LEFT3;
    uint64_t * HASH_RIGHT0;
    uint64_t * HASH_RIGHT1;
    uint64_t * HASH_RIGHT2;
    uint64_t * HASH_RIGHT3;
    uint64_t * OLD_ROOT0;
    uint64_t * OLD_ROOT1;
    uint64_t * OLD_ROOT2;
    uint64_t * OLD_ROOT3;
    uint64_t * NEW_ROOT0;
    uint64_t * NEW_ROOT1;
    uint64_t * NEW_ROOT2;
    uint64_t * NEW_ROOT3;
    FieldElement * VALUE_LOW0;
    FieldElement * VALUE_LOW1;
    FieldElement * VALUE_LOW2;
    FieldElement * VALUE_LOW3;
    FieldElement * VALUE_HIGH0;
    FieldElement * VALUE_HIGH1;
    FieldElement * VALUE_HIGH2;
    FieldElement * VALUE_HIGH3;
    uint64_t * SIBLING_VALUE_HASH0;
    uint64_t * SIBLING_VALUE_HASH1;
    uint64_t * SIBLING_VALUE_HASH2;
    uint64_t * SIBLING_VALUE_HASH3;
    uint64_t * RKEY0;
    uint64_t * RKEY1;
    uint64_t * RKEY2;
    uint64_t * RKEY3;
    uint64_t * SIBLING_RKEY0;
    uint64_t * SIBLING_RKEY1;
    uint64_t * SIBLING_RKEY2;
    uint64_t * SIBLING_RKEY3;
    uint64_t * LEVEL0;
    uint64_t * LEVEL1;
    uint64_t * LEVEL2;
    uint64_t * LEVEL3;
    uint64_t * FREE0;
    uint64_t * FREE1;
    uint64_t * FREE2;
    uint64_t * FREE3;
    uint64_t * CONST0;
    uint64_t * CONST1;
    uint64_t * CONST2;
    uint64_t * CONST3;

    // Instructions
    uint64_t * iJmpz;
    uint64_t * iJmp;
    uint64_t * iRotateLevel;
    uint64_t * iHash;
    uint64_t * iHashType;
    uint64_t * iClimbRkey;
    uint64_t * iClimbSiblingRkey;
    uint64_t * iLatchGet;
    uint64_t * iLatchSet;
    uint64_t * iAddress;

    // Selectors
    uint64_t * inFREE;
    uint64_t * inOLD_ROOT;
    uint64_t * inNEW_ROOT;
    uint64_t * inRKEY_BIT;
    uint64_t * inVALUE_LOW;
    uint64_t * inVALUE_HIGH;
    uint64_t * inRKEY;
    uint64_t * inSIBLING_RKEY;
    uint64_t * inSIBLING_VALUE_HASH;

//pol bool commit selPC; // TODO

    // Setters
    uint64_t * setRKEY;
    uint64_t * setRKEY_BIT;
    uint64_t * setVALUE_LOW;
    uint64_t * setVALUE_HIGH;
    uint64_t * setLEVEL;
    uint64_t * setOLD_ROOT;
    uint64_t * setNEW_ROOT;
    uint64_t * setHASH_LEFT;
    uint64_t * setHASH_RIGHT;
    uint64_t * setSIBLING_RKEY;
    uint64_t * setSIBLING_VALUE_HASH;

private:
    // Internal attributes
    uint64_t nCommitments;
    uint64_t firstPolId;
    uint64_t length;
    uint64_t polSize;
    uint64_t numberOfPols;
    uint64_t totalSize;
    uint64_t * pAddress;

public:
    StoragePols(const Config &config) : config(config)
    {
        pAddress = NULL;
    }

    void alloc (uint64_t len, json &j);
    void dealloc (void);

    uint64_t getPolOrder (json &j, const char * pPolName);
};

#endif