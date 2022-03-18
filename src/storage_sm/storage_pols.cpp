#include "storage_pols.hpp"

void StoragePols::alloc (uint64_t len)
{
    zkassert(pAddress == NULL);
    length = len;
    polSize = length*sizeof(uint64_t);
    numberOfPols = 71;
    totalSize = polSize*numberOfPols;
    pAddress = (uint64_t *)malloc(totalSize);
    if (pAddress==NULL)
    {
        cerr << "StoragePols::alloc() failed calling malloc of size " << totalSize << endl;
        exit(-1);
    }
    memset(pAddress, 0, totalSize);

    PC = pAddress;
    RKEY_BIT = pAddress + length;

    HASH_LEFT0 = pAddress + 2*length;
    HASH_LEFT1 = pAddress + 3*length;
    HASH_LEFT2 = pAddress + 4*length;
    HASH_LEFT3 = pAddress + 5*length;
    HASH_RIGHT0 = pAddress + 6*length;
    HASH_RIGHT1 = pAddress + 7*length;
    HASH_RIGHT2 = pAddress + 8*length;
    HASH_RIGHT3 = pAddress + 9*length;
    OLD_ROOT0 = pAddress + 10*length;
    OLD_ROOT1 = pAddress + 11*length;
    OLD_ROOT2 = pAddress + 12*length;
    OLD_ROOT3 = pAddress + 13*length;
    NEW_ROOT0 = pAddress + 14*length;
    NEW_ROOT1 = pAddress + 15*length;
    NEW_ROOT2 = pAddress + 16*length;
    NEW_ROOT3 = pAddress + 17*length;
    VALUE_LOW0 = pAddress + 18*length;
    VALUE_LOW1 = pAddress + 19*length;
    VALUE_LOW2 = pAddress + 20*length;
    VALUE_LOW3 = pAddress + 21*length;
    VALUE_HIGH0 = pAddress + 22*length;
    VALUE_HIGH1 = pAddress + 23*length;
    VALUE_HIGH2 = pAddress + 24*length;
    VALUE_HIGH3 = pAddress + 25*length;
    SIBLING_VALUE_HASH0 = pAddress + 26*length;
    SIBLING_VALUE_HASH1 = pAddress + 27*length;
    SIBLING_VALUE_HASH2 = pAddress + 28*length;
    SIBLING_VALUE_HASH3 = pAddress + 29*length;
    RKEY0 = pAddress + 30*length;
    RKEY1 = pAddress + 31*length;
    RKEY2 = pAddress + 32*length;
    RKEY3 = pAddress + 33*length;
    SIBLING_RKEY0 = pAddress + 34*length;
    SIBLING_RKEY1 = pAddress + 35*length;
    SIBLING_RKEY2 = pAddress + 36*length;
    SIBLING_RKEY3 = pAddress + 37*length;
    LEVEL0 = pAddress + 38*length;
    LEVEL1 = pAddress + 39*length;
    LEVEL2 = pAddress + 40*length;
    LEVEL3 = pAddress + 41*length;

    iJmpz = pAddress + 42*length;
    iJmp = pAddress + 43*length;
    iRotateLevel = pAddress + 44*length;
    iHash = pAddress + 45*length;
    iClimbRkey = pAddress + 46*length;
    iClimbSiblingRkey = pAddress + 47*length;
    iLatchGet = pAddress + 48*length;
    iLatchSet = pAddress + 49*length;

    inFREE = pAddress + 50*length;
    inOLD_ROOT = pAddress + 51*length;
    inNEW_ROOT = pAddress + 52*length;
    inRKEY_BIT = pAddress + 53*length;
    inVALUE_LOW = pAddress + 54*length;
    inVALUE_HIGH = pAddress + 55*length;
    inRKEY = pAddress + 56*length;
    inSIBLING_RKEY = pAddress + 57*length;
    inSIBLING_VALUE_HASH = pAddress + 58*length;

    setRKEY = pAddress + 59*length;
    setRKEY_BIT = pAddress + 60*length;
    setVALUE_LOW = pAddress + 61*length;
    setVALUE_HIGH = pAddress + 62*length;
    setLEVEL = pAddress + 63*length;
    setOLD_ROOT = pAddress + 64*length;
    setNEW_ROOT = pAddress + 65*length;
    setHASH_LEFT = pAddress + 66*length;
    setHASH_RIGHT = pAddress + 67*length;
    setSIBLING_RKEY = pAddress + 68*length;
    setSIBLING_VALUE_HASH = pAddress + 69*length;

    CONST = pAddress + 70*length;
}


void StoragePols::dealloc (void)
{
    zkassert(pAddress != NULL);
    free(pAddress);
    pAddress = NULL;
}