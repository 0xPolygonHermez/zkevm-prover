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
    RKEY_BIT = pAddress + polSize;

    HASH_LEFT0 = pAddress + 2*polSize;
    HASH_LEFT1 = pAddress + 3*polSize;
    HASH_LEFT2 = pAddress + 4*polSize;
    HASH_LEFT3 = pAddress + 5*polSize;
    HASH_RIGHT0 = pAddress + 6*polSize;
    HASH_RIGHT1 = pAddress + 7*polSize;
    HASH_RIGHT2 = pAddress + 8*polSize;
    HASH_RIGHT3 = pAddress + 9*polSize;
    OLD_ROOT0 = pAddress + 10*polSize;
    OLD_ROOT1 = pAddress + 11*polSize;
    OLD_ROOT2 = pAddress + 12*polSize;
    OLD_ROOT3 = pAddress + 13*polSize;
    NEW_ROOT0 = pAddress + 14*polSize;
    NEW_ROOT1 = pAddress + 15*polSize;
    NEW_ROOT2 = pAddress + 16*polSize;
    NEW_ROOT3 = pAddress + 17*polSize;
    VALUE_LOW0 = pAddress + 18*polSize;
    VALUE_LOW1 = pAddress + 19*polSize;
    VALUE_LOW2 = pAddress + 20*polSize;
    VALUE_LOW3 = pAddress + 21*polSize;
    VALUE_HIGH0 = pAddress + 22*polSize;
    VALUE_HIGH1 = pAddress + 23*polSize;
    VALUE_HIGH2 = pAddress + 24*polSize;
    VALUE_HIGH3 = pAddress + 25*polSize;
    SIBLING_VALUE_HASH0 = pAddress + 26*polSize;
    SIBLING_VALUE_HASH1 = pAddress + 27*polSize;
    SIBLING_VALUE_HASH2 = pAddress + 28*polSize;
    SIBLING_VALUE_HASH3 = pAddress + 29*polSize;
    RKEY0 = pAddress + 30*polSize;
    RKEY1 = pAddress + 31*polSize;
    RKEY2 = pAddress + 32*polSize;
    RKEY3 = pAddress + 33*polSize;
    SIBLING_RKEY0 = pAddress + 34*polSize;
    SIBLING_RKEY1 = pAddress + 35*polSize;
    SIBLING_RKEY2 = pAddress + 36*polSize;
    SIBLING_RKEY3 = pAddress + 37*polSize;
    LEVEL0 = pAddress + 38*polSize;
    LEVEL1 = pAddress + 39*polSize;
    LEVEL2 = pAddress + 40*polSize;
    LEVEL3 = pAddress + 41*polSize;

    iJmpz = pAddress + 42*polSize;
    iJmp = pAddress + 43*polSize;
    iRotateLevel = pAddress + 44*polSize;
    iHash = pAddress + 45*polSize;
    iClimbRkey = pAddress + 46*polSize;
    iClimbSiblingRkey = pAddress + 47*polSize;
    iLatchGet = pAddress + 48*polSize;
    iLatchSet = pAddress + 49*polSize;

    inFREE = pAddress + 50*polSize;
    inOLD_ROOT = pAddress + 51*polSize;
    inNEW_ROOT = pAddress + 52*polSize;
    inRKEY_BIT = pAddress + 53*polSize;
    inVALUE_LOW = pAddress + 54*polSize;
    inVALUE_HIGH = pAddress + 55*polSize;
    inRKEY = pAddress + 56*polSize;
    inSIBLING_RKEY = pAddress + 57*polSize;
    inSIBLING_VALUE_HASH = pAddress + 58*polSize;

    setRKEY = pAddress + 59*polSize;
    setRKEY_BIT = pAddress + 60*polSize;
    setVALUE_LOW = pAddress + 61*polSize;
    setVALUE_HIGH = pAddress + 62*polSize;
    setLEVEL = pAddress + 63*polSize;
    setOLD_ROOT = pAddress + 64*polSize;
    setNEW_ROOT = pAddress + 65*polSize;
    setHASH_LEFT = pAddress + 66*polSize;
    setHASH_RIGHT = pAddress + 67*polSize;
    setSIBLING_RKEY = pAddress + 68*polSize;
    setSIBLING_VALUE_HASH = pAddress + 69*polSize;

    CONST = pAddress + 70*polSize;
}


void StoragePols::dealloc (void)
{
    zkassert(pAddress != NULL);
    free(pAddress);
    pAddress = NULL;
}