#include "storage_pols.hpp"
#include "utils.hpp"

uint64_t StoragePols::getPolOrder (json &j, const char * pPolName)
{    
    zkassert(j.contains("references"));
    zkassert(j["references"].is_object());
    zkassert(j["references"].contains(pPolName));
    zkassert(j["references"][pPolName].is_object());
    zkassert(j["references"][pPolName].contains("type"));
    zkassert(j["references"][pPolName]["type"].is_string());
    zkassert(j["references"][pPolName]["type"] == "cmP");
    zkassert(j["references"][pPolName].contains("id"));
    zkassert(j["references"][pPolName]["id"].is_number_unsigned());

    uint64_t id;
    id = j["references"][pPolName]["id"];

    zkassert(id>=firstPolId);
    id -= firstPolId;

#ifdef LOG_STORAGE_EXECUTOR
    cout << "StoragePols::getPolOrder() name=" << pPolName << " id=" << id << endl;
#endif
    return id;
}

void StoragePols::alloc (uint64_t len, json &j)
{
    zkassert(pAddress == NULL);
    length = len;
    polSize = length*sizeof(uint64_t);
    
    zkassert(j.contains("nCommitments"));
    zkassert(j["nCommitments"].is_number_unsigned());
    nCommitments = j["nCommitments"];

#ifdef LOG_STORAGE_EXECUTOR
    cout << "StoragePols::alloc() got nCommitments=" << nCommitments << endl;
#endif

    #define FIRST_STORAGE_POL "Storage.free0"
    zkassert(j.contains("references"));
    zkassert(j["references"].is_object());
    zkassert(j["references"].contains(FIRST_STORAGE_POL));
    zkassert(j["references"][FIRST_STORAGE_POL].is_object());
    zkassert(j["references"][FIRST_STORAGE_POL].contains("id"));
    zkassert(j["references"][FIRST_STORAGE_POL]["id"].is_number_unsigned());

    firstPolId = j["references"][FIRST_STORAGE_POL]["id"];
    zkassert(firstPolId < nCommitments);

    numberOfPols = nCommitments - firstPolId;

    totalSize = polSize*numberOfPols;

    if (config.storagePolsFile.size() == 0)
    {
        pAddress = (uint64_t *)malloc(totalSize);
        if (pAddress==NULL)
        {
            cerr << "StoragePols::alloc() failed calling malloc of size " << totalSize << endl;
            exit(-1);
        }
        memset(pAddress, 0, totalSize);
    }
    else
    {
        pAddress = (uint64_t *)mapFile(config.storagePolsFile, totalSize, true);
        zkassert(pAddress!=NULL);
    }

    PC = pAddress + getPolOrder(j, "Storage.pc")*length;;
    RKEY_BIT = pAddress + getPolOrder(j, "Storage.rkeyBit")*length;
    HASH_LEFT0 = pAddress + getPolOrder(j, "Storage.hashLeft0")*length;
    HASH_LEFT1 = pAddress + getPolOrder(j, "Storage.hashLeft1")*length;
    HASH_LEFT2 = pAddress + getPolOrder(j, "Storage.hashLeft2")*length;
    HASH_LEFT3 = pAddress + getPolOrder(j, "Storage.hashLeft3")*length;
    HASH_RIGHT0 = pAddress + getPolOrder(j, "Storage.hashRight0")*length;
    HASH_RIGHT1 = pAddress + getPolOrder(j, "Storage.hashRight1")*length;
    HASH_RIGHT2 = pAddress + getPolOrder(j, "Storage.hashRight2")*length;
    HASH_RIGHT3 = pAddress + getPolOrder(j, "Storage.hashRight3")*length;
    OLD_ROOT0 = pAddress + getPolOrder(j, "Storage.oldRoot0")*length;
    OLD_ROOT1 = pAddress + getPolOrder(j, "Storage.oldRoot1")*length;
    OLD_ROOT2 = pAddress + getPolOrder(j, "Storage.oldRoot2")*length;
    OLD_ROOT3 = pAddress + getPolOrder(j, "Storage.oldRoot3")*length;
    NEW_ROOT0 = pAddress + getPolOrder(j, "Storage.newRoot0")*length;
    NEW_ROOT1 = pAddress + getPolOrder(j, "Storage.newRoot1")*length;
    NEW_ROOT2 = pAddress + getPolOrder(j, "Storage.newRoot2")*length;
    NEW_ROOT3 = pAddress + getPolOrder(j, "Storage.newRoot3")*length;
    VALUE_LOW0 = (FieldElement*)(pAddress + getPolOrder(j, "Storage.valueLow0")*length);
    VALUE_LOW1 = (FieldElement*)(pAddress + getPolOrder(j, "Storage.valueLow1")*length);
    VALUE_LOW2 = (FieldElement*)(pAddress + getPolOrder(j, "Storage.valueLow2")*length);
    VALUE_LOW3 = (FieldElement*)(pAddress + getPolOrder(j, "Storage.valueLow3")*length);
    VALUE_HIGH0 = (FieldElement*)(pAddress + getPolOrder(j, "Storage.valueHigh0")*length);
    VALUE_HIGH1 = (FieldElement*)(pAddress + getPolOrder(j, "Storage.valueHigh1")*length);
    VALUE_HIGH2 = (FieldElement*)(pAddress + getPolOrder(j, "Storage.valueHigh2")*length);
    VALUE_HIGH3 = (FieldElement*)(pAddress + getPolOrder(j, "Storage.valueHigh3")*length);
    SIBLING_VALUE_HASH0 = pAddress + getPolOrder(j, "Storage.siblingValueHash0")*length;
    SIBLING_VALUE_HASH1 = pAddress + getPolOrder(j, "Storage.siblingValueHash1")*length;
    SIBLING_VALUE_HASH2 = pAddress + getPolOrder(j, "Storage.siblingValueHash2")*length;
    SIBLING_VALUE_HASH3 = pAddress + getPolOrder(j, "Storage.siblingValueHash3")*length;
    RKEY0 = pAddress + getPolOrder(j, "Storage.rkey0")*length;
    RKEY1 = pAddress + getPolOrder(j, "Storage.rkey1")*length;
    RKEY2 = pAddress + getPolOrder(j, "Storage.rkey2")*length;
    RKEY3 = pAddress + getPolOrder(j, "Storage.rkey3")*length;
    SIBLING_RKEY0 = pAddress + getPolOrder(j, "Storage.siblingRkey0")*length;
    SIBLING_RKEY1 = pAddress + getPolOrder(j, "Storage.siblingRkey1")*length;
    SIBLING_RKEY2 = pAddress + getPolOrder(j, "Storage.siblingRkey2")*length;
    SIBLING_RKEY3 = pAddress + getPolOrder(j, "Storage.siblingRkey3")*length;
    LEVEL0 = pAddress + getPolOrder(j, "Storage.level0")*length;
    LEVEL1 = pAddress + getPolOrder(j, "Storage.level1")*length;
    LEVEL2 = pAddress + getPolOrder(j, "Storage.level2")*length;
    LEVEL3 = pAddress + getPolOrder(j, "Storage.level3")*length;
    FREE0 = pAddress + getPolOrder(j, "Storage.free0")*length;
    FREE1 = pAddress + getPolOrder(j, "Storage.free1")*length;
    FREE2 = pAddress + getPolOrder(j, "Storage.free2")*length;
    FREE3 = pAddress + getPolOrder(j, "Storage.free3")*length;
    CONST0 = pAddress + getPolOrder(j, "Storage.iConst0")*length;
    CONST1 = pAddress + getPolOrder(j, "Storage.iConst1")*length;
    CONST2 = pAddress + getPolOrder(j, "Storage.iConst2")*length;
    CONST3 = pAddress + getPolOrder(j, "Storage.iConst3")*length;

    iJmpz = pAddress + getPolOrder(j, "Storage.iJmpz")*length;
    iJmp = pAddress + getPolOrder(j, "Storage.iJmp")*length;
    iRotateLevel = pAddress + getPolOrder(j, "Storage.iRotateLevel")*length;
    iHash = pAddress + getPolOrder(j, "Storage.iHash")*length;
    iHashType = pAddress + getPolOrder(j, "Storage.iHashType")*length;
    iClimbRkey = pAddress + getPolOrder(j, "Storage.iClimbRkey")*length;
    iClimbSiblingRkey = pAddress + getPolOrder(j, "Storage.iClimbSiblingRkey")*length;
    iLatchGet = pAddress + getPolOrder(j, "Storage.iLatchGet")*length;
    iLatchSet = pAddress + getPolOrder(j, "Storage.iLatchSet")*length;
    iAddress = pAddress + getPolOrder(j, "Storage.iAddress")*length;

    inFREE = pAddress + getPolOrder(j, "Storage.selFree")*length;
    inOLD_ROOT = pAddress + getPolOrder(j, "Storage.selOldRoot")*length;
    inNEW_ROOT = pAddress + getPolOrder(j, "Storage.selNewRoot")*length;
    inRKEY_BIT = pAddress + getPolOrder(j, "Storage.selRkeyBit")*length; // TODO: slRkeyBit
    inVALUE_LOW = pAddress + getPolOrder(j, "Storage.selValueLow")*length;
    inVALUE_HIGH = pAddress + getPolOrder(j, "Storage.selValueHigh")*length;
    inRKEY = pAddress + getPolOrder(j, "Storage.selRkey")*length;
    inSIBLING_RKEY = pAddress + getPolOrder(j, "Storage.selSiblingRkey")*length;
    inSIBLING_VALUE_HASH = pAddress + getPolOrder(j, "Storage.selSiblingValueHash")*length;

    setRKEY = pAddress + getPolOrder(j, "Storage.setRkey")*length;
    setRKEY_BIT = pAddress + getPolOrder(j, "Storage.setRkeyBit")*length;
    setVALUE_LOW = pAddress + getPolOrder(j, "Storage.setValueLow")*length;
    setVALUE_HIGH = pAddress + getPolOrder(j, "Storage.setValueHigh")*length;
    setLEVEL = pAddress + getPolOrder(j, "Storage.setLevel")*length;
    setOLD_ROOT = pAddress + getPolOrder(j, "Storage.setOldRoot")*length;
    setNEW_ROOT = pAddress + getPolOrder(j, "Storage.setNewRoot")*length;
    setHASH_LEFT = pAddress + getPolOrder(j, "Storage.setHashLeft")*length;
    setHASH_RIGHT = pAddress + getPolOrder(j, "Storage.setHashRight")*length;
    setSIBLING_RKEY = pAddress + getPolOrder(j, "Storage.setSiblingRkey")*length;
    setSIBLING_VALUE_HASH = pAddress + getPolOrder(j, "Storage.setSiblingValueHash")*length;
}


void StoragePols::dealloc (void)
{
    zkassert(pAddress != NULL);
    if (config.storagePolsFile.size() == 0)
    {
        free(pAddress);
    }
    else
    {
        unmapFile(pAddress, totalSize);
    }
    pAddress = NULL;
}