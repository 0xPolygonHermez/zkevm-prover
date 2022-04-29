#include "memory_pols.hpp"
#include "utils.hpp"

uint64_t MemoryPols::getPolOrder (json &j, const char * pPolName)
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

    return id;
}

void MemoryPols::alloc (uint64_t len, json &j)
{
    zkassert(pAddress == NULL);
    length = len;
    polSize = length*sizeof(uint64_t);
    
    zkassert(j.contains("nCommitments"));
    zkassert(j["nCommitments"].is_number_unsigned());
    nCommitments = j["nCommitments"];

    numberOfPols = nCommitments;

    totalSize = polSize*numberOfPols;

    if (config.memoryPolsFile.size() == 0)
    {
        pAddress = (uint64_t *)malloc(totalSize);
        if (pAddress==NULL)
        {
            cerr << "MemoryPols::alloc() failed calling malloc of size " << totalSize << endl;
            exit(-1);
        }
        memset(pAddress, 0, totalSize);
    }
    else
    {
        pAddress = (uint64_t *)mapFile(config.memoryPolsFile, totalSize, true);
        zkassert(pAddress!=NULL);
    }

    addr = pAddress + getPolOrder(j, "Ram.addr")*length;
    step = pAddress + getPolOrder(j, "Ram.step")*length;
    mOp = pAddress + getPolOrder(j, "Ram.mOp")*length;
    mRd = pAddress + getPolOrder(j, "Ram.mRd")*length;
    mWr = pAddress + getPolOrder(j, "Ram.mWr")*length;
    zkassert(j["references"]["Ram.val"]["isArray"]==true);
    zkassert(j["references"]["Ram.val"]["len"]==8);
    uint64_t valueOrder = getPolOrder(j, "Ram.val");
    for (uint64_t i=0; i<8; i++) val[i] = (FieldElement*)(pAddress + (valueOrder+i)*length);
    lastAccess = pAddress + getPolOrder(j, "Ram.lastAccess")*length;

}

void MemoryPols::dealloc (void)
{
    zkassert(pAddress != NULL);
    if (config.memoryPolsFile.size() == 0)
    {
        free(pAddress);
    }
    else
    {
        unmapFile(pAddress, totalSize);
    }
    pAddress = NULL;
}