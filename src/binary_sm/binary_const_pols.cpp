#include "binary_const_pols.hpp"
#include "utils.hpp"

uint64_t BinaryConstPols::getPolOrder (json &j, const char * pPolName)
{    
    zkassert(j.contains("references"));
    zkassert(j["references"].is_object());
    zkassert(j["references"].contains(pPolName));
    zkassert(j["references"][pPolName].is_object());
    zkassert(j["references"][pPolName].contains("type"));
    zkassert(j["references"][pPolName]["type"].is_string());
    zkassert(j["references"][pPolName]["type"] == "constP");
    zkassert(j["references"][pPolName].contains("id"));
    zkassert(j["references"][pPolName]["id"].is_number_unsigned());

    uint64_t id;
    id = j["references"][pPolName]["id"];

#ifdef LOG_BINARY_EXECUTOR
    cout << "BinaryConstPols::getPolOrder() name=" << pPolName << " id=" << id << endl;
#endif
    return id;
}

void BinaryConstPols::alloc (uint64_t len, json &j)
{
    zkassert(pAddress == NULL);
    length = len;
    polSize = length*sizeof(uint64_t);
    
    zkassert(j.contains("nCommitments"));
    zkassert(j["nCommitments"].is_number_unsigned());
    nCommitments = j["nCommitments"];

#ifdef LOG_BINARY_EXECUTOR
    cout << "BinaryConstPols::alloc() got nCommitments=" << nCommitments << endl;
#endif

    numberOfPols = nCommitments;

    totalSize = polSize*numberOfPols;

    if (config.binaryPolsFile.size() == 0)
    {
        pAddress = (uint64_t *)malloc(totalSize);
        if (pAddress==NULL)
        {
            cerr << "BinaryPols::alloc() failed calling malloc of size " << totalSize << endl;
            exit(-1);
        }
        memset(pAddress, 0, totalSize);
    }
    else
    {
        pAddress = (uint64_t *)mapFile(config.binaryPolsFile, totalSize, false);
        zkassert(pAddress!=NULL);
    }

    P_OPCODE = pAddress + getPolOrder(j, "Binary.P_OPCODE")*length;
    P_A = pAddress + getPolOrder(j, "Binary.P_A")*length;
    P_B = pAddress + getPolOrder(j, "Binary.P_B")*length;
    P_CIN = pAddress + getPolOrder(j, "Binary.P_CIN")*length;
    P_C = pAddress + getPolOrder(j, "Binary.P_C")*length;
    P_COUT = pAddress + getPolOrder(j, "Binary.P_COUT")*length;
    RESET = pAddress + getPolOrder(j, "Binary.RESET")*length;
    for (uint64_t i=0; i<REGISTERS_NUM; i++)
        FACTOR[i] = pAddress + (getPolOrder(j, "Binary.FACTOR") + i)*length;
}

void BinaryConstPols::dealloc (void)
{
    zkassert(pAddress != NULL);
    if (config.binaryPolsFile.size() == 0)
    {
        free(pAddress);
    }
    else
    {
        unmapFile(pAddress, totalSize);
    }
    pAddress = NULL;
}