#include "binary_pols.hpp"
#include "utils.hpp"

uint64_t BinaryPols::getPolOrder (json &j, const char * pPolName)
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

#ifdef LOG_BINARY_EXECUTOR
    cout << "BinaryPols::getPolOrder() name=" << pPolName << " id=" << id << endl;
#endif
    return id;
}

void BinaryPols::alloc (uint64_t len, json &j)
{
    zkassert(pAddress == NULL);
    length = len;
    polSize = length*sizeof(uint64_t);
    
    zkassert(j.contains("nCommitments"));
    zkassert(j["nCommitments"].is_number_unsigned());
    nCommitments = j["nCommitments"];

#ifdef LOG_BINARY_EXECUTOR
    cout << "BinaryPols::alloc() got nCommitments=" << nCommitments << endl;
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
        pAddress = (uint64_t *)mapFile(config.binaryPolsFile, totalSize, true);
        zkassert(pAddress!=NULL);
    }

    freeInA = pAddress + getPolOrder(j, "Binary.freeInA")*length;
    freeInB = pAddress + getPolOrder(j, "Binary.freeInB")*length;
    freeInC = pAddress + getPolOrder(j, "Binary.freeInC")*length;
    a0 = pAddress + getPolOrder(j, "Binary.a0")*length;
    a1 = pAddress + getPolOrder(j, "Binary.a1")*length;
    a2 = pAddress + getPolOrder(j, "Binary.a2")*length;
    a3 = pAddress + getPolOrder(j, "Binary.a3")*length;
    a4 = pAddress + getPolOrder(j, "Binary.a4")*length;
    a5 = pAddress + getPolOrder(j, "Binary.a5")*length;
    a6 = pAddress + getPolOrder(j, "Binary.a6")*length;
    a7 = pAddress + getPolOrder(j, "Binary.a7")*length;
    b0 = pAddress + getPolOrder(j, "Binary.b0")*length;
    b1 = pAddress + getPolOrder(j, "Binary.b1")*length;
    b2 = pAddress + getPolOrder(j, "Binary.b2")*length;
    b3 = pAddress + getPolOrder(j, "Binary.b3")*length;
    b4 = pAddress + getPolOrder(j, "Binary.b4")*length;
    b5 = pAddress + getPolOrder(j, "Binary.b5")*length;
    b6 = pAddress + getPolOrder(j, "Binary.b6")*length;
    b7 = pAddress + getPolOrder(j, "Binary.b7")*length;
    c0 = pAddress + getPolOrder(j, "Binary.c0")*length;
    c1 = pAddress + getPolOrder(j, "Binary.c1")*length;
    c2 = pAddress + getPolOrder(j, "Binary.c2")*length;
    c3 = pAddress + getPolOrder(j, "Binary.c3")*length;
    c4 = pAddress + getPolOrder(j, "Binary.c4")*length;
    c5 = pAddress + getPolOrder(j, "Binary.c5")*length;
    c6 = pAddress + getPolOrder(j, "Binary.c6")*length;
    c7 = pAddress + getPolOrder(j, "Binary.c7")*length;
    c0Temp = pAddress + getPolOrder(j, "Binary.c0Temp")*length;
    opcode = pAddress + getPolOrder(j, "Binary.opcode")*length;
    cIn = pAddress + getPolOrder(j, "Binary.cIn")*length;
    cOut = pAddress + getPolOrder(j, "Binary.cOut")*length;
    last = pAddress + getPolOrder(j, "Binary.last")*length;
    useCarry = pAddress + getPolOrder(j, "Binary.useCarry")*length;
}

void BinaryPols::dealloc (void)
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