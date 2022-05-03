#include "binary_const_pols.hpp"
#include "utils.hpp"

void BinaryConstPols::alloc (uint64_t len, json &j)
{
    zkassert(pAddress == NULL);
    length = len;

    zkassert(j["nConstants"]==16);
    totalSize = length*(8 + 4*REGISTERS_NUM);

    if (config.binaryConstPolsFile.size() != 0)
    {
        pAddress = (uint8_t *)mapFile(config.binaryConstPolsFile, totalSize, false);
        zkassert(pAddress!=NULL);
    }
    else
    {
        pAddress = (uint8_t *)malloc(totalSize);
        if (pAddress==NULL)
        {
            cerr << "BinaryConstPols::alloc() failed calling malloc of size " << totalSize << endl;
            exit(-1);
        }
        memset(pAddress, 0, totalSize);
    }

    uint64_t offset = 0;
    zkassert(j["references"]["Binary.P_OPCODE"]["type"]=="constP");
    zkassert(j["references"]["Binary.P_OPCODE"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.P_OPCODE"]["id"]==0);
    P_OPCODE = pAddress + offset;
    offset += length;
    
    zkassert(j["references"]["Binary.P_A"]["type"]=="constP");
    zkassert(j["references"]["Binary.P_A"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.P_A"]["id"]==1);
    P_A = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.P_B"]["type"]=="constP");
    zkassert(j["references"]["Binary.P_B"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.P_B"]["id"]==2);
    P_B = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.P_CIN"]["type"]=="constP");
    zkassert(j["references"]["Binary.P_CIN"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.P_CIN"]["id"]==3);
    P_CIN = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.P_LAST"]["type"]=="constP");
    zkassert(j["references"]["Binary.P_LAST"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.P_LAST"]["id"]==4);
    P_LAST = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.P_C"]["type"]=="constP");
    zkassert(j["references"]["Binary.P_C"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.P_C"]["id"]==5);
    P_C = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.P_COUT"]["type"]=="constP");
    zkassert(j["references"]["Binary.P_COUT"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.P_COUT"]["id"]==6);
    P_COUT = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.RESET"]["type"]=="constP");
    zkassert(j["references"]["Binary.RESET"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.RESET"]["id"]==7);
    RESET = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.FACTOR"]["type"]=="constP");
    zkassert(j["references"]["Binary.FACTOR"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.FACTOR"]["id"]==8);
    zkassert(j["references"]["Binary.FACTOR"]["isArray"]==true);
    zkassert(j["references"]["Binary.FACTOR"]["len"]==REGISTERS_NUM);

    for (uint64_t i=0; i<REGISTERS_NUM; i++)
    {
        FACTOR[i] = (uint32_t *)(pAddress + offset);
        offset += 4*length;
    }

    zkassert(offset==totalSize);
}

void BinaryConstPols::dealloc (void)
{
    zkassert(pAddress != NULL);
    if (config.binaryPolsFile.size() != 0)
    {
        unmapFile(pAddress, totalSize);
    }
    else
    {
        free(pAddress);
    }
    pAddress = NULL;
}