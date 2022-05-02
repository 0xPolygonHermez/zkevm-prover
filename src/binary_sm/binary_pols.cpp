#include "binary_pols.hpp"
#include "utils.hpp"

void BinaryPols::alloc (uint64_t len, json &j)
{
    zkassert(pAddress == NULL);
    length = len;

    zkassert(j["nCommitments"]==33);
    totalSize = length*(8 + 25*4);

    if (config.binaryPolsFile.size() != 0)
    {
        pAddress = (uint8_t *)mapFile(config.binaryPolsFile, totalSize, true);
        zkassert(pAddress!=NULL);
    }
    else
    {
        pAddress = (uint8_t *)malloc(totalSize);
        if (pAddress==NULL)
        {
            cerr << "BinaryPols::alloc() failed calling malloc of size " << totalSize << endl;
            exit(-1);
        }
        memset(pAddress, 0, totalSize);
    }

    uint64_t offset = 0;

    zkassert(j["references"]["Binary.freeInA"]["type"]=="cmP");
    zkassert(j["references"]["Binary.freeInA"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.freeInA"]["id"]==0);
    freeInA = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.freeInB"]["type"]=="cmP");
    zkassert(j["references"]["Binary.freeInB"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.freeInB"]["id"]==1);
    freeInB = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.freeInC"]["type"]=="cmP");
    zkassert(j["references"]["Binary.freeInC"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.freeInC"]["id"]==2);
    freeInC = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.a0"]["type"]=="cmP");
    zkassert(j["references"]["Binary.a0"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.a0"]["id"]==3);
    a0 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.a1"]["type"]=="cmP");
    zkassert(j["references"]["Binary.a1"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.a1"]["id"]==4);
    a1 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.a2"]["type"]=="cmP");
    zkassert(j["references"]["Binary.a2"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.a2"]["id"]==5);
    a2 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.a3"]["type"]=="cmP");
    zkassert(j["references"]["Binary.a3"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.a3"]["id"]==6);
    a3 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.a4"]["type"]=="cmP");
    zkassert(j["references"]["Binary.a4"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.a4"]["id"]==7);
    a4 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.a5"]["type"]=="cmP");
    zkassert(j["references"]["Binary.a5"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.a5"]["id"]==8);
    a5 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.a6"]["type"]=="cmP");
    zkassert(j["references"]["Binary.a6"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.a6"]["id"]==9);
    a6 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.a7"]["type"]=="cmP");
    zkassert(j["references"]["Binary.a7"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.a7"]["id"]==10);
    a7 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.b0"]["type"]=="cmP");
    zkassert(j["references"]["Binary.b0"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.b0"]["id"]==11);
    b0 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.b1"]["type"]=="cmP");
    zkassert(j["references"]["Binary.b1"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.b1"]["id"]==12);
    b1 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.b2"]["type"]=="cmP");
    zkassert(j["references"]["Binary.b2"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.b2"]["id"]==13);
    b2 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.b3"]["type"]=="cmP");
    zkassert(j["references"]["Binary.b3"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.b3"]["id"]==14);
    b3 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.b4"]["type"]=="cmP");
    zkassert(j["references"]["Binary.b4"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.b4"]["id"]==15);
    b4 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.b5"]["type"]=="cmP");
    zkassert(j["references"]["Binary.b5"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.b5"]["id"]==16);
    b5 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.b6"]["type"]=="cmP");
    zkassert(j["references"]["Binary.b6"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.b6"]["id"]==17);
    b6 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.b7"]["type"]=="cmP");
    zkassert(j["references"]["Binary.b7"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.b7"]["id"]==18);
    b7 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.c0"]["type"]=="cmP");
    zkassert(j["references"]["Binary.c0"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.c0"]["id"]==19);
    c0 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.c1"]["type"]=="cmP");
    zkassert(j["references"]["Binary.c1"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.c1"]["id"]==20);
    c1 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.c2"]["type"]=="cmP");
    zkassert(j["references"]["Binary.c2"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.c2"]["id"]==21);
    c2 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.c3"]["type"]=="cmP");
    zkassert(j["references"]["Binary.c3"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.c3"]["id"]==22);
    c3 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.c4"]["type"]=="cmP");
    zkassert(j["references"]["Binary.c4"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.c4"]["id"]==23);
    c4 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.c5"]["type"]=="cmP");
    zkassert(j["references"]["Binary.c5"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.c5"]["id"]==24);
    c5 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.c6"]["type"]=="cmP");
    zkassert(j["references"]["Binary.c6"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.c6"]["id"]==25);
    c6 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.c7"]["type"]=="cmP");
    zkassert(j["references"]["Binary.c7"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.c7"]["id"]==26);
    c7 = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.c0Temp"]["type"]=="cmP");
    zkassert(j["references"]["Binary.c0Temp"]["elementType"]=="u32");
    zkassert(j["references"]["Binary.c0Temp"]["id"]==27);
    c0Temp = (uint32_t*)(pAddress + offset);
    offset += 4*length;

    zkassert(j["references"]["Binary.opcode"]["type"]=="cmP");
    zkassert(j["references"]["Binary.opcode"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.opcode"]["id"]==28);
    opcode = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.cIn"]["type"]=="cmP");
    zkassert(j["references"]["Binary.cIn"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.cIn"]["id"]==29);
    cIn = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.cOut"]["type"]=="cmP");
    zkassert(j["references"]["Binary.cOut"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.cOut"]["id"]==30);
    cOut = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.last"]["type"]=="cmP");
    zkassert(j["references"]["Binary.last"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.last"]["id"]==31);
    last = pAddress + offset;
    offset += length;

    zkassert(j["references"]["Binary.useCarry"]["type"]=="cmP");
    zkassert(j["references"]["Binary.useCarry"]["elementType"]=="u8");
    zkassert(j["references"]["Binary.useCarry"]["id"]==32);
    useCarry = pAddress + offset;
    offset += length;

    zkassert(offset==totalSize);
}

void BinaryPols::dealloc (void)
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