#ifndef POL_TYPES_HPP
#define POL_TYPES_HPP

#include <sys/types.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include "ffiasm/fr.hpp"

using namespace std;

enum eElementType {
   et_unknown = 0,
   et_bool = 1,
   et_s8 = 2,
   et_u8 = 3,
   et_s16 = 4,
   et_u16 = 5,
   et_s32 = 6,
   et_u32 = 7,
   et_s64 = 8,
   et_u64 = 9,
   et_field = 10
};

uint64_t type2size (eElementType elementType);
eElementType string2et (const string &s);

#define INVALID_ID 0xFFFFFFFFFFFFFFFF

class Pol
{
public:

    uint64_t     id;
    eElementType elementType;
    uint64_t     elementSize;
    string       name;
    Pol() {
        id = INVALID_ID;
        elementType = et_unknown;
        };
};

class PolBool: public Pol 
{
public:
    uint8_t * pData;
    PolBool() : Pol() {
        elementType = et_bool;
        elementSize = type2size(elementType);
        pData = NULL;
    }
    void map (void *p) { pData = (uint8_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolS8: public Pol
{
public:
    int8_t * pData;
    PolS8() : Pol() {
        elementType = et_s8;
        elementSize = type2size(elementType);
        pData = NULL;
    }
    void map (void *p) { pData = (int8_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolU8: public Pol 
{
public:
    uint8_t * pData;
    PolU8() : Pol() {
        elementType = et_u8;
        elementSize = type2size(elementType);
        pData = NULL;
    }
    void map (void *p) { pData = (uint8_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolS16: public Pol
{
public:
    int16_t * pData;
    PolS16() : Pol() {
        elementType = et_s16;
        elementSize = type2size(elementType);
        pData = NULL;
    }
    void map (void *p) { pData = (int16_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolU16: public Pol 
{
public:
    uint16_t * pData;
    PolU16() : Pol() {
        elementType = et_u16;
        elementSize = type2size(elementType);
        pData = NULL;
    }
    void map (void *p) { pData = (uint16_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolS32: public Pol
{
public:
    int32_t * pData;
    PolS32() : Pol() {
        elementType = et_s32;
        elementSize = type2size(elementType);
        pData = NULL;
    }
    void map (void *p) { pData = (int32_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolU32: public Pol 
{
public:
    uint32_t * pData;
    PolU32() : Pol() {
        elementType = et_u32;
        elementSize = type2size(elementType);
        pData = NULL;
    }
    void map (void *p) { pData = (uint32_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolS64: public Pol
{
public:
    int64_t * pData;
    PolS64() : Pol() {
        elementType = et_s64;
        elementSize = type2size(elementType);
        pData = NULL;
    }
    void map (void *p) { pData = (int64_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolU64: public Pol 
{
public:
    uint64_t * pData;
    PolU64() : Pol() {
        elementType = et_u64;
        elementSize = type2size(elementType);
        pData = NULL;
    }
    void map (void *p) { pData = (uint64_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolFieldElement: public Pol
{
public:
    RawFr::Element * pData;
    PolFieldElement() : Pol() {
        elementType = et_field;
        elementSize = type2size(elementType);
        pData = NULL;
    }
    void map (void *p) { pData = (RawFr::Element *)p; }
    void unmap (void) { pData = NULL; }
};

#endif