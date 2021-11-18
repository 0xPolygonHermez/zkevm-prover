#ifndef POL_TYPES_HPP
#define POL_TYPES_HPP

#include <sys/types.h>
#include <stdio.h>
#include <stdint.h>
#include "ffiasm/fr.hpp"

using namespace std;

enum eElementType {
   et_unknown,
   et_bool,
   et_s8,
   et_u8,
   et_s16,
   et_u16,
   et_s32,
   et_u32,
   et_s64,
   et_u64,
   et_field
};

#define INVALID_ID 0xFFFFFFFFFFFFFFFF

class Pol
{
public:

    uint64_t id;
    eElementType elementType;
    Pol() {
        id = INVALID_ID;
        elementType = et_unknown;
        };
    uint64_t elementSize(void) {
        switch (elementType) {
            case et_bool:
            case et_s8:
            case et_u8: return 1;
            case et_s16:
            case et_u16: return 2;
            case et_s32:
            case et_u32: return 4;
            case et_s64:
            case et_u64: return 8;
            case et_field: return sizeof(RawFr::Element);
        }
        cerr << "Error: Pol::elementSize() caled with invalid elementType" << endl;
        exit(-1);
    }
};

class PolBool: public Pol 
{
public:
    uint8_t * pData;
    PolBool() : Pol() {
        elementType = et_bool;
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
        pData = NULL;
    }
    void map (void *p) { pData = (int16_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolU16: public Pol 
{
public:
    uint16_t * pData;
    PolU16() : Pol() { elementType = et_u16; pData = NULL; }
    void map (void *p) { pData = (uint16_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolS32: public Pol
{
public:
    int32_t * pData;
    PolS32() : Pol() { elementType = et_s32; pData = NULL; }
    void map (void *p) { pData = (int32_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolU32: public Pol 
{
public:
    uint32_t * pData;
    PolU32() : Pol() { elementType = et_u32; pData = NULL; }
    void map (void *p) { pData = (uint32_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolS64: public Pol
{
public:
    int64_t * pData;
    PolS64() : Pol() { elementType = et_s64; pData = NULL; }
    void map (void *p) { pData = (int64_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolU64: public Pol 
{
public:
    uint64_t * pData;
    PolU64() : Pol() { elementType = et_u64; pData = NULL; }
    void map (void *p) { pData = (uint64_t *)p; }
    void unmap (void) { pData = NULL; }
};

class PolFieldElement: public Pol
{
public:
    RawFr::Element * pData;
    PolFieldElement() : Pol() { elementType = et_field; pData = NULL; }
    void map (void *p) { pData = (RawFr::Element *)p; }
    void unmap (void) { pData = NULL; }
};

#endif