#ifndef MEMORY_ACCESS_HPP
#define MEMORY_ACCESS_HPP

#include "ff/ff.hpp"

class MemoryAccess
{
public:
    bool bIsWrite;
    uint64_t address;
    uint64_t pc;
    FieldElement fe0;
    FieldElement fe1;
    FieldElement fe2;
    FieldElement fe3;
    FieldElement fe4;
    FieldElement fe5;
    FieldElement fe6;
    FieldElement fe7;
};

#endif