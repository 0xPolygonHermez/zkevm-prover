#ifndef MEMORY_ACCESS_LIST_HPP
#define MEMORY_ACCESS_LIST_HPP

#include <vector>
#include "ff/ff.hpp"

using namespace std;

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

class MemoryAccessList
{
public:
    vector<MemoryAccess> access;

    /* Reorder access list by the following criteria:
        - In order of incremental address
        - If addresses are the same, in order ov incremental pc
    */
    void reorder (void);

    /* Prints access list contents, for debugging purposes */
    void print (FiniteField &fr);
};

#endif