#ifndef MEMORY_ACCESS_LIST_HPP
#define MEMORY_ACCESS_LIST_HPP

#include <vector>
#include "ffiasm/fr.hpp"

using namespace std;

class MemoryAccess
{
public:
    bool bIsWrite;
    uint64_t address;
    uint64_t pc;
    RawFr::Element fe0;
    RawFr::Element fe1;
    RawFr::Element fe2;
    RawFr::Element fe3;
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
    void print (RawFr &fr);
};

#endif