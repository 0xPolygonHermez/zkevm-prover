#ifndef MEMORY_ACCESS_LIST_HPP
#define MEMORY_ACCESS_LIST_HPP

#include <vector>
#include "memory_access.hpp"

using namespace std;

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