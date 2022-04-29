#include <iostream>
#include "memory_access_list.hpp"

/* Reorder access list by the following criteria:
    - In order of incremental address
    - If addresses are the same, in order ov incremental pc
*/
void MemoryAccessList::reorder (void)
{
    vector<MemoryAccess> aux;
    for (uint64_t i=0; i<access.size(); i++)
    {
        MemoryAccess ma = access[i];
        vector<MemoryAccess>::iterator it;
        for (it=aux.begin(); it!=aux.end(); it++)
        {
            if ( it->address > ma.address) break;
            else if ( (it->address == ma.address) && (it->pc > ma.pc) ) break;
        }
        aux.insert(it, ma);
    }
    access = aux;
}

void MemoryAccessList::print (FiniteField &fr)
{
    for (uint64_t i=0; i<access.size(); i++)
    {
        mpz_class aux = access[i].address;
        cout << "Memory access i=" << i << " address=" << aux.get_str(16) << " pc=" << access[i].pc << " " << (access[i].bIsWrite?"WRITE":"READ") << " value="
        << fr.toString(access[i].fe7,16) << ":" << fr.toString(access[i].fe6,16) << ":" << fr.toString(access[i].fe5,16) << ":" << fr.toString(access[i].fe4,16) << ":"
        << fr.toString(access[i].fe3,16) << ":" << fr.toString(access[i].fe2,16) << ":" << fr.toString(access[i].fe1,16) << ":" << fr.toString(access[i].fe0,16)
        << endl;
    }
}