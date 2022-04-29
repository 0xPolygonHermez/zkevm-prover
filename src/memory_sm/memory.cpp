#include <nlohmann/json.hpp>
#include "memory.hpp"
#include "memory_pols.hpp"
#include "utils.hpp"
#include "scalar.hpp"

using json = nlohmann::json;

void MemoryExecutor::execute (vector<MemoryAccess> &access)
{
    // Allocate polynomials
    MemoryPols pols(config);
    uint64_t polSize = 1<<16;
    json pilJson;
    file2json(config.memoryPilFile, pilJson);
    pols.alloc(polSize, pilJson);

    uint64_t a=0; // access number, so current access is access[a]

    // For all polynomial evaluations
    for (uint64_t i=0; i<polSize; i++)
    {
        // Set the next evaluation index, which will be 0 when we reach the last evaluation
        //uint64_t nexti = (i+1)%polSize;

        // If we still have accesses to process
        if ( a < access.size() )
        {
            pols.addr[i] = access[a].address;
            pols.step[i] = access[a].pc;
            pols.mOp[i] = 1;
            if (access[a].bIsWrite)
            {
                pols.mRd[i] = 0;
                pols.mWr[i] = 1;
            }
            else
            {
                pols.mRd[i] = 1;
                pols.mWr[i] = 0;
            }
            pols.val[0][i] = access[a].fe0;
            pols.val[1][i] = access[a].fe1;
            pols.val[2][i] = access[a].fe2;
            pols.val[3][i] = access[a].fe3;
            pols.val[4][i] = access[a].fe4;
            pols.val[5][i] = access[a].fe5;
            pols.val[6][i] = access[a].fe6;
            pols.val[7][i] = access[a].fe7;           
        
            if ( (a < (access.size()-1)) && 
                 (access[a].address == access[a+1].address) )
            {
                pols.lastAccess[i] = 0;
            }
            else
            {
                pols.lastAccess[i] = 1;
            }

#ifdef LOG_MEMORY_EXECUTOR
            mpz_class addr = pols.addr[i];
            cout << "Memory executor: i=" << i << 
            " addr=" << addr.get_str(16) << 
            " step=" << pols.step[i] << 
            " mOp=" << pols.mOp[i] << 
            " mRd=" << pols.mRd[i] << 
            " mWr=" << pols.mWr[i] <<
            " val=" << fr.toString(pols.val[7][i],16) << 
                ":" << fr.toString(pols.val[6][i],16) << 
                ":" << fr.toString(pols.val[5][i],16) << 
                ":" << fr.toString(pols.val[4][i],16) << 
                ":" << fr.toString(pols.val[3][i],16) << 
                ":" << fr.toString(pols.val[2][i],16) << 
                ":" << fr.toString(pols.val[1][i],16) << 
                ":" << fr.toString(pols.val[0][i],16) <<
            " lastAccess=" << pols.lastAccess[i] << endl;
#endif
            // Increment memory access counter
            a++;
        }

        // If access list has been completely consumed
        else
        {
            pols.lastAccess[i] = 1;
        }

    }
    
    // Deallocate polynomials
    pols.dealloc();

    cout << "MemoryExecutor successfully processed " << access.size() << " memory accesses" << endl;
}