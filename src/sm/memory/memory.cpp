#include <nlohmann/json.hpp>
#include "memory.hpp"
#include "utils.hpp"
#include "scalar.hpp"

using json = nlohmann::json;

void MemoryExecutor::execute (vector<MemoryAccess> &access, RamCommitPols &pols)
{
    uint64_t a=0; // access number, so current access is access[a]

    //We use variables to store the previous values of addr and step. We need this
    //to complete the "empty" evaluations of the polynomials addr and step. We cannot
    //do it with i-1 because we have to "protect" the case that the access list is empty    
    uint64_t lastAddr = 0;
    uint64_t prevStep = 0;

    // For all polynomial evaluations
    uint64_t degree = pols.degree();
    for (uint64_t i=0; i<degree; i++)
    {
        // If we still have accesses to process
        if ( a < access.size() )
        {
            pols.addr[i] = access[a].address;
            pols.step[i] = access[a].pc;
            pols.mOp[i] = 1;
            pols.mWr[i] = (access[a].bIsWrite) ? 1 : 0;
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

            lastAddr = pols.addr[i];
            prevStep = pols.step[i];

            // Increment memory access counter
            a++;
        }

        // If access list has been completely consumed
        else
        {
            //We complete the remaining polynomial evaluations. To validate the pil correctly
            //keep last addr incremented +1 and increment the step respect to the previous value
            pols.addr[i] = lastAddr+1;
            prevStep++;
            pols.step[i] = prevStep;

            //lastAccess = 1 in the last evaluation to ensure ciclical validation
            pols.lastAccess[i] = (i==degree-1) ? 1 : 0;
        }

    }

    cout << "MemoryExecutor successfully processed " << access.size() << " memory accesses" << endl;
}