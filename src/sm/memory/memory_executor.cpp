#include <nlohmann/json.hpp>
#include "memory_executor.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "timer.hpp"

using json = nlohmann::json;

void MemoryExecutor::execute (vector<MemoryAccess> &input, MemCommitPols &pols)
{
    // Check input size
    if (input.size() > N)
    {
        cerr << "Error: MemoryExecutor::execute() Too many entries input.size()=" << input.size() << " > N=" << N << endl;
        exitProcess();
    }

    // Reorder
    TimerStart(MEMORY_EXECUTOR_REORDER);
    vector<MemoryAccess> access;
    reorder(input, access);
    TimerStopAndLog(MEMORY_EXECUTOR_REORDER);

    uint64_t a=0; // access number, so current access is access[a]

    //We use variables to store the previous values of addr and step. We need this
    //to complete the "empty" evaluations of the polynomials addr and step. We cannot
    //do it with i-1 because we have to "protect" the case that the access list is empty    
    uint64_t lastAddr = 0;
    uint64_t prevStep = 0;

    // For all polynomial evaluations
    for (uint64_t i=0; i<N; i++)
    {
        // If we still have accesses to process
        if ( a < access.size() )
        {
            pols.addr[i] = fr.fromU64(access[a].address);
            pols.step[i] = fr.fromU64(access[a].pc);
            pols.mOp[i] = fr.one();
            if (access[a].bIsWrite)
            {
                pols.mWr[i] = fr.one();
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
                //pols.lastAccess[i] = fr.zero(); // Committed pols memory is zero by default
            }
            else
            {
                pols.lastAccess[i] = fr.one();
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

            lastAddr = fr.toU64(pols.addr[i]);
            prevStep = fr.toU64(pols.step[i]);

            // Increment memory access counter
            a++;
        }

        // If access list has been completely consumed
        else
        {
            //We complete the remaining polynomial evaluations. To validate the pil correctly
            //keep last addr incremented +1 and increment the step respect to the previous value
            pols.addr[i] = fr.fromU64(lastAddr+1);
            prevStep++;
            pols.step[i] = fr.fromU64(prevStep);

            //lastAccess = 1 in the last evaluation to ensure ciclical validation
            if (i == (N-1))
            {
                pols.lastAccess[i] = fr.one(); // Committed pols memory is zero by default
            }
        }

    }

    cout << "MemoryExecutor successfully processed " << access.size() << " memory accesses (" << (double(access.size())*100)/N << "%)" << endl;
}

class MemoryAccessCompare
{
public:
    bool operator()(const MemoryAccess &a, const MemoryAccess &b) const
    {
        if (a.address == b.address) return a.pc < b.pc;
        else return a.address < b.address;
    }
};

void MemoryExecutor::reorder (const vector<MemoryAccess> &input, vector<MemoryAccess> &output)
{
    // Clear output vector
    output.clear();

    // Map input MemoryAccess entries using the MemoryAccessCompare class to order them
    map<MemoryAccess, uint64_t, MemoryAccessCompare> auxMap;
    for (uint64_t i=0; i<input.size(); i++)
    {
        auxMap[input[i]] = i;
    }

    // Copy data from the map to the output vector, in the map order
    map<MemoryAccess, uint64_t, MemoryAccessCompare>::const_iterator auxMapIterator;
    for (auxMapIterator = auxMap.begin(); auxMapIterator != auxMap.end(); auxMapIterator++)
    {
        output.push_back(auxMapIterator->first);
    }
}

void MemoryExecutor::print (const vector<MemoryAccess> &access, Goldilocks &fr)
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