#include <nlohmann/json.hpp>
#include "memory_executor.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "timer.hpp"
#include "zklog.hpp"

using json = nlohmann::json;

void MemoryExecutor::execute (vector<MemoryAccess> &input, MemCommitPols &pols)
{
    // Get input size
    uint64_t inputSize = input.size();
    uint64_t inputSizeMinusOne = inputSize - 1;

    // Check input size does not exceed the number of evaluations
    if (inputSize > N)
    {
        zklog.error("MemoryExecutor::execute() Too many entries input.size()=" + to_string(input.size()) + " > N=" + to_string(N));
        exitProcess();
    }

    // Reorder
    TimerStart(MEMORY_EXECUTOR_REORDER);
    vector<MemoryAccess> access;
    reorder(input, access);
    TimerStopAndLog(MEMORY_EXECUTOR_REORDER);

    // We use variables to store the previous values of addr and step
    // We need this to complete the "empty" evaluations of the polynomials addr and step
    // We cannot do it with i-1 because we have to "protect" the case that the access list is empty
    Goldilocks::Element lastAddr = fr.zero();
    uint64_t prevStep = 0;

    // Counter of the current polynomial evaluation
    uint64_t i=0;

    // For every input we consume one evaluation
    for (; i<inputSize; i++)
    {
        pols.addr[i] = fr.fromU64(access[i].address);
        pols.step[i] = fr.fromU64(access[i].pc);
        pols.mOp[i] = fr.one();
        if (access[i].bIsWrite)
        {
            pols.mWr[i] = fr.one();
        }
        pols.val[0][i] = access[i].fe0;
        pols.val[1][i] = access[i].fe1;
        pols.val[2][i] = access[i].fe2;
        pols.val[3][i] = access[i].fe3;
        pols.val[4][i] = access[i].fe4;
        pols.val[5][i] = access[i].fe5;
        pols.val[6][i] = access[i].fe6;
        pols.val[7][i] = access[i].fe7;           
    
        if ( (i < (inputSizeMinusOne)) && 
             (access[i].address == access[i+1].address) )
        {
            //pols.lastAccess[i] = fr.zero(); // Committed pols memory is zero by default
        }
        else
        {
            pols.lastAccess[i] = fr.one();
        }

#ifdef LOG_MEMORY_EXECUTOR
        mpz_class addr = pols.addr[i];
        zklog.info( "Memory executor: i=" + to_string(i) + 
        " addr=" + addr.get_str(16) +
        " step=" + fr.toString(pols.step[i],10) + 
        " mOp=" + fr.toString(pols.mOp[i],10) + 
        " mWr=" + fr.toString(pols.mWr[i], 10) +
        " val=" + fr.toString(pols.val[7][i],16) + 
            ":" + fr.toString(pols.val[6][i],16) + 
            ":" + fr.toString(pols.val[5][i],16) + 
            ":" + fr.toString(pols.val[4][i],16) +
            ":" + fr.toString(pols.val[3][i],16) + 
            ":" + fr.toString(pols.val[2][i],16) + 
            ":" + fr.toString(pols.val[1][i],16) + 
            ":" + fr.toString(pols.val[0][i],16) +
        " lastAccess=" + fr.toString(pols.lastAccess[i],10));
#endif
    }

    // If the input list was not empty, get the values from the previous evaluation before breaking the loop
    if (i > 0)
    {
        lastAddr = fr.add(pols.addr[i-1], fr.one());
        prevStep = fr.toU64(pols.step[i-1]);
    }

    // After all inputs have been processed, consume the rest of evaluations
    for (; i<N; i++)
    {
        // We complete the remaining polynomial evaluations
        // To validate the pil correctly keep last addr incremented +1 and increment the step respect to the previous value
        pols.addr[i] = lastAddr;
        prevStep++;
        pols.step[i] = fr.fromU64(prevStep);
    }
    
    // pols.lastAccess = 1 in the last evaluation to ensure ciclical validation
    pols.lastAccess[N-1] = fr.one();

    zklog.info("MemoryExecutor successfully processed " + to_string(access.size()) + " memory accesses (" + to_string((double(access.size())*100)/N) + "%)");
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
        zklog.info("Memory access i=" + to_string(i) + " address=" + aux.get_str(16) + " pc=" + to_string(access[i].pc) + " " + (access[i].bIsWrite?"WRITE":"READ") + " value="
        + fr.toString(access[i].fe7,16) + ":" + fr.toString(access[i].fe6,16) + ":" + fr.toString(access[i].fe5,16) + ":" + fr.toString(access[i].fe4,16) + ":"
        + fr.toString(access[i].fe3,16) + ":" + fr.toString(access[i].fe2,16) + ":" + fr.toString(access[i].fe1,16) + ":" + fr.toString(access[i].fe0,16));
    }
}