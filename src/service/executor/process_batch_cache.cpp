#include "process_batch_cache.hpp"

#define PROCESS_BATCH_CACHE_MAX_ITEMS 1000

bool ProcessBatchCache::Read (ProverRequest & proverRequest)
{
    lock();

    for (uint64_t i=0; i<cache.size(); i++)
    {
        if (cache[i].input == proverRequest.input)
        {
            proverRequest.fullTracer = cache[i].fullTracer;
            proverRequest.counters = cache[i].counters;
            unlock();
            return true;
        }
    }

    unlock();
    return false;
}

void ProcessBatchCache::Write (const ProverRequest & proverRequest)
{
    lock();
    if (cache.size() < PROCESS_BATCH_CACHE_MAX_ITEMS)
    {
        cache.push_back(proverRequest);
    }
    else
    {
        uint64_t pos = next % PROCESS_BATCH_CACHE_MAX_ITEMS;
        cache[pos].input = proverRequest.input;
        cache[pos].fullTracer = proverRequest.fullTracer;
        cache[pos].counters = proverRequest.counters;
    }
    next += 1;
    //cout << "ProcessBatchCache::Write() next=" << next << endl;
    unlock();
}