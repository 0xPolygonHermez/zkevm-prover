#include "process_batch_cache.hpp"

#define PROCESS_BATCH_CACHE_MAX_ITEMS 100

bool ProcessBatchCache::Read (ProverRequest & proverRequest)
{
    lock();

#ifdef LOG_PROCESS_BATCH_CACHE
    cout << "--> ProcessBatchCache::Read() writes=" << writes << endl;
#endif

    for (uint64_t i=0; i<cache.size(); i++)
    {
        if (cache[i]->input == proverRequest.input)
        {
            proverRequest.fullTracer = cache[i]->fullTracer;
            proverRequest.counters = cache[i]->counters;
            proverRequest.result = ZKR_SUCCESS;
            readsFound++;

#ifdef LOG_PROCESS_BATCH_CACHE
            cout << "<-- ProcessBatchCache::read() found i=" << i << " writes=" << writes << " readsFound=" << readsFound << " readsNotFound=" << readsNotFound << endl;
#endif
            unlock();
            return true;
        }
    }

    readsNotFound++;

#ifdef LOG_PROCESS_BATCH_CACHE
    cout << "<-- ProcessBatchCache::read() not found writes=" << writes << " readsFound=" << readsFound << " readsNotFound=" << readsNotFound << endl;
#endif

    unlock();
    return false;
}

void ProcessBatchCache::Write (const ProverRequest & proverRequest)
{
    if (proverRequest.result != ZKR_SUCCESS)
    {
        cerr << "Error: ProcessBatchCache::Write() got proverRequest.result != ZKR_SUCCESS result=" << proverRequest.result << "=" << zkresult2string(proverRequest.result) << endl;
        exitProcess();
    }

    lock();

#ifdef LOG_PROCESS_BATCH_CACHE
    cout << "--> ProcessBatchCache::Write() writes=" << writes << " readsFound=" << readsFound << " readsNotFound=" << readsNotFound << endl;
#endif

    if (cache.size() < PROCESS_BATCH_CACHE_MAX_ITEMS)
    {
        ProverRequest * pProverRequest = new ProverRequest(proverRequest.fr, proverRequest.config, proverRequest.type);
        if (pProverRequest == NULL)
        {
            cerr << "Error: ProcessBatchCache::Write() failed calling new ProverRequest()" << endl;
            exitProcess();
        }
        pProverRequest->input = proverRequest.input;
        pProverRequest->fullTracer = proverRequest.fullTracer;
        pProverRequest->counters = proverRequest.counters;
        cache.push_back(pProverRequest);
    }
    else
    {
        uint64_t pos = writes % PROCESS_BATCH_CACHE_MAX_ITEMS;
        cache[pos]->input = proverRequest.input;
        cache[pos]->fullTracer = proverRequest.fullTracer;
        cache[pos]->counters = proverRequest.counters;
    }
    writes++;

#ifdef LOG_PROCESS_BATCH_CACHE
    cout << "<-- ProcessBatchCache::Write() writes=" << writes << " readsFound=" << readsFound << " readsNotFound=" << readsNotFound << endl;
#endif

    unlock();
}