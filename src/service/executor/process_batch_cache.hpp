#ifndef PROCESS_BATCH_CACHE
#define PROCESS_BATCH_CACHE

#include "prover_request.hpp"

class ProcessBatchCache
{
    vector<ProverRequest> cache;
    pthread_mutex_t mutex; // Mutex to protect the access to the throughput attributes
    uint64_t next;
public:
    ProcessBatchCache()
    {
        pthread_mutex_init(&mutex, NULL);
        next = 0;
    }
private:
    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };
public:
    bool Read (ProverRequest & proverRequest);
    void Write (const ProverRequest & proverRequest);
};

#endif