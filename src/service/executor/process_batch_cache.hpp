#ifndef PROCESS_BATCH_CACHE
#define PROCESS_BATCH_CACHE

#include "prover_request.hpp"

class ProcessBatchCache
{
    vector<ProverRequest *> cache;
    pthread_mutex_t mutex; // Mutex to protect the access to the throughput attributes
    uint64_t readsFound;
    uint64_t readsNotFound;
    uint64_t writes;
public:
    ProcessBatchCache() : readsFound(0), readsNotFound(0), writes(0)
    {
        pthread_mutex_init(&mutex, NULL);
    }
private:
    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };
public:
    bool Read (ProverRequest & proverRequest);
    void Write (const ProverRequest & proverRequest);
};

#endif