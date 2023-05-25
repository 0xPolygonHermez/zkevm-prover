#ifndef MULTI_WRITE_HPP
#define MULTI_WRITE_HPP

#include "multi_write_data.hpp"

using namespace std;

class MultiWrite
{
public:

    uint64_t lastFlushId;
    uint64_t storedFlushId;
    uint64_t storingFlushId;

    uint64_t pendingToFlushDataIndex; // Index of data to store data of batches being processed
    uint64_t storingDataIndex; // Index of data being sent to database
    uint64_t synchronizingDataIndex; // Index of data being synchronized to other database caches

    MultiWriteData data[3];

    pthread_mutex_t mutex; // Mutex to protect the multi write queues
    
    // Constructor
    MultiWrite() :
        lastFlushId(0),
        storedFlushId(0),
        storingFlushId(0),
        pendingToFlushDataIndex(0),
        storingDataIndex(2),
        synchronizingDataIndex(2)
    {
        // Init mutex
        pthread_mutex_init(&mutex, NULL);

        // Reset data
        data[0].Reset();
        data[1].Reset();
        data[2].Reset();
    };

    // Lock/Unlock
    void Lock(void) { pthread_mutex_lock(&mutex); };
    void Unlock(void) { pthread_mutex_unlock(&mutex); };
    bool IsEmpty(void) { return data[0].IsEmpty() && data[1].IsEmpty() && data[2].IsEmpty(); };
};

#endif