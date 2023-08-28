#ifndef MULTI_WRITE_HPP
#define MULTI_WRITE_HPP

#include "goldilocks_base_field.hpp"
#include "multi_write_data.hpp"
#include <vector>

using namespace std;

class MultiWrite
{
public:
    Goldilocks &fr;

    uint64_t lastFlushId;
    uint64_t storedFlushId;
    uint64_t storingFlushId;

    uint64_t pendingToFlushDataIndex; // Index of data to store data of batches being processed
    uint64_t storingDataIndex; // Index of data being sent to database
    uint64_t synchronizingDataIndex; // Index of data being synchronized to other database caches

    MultiWriteData data[3];

    pthread_mutex_t mutex; // Mutex to protect the multi write queues
    
    // Constructor
    MultiWrite(Goldilocks & fr);

    // Lock/Unlock
    void Lock(void) { pthread_mutex_lock(&mutex); };
    void Unlock(void) { pthread_mutex_unlock(&mutex); };
    bool IsEmpty(void) { return data[0].IsEmpty() && data[1].IsEmpty() && data[2].IsEmpty(); };
    string print(void);

    bool findNode(const string &key, vector<Goldilocks::Element> &value);
    bool findProgram(const string &key, vector<uint8_t> &value);
};

#endif