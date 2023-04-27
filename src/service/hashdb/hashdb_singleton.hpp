#ifndef HASHDB_SINGLETON_HPP
#define HASHDB_SINGLETON_HPP

#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "hashdb.hpp"

class HashDBSingleton
{
private:
    HashDB * pHashDB;
    pthread_mutex_t mutex;    // Mutex to protect the access to the singleton
public:
    HashDBSingleton();
    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };

public:
    // Returns a valid HashDB pointer, creating it the first time it is called
    HashDB * get(Goldilocks &fr, const Config &config);

    // Gets the current value of pHashDB, without creating any
    HashDB * get(void);
};

extern HashDBSingleton hashDBSingleton;

#endif