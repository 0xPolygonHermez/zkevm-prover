#ifndef STATEDB_SINGLETON_HPP
#define STATEDB_SINGLETON_HPP

#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "statedb.hpp"

class StateDBSingleton
{
private:
    StateDB * pStateDB;
    pthread_mutex_t mutex;    // Mutex to protect the access to the singleton
public:
    StateDBSingleton();
    void lock(void) { pthread_mutex_lock(&mutex); };
    void unlock(void) { pthread_mutex_unlock(&mutex); };

public:
    // Returns a valid StateDB pointer, creating it the first time it is called
    StateDB * get(Goldilocks &fr, const Config &config);

    // Gets the current value of pStateDB, without creating any
    StateDB * get(void);
};

extern StateDBSingleton stateDBSingleton;

#endif