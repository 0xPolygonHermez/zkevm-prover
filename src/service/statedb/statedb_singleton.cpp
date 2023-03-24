#include <iostream>
#include "statedb_singleton.hpp"
#include "zklog.hpp"

using namespace std;

StateDBSingleton stateDBSingleton;

StateDBSingleton::StateDBSingleton() : pStateDB(NULL)
{
    pthread_mutex_init(&mutex, NULL);
};

StateDB * StateDBSingleton::get(Goldilocks &fr, const Config &config)
{
    lock();
    if (pStateDB == NULL)
    {
        pStateDB = new StateDB(fr, config);
        if (pStateDB == NULL)
        {
            zklog.error("StateDBSingleton::get() failed creating a new StateDB instance");
            exitProcess();
        }
    }
    unlock();
    return pStateDB;
}

StateDB * StateDBSingleton::get(void)
{
    StateDB * pResult;
    lock();
    pResult = pStateDB;
    unlock();
    return pResult;
}