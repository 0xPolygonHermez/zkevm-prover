#include <iostream>
#include "statedb_singleton.hpp"

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
            cerr << "Error: StateDBSingleton::get() failed creating a new StateDB instance" << endl;
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