#include <iostream>
#include "hashdb_singleton.hpp"
#include "zklog.hpp"

using namespace std;

HashDBSingleton hashDBSingleton;

HashDBSingleton::HashDBSingleton() : pHashDB(NULL)
{
    pthread_mutex_init(&mutex, NULL);
};

HashDB * HashDBSingleton::init (Goldilocks &fr, const Config &config)
{
    lock();
    if (pHashDB != NULL)
    {
        zklog.error("HashDBSingleton::init() found pHashDB != NULL");
        exitProcess();
    }
    pHashDB = new HashDB(fr, config);
    if (pHashDB == NULL)
    {
        zklog.error("HashDBSingleton::get() failed creating a new HashDB instance");
        exitProcess();
    }
    unlock();
    return pHashDB;
}

HashDB * HashDBSingleton::get(void)
{
    HashDB * pResult;
    lock();
    pResult = pHashDB;
    unlock();
    return pResult;
}