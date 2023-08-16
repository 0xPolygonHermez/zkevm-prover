#ifndef STATE_MANAGER_64_HPP
#define STATE_MANAGER_64_HPP

#include <unordered_map>
#include <vector>
#include <string>
#include "goldilocks_base_field.hpp"
#include "zkresult.hpp"
#include "database_map.hpp"
#include "persistence.hpp"
#include "database_64.hpp"
#include "utils/time_metric.hpp"

using namespace std;

class TxSubState64
{
public:
    string oldStateRoot;
    string newStateRoot;
    uint64_t previousSubState;
    bool bValid;
    unordered_map<string, vector<Goldilocks::Element>> dbWrite;
    vector<string> dbDelete;
    TxSubState64() : previousSubState(0), bValid(false)
    {
        dbWrite.reserve(128);
        dbDelete.reserve(128);
    };
};

class TxPersistenceState64
{
public:
    string oldStateRoot;
    string newStateRoot;
    uint64_t currentSubState;
    vector<TxSubState64> subState;
    TxPersistenceState64() : currentSubState(0)
    {
        subState.reserve(128);
    };
};

class TxState64
{
public:
    TxPersistenceState64 persistence[PERSISTENCE_SIZE];
};

class BatchState64
{
public:
    string oldStateRoot;
    string currentStateRoot;
    uint64_t currentTx;
    vector<TxState64> txState;
    unordered_map<string, vector<Goldilocks::Element>> dbWrite;
#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    TimeMetricStorage timeMetricStorage;
#endif
    BatchState64() : currentTx(0)
    {
        txState.reserve(32);
        dbWrite.reserve(1024);
    };
};

class StateManager64
{
private:
    unordered_map<string, BatchState64> state;
    Config config;
    pthread_mutex_t mutex; // Mutex to protect the multi write queues
public:
    StateManager64 ()
    {        
        // Init mutex
        pthread_mutex_init(&mutex, NULL);
    };
private:
    zkresult setStateRoot (const string &batchUUID, uint64_t tx, const string &stateRoot, bool bIsOldStateRoot, const Persistence persistence);
public:
    void init (const Config &_config)
    {
        config = _config;
    }
    zkresult setOldStateRoot (const string &batchUUID, uint64_t tx, const string &stateRoot, const Persistence persistence)
    {
        return setStateRoot(batchUUID, tx, stateRoot, true, persistence);
    }
    zkresult setNewStateRoot (const string &batchUUID, uint64_t tx, const string &stateRoot, const Persistence persistence)
    {
        return setStateRoot(batchUUID, tx, stateRoot, false, persistence);
    }
    zkresult write (const string &batchUUID, uint64_t tx, const string &_key, const vector<Goldilocks::Element> &value, const Persistence persistence);
    zkresult deleteNode (const string &batchUUID, uint64_t tx, const string &_key, const Persistence persistence);
    zkresult read (const string &batchUUID, const string &_key, vector<Goldilocks::Element> &value, DatabaseMap *dbReadLog);
    zkresult semiFlush (const string &batchUUID, const string &newStateRoot, const Persistence persistence);
    zkresult flush (const string &batchUUID, Database64 &db, uint64_t &flushId, uint64_t &lastSentFlushId);
    void print (bool bDbContent = false);

    // Lock/Unlock
    void Lock(void) { pthread_mutex_lock(&mutex); };
    void Unlock(void) { pthread_mutex_unlock(&mutex); };
};

extern StateManager64 stateManager64;

#endif