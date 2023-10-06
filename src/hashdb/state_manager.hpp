#ifndef STATE_MANAGER_HPP
#define STATE_MANAGER_HPP

#include <unordered_map>
#include <vector>
#include <string>
#include "goldilocks_base_field.hpp"
#include "zkresult.hpp"
#include "database_map.hpp"
#include "persistence.hpp"
#include "database.hpp"
#include "utils/time_metric.hpp"

using namespace std;

class TxSubState
{
public:
    string oldStateRoot;
    string newStateRoot;
    uint64_t previousSubState;
    bool bValid;
    unordered_map<string, vector<Goldilocks::Element>> dbWrite;
    vector<string> dbDelete;
    TxSubState() : previousSubState(0), bValid(false)
    {
        dbWrite.reserve(128);
        dbDelete.reserve(128);
    };
};

class TxPersistenceState
{
public:
    string oldStateRoot;
    string newStateRoot;
    uint64_t currentSubState;
    vector<TxSubState> subState;
    TxPersistenceState() : currentSubState(0)
    {
        subState.reserve(128);
    };
};

class TxState
{
public:
    TxPersistenceState persistence[PERSISTENCE_SIZE];
};

class BatchState
{
public:
    string oldStateRoot;
    string currentStateRoot;
    uint64_t currentTx;
    vector<TxState> txState;
    unordered_map<string, vector<Goldilocks::Element>> dbWrite;
#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    TimeMetricStorage timeMetricStorage;
#endif
    BatchState() : currentTx(0)
    {
        txState.reserve(32);
        dbWrite.reserve(1024);
    };
};

class StateManager
{
private:
    unordered_map<string, BatchState> state;
    Config config;
    pthread_mutex_t mutex; // Mutex to protect the multi write queues

public:
    StateManager ()
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
    zkresult flush (const string &batchUUID, const string &newStateRoot, const Persistence persistence, Database &db, uint64_t &flushId, uint64_t &lastSentFlushId);
    void print (bool bDbContent = false);

    // Lock/Unlock
    void Lock(void) { pthread_mutex_lock(&mutex); };
    void Unlock(void) { pthread_mutex_unlock(&mutex); };
};

extern StateManager stateManager;

#endif