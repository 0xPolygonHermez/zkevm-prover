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
    unordered_map<string, vector<Goldilocks::Element>> dbWriteNodes;
    vector<string> dbDeleteNodes;
    TxSubState() : previousSubState(0), bValid(false)
    {
        dbWriteNodes.reserve(128);
        dbDeleteNodes.reserve(128);
    };
};

class TxPersistenceState
{
public:
    string oldStateRoot;
    string newStateRoot;
    uint64_t currentSubState;
    vector<TxSubState> subState;
    unordered_map<string, vector<uint8_t>> dbWritePrograms;
    TxPersistenceState() : currentSubState(0)
    {
        subState.reserve(128);
        dbWritePrograms.reserve(8);
    };
};

class TxState
{
public:
    TxPersistenceState persistence[PERSISTENCE_SIZE];
};

class BlockState
{
public:
    string oldStateRoot;
    string currentStateRoot;
    uint64_t currentTx;
    vector<TxState> txState;
    BlockState() : currentTx(0)
    {
        txState.reserve(32);
    };
};

class BatchState
{
public:
    string oldStateRoot;
    string currentStateRoot;
    uint64_t currentBlock;
    vector<BlockState> blockState;
    unordered_map<string, vector<Goldilocks::Element>> dbWriteNodes;
    unordered_map<string, vector<uint8_t>> dbWritePrograms;
#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    TimeMetricStorage timeMetricStorage;
#endif
    BatchState() : currentBlock(0)
    {
        blockState.reserve(32);
        dbWriteNodes.reserve(1024);
        dbWritePrograms.reserve(8);
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
    zkresult setStateRoot (const string &batchUUID, uint64_t block, uint64_t tx, const string &stateRoot, bool bIsOldStateRoot, const Persistence persistence);
public:
    void init (const Config &_config)
    {
        config = _config;
    }
    zkresult setOldStateRoot (const string &batchUUID, uint64_t block, uint64_t tx, const string &stateRoot, const Persistence persistence)
    {
        return setStateRoot(batchUUID, block, tx, stateRoot, true, persistence);
    }
    zkresult setNewStateRoot (const string &batchUUID, uint64_t block, uint64_t tx, const string &stateRoot, const Persistence persistence)
    {
        return setStateRoot(batchUUID, block, tx, stateRoot, false, persistence);
    }
    zkresult writeNode (const string &batchUUID, uint64_t block, uint64_t tx, const string &_key, const vector<Goldilocks::Element> &value, const Persistence persistence);
    zkresult deleteNode (const string &batchUUID, uint64_t block, uint64_t tx, const string &_key, const Persistence persistence);
    zkresult readNode (const string &batchUUID, const string &_key, vector<Goldilocks::Element> &value, DatabaseMap *dbReadLog);
    zkresult writeProgram (const string &batchUUID, uint64_t block, uint64_t tx, const string &_key, const vector<uint8_t> &value, const Persistence persistence);
    zkresult readProgram (const string &batchUUID, const string &_key, vector<uint8_t> &value, DatabaseMap *dbReadLog);
    zkresult finishTx (const string &batchUUID, const string &newStateRoot, const Persistence persistence);
    zkresult startBlock (const string &batchUUID, const string &oldStateRoot, const Persistence persistence);
    zkresult finishBlock (const string &batchUUID, const string &newStateRoot, const Persistence persistence);
    zkresult flush (const string &batchUUID, const string &newStateRoot, const Persistence persistence, Database &db, uint64_t &flushId, uint64_t &lastSentFlushId);
    zkresult cancelBatch (const string &batchUUID);
    void print (bool bDbContent = false);
private:
    // Lock/Unlock
    void Lock(void) { pthread_mutex_lock(&mutex); };
    void Unlock(void) { pthread_mutex_unlock(&mutex); };
};

extern StateManager stateManager;

#endif