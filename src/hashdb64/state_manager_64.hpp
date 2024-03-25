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
#include "poseidon_goldilocks.hpp"
#include "smt_get_result.hpp"
#include "smt_set_result.hpp"
#include "key_value_tree.hpp"
#include "level_tree_key_value.hpp"

using namespace std;

class TxSubState64
{
public:
    string oldStateRoot;
    string newStateRoot;
    uint64_t previousSubState;
    bool bValid;
    unordered_map<string, mpz_class> dbWrite;
    TxSubState64() : previousSubState(0), bValid(false)
    {
        dbWrite.reserve(1);
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
    string newStateRoot;
    uint64_t currentTx;
    vector<TxState64> txState;
#ifndef USE_NEW_KVTREE
    KeyValueTree keyValueTree;
#else
    KVTree keyValueTree;
#endif
#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    TimeMetricStorage timeMetricStorage;
#endif
    BatchState64() : currentTx(0)
    {
        txState.reserve(32);
 #ifdef USE_NEW_KVTREE
        keyValueTree.postConstruct(4);
#endif       
    };
};

class StateManager64
{
private:
    Goldilocks &fr;
    PoseidonGoldilocks &poseidon;
    unordered_map<string, BatchState64> state;
    vector<string> stateOrder;
    Config config;
    pthread_mutex_t mutex; // Mutex to protect the multi write queues
    uint64_t lastVirtualStateRoot;
    Goldilocks::Element lastConsolidatedStateRoot[4];
    string lastConsolidatedStateRootString;
public:
    StateManager64(Goldilocks &fr, PoseidonGoldilocks &poseidon) : fr(fr), poseidon(poseidon), lastVirtualStateRoot(0)
    {        
        // Init mutex
        pthread_mutex_init(&mutex, NULL);
        lastConsolidatedStateRoot[0] = fr.zero();
        lastConsolidatedStateRoot[1] = fr.zero();
        lastConsolidatedStateRoot[2] = fr.zero();
        lastConsolidatedStateRoot[3] = fr.zero();
    };
private:
    zkresult setStateRoot (const string &batchUUID, uint64_t tx, const string &stateRoot, const bool bIsOldStateRoot, const Persistence persistence);
    zkresult purgeBatch (BatchState64 &batchState, const string &newStateRoot);
    zkresult purgeTxPersistence (TxPersistenceState64 &txPersistence, const Config &config);

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
    zkresult write (const string &batchUUID, uint64_t tx, const string &_key, const mpz_class &value, const Persistence persistence, uint64_t &level);
    zkresult read (const string &batchUUID, const string &_key, mpz_class &value, uint64_t &level, DatabaseMap *dbReadLog);
    zkresult semiFlush (const string &batchUUID, const string &newStateRoot, const Persistence persistence);
    zkresult purge (const string &batchUUID, const string &_newStateRoot, const Persistence _persistence, Database64 &db);
    zkresult consolidateState (const string &newStateRoot, const Persistence _persistence, string & consolidatedStateRoot, Database64 &db, uint64_t &flushId, uint64_t &lastSentFlushId);
    zkresult cancelBatch (const string &batchUUID);

    void print (bool bDbContent = false);
    void getVirtualStateRoot (Goldilocks::Element (&newStateRoot)[4], string &newStateRootString);
    bool isVirtualStateRoot (const string &stateRoot);

    // Lock/Unlock
    void Lock(void) { pthread_mutex_lock(&mutex); };
    void Unlock(void) { pthread_mutex_unlock(&mutex); };

    zkresult set(const string &batchUUID, uint64_t tx, Database64 &db, const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const Persistence persistence, SmtSetResult &result, DatabaseMap *dbReadLog = NULL);
    zkresult get(const string &batchUUID, Database64 &db, const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], SmtGetResult &result, DatabaseMap *dbReadLog = NULL);

};

extern StateManager64 stateManager64;

#endif