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
    //unordered_map<string, vector<Goldilocks::Element>> dbWriteNodes;
    //vector<string> dbDeleteNodes;
    TxSubState64() : previousSubState(0), bValid(false)
    {
        dbWrite.reserve(1);
        //dbWriteNodes.reserve(128);
        //dbDeleteNodes.reserve(128);
    };
};

class TxPersistenceState64
{
public:
    string oldStateRoot;
    string newStateRoot;
    uint64_t currentSubState;
    vector<TxSubState64> subState;
    unordered_map<string, vector<uint8_t>> dbPrograms;
    TxPersistenceState64() : currentSubState(0)
    {
        subState.reserve(128);
        dbPrograms.reserve(8);
    };
};

class TxState64
{
public:
    TxPersistenceState64 persistence[PERSISTENCE_SIZE];
};

class BlockState64
{
public:
    string oldStateRoot;
    string currentStateRoot;
    uint64_t currentTx;
    vector<TxState64> txState;
    BlockState64() : currentTx(0)
    {
        txState.reserve(32);
    };
};

class BatchState64
{
public:
    string oldStateRoot;
    string currentStateRoot;
    uint64_t currentBlock;
    vector<BlockState64> blockState;
#ifndef USE_NEW_KVTREE
    KeyValueTree keyValueTree;
#else
    KVTree keyValueTree;
    unordered_map<string, vector<uint8_t>> dbPrograms;
#endif
    //unordered_map<string, vector<Goldilocks::Element>> dbWriteNodes;
    //unordered_map<string, vector<uint8_t>> dbWritePrograms;
#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    TimeMetricStorage timeMetricStorage;
#endif
    BatchState64() : currentBlock(0)
    {
        blockState.reserve(32);
#ifdef USE_NEW_KVTREE
        keyValueTree.postConstruct(4);
#endif 
        //dbWriteNodes.reserve(1024);
        //dbWritePrograms.reserve(8);
    };
};

class StateManager64
{
private:
    unordered_map<string, BatchState64> state;
    vector<string> stateOrder; // Store batch UUIDs in order of creation
    pthread_mutex_t mutex; // Mutex to protect the multi write queues
    uint64_t lastVirtualStateRoot;
    Goldilocks::Element lastConsolidatedStateRoot[4];
    string lastConsolidatedStateRootString;

public:
    StateManager64 ()
    {        
        // Init mutex
        pthread_mutex_init(&mutex, NULL);
        lastConsolidatedStateRoot[0] = fr.zero();
        lastConsolidatedStateRoot[1] = fr.zero();
        lastConsolidatedStateRoot[2] = fr.zero();
        lastConsolidatedStateRoot[3] = fr.zero();
    };
private:
    zkresult setStateRoot (const string &batchUUID, uint64_t block, uint64_t tx, const string &stateRoot, bool bIsOldStateRoot, const Persistence persistence);
public:
    void init (void)
    {
    }
    zkresult setOldStateRoot (const string &batchUUID, uint64_t block, uint64_t tx, const string &stateRoot, const Persistence persistence)
    {
        return setStateRoot(batchUUID, block, tx, stateRoot, true, persistence);
    }
    zkresult setNewStateRoot (const string &batchUUID, uint64_t block, uint64_t tx, const string &stateRoot, const Persistence persistence)
    {
        return setStateRoot(batchUUID, block, tx, stateRoot, false, persistence);
    }
    zkresult write (const string &batchUUID, uint64_t block, uint64_t tx, const string &key, const mpz_class &value, const Persistence persistence, uint64_t &level);
    zkresult read (const string &batchUUID, const string &key, mpz_class &value, uint64_t &level, DatabaseMap *dbReadLog);
    zkresult writeProgram (const string &batchUUID, uint64_t block, uint64_t tx, const string &key, const vector<uint8_t> &value, const Persistence persistence);
    zkresult readProgram (const string &batchUUID, const string &key, vector<uint8_t> &value, DatabaseMap *dbReadLog);
    zkresult finishTx (const string &batchUUID, const string &newStateRoot, const Persistence persistence);
    zkresult startBlock (const string &batchUUID, const string &oldStateRoot, const Persistence persistence);
    zkresult finishBlock (const string &batchUUID, const string &newStateRoot, const Persistence persistence);
private:    
    zkresult purgeBatch (const string &batchUUID, BatchState64 &batchState, const Persistence persistence);
public:
    zkresult cancelBatch (const string &batchUUID);


    void print (bool bDbContent = false);
    void getVirtualStateRoot (Goldilocks::Element (&newStateRoot)[4], string &newStateRootString);
    static bool isVirtualStateRoot (const string &stateRoot);
private:
    // Lock/Unlock
    void Lock(void) { pthread_mutex_lock(&mutex); };
    void Unlock(void) { pthread_mutex_unlock(&mutex); };

public:
    // Virtual state
    zkresult consolidateState (const string &newStateRoot, const Persistence _persistence, string & consolidatedStateRoot, Database64 &db, uint64_t &flushId, uint64_t &lastSentFlushId);

    void inline setLastConsolidatedStateRoot(const Goldilocks::Element (&root)[4])
    {
        lastConsolidatedStateRoot[0] = root[0];
        lastConsolidatedStateRoot[1] = root[1];
        lastConsolidatedStateRoot[2] = root[2];
        lastConsolidatedStateRoot[3] = root[3];
        lastConsolidatedStateRootString = fea2string(fr, lastConsolidatedStateRoot);
    }

    // SMT
    zkresult set(const string &batchUUID, uint64_t block, uint64_t tx, Database64 &db, const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const Persistence persistence, SmtSetResult &result, DatabaseMap *dbReadLog = NULL);
    zkresult get(const string &batchUUID, Database64 &db, const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], SmtGetResult &result, DatabaseMap *dbReadLog = NULL);

};

extern StateManager64 stateManager64;

#endif