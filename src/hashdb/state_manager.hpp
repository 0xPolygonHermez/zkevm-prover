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
    TxSubState() : previousSubState(0), bValid(false) {};
};

class TxPersistenceState
{
public:
    string oldStateRoot;
    string newStateRoot;
    uint64_t currentSubState;
    vector<TxSubState> subState;
    TxPersistenceState() : currentSubState(0) {};
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
    BatchState() : currentTx(0) {};
};

class StateManager
{
private:
    unordered_map<string, BatchState> state;
public:
    StateManager () {;};
private:
    zkresult setStateRoot (const string &batchUUID, uint64_t tx, const string &stateRoot, bool bIsOldStateRoot, const Persistence persistence);
public:
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
    zkresult flush (const string &batchUUID, Database &db, uint64_t &flushId, uint64_t &lastSentFlushId);
    void print (bool bDbContent = false);
};

extern StateManager stateManager;

#endif