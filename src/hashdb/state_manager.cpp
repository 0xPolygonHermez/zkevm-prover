#include "state_manager.hpp"
#include "zkassert.hpp"
#include "zklog.hpp"
#include "scalar.hpp"
#include "timer.hpp"
#include "persistence.hpp"

StateManager stateManager;

zkresult StateManager::setStateRoot (const string &batchUUID, uint64_t tx, const string &_stateRoot, bool bIsOldStateRoot, const Persistence persistence)
{
    zkassert(persistence < PERSISTENCE_SIZE);

    // Normalize state root format
    string stateRoot = NormalizeToNFormat(_stateRoot, 64);
    stateRoot = stringToLower(stateRoot);

#ifdef LOG_STATE_MANAGER
    zklog.info("StateManager::setStateRoot() batchUUID=" + batchUUID + " tx=" + to_string(tx) + " stateRoot=" + stateRoot + " bIsOldStateRoot=" + to_string(bIsOldStateRoot));
#endif

    unordered_map<string, BatchState>::iterator it;

    // Find batch state for this uuid, or create it if not existing
    it = state.find(batchUUID);
    if (it == state.end())
    {
        if (!bIsOldStateRoot)
        {
            zklog.error("StateManager::setStateRoot() called with bIsOldStateRoot=false, but batchUUID=" + batchUUID + " does not previously exist");
            return ZKR_UNSPECIFIED;
        }
        BatchState batchState;
        batchState.oldStateRoot = stateRoot;
        state[batchUUID] = batchState;
    }
    it = state.find(batchUUID);
    BatchState &batchState = it->second;
    batchState.currentStateRoot = stateRoot;

    // Check tx range
    if (tx > batchState.txState.size()) // TODO: Should we fill it up with missing txs?
    {
        zklog.error("StateManager::setStateRoot() invalid tx=" + to_string(tx) + " txState.size=" + to_string(batchState.txState.size()) + " batchUUID=" + batchUUID);
        return ZKR_UNSPECIFIED;
    }

    // Find tx state for this tx, or create it if not existing
    if (tx == batchState.txState.size())
    {
        if (!bIsOldStateRoot)
        {
            zklog.error("StateManager::setStateRoot() called with bIsOldStateRoot=false, but tx=" + to_string(tx) + " does not previously exist");
            return ZKR_UNSPECIFIED;
        }

        // Create TX state
        TxState txState;
        
        // Insert TX state
        batchState.txState.push_back(txState);
        batchState.currentTx = tx;
    }

    TxState &txState = batchState.txState[tx];
    
    if (bIsOldStateRoot)
    {
        uint64_t currentSubStateSize = txState.persistence[persistence].subState.size();

        if ( currentSubStateSize == 0)
        {
            // Record the old state root
            txState.persistence[persistence].oldStateRoot = stateRoot;
        }
        else
        {
            if (txState.persistence[persistence].subState[currentSubStateSize-1].newStateRoot.size() == 0)
            {
                zklog.error("StateManager::setStateRoot() oldStateRoot found newStateRoot empty");
            }
        }
        // Create TX sub-state
        TxSubState txSubState;
        txSubState.oldStateRoot = stateRoot;
        txSubState.previousSubState = txState.persistence[persistence].currentSubState;

        // Insert it
        txState.persistence[persistence].subState.push_back(txSubState);
        txState.persistence[persistence].currentSubState = txState.persistence[persistence].subState.size() - 1;
    }
    else
    {
        if (txState.persistence[persistence].subState[txState.persistence[persistence].currentSubState].newStateRoot.size() != 0)
        {
            zklog.error("StateManager::setStateRoot() found nesStateRoot busy");
            return ZKR_UNSPECIFIED;
        }

        // Record the new state root
        txState.persistence[persistence].subState[txState.persistence[persistence].currentSubState].newStateRoot = stateRoot;
        txState.persistence[persistence].newStateRoot = stateRoot;
    }

    return ZKR_SUCCESS;

}

zkresult StateManager::write (const string &batchUUID, uint64_t tx, const string &_key, const vector<Goldilocks::Element> &value, const Persistence persistence)
{
    zkassert(persistence < PERSISTENCE_SIZE);

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

#ifdef LOG_STATE_MANAGER
    zklog.info("StateManager::write() batchUUID=" + batchUUID + " tx=" + to_string(tx) + " key=" + key + " persistence=" + persistence2string(persistence));
#endif

    // Find batch state for this uuid
    unordered_map<string, BatchState>::iterator it;
    it = state.find(batchUUID);
    if (it == state.end())
    {
        zklog.error("StateManager::write() found no batch state for batch UUID=" + batchUUID);
        return ZKR_UNSPECIFIED;
    }
    BatchState &batchState = it->second;

    // Check tx range
    if (tx > batchState.txState.size())
    {
        zklog.error("StateManager::write() got tx=" + to_string(tx) + " bigger than txState size=" + to_string(it->second.txState.size()));
        return ZKR_UNSPECIFIED;
    }

    // Create TxState, if not existing
    if (tx == batchState.txState.size())
    {
        TxState aux;
        aux.persistence[persistence].oldStateRoot = it->second.currentStateRoot;
        it->second.txState.push_back(aux);
    }
    TxState &txState = batchState.txState[tx];

    // Create TxSubState, if not existing
    if (txState.persistence[persistence].subState.size() == 0)
    {
        TxSubState subState;
        subState.previousSubState = 0;
        subState.oldStateRoot = batchState.currentStateRoot;
        txState.persistence[persistence].subState.push_back(subState);
        txState.persistence[persistence].currentSubState = 0;
    }

    txState.persistence[persistence].subState[txState.persistence[persistence].currentSubState].dbWrite[key] = value;

    return ZKR_SUCCESS;
}

zkresult StateManager::deleteNode (const string &batchUUID, uint64_t tx, const string &_key, const Persistence persistence)
{
    zkassert(persistence < PERSISTENCE_SIZE);

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

#ifdef LOG_STATE_MANAGER
    zklog.info("StateManager::deleteNode() batchUUID=" + batchUUID + " tx=" + to_string(tx) + " key=" + key + " persistence=" + persistence2string(persistence));
#endif

    unordered_map<string, BatchState>::iterator it;

    // Find batch state for this batch uuid
    it = state.find(batchUUID);
    if (it == state.end())
    {
        zklog.error("StateManager::deleteNode() found no batch state for batch UUID=" + batchUUID);
        return ZKR_UNSPECIFIED;
    }
    BatchState &batchState = it->second;

    // Check tx range
    if (tx >= batchState.txState.size())
    {
        zklog.error("StateManager::deleteNode() got tx=" + to_string(tx) + " bigger than txState size=" + to_string(it->second.txState.size()));
        return ZKR_UNSPECIFIED;
    }

    // Find TX state for this tx
    TxState &txState = batchState.txState[tx];
    
    // Find TX current sub-state
    if (txState.persistence[persistence].subState.size() == 0)
    {
        return ZKR_SUCCESS;
    }
    TxSubState &txSubState = txState.persistence[persistence].subState[txState.persistence[persistence].currentSubState];

    // Delete this key in the surrent TX sub-state
    unordered_map<string, vector<Goldilocks::Element>>::iterator dbIt;
    dbIt = txSubState.dbWrite.find(key);
    if (dbIt != txSubState.dbWrite.end())
    {
        txSubState.dbWrite.erase(dbIt);
        zklog.info("StateManager::deleteNode() batchUUID=" + batchUUID + " tx=" + to_string(tx) + " key=" + key);
    }
    txSubState.dbDelete.push_back(key); // TODO: delete key

    return ZKR_SUCCESS;
}

zkresult StateManager::read(const string &batchUUID, const string &_key, vector<Goldilocks::Element> &value, DatabaseMap *dbReadLog)
{
    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

    struct timeval t;
    if (dbReadLog != NULL) gettimeofday(&t, NULL);

    // Find batch state for this uuid
    unordered_map<string, BatchState>::iterator it;
    it = state.find(batchUUID);
    if (it == state.end())
    {
        //zklog.error("StateManager::read() found no batch state for batch UUID=" + batchUUID);
        return ZKR_DB_KEY_NOT_FOUND;
    }
    BatchState &batchState = it->second;

    // For all txs, search for this key
    for (uint64_t tx=0; tx<batchState.txState.size(); tx++)
    {
        TxState &txState = batchState.txState[tx];
        for (uint64_t persistence=0; persistence<PERSISTENCE_SIZE; persistence++)
        {
            for (uint64_t i=0; i<txState.persistence[persistence].subState.size(); i++)
            {
                TxSubState &txSubState = txState.persistence[persistence].subState[i];
                unordered_map<string, vector<Goldilocks::Element>>::iterator dbIt;
                dbIt = txSubState.dbWrite.find(key);
                if (dbIt != txSubState.dbWrite.end())
                {
                    value = dbIt->second;
                        
                    // Add to the read log
                    if (dbReadLog != NULL) dbReadLog->add(key, value, true, TimeDiff(t));

#ifdef LOG_STATE_MANAGER
                    zklog.info("StateManager::read() batchUUID=" + batchUUID + " key=" + key);
#endif
                    return ZKR_SUCCESS;
                }
            }
        }
    }

    return ZKR_DB_KEY_NOT_FOUND;
}

bool IsInvalid(TxSubState &txSubState)
{
    return !txSubState.bValid;
}

zkresult StateManager::flush (const string &batchUUID, Database &db, uint64_t &flushId, uint64_t &lastSentFlushId)
{
    // For every TX, track backwards from newStateRoot to oldStateRoot, marking sub-states as valid
    //print(false);
    //print(true);

    TimerStart(STATE_MANAGER_FLUSH);

#ifdef LOG_STATE_MANAGER
    zklog.info("StateManager::flush() batchUUID=" + batchUUID);
#endif

    zkresult zkr;

    // Find batch state for this uuid
    unordered_map<string, BatchState>::iterator it;
    it = state.find(batchUUID);
    if (it == state.end())
    {
        zklog.error("StateManager::flush() found no batch state for batch UUID=" + batchUUID);
        return ZKR_DB_KEY_NOT_FOUND;
    }
    BatchState &batchState = it->second;

    // For all txs
    for (uint64_t tx=0; tx<batchState.txState.size(); tx++)
    {
        for (uint64_t persistence = 0; persistence < PERSISTENCE_SIZE; persistence++)
        {
            TxState &txState = batchState.txState[tx];

            // Temporary data can be deleted at the end of a batch
            if (persistence == PERSISTENCE_TEMPORARY)
            {
                txState.persistence[persistence].subState.clear();
                continue;
            }

            // If there's no data, there's nothing to do
            if (txState.persistence[persistence].subState.size() == 0)
            {
                continue;
            }

            // Check that current sub-state newStateRoot matches the TX one
            if (txState.persistence[persistence].subState[txState.persistence[persistence].currentSubState].newStateRoot != txState.persistence[persistence].newStateRoot)
            {
                zklog.error("StateManager::flush() found inconsistent new state roots: batchUUID=" + batchUUID +
                    " tx=" + to_string(tx) + " txState.newStateRoot=" + txState.persistence[persistence].newStateRoot +
                    " currentSubState=" + to_string(txState.persistence[persistence].currentSubState) +
                    " substate.newStateRoot=" + txState.persistence[persistence].subState[txState.persistence[persistence].currentSubState].newStateRoot);
                return ZKR_UNSPECIFIED;
            }

            uint64_t currentSubState = txState.persistence[persistence].currentSubState;
            while (true)
            {
                txState.persistence[persistence].subState[currentSubState].bValid = true;
                if (currentSubState == 0)
                {
                    if (txState.persistence[persistence].subState[currentSubState].oldStateRoot != txState.persistence[persistence].oldStateRoot)
                    {
                        zklog.error("StateManager::flush() found inconsistent old state roots: batchUUID=" + batchUUID +
                            " tx=" + to_string(tx) + " txState.oldStateRoot=" + txState.persistence[persistence].oldStateRoot +
                            " currentSubState=" + to_string(txState.persistence[persistence].currentSubState) +
                            " substate.oldStateRoot=" + txState.persistence[persistence].subState[currentSubState].oldStateRoot);
                        return ZKR_UNSPECIFIED;
                    }
                    break;
                }
                uint64_t previousSubState = txState.persistence[persistence].subState[currentSubState].previousSubState;
                if (txState.persistence[persistence].subState[previousSubState].newStateRoot == txState.persistence[persistence].subState[currentSubState].oldStateRoot)
                {
                    currentSubState = previousSubState;
                    continue;
                }

                // Search for the previous state
                uint64_t i=0;
                for (; i<currentSubState; i++)
                {
                    if (txState.persistence[persistence].subState[i].newStateRoot == txState.persistence[persistence].subState[currentSubState].oldStateRoot)
                    {
                        previousSubState = i;
                        break;
                    }
                }
                if (i == currentSubState)
                {
                    zklog.error("StateManager::flush() could not find previous tx sub-state: batchUUID=" + batchUUID +
                        " tx=" + to_string(tx) +
                        " txState.oldStateRoot=" + txState.persistence[persistence].oldStateRoot +
                        " currentSubState=" + to_string(txState.persistence[persistence].currentSubState) +
                        " substate.oldStateRoot=" + txState.persistence[persistence].subState[currentSubState].oldStateRoot);
                    return ZKR_UNSPECIFIED;
                }
                currentSubState = previousSubState;
            }

            // Delete invalid TX sub-states
            for (int64_t i = txState.persistence[persistence].subState.size()-1; i>=0; i--)
            {
                if (!txState.persistence[persistence].subState[i].bValid)
                {
                    txState.persistence[persistence].subState.erase(txState.persistence[persistence].subState.begin() + i);
                }
            }

            // Delete unneeded hashes: delete only hashes written previously to the deletion time

            // For all sub-states
            for (uint64_t ss = 0; ss < txState.persistence[persistence].subState.size(); ss++)
            {
                // For all keys to delete
                for (uint64_t k = 0; k < txState.persistence[persistence].subState[ss].dbDelete.size(); k++)
                {
                    // For all previouse sub-states, previous to the current sub-state
                    for (uint64_t pss = 0; pss < ss; pss++)
                    {
                        txState.persistence[persistence].subState[pss].dbWrite.erase(txState.persistence[persistence].subState[ss].dbDelete[k]);
                    }
                }
                txState.persistence[persistence].subState[ss].dbDelete.clear();
            }

            // Save data to database

            // For all sub-states
            for (uint64_t ss = 0; ss < txState.persistence[persistence].subState.size(); ss++)
            {
                // For all keys to write
                unordered_map<string, vector<Goldilocks::Element>>::const_iterator writeIt;
                for ( writeIt = txState.persistence[persistence].subState[ss].dbWrite.begin();
                      writeIt != txState.persistence[persistence].subState[ss].dbWrite.end();
                      writeIt++ )
                {
                    zkr = db.write(writeIt->first, writeIt->second, persistence == PERSISTENCE_DATABASE ? 1 : 0);
                    if (zkr != ZKR_SUCCESS)
                    {
                        zklog.error("StateManager::flush() failed calling db.write() result=" + zkresult2string(zkr));
                        //return zkr;
                    }
                }
            }
        }
    }
    
    // Delete this batch UUID state
    state.erase(it);

    zkr = db.flush(flushId, lastSentFlushId);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("StateManager::flush() failed calling db.flush() result=" + zkresult2string(zkr));
    }


    TimerStopAndLog(STATE_MANAGER_FLUSH);

    //print(false);
    //print(true);

    return zkr;
}

void StateManager::print (bool bDbContent)
{
    uint64_t totalDbWrites[PERSISTENCE_SIZE] = {0, 0, 0};
    uint64_t totalDbDeletes[PERSISTENCE_SIZE] = {0, 0, 0};
    zklog.info("StateManager::print():");
    zklog.info("state.size=" + to_string(state.size()));
    unordered_map<string, BatchState>::const_iterator stateIt;
    uint64_t batchStateCounter = 0;
    for (stateIt = state.begin(); stateIt != state.end(); stateIt++)
    {
        const BatchState &batchState = stateIt->second;
        zklog.info("  batchState=" + to_string(batchStateCounter));
        batchStateCounter++;
        zklog.info("  BatchUUID=" + stateIt->first);
        zklog.info("  oldStateRoot=" + batchState.oldStateRoot);
        zklog.info("  currentStateRoot=" + batchState.currentStateRoot);
        zklog.info("  currentTx=" + to_string(batchState.currentTx));

        for (uint64_t tx=0; tx<batchState.txState.size(); tx++)
        {

            zklog.info("    TX=" + to_string(tx));
            const TxState &txState = batchState.txState[tx];

            for (uint64_t persistence = 0; persistence < PERSISTENCE_SIZE; persistence++)
            {
                zklog.info("      persistence=" + to_string(persistence) + "=" + persistence2string((Persistence)persistence));
                zklog.info("        oldStateRoot=" + txState.persistence[persistence].oldStateRoot);
                zklog.info("        newStateRoot=" + txState.persistence[persistence].newStateRoot);
                zklog.info("        currentSubState=" + to_string(txState.persistence[persistence].currentSubState));
                zklog.info("        txSubState.size=" + to_string(txState.persistence[persistence].subState.size()));
                for (uint64_t i=0; i<txState.persistence[persistence].subState.size(); i++)
                {
                    const TxSubState &txSubState = txState.persistence[persistence].subState[i];
                    zklog.info("          txSubState=" + to_string(i));
                    zklog.info("            oldStateRoot=" + txSubState.oldStateRoot);
                    zklog.info("            newStateRoot=" + txSubState.newStateRoot);
                    zklog.info("            valid=" + to_string(txSubState.bValid));
                    zklog.info("            previousSubState=" + to_string(txSubState.previousSubState));
                    zklog.info("            dbWrite.size=" + to_string(txSubState.dbWrite.size()));

                    totalDbWrites[persistence] += txSubState.dbWrite.size();
                    if (bDbContent)
                    {
                        unordered_map<string, vector<Goldilocks::Element>>::const_iterator dbIt;
                        for (dbIt = txSubState.dbWrite.begin(); dbIt != txSubState.dbWrite.end(); dbIt++)
                        {
                            zklog.info("              " + dbIt->first);
                        }
                    }
                    zklog.info("            dbDelete.size=" + to_string(txSubState.dbDelete.size()));
                    totalDbDeletes[persistence] += txSubState.dbDelete.size();
                    if (bDbContent)
                    {
                        for (uint64_t j=0; j<txSubState.dbDelete.size(); j++)
                        {
                            zklog.info("              " + txSubState.dbDelete[j]);
                        }
                    }
                }
            }
        }
    }

    uint64_t totalWrites = 0;
    uint64_t totalDeletes = 0;
    for (uint64_t persistence=0; persistence<PERSISTENCE_SIZE; persistence++)
    {
        zklog.info("total db writes[" + persistence2string((Persistence)persistence) + "]=" + to_string(totalDbWrites[persistence]));
        totalWrites += totalDbWrites[persistence];
        zklog.info("total db deletes[" + persistence2string((Persistence)persistence) + "]=" + to_string(totalDbDeletes[persistence]));
        totalDeletes += totalDbDeletes[persistence];
    }
    zklog.info("total writes=" + to_string(totalWrites));
    zklog.info("total deletes=" + to_string(totalDeletes));
}