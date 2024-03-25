#include "state_manager_64.hpp"
#include "zkassert.hpp"
#include "zklog.hpp"
#include "scalar.hpp"
#include "timer.hpp"
#include "persistence.hpp"
#include "definitions.hpp"
#include "zkglobals.hpp"

Goldilocks frSM64;
PoseidonGoldilocks poseidonSM64;
StateManager64 stateManager64(frSM64, poseidonSM64);

zkresult StateManager64::setStateRoot(const string &batchUUID, uint64_t tx, const string &_stateRoot, const bool bIsOldStateRoot, const Persistence persistence)
{
#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    struct timeval t;
    gettimeofday(&t, NULL);
#endif

    zkresult zkr;

    // Normalize state root format
    string stateRoot = NormalizeToNFormat(_stateRoot, 64);
    stateRoot = stringToLower(stateRoot);

    // Check persistence range
    if (persistence >= PERSISTENCE_SIZE)
    {
        zklog.error("StateManager64::setStateRoot() invalid persistence batchUUID=" + batchUUID + " tx=" + to_string(tx) + " stateRoot=" + stateRoot + " bIsOldStateRoot=" + to_string(bIsOldStateRoot) + " persistence=" + persistence2string(persistence));
        return ZKR_STATE_MANAGER;
    }

#ifdef LOG_STATE_MANAGER
    zklog.info("StateManager64::setStateRoot() batchUUID=" + batchUUID + " tx=" + to_string(tx) + " stateRoot=" + stateRoot + " bIsOldStateRoot=" + to_string(bIsOldStateRoot) + " persistence=" + persistence2string(persistence));
#endif

    Lock();

    unordered_map<string, BatchState64>::iterator it;

    // Find batch state for this uuid, or create it if it does not exist
    it = state.find(batchUUID);
    if (it == state.end())
    {
        if (!bIsOldStateRoot)
        {
            zklog.error("StateManager64::setStateRoot() called with bIsOldStateRoot=false, but batchUUID=" + batchUUID + " does not previously exist");
            Unlock();
            return ZKR_STATE_MANAGER;
        }
        BatchState64 batchState;
        batchState.oldStateRoot = stateRoot;
        state[batchUUID] = batchState;
        it = state.find(batchUUID);
        zkassert(it != state.end());

        // Copy the previous batch state keyValueTree
        for (int64_t i = stateOrder.size() - 1; i >= 0; i--)
        {
            unordered_map<string, BatchState64>::iterator previt;
            previt = state.find(stateOrder[i]);
            zkassert(previt != state.end());
            if (previt->second.currentStateRoot == stateRoot)
            {
                it->second.keyValueTree = previt->second.keyValueTree;
                break;
            }
        }

        stateOrder.emplace_back(batchUUID);
    }
    BatchState64 &batchState = it->second;

    // Set the current state root
    batchState.currentStateRoot = stateRoot;

    // Create tx states, if needed
    if (tx >= batchState.txState.size())
    {
        // If this is the first state of a new tx, check that it is the old state root
        if (!bIsOldStateRoot)
        {
            zklog.error("StateManager64::setStateRoot() called with bIsOldStateRoot=false, but tx=" + to_string(tx) + " does not previously exist");
            Unlock();
            return ZKR_STATE_MANAGER;
        }

        // Calculate the number of tx slots to create
        uint64_t txsToCreate = tx - batchState.txState.size() + 1;

        // Create TX state to insert
        TxState64 txState;

        // Insert TX state
        for (uint64_t i = 0; i < txsToCreate; i++)
        {
            batchState.txState.emplace_back(txState);
        }

        // Set current TX
        batchState.currentTx = tx;
    }

    // Get a reference to the tx state
    TxState64 &txState = batchState.txState[tx];

    // Get the current sub-state list size
    uint64_t currentSubStateSize = txState.persistence[persistence].subState.size();

    // In case it is an old state root, we need to create a new sub-state, and check that everything makes sense
    if (bIsOldStateRoot)
    {
        // If this is the first sub-state of the tx state, record the tx old state root
        if (currentSubStateSize == 0)
        {
            // Check current sub-state
            if (txState.persistence[persistence].currentSubState != 0)
            {
                zklog.error("StateManager64::setStateRoot() currentSubState=" + to_string(txState.persistence[persistence].currentSubState) + "!=0 batchUUID=" + batchUUID + " tx=" + to_string(tx) + " stateRoot=" + stateRoot + " bIsOldStateRoot=" + to_string(bIsOldStateRoot) + " persistence=" + persistence2string(persistence));
                Unlock();
                return ZKR_STATE_MANAGER;
            }

            // Record the old state root
            txState.persistence[persistence].oldStateRoot = stateRoot;
        }

        // If it is not the first sub-state, it must have been called with the previous new state root
        else
        {
            // Check current sub-state
            if (txState.persistence[persistence].currentSubState >= currentSubStateSize)
            {
                zklog.error("StateManager64::setStateRoot() currentSubState=" + to_string(txState.persistence[persistence].currentSubState) + " > currentSubStateSize=" + to_string(currentSubStateSize) + " batchUUID=" + batchUUID + " tx=" + to_string(tx) + " stateRoot=" + stateRoot + " bIsOldStateRoot=" + to_string(bIsOldStateRoot) + " persistence=" + persistence2string(persistence));
                Unlock();
                return ZKR_STATE_MANAGER;
            }

            // Check that new state root is empty
            if (txState.persistence[persistence].subState[currentSubStateSize - 1].newStateRoot.empty())
            {
                zklog.error("StateManager64::setStateRoot() oldStateRoot found previous newStateRoot empty");
                Unlock();
                return ZKR_STATE_MANAGER;
            }
        }

        // Create TX sub-state
        TxSubState64 txSubState;
        txSubState.oldStateRoot = stateRoot;
        txSubState.previousSubState = txState.persistence[persistence].currentSubState;

        // Find the previous state, if it exists
        for (uint64_t i = 0; i < currentSubStateSize; i++)
        {
            if (txState.persistence[persistence].subState[i].newStateRoot == stateRoot)
            {
                // Delete the keys of the rest of sub states
                for (uint64_t j = i+1; j < currentSubStateSize; j++)
                {
                    unordered_map<string, mpz_class> &dbWrite = txState.persistence[persistence].subState[j].dbWrite;
                    unordered_map<string, mpz_class>::const_iterator it;
                    for (it = dbWrite.begin(); it != dbWrite.end(); it++)
                    {
                        Goldilocks::Element key_[4];
                        string2fea(frSM64,it->first, key_);
                        zkr = batchState.keyValueTree.extract(key_, it->second);
                        if (zkr != ZKR_SUCCESS)
                        {
                            zklog.error("StateManager64::setStateRoot() failed calling batchState.keyValueTree.extract()");
                            Unlock();
                            return ZKR_STATE_MANAGER;
                        }
                    }
                }
            }
        }

        // Insert it
        txState.persistence[persistence].subState.emplace_back(txSubState);

        // Record the current state
        txState.persistence[persistence].currentSubState = txState.persistence[persistence].subState.size() - 1;
    }

    // If it is a new state root, we need to complete the current sub-state
    else
    {
        if (txState.persistence[persistence].currentSubState >= currentSubStateSize)
        {
            zklog.error("StateManager64::setStateRoot() currentSubState=" + to_string(txState.persistence[persistence].currentSubState) + " > currentSubStateSize=" + to_string(currentSubStateSize) + " batchUUID=" + batchUUID + " tx=" + to_string(tx) + " stateRoot=" + stateRoot + " bIsOldStateRoot=" + to_string(bIsOldStateRoot) + " persistence=" + persistence2string(persistence));
            Unlock();
            return ZKR_STATE_MANAGER;
        }

        // Check that the new state root is empty
        if (!txState.persistence[persistence].subState[txState.persistence[persistence].currentSubState].newStateRoot.empty())
        {
            zklog.error("StateManager64::setStateRoot() found nesStateRoot busy");
            Unlock();
            return ZKR_STATE_MANAGER;
        }

        // Record the new state root in the tx sub-state, and in the tx state
        txState.persistence[persistence].subState[txState.persistence[persistence].currentSubState].newStateRoot = stateRoot;
        txState.persistence[persistence].newStateRoot = stateRoot;
    }

#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    batchState.timeMetricStorage.add("setStateRoot", TimeDiff(t));
#endif

    Unlock();

    return ZKR_SUCCESS;
}

zkresult StateManager64::write (const string &batchUUID, uint64_t tx, const string &key, const mpz_class &value, const Persistence persistence, uint64_t &level)
{
#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    struct timeval t;
    gettimeofday(&t, NULL);
#endif

#ifdef LOG_STATE_MANAGER
    zklog.info("StateManager64::write() batchUUID=" + batchUUID + " tx=" + to_string(tx) + " key=" + key + " persistence=" + persistence2string(persistence));
#endif

    // Check persistence range
    if (persistence >= PERSISTENCE_SIZE)
    {
        zklog.error("StateManager64::write() wrong persistence batchUUID=" + batchUUID + " tx=" + to_string(tx) + " key=" + key + " persistence=" + persistence2string(persistence));
        return ZKR_STATE_MANAGER;
    }

    Lock();

    // Find batch state for this uuid
    unordered_map<string, BatchState64>::iterator it;
    it = state.find(batchUUID);
    if (it == state.end())
    {
        zklog.error("StateManager64::write() found no batch state for batch UUID=" + batchUUID);
        Unlock();
        return ZKR_STATE_MANAGER;
    }
    BatchState64 &batchState = it->second;

    // Check tx range
    if (tx >= batchState.txState.size())
    {
        zklog.error("StateManager64::write() got tx=" + to_string(tx) + " bigger than txState size=" + to_string(it->second.txState.size()));
        Unlock();
        return ZKR_STATE_MANAGER;
    }

    // Create TxState, if not existing
    /*if (tx == batchState.txState.size())
    {
        TxState64 aux;
        aux.persistence[persistence].oldStateRoot = it->second.currentStateRoot;
        it->second.txState.emplace_back(aux);
    }*/
    TxState64 &txState = batchState.txState[tx];

    // Create TxSubState, if not existing
    if (txState.persistence[persistence].subState.size() == 0)
    {
        TxSubState64 subState;
        subState.previousSubState = 0;
        subState.oldStateRoot = batchState.currentStateRoot;
        txState.persistence[persistence].subState.emplace_back(subState);
        txState.persistence[persistence].currentSubState = 0;
    }

    // Add to sub-state
    txState.persistence[persistence].subState[txState.persistence[persistence].currentSubState].dbWrite[key] = value;

    // Add to common write pool to speed up read
    Goldilocks::Element key_[4];
    string2fea(frSM64, key, key_);
    batchState.keyValueTree.write(key_, value, level);
    // TODO: Get level from DB and return the max

#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    batchState.timeMetricStorage.add("write", TimeDiff(t));
#endif

    Unlock();

    return ZKR_SUCCESS;
}

zkresult StateManager64::read(const string &batchUUID, const string &key, mpz_class &value, uint64_t &level, DatabaseMap *dbReadLog)
{
    struct timeval t;
    gettimeofday(&t, NULL);

    Lock();

    // Find batch state for this uuid
    unordered_map<string, BatchState64>::iterator it;
    it = state.find(batchUUID);
    if (it == state.end())
    {
        // zklog.error("StateManager64::read() found no batch state for batch UUID=" + batchUUID);
        Unlock();
        return ZKR_DB_KEY_NOT_FOUND;
    }
    BatchState64 &batchState = it->second;

    // Search in the common key-value tree
    Goldilocks::Element key_[4];
    string2fea(frSM64, key, key_);
    zkresult zkr = batchState.keyValueTree.read(key_, value, level);
    if (zkr == ZKR_SUCCESS)
    {
        // Add to the read log
        if (dbReadLog != NULL)
            dbReadLog->add(key, value.get_str(16), true, TimeDiff(t));

#ifdef LOG_STATE_MANAGER
        zklog.info("StateManager64::read() batchUUID=" + batchUUID + " key=" + key);
#endif

#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
        batchState.timeMetricStorage.add("read success", TimeDiff(t));
#endif
        Unlock();

        return ZKR_SUCCESS;
    }

#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    batchState.timeMetricStorage.add("read not found", TimeDiff(t));
#endif

    Unlock();

    return ZKR_DB_KEY_NOT_FOUND;
}

bool IsInvalid(TxSubState64 &txSubState)
{
    return !txSubState.bValid;
}

zkresult StateManager64::semiFlush(const string &batchUUID, const string &_stateRoot, const Persistence persistence)
{
#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    struct timeval t;
    gettimeofday(&t, NULL);
#endif

    // Normalize state root format
    string stateRoot = NormalizeToNFormat(_stateRoot, 64);
    stateRoot = stringToLower(stateRoot);

    // Check persistence range
    if (persistence >= PERSISTENCE_SIZE)
    {
        zklog.error("StateManager64::semiFlush() invalid persistence batchUUID=" + batchUUID + " stateRoot=" + stateRoot + " persistence=" + persistence2string(persistence));
        return ZKR_STATE_MANAGER;
    }

#ifdef LOG_STATE_MANAGER
    zklog.info("StateManager64::semiFlush() batchUUID=" + batchUUID + " stateRoot=" + stateRoot + " persistence=" + persistence2string(persistence));
#endif

    Lock();

    unordered_map<string, BatchState64>::iterator it;

    // Find batch state for this uuid
    it = state.find(batchUUID);
    if (it == state.end())
    {
        zklog.warning("StateManager64::semiFlush() found no batch state for batch UUID=" + batchUUID + "; normal if no SMT activity happened");

#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
        // timeMetricStorage.add("semiFlush UUID not found", TimeDiff(t));
        // timeMetricStorage.print("State Manager calls");
#endif
        Unlock();
        return ZKR_SUCCESS;
    }
    BatchState64 &batchState = it->second;

    // Check currentTx range
    if (batchState.currentTx >= batchState.txState.size())
    {
        zklog.error("StateManager64::semiFlush() found batchState.currentTx=" + to_string(batchState.currentTx) + " >= batchState.txState.size=" + to_string(batchState.txState.size()) + " batchUUID=" + batchUUID + " stateRoot=" + stateRoot + " persistence=" + persistence2string(persistence));
        Unlock();
        return ZKR_STATE_MANAGER;
    }

    // Get a reference to the tx state
    TxState64 &txState = batchState.txState[batchState.currentTx];
    TxPersistenceState64 &txPersistenceState = txState.persistence[persistence];

    if (txPersistenceState.newStateRoot == stateRoot)
    {
        // This is the expected case
    }
    else if (txPersistenceState.oldStateRoot == stateRoot)
    {
        if (config.stateManagerPurge)
        {
            // The TX ended up with the same state root as the beginning, so we can delete all data
            txPersistenceState.subState.clear();
            txPersistenceState.newStateRoot = stateRoot;
            txPersistenceState.currentSubState = 0;
        }
    }
    else
    {
        if (config.stateManagerPurge)
        {
            // Search for the point at which we reach this state, and delete the rest
            bool bFound = false;
            uint64_t i = 0;
            uint64_t subStateSize = txPersistenceState.subState.size();
            for (i = 0; i < subStateSize; i++)
            {
                if (!bFound && (txPersistenceState.subState[i].oldStateRoot == stateRoot))
                {
                    bFound = true;
                    break;
                }
            }
            if (bFound)
            {
                txPersistenceState.newStateRoot = stateRoot;
                txPersistenceState.currentSubState = (i == 0) ? 0 : i - 1;
                for (; i < subStateSize; i++)
                {
                    txPersistenceState.subState.pop_back();
                }
            }
        }
    }

#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    batchState.timeMetricStorage.add("semi flush", TimeDiff(t));
#endif

    Unlock();

    return ZKR_SUCCESS;
}

zkresult StateManager64::purgeBatch (BatchState64 &batchState, const string &newStateRoot)
{
    zkassert(config.stateManagerPurgeTxs == true);
    zkassert(newStateRoot.size() > 0);

    // For all txs, delete the ones that are not part of the final state root chain

    // Start searching from the last TX, and when a tx with the same new state root is found,
    // delete the ones after this one (typically fruit of an out-of-counters condition)
    int64_t tx = -1;
    for (tx = batchState.txState.size() - 1; tx >= 0; tx--)
    {
        if (batchState.txState[tx].persistence[PERSISTENCE_DATABASE].newStateRoot == newStateRoot)
        {
            break;
        }
    }

    // Check if we found the last valid TX
    if (tx < 0)
    {
        zklog.error("StateManager::purgeBatch() called with newStateRoot=" + newStateRoot + " but could not find it");
        return ZKR_STATE_MANAGER;
    }

    // Delete the TXs after this one
    while ((int64_t)batchState.txState.size() > (tx + 1))
    {
        batchState.txState.pop_back();
    }

    return ZKR_SUCCESS;
}

zkresult StateManager64::purgeTxPersistence (TxPersistenceState64 &txPersistence, const Config &config)
{
    // Check that current sub-state newStateRoot matches the TX one,
    // i.e. that setNewStateRoot() was called before flush()
    if (txPersistence.subState[txPersistence.currentSubState].newStateRoot != txPersistence.newStateRoot)
    {
        zklog.error("StateManager64::purgeTxPersistence() found inconsistent new state roots: currentSubState=" + to_string(txPersistence.currentSubState) +
                    " substate.newStateRoot=" + txPersistence.subState[txPersistence.currentSubState].newStateRoot +
                    " txPersistence.newStateRoot=" + txPersistence.newStateRoot);
        return ZKR_STATE_MANAGER;
    }

    uint64_t currentSubState = txPersistence.currentSubState;

    // TODO: check that currentSubState == size(), or simply don't use it

    // Search for the chain of sub-states that end with this new state root
    // Mark the ones that are part of the chain with bValid = true
    while (true)
    {
        // Mark it as a valid sub-state
        txPersistence.subState[currentSubState].bValid = true;

        // If we went back to the first sub-state, we are done, as long as the old state roots match
        if (currentSubState == 0)
        {
            // Check that both old state roots match
            if (txPersistence.subState[currentSubState].oldStateRoot != txPersistence.oldStateRoot)
            {
                zklog.error("StateManager64::purgeTxPersistence() found inconsistent old state roots: txState.oldStateRoot=" + txPersistence.oldStateRoot +
                            " currentSubState=" + to_string(txPersistence.currentSubState) +
                            " substate.oldStateRoot=" + txPersistence.subState[currentSubState].oldStateRoot);
                return ZKR_STATE_MANAGER;
            }

            // We are done
            break;
        }

        // If the previous sub-state ended the same way this sub-state started, then it is part of the chain
        uint64_t previousSubState = txPersistence.subState[currentSubState].previousSubState;
        if (txPersistence.subState[previousSubState].newStateRoot == txPersistence.subState[currentSubState].oldStateRoot)
        {
            currentSubState = previousSubState;
            continue;
        }

        // Otherwise, we resumed the chain from a previous state, maybe due to a revert
        // Search for the previous state that ends the same way this sub-state starts
        uint64_t i = 0;
        for (; i < currentSubState; i++)
        {
            if (txPersistence.subState[i].newStateRoot == txPersistence.subState[currentSubState].oldStateRoot)
            {
                previousSubState = i;
                break;
            }
        }

        // Check that we actually found it
        if (i == currentSubState)
        {
            zklog.error("StateManager64::purgeTxPersistence() could not find previous tx sub-state: txState.oldStateRoot=" + txPersistence.oldStateRoot +
                        " currentSubState=" + to_string(txPersistence.currentSubState) +
                        " substate.oldStateRoot=" + txPersistence.subState[currentSubState].oldStateRoot);
            return ZKR_STATE_MANAGER;
        }

        // Iterate with the previous state
        currentSubState = previousSubState;
    }

    // Delete invalid TX sub-states, i.e. the ones with bValid = false
    if (config.stateManagerPurge)
    {
        // Delete all substates that are not valid or that did not change the state root (i.e. SMT set update in which new value was equal to old value)
        for (int64_t i = txPersistence.subState.size() - 1; i >= 0; i--)
        {
            if (!txPersistence.subState[i].bValid ||
                (txPersistence.subState[i].oldStateRoot == txPersistence.subState[i].newStateRoot))
            {
                txPersistence.subState.erase(txPersistence.subState.begin() + i);
            }
        }
    }

    return ZKR_SUCCESS;
}

zkresult StateManager64::purge (const string &batchUUID, const string &_newStateRoot, const Persistence persistence, Database64 &db)
{
#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    struct timeval t;
    gettimeofday(&t, NULL);
#endif

#ifdef LOG_STATE_MANAGER
    zklog.info("StateManager64::purge() batchUUID=" + batchUUID);
#endif

    // For every TX, track backwards from newStateRoot to oldStateRoot, marking sub-states as valid

    Lock();

    //print(false);

    // Format the new state root
    string newStateRoot = NormalizeToNFormat(_newStateRoot, 64);
    newStateRoot = stringToLower(newStateRoot);

    // Find batch state for this uuid
    // If it does not exist, we call db.flush() directly
    unordered_map<string, BatchState64>::iterator it;
    it = state.find(batchUUID);
    if (it == state.end())
    {
        zklog.warning("StateManager64::purge() found no batch state for batch UUID=" + batchUUID + "; normal if no SMT activity happened");
        Unlock();
        return ZKR_STATE_MANAGER;
    }
    BatchState64 &batchState = it->second;

    // Purge the batch from unneeded TXs
    zkresult zkr;
    if (config.stateManagerPurgeTxs && (_newStateRoot.size() > 0) && (persistence == PERSISTENCE_DATABASE))
    {
        zkr = purgeBatch(batchState, newStateRoot);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("StateManager64::purge() failed calling purgeBatch() zkr=" + zkresult2string(zkr));
        }
    }

    // For all tx sub-states, purge the data to write:
    // - Delete all temporary data
    // - Mark sub-states that are part of the chain
    // - Delete the rest of sub-states

    // For all transactions
    for (uint64_t tx = 0; tx < batchState.txState.size(); tx++)
    {
        // Get a reference to the current transaction state
        TxState64 &txState = batchState.txState[tx];

        // For all persistences
        for (uint64_t persistence = 0; persistence < PERSISTENCE_SIZE; persistence++)
        {
            // If there's no data, then no purge is required
            if (txState.persistence[persistence].subState.size() == 0)
            {
                continue;
            }

            // All data with temporary persistence can be deleted at the end of a batch
            if (persistence == PERSISTENCE_TEMPORARY)
            {
                txState.persistence[persistence].subState.clear();
                txState.persistence[persistence].currentSubState = 0;
                continue;
            }

            zkr = purgeTxPersistence(txState.persistence[persistence], config);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("StateManager64::purge() failed calling purgeTxPersistence() zkr=" + zkresult2string(zkr) +
                            " batchUUID=" + batchUUID +
                            " tx=" + to_string(tx));
#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
                batchState.timeMetricStorage.add("purgeTxPersistence failed", TimeDiff(t));
                batchState.timeMetricStorage.print("State Manager calls");
#endif
                Unlock();
                return zkr;

            }

        } // For all persistences

    } // For all transactions

    // Take note of the batch new state root
    batchState.newStateRoot = newStateRoot;

    Unlock();

    return ZKR_SUCCESS;
}

zkresult StateManager64::consolidateState(const string &_virtualStateRoot, const Persistence persistence, string & consolidatedStateRoot, Database64 &db, uint64_t &flushId, uint64_t &lastSentFlushId)
{
#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    struct timeval t;
    gettimeofday(&t, NULL);
#endif

    TimerStart(STATE_MANAGER_CONSOLIDATE_STATE);

    // Format the new state root
    string virtualStateRoot = NormalizeToNFormat(_virtualStateRoot, 64);

    if (!isVirtualStateRoot(virtualStateRoot))
    {
        zklog.error("StateManager64::consolidateState() called with non-virtual virtualStateRoot=" + virtualStateRoot);
        return ZKR_STATE_MANAGER;
    }

#ifdef LOG_STATE_MANAGER
    zklog.info("StateManager64::consolidateState() newStateRoot=" + newStateRoot);
#endif

    // For every TX, track backwards from newStateRoot to oldStateRoot, marking sub-states as valid

    Lock();

    //print(false);

    zkresult zkr;

    // Find the batch that ends up with the virtual state root
    uint64_t virtualStatePosition = 0;
    string oldStateRoot;
    unordered_map<string, BatchState64>::iterator it;
    for (virtualStatePosition = 0; virtualStatePosition < stateOrder.size(); virtualStatePosition++)
    {
        it = state.find(stateOrder[virtualStatePosition]);
        if (it == state.end())
        {
            zklog.error("StateManager64::consolidateState() found unmatching state i=" + to_string(virtualStatePosition) + " batchUUI=" + stateOrder[virtualStatePosition]);
            TimerStopAndLog(STATE_MANAGER_CONSOLIDATE_STATE);        
            Unlock();
            return ZKR_STATE_MANAGER;
        }
        if (it->second.newStateRoot == virtualStateRoot)
        {
            oldStateRoot = it->second.oldStateRoot;
            break;
        }
    }
    if (virtualStatePosition == stateOrder.size())
    {
        zklog.error("StateManager64::consolidateState() could not find a matching virtual state=" + virtualStateRoot);
        TimerStopAndLog(STATE_MANAGER_CONSOLIDATE_STATE);        
        Unlock();
        return ZKR_STATE_MANAGER;
    }

    // Determine the chain of state roots (i.e. previous batches) that leads to newStateRoot
    // Delete previous batches that are not part of the chain

    for (int64_t i = (int64_t)virtualStatePosition - 1; i >= 0; i--)
    {
        // Find the batch state corresponding to this position
        it = state.find(stateOrder[i]);
        if (it == state.end())
        {
            zklog.error("StateManager64::consolidateState() could not find a matching virtual state=" + stateOrder[i]);
            TimerStopAndLog(STATE_MANAGER_CONSOLIDATE_STATE);        
            Unlock();
            return ZKR_STATE_MANAGER;
        }

        // If this state is part of the state chain, continue searching
        if (oldStateRoot == it->second.newStateRoot)
        {
            oldStateRoot = it->second.oldStateRoot;
            continue;
        }

        // Otherwise, we must delete this state, and shift the virtual state position
        state.erase(it);
        stateOrder.erase(stateOrder.begin() + i);
        virtualStatePosition--;
    }

    // Calculate the real state roots, and write them to the database

    // We will store here the latest tx new state root
    Goldilocks::Element newRoot[4];

    // For all batches in the state root chain

    for (uint64_t batch = 0; batch <= virtualStatePosition; batch++)
    {
        // Find the batch state corresponding to this position
        it = state.find(stateOrder[batch]);
        if (it == state.end())
        {
            zklog.error("StateManager64::consolidateState() could not find a matching virtual state=" + stateOrder[batch]);
            TimerStopAndLog(STATE_MANAGER_CONSOLIDATE_STATE);        
            Unlock();
            return ZKR_STATE_MANAGER;
        }

        BatchState64 &batchState = state[stateOrder[batch]];

        if (batch > 0)
        {
            string newStateRootString = fea2string(fr, newRoot);
            batchState.oldStateRoot = newStateRootString;
            if (batchState.txState.size() > 0)
            {
                batchState.txState[0].persistence[persistence].oldStateRoot = newStateRootString;
            }
        }

        // For all transactions
        for (uint64_t tx = 0; tx < batchState.txState.size(); tx++)
        {
            // Get a reference to the current transaction state
            TxState64 &txState = batchState.txState[tx];

            // Get old state root for this tx
            Goldilocks::Element oldRoot[4];
            string2fea(fr, txState.persistence[persistence].oldStateRoot, oldRoot);

            // Get the key-values for this tx
            vector<KeyValue> keyValues;
            for (uint64_t i=0; i<txState.persistence[persistence].subState.size(); i++)
            {
                unordered_map<string, mpz_class> &dbWrite = txState.persistence[persistence].subState[i].dbWrite;
                unordered_map<string, mpz_class>::const_iterator it;
                for (it = dbWrite.begin(); it != dbWrite.end(); it++)
                {
                    KeyValue keyValue;
                    string2fea(fr, it->first, keyValue.key);
                    keyValue.value = it->second;
                    keyValues.emplace_back(keyValue);
                }
            }

            // Call WriteTree and get the new state root
            zkr = db.WriteTree(oldRoot, keyValues, newRoot, persistence == PERSISTENCE_DATABASE ? true : false);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("StateManager64::consolidateState() failed calling WriteTree zkr=" + zkresult2string(zkr) +
                            " tx=" + to_string(tx) +
                            " txState.oldStateRoot=" + txState.persistence[persistence].oldStateRoot);
    #ifdef LOG_TIME_STATISTICS_STATE_MANAGER
                batchState.timeMetricStorage.add("consolidateState WriteTree failed", TimeDiff(t));
                batchState.timeMetricStorage.print("State Manager calls");
    #endif
                Unlock();
                return ZKR_STATE_MANAGER;
            }

            // Save the real state root of this tx
            string newRootString = fea2string(fr, newRoot);
            txState.persistence[persistence].newStateRoot = newRootString;

            // Save the old state root of the next tx, if any
            if (tx < batchState.txState.size() - 1)
            {
                batchState.txState[tx+1].persistence[persistence].oldStateRoot = newRootString;
            }
            // If this is the last tx, then save the new state root of the batch
            else
            {
                batchState.newStateRoot = newRootString;
            }

        } // For all transactions

#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
        batchState.timeMetricStorage.add("consolidateState success", TimeDiff(t));
        batchState.timeMetricStorage.print("State Manager calls");
#endif

    } // For all batches

    // Return the consolidated state root
    consolidatedStateRoot = fea2string(fr, newRoot);

    // Copy it to use it when reading a KV that is not in the StateManager,
    // and we have to read it from database using a consolidated state root
    lastConsolidatedStateRoot[0] = newRoot[0];
    lastConsolidatedStateRoot[1] = newRoot[1];
    lastConsolidatedStateRoot[2] = newRoot[2];
    lastConsolidatedStateRoot[3] = newRoot[3];
    lastConsolidatedStateRootString = consolidatedStateRoot;

    // Call flush and get the flush ID

    zkr = db.flush(flushId, lastSentFlushId);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("StateManager64::consolidateState() failed calling db.flush() result=" + zkresult2string(zkr));
    }

    // Delete all batches of the virtual state chain

    for (int64_t i = (int64_t)virtualStatePosition; i >= 0; i--)
    {
        // Find the batch state corresponding to this position
        it = state.find(stateOrder[i]);
        if (it == state.end())
        {
            zklog.error("StateManager64::consolidateState() could not find a matching virtual state=" + stateOrder[i]);
            TimerStopAndLog(STATE_MANAGER_CONSOLIDATE_STATE);        
            Unlock();
            return ZKR_STATE_MANAGER;
        }

        // Delete this state, and shift the virtual state position
        state.erase(it);
        stateOrder.erase(stateOrder.begin() + i);
    }

    Unlock();

    TimerStopAndLog(STATE_MANAGER_CONSOLIDATE_STATE);

    return zkr;
}

zkresult StateManager64::cancelBatch (const string &batchUUID)
{
#ifdef LOG_TIME_STATISTICS_STATE_MANAGER
    struct timeval t;
    gettimeofday(&t, NULL);
#endif

    TimerStart(STATE_MANAGER_CANCEL_BATCH);

#ifdef LOG_STATE_MANAGER
    zklog.info("StateManager64::cancelBatch() batchUUID=" + batchUUID);
#endif

    Lock();

    // Find batch state for this uuid
    unordered_map<string, BatchState64>::iterator it;
    it = state.find(batchUUID);
    if (it == state.end())
    {
        zklog.warning("StateManager64::cancelBatch() found no batch state for batch UUID=" + batchUUID + "; normal if no SMT activity happened");
        TimerStopAndLog(STATE_MANAGER_CANCEL_BATCH);
        Unlock();
        return ZKR_STATE_MANAGER;
    }
    state.erase(it);

    Unlock();

    TimerStopAndLog(STATE_MANAGER_CANCEL_BATCH);

    return ZKR_SUCCESS;
}

void StateManager64::print(bool bDbContent)
{
    uint64_t totalDbWrites[PERSISTENCE_SIZE] = {0, 0, 0};
    zklog.info("StateManager64::print():");
    zklog.info("state.size=" + to_string(state.size()));
    unordered_map<string, BatchState64>::const_iterator stateIt;
    uint64_t batchStateCounter = 0;
    for (stateIt = state.begin(); stateIt != state.end(); stateIt++)
    {
        const BatchState64 &batchState = stateIt->second;
        zklog.info("  batchState=" + to_string(batchStateCounter));
        batchStateCounter++;
        zklog.info("  BatchUUID=" + stateIt->first);
        zklog.info("  oldStateRoot=" + batchState.oldStateRoot);
        zklog.info("  currentStateRoot=" + batchState.currentStateRoot);
        zklog.info("  currentTx=" + to_string(batchState.currentTx));

        for (uint64_t tx = 0; tx < batchState.txState.size(); tx++)
        {

            zklog.info("    TX=" + to_string(tx));
            const TxState64 &txState = batchState.txState[tx];

            for (uint64_t persistence = 0; persistence < PERSISTENCE_SIZE; persistence++)
            {
                zklog.info("      persistence=" + to_string(persistence) + "=" + persistence2string((Persistence)persistence));
                zklog.info("        oldStateRoot=" + txState.persistence[persistence].oldStateRoot);
                zklog.info("        newStateRoot=" + txState.persistence[persistence].newStateRoot);
                zklog.info("        currentSubState=" + to_string(txState.persistence[persistence].currentSubState));
                zklog.info("        txSubState.size=" + to_string(txState.persistence[persistence].subState.size()));
                for (uint64_t i = 0; i < txState.persistence[persistence].subState.size(); i++)
                {
                    const TxSubState64 &txSubState = txState.persistence[persistence].subState[i];
                    zklog.info("          txSubState=" + to_string(i));
                    zklog.info("            oldStateRoot=" + txSubState.oldStateRoot);
                    zklog.info("            newStateRoot=" + txSubState.newStateRoot);
                    zklog.info("            valid=" + to_string(txSubState.bValid));
                    zklog.info("            previousSubState=" + to_string(txSubState.previousSubState));
                    zklog.info("            dbWrite.size=" + to_string(txSubState.dbWrite.size()));

                    totalDbWrites[persistence] += txSubState.dbWrite.size();
                    if (bDbContent)
                    {
                        unordered_map<string, mpz_class>::const_iterator dbIt;
                        for (dbIt = txSubState.dbWrite.begin(); dbIt != txSubState.dbWrite.end(); dbIt++)
                        {
                            zklog.info("              key=" + dbIt->first + " value=" + dbIt->second.get_str(16));
                        }
                    }
                }
            }
        }
    }

    uint64_t totalWrites = 0;
    for (uint64_t persistence = 0; persistence < PERSISTENCE_SIZE; persistence++)
    {
        zklog.info("total db writes[" + persistence2string((Persistence)persistence) + "]=" + to_string(totalDbWrites[persistence]));
        totalWrites += totalDbWrites[persistence];
    }
    zklog.info("total writes=" + to_string(totalWrites));

    zklog.info("stateOrder.size=" + to_string(stateOrder.size()));
    for (uint64_t i=0; i<stateOrder.size(); i++)
    {
        zklog.info("  uuid=" + stateOrder[i]);
    }
}

void StateManager64::getVirtualStateRoot(Goldilocks::Element (&newStateRoot)[4], string &newStateRootString)
{
    lastVirtualStateRoot++;
    newStateRoot[0] = fr.fromU64(lastVirtualStateRoot);
    newStateRoot[1] = fr.zero();
    newStateRoot[2] = fr.zero();
    newStateRoot[3] = fr.zero();
    newStateRootString = fea2string(fr, newStateRoot);
}

bool StateManager64::isVirtualStateRoot(const string &stateRoot)
{
    Goldilocks::Element root[4];
    string2fea(fr, stateRoot, root);
    return fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]);
}

zkresult StateManager64::set (const string &batchUUID, uint64_t tx, Database64 &db, const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const Persistence persistence, SmtSetResult &result, DatabaseMap *dbReadLog)
{
#ifdef LOG_STATE_MANAGER
    zklog.info("StateManager64::set() called with oldRoot=" + fea2string(fr,oldRoot) + " key=" + fea2string(fr,key) + " value=" + value.get_str(16) + " persistent=" + to_string(persistent));
#endif

    zkresult zkr;

    bool bUseStateManager = config.stateManager && (batchUUID.size() > 0);

    if (bUseStateManager)
    {
        // Set the old state root
        string oldRootString;
        oldRootString = fea2string(fr, oldRoot);
        stateManager64.setOldStateRoot(batchUUID, tx, oldRootString, persistence);

        // Write the key-value pair
        string hashString = fea2string(fr, key);
        uint64_t level=0;
        uint64_t stateManagerLevel;
        uint64_t databaseLevel;
        zkr = stateManager64.write(batchUUID, tx, hashString, value, persistence, stateManagerLevel);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("StateManager64::set() failed calling stateManager.write() key=" + hashString + " result=" + to_string(zkr) + "=" + zkresult2string(zkr));
        }
        else
        {
            zkresult dbzkr = db.readLevel(key, databaseLevel);
            if (dbzkr != ZKR_SUCCESS)
            {
                zklog.error("StateManager64::set() failed calling db.readLevel() key=" + hashString + " result=" + to_string(dbzkr) + "=" + zkresult2string(dbzkr));
                level = 128;
            }
            else
            {
                level = zkmax(stateManagerLevel, databaseLevel);
            }

        }

        // Get a new state root
        Goldilocks::Element newRoot[4]; // TODO: Get a new state root
        string newRootString;
        stateManager64.getVirtualStateRoot(newRoot, newRootString);

        // Set the new sttae root
        stateManager64.setNewStateRoot(batchUUID, tx, newRootString, persistence);

        result.newRoot[0] = newRoot[0];
        result.newRoot[1] = newRoot[1];
        result.newRoot[2] = newRoot[2];
        result.newRoot[3] = newRoot[3];
        result.proofHashCounter = level + 2;
    }
    else
    {
        // TODO: implementation
    }

    return ZKR_SUCCESS;
}

zkresult StateManager64::get (const string &batchUUID, Database64 &db, const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], SmtGetResult &result, DatabaseMap *dbReadLog)
{
#ifdef LOG_STATE_MANAGER
    zklog.info("StateManager64::get() called with root=" + fea2string(fr,root) + " and key=" + fea2string(fr,key));
#endif

    bool bUseStateManager = config.stateManager && (batchUUID.size() > 0);

    string keyString = fea2string(fr, key);
    mpz_class value;
    zkresult zkr = ZKR_UNSPECIFIED;
    uint64_t level = 0;
    if (bUseStateManager)
    {
        uint64_t stateManagerLevel;
        uint64_t databaseLevel;
        zkr = stateManager64.read(batchUUID, keyString, value, stateManagerLevel, dbReadLog);
        if (zkr == ZKR_SUCCESS)
        {
            zkresult dbzkr = db.readLevel(key, databaseLevel);
            if (dbzkr != ZKR_SUCCESS)
            {
                zklog.error("StateManager64::get() failed calling db.readLevel() result=" + zkresult2string(dbzkr));
                level = 128;
            }
            else
            {
                level = zkmax(stateManagerLevel, databaseLevel);
            }
        }
    }
    if (zkr != ZKR_SUCCESS)
    {
        zkr = db.readKV(lastConsolidatedStateRoot, key, value, level, dbReadLog);
    }
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("StateManager64::get() db.read error=" + zkresult2string(zkr) + " root=" + fea2string(fr, lastConsolidatedStateRoot) + " key=" + fea2string(fr, key));
        return zkr;
    }
    
    result.value = value;
    result.proofHashCounter = level + 2;

    return ZKR_SUCCESS;
}