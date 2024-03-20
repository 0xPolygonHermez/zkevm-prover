#include "hashdb.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sys/time.h>
#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "scalar.hpp"
#include "zkresult.hpp"
#include "database_map.hpp"
#include "state_manager.hpp"
#include "state_manager_64.hpp"
#include "key_utils.hpp"

HashDB::HashDB(Goldilocks &fr, const Config &config) : fr(fr), config(config), db(fr, config), db64(fr, config), smt(fr)
{
    if (config.hashDB64)
    {
        db64.init();
    }
    else
    {
        db.init();
    }
}

HashDB::~HashDB()
{    
#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.print("HashDB");
#endif    
}

zkresult HashDB::getLatestStateRoot (Goldilocks::Element (&stateRoot)[4])
{
#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    zkresult zkr;
    
    if (config.hashDB64)
    {
        zkr = db64.getLatestStateRoot(stateRoot);
    }
    else
    {
        zklog.error("HashDB::getLatestStateRoot() not suported with option config.hashDB64=false");
        return ZKR_DB_ERROR;
    }
    return zkr;
}
zkresult HashDB::set (const string &batchUUID, uint64_t block, uint64_t tx, const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const Persistence persistence, Goldilocks::Element (&newRoot)[4], SmtSetResult *result, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    SmtSetResult *r;
    if (result == NULL) r = new SmtSetResult;
    else r = result;

    zkresult zkr;

    if (config.hashDB64)
    {
        zkr = stateManager64.set(batchUUID, block, tx, db64, oldRoot, key, value, persistence, *r, dbReadLog);
    }
    else
    {
        zkr = smt.set(batchUUID, block, tx, db, oldRoot, key, value, persistence, *r, dbReadLog);
    }
    for (int i = 0; i < 4; i++) newRoot[i] = r->newRoot[i];

    if (result == NULL) delete r;

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("set", TimeDiff(t));
#endif

    return zkr;
}

zkresult HashDB::get (const string &batchUUID, const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    SmtGetResult *r;
    if (result == NULL) r = new SmtGetResult;
    else r = result;

    zkresult zkr;
    
    if (config.hashDB64)
    {
        zkr = stateManager64.get(batchUUID, db64, root, key, *r, dbReadLog);
    }
    else
    {
        zkr = smt.get(batchUUID, db, root, key, *r, dbReadLog);
    }

    value = r->value;

    if (result == NULL) delete r;

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("get", TimeDiff(t));
#endif

    return zkr;
}

zkresult HashDB::setProgram (const string &batchUUID, uint64_t block, uint64_t tx, const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const Persistence persistence)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    // Get the key as an hexa string
    string keyString = fea2string(fr, key);

    // Call writeProgram()
    zkresult zkr;
    if (config.hashDB64)
    {
        zkr = stateManager64.writeProgram(batchUUID, block, tx, keyString, data, persistence);
    }
    else
    {
        zkr = stateManager.writeProgram(batchUUID, block, tx, keyString, data, persistence);
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("setProgram", TimeDiff(t));
#endif

    return zkr;
}

zkresult HashDB::getProgram (const string &batchUUID, const Goldilocks::Element (&key)[4], vector<uint8_t> &data, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    // Get the key as an hexa string
    string keyString = fea2string(fr, key);

    // Clear data
    data.clear();

    // Call readProgram
    zkresult zkr;
    if (config.hashDB64)
    {
        zkr = stateManager64.readProgram(batchUUID, keyString, data, dbReadLog);    
        if (zkr != ZKR_SUCCESS)
        {
            zkr = db64.getProgram(keyString, data, dbReadLog);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("HashDB::getProgram() failed result=" + zkresult2string(zkr) + " key=" + keyString);
            }
        }    
    }
    else
    {
        zkr = stateManager.readProgram(batchUUID, keyString, data, dbReadLog);
        if (zkr != ZKR_SUCCESS)
        {
            zkr = db.getProgram(keyString, data, dbReadLog);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("HashDB::getProgram() failed result=" + zkresult2string(zkr) + " key=" + keyString);
            }
        }
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("getProgram", TimeDiff(t));
#endif

    return zkr;
}

zkresult hashValue2keyValue (const DatabaseMap::MTMap &input, const Goldilocks::Element (&stateRoot)[4], vector<KeyValue> &keyValues, const uint64_t level, vector<uint64_t> &bits)
{
    zkresult zkr;

    string root = fea2string(fr, stateRoot);
    
    DatabaseMap::MTMap::const_iterator it;
    it = input.find(root);
    if (it == input.end())
    {
        zklog.error("hashValue2keyValue() failed searching for root=" + root);
        return ZKR_DB_KEY_NOT_FOUND;
    }

    const vector<Goldilocks::Element> &value = it->second;
    if (value.size() != 12)
    {
        zklog.error("hashValue2keyValue() found value.size=" + to_string(value.size()));
        return ZKR_DB_ERROR;
    }

    // If capacity is {0,0,0,0} then this is an intermediate node
    if (fr.isZero(value[8]) && fr.isZero(value[9]) && fr.isZero(value[10]) && fr.isZero(value[11]))
    {
        Goldilocks::Element valueFea[4];

        // If left hash is not {0,0,0,0} then iterate
        valueFea[0] = value[0];
        valueFea[1] = value[1];
        valueFea[2] = value[2];
        valueFea[3] = value[3];
        if (!feaIsZero(valueFea))
        {
            vector<uint64_t> leftBits = bits;
            leftBits.emplace_back(0);
            zkr = hashValue2keyValue(input, valueFea, keyValues, level + 1, leftBits);
            if (zkr != ZKR_SUCCESS)
            {
                return zkr;
            }
        }

        // If right hash is not {0,0,0,0} then iterate
        valueFea[0] = value[4];
        valueFea[1] = value[5];
        valueFea[2] = value[6];
        valueFea[3] = value[7];
        if (!feaIsZero(valueFea))
        {
            vector<uint64_t> rightBits = bits;
            rightBits.emplace_back(1);
            zkr = hashValue2keyValue(input, valueFea, keyValues, level + 1, rightBits);
            if (zkr != ZKR_SUCCESS)
            {
                return zkr;
            }
        }

        return ZKR_SUCCESS;
    }

    // If capacity is {1,0,0,0} then this is a leaf node
    else if (fr.isOne(value[8]) && fr.isZero(value[9]) && fr.isZero(value[10]) && fr.isZero(value[11]))
    {
        KeyValue keyValue;

        // Re-build the key
        Goldilocks::Element remainingKey[4];
        remainingKey[0] = value[0];
        remainingKey[1] = value[1];
        remainingKey[2] = value[2];
        remainingKey[3] = value[3];
        joinKey(fr, bits, remainingKey, keyValue.key);

        // Get the value hash
        Goldilocks::Element valueHash[4];
        valueHash[0] = value[4];
        valueHash[1] = value[5];
        valueHash[2] = value[6];
        valueHash[3] = value[7];

        // Get the value
        string valueHashString = fea2string(fr, valueHash);
        DatabaseMap::MTMap::const_iterator it;
        it = input.find(valueHashString);
        if (it == input.end())
        {
            zklog.error("hashValue2keyValue() failed searching for valueHash=" + valueHashString);
            return ZKR_DB_KEY_NOT_FOUND;
        }
        const vector<Goldilocks::Element> &value = it->second;
        if (value.size() != 12)
        {
            zklog.error("hashValue2keyValue() value vector with size=" + value.size());
            return ZKR_DB_ERROR;
        }
        if (!fr.isZero(value[8]) || !fr.isZero(value[9]) || !fr.isZero(value[10]) || !fr.isZero(value[11]))
        {
            zklog.error("hashValue2keyValue() value vector with invalid capacity");
            return ZKR_DB_ERROR;
        }
        fea2scalar(fr, keyValue.value, value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7]);

        // Store the key value
        keyValues.emplace_back(keyValue);

        return ZKR_SUCCESS;
    }

    // Invalid capacity
    zklog.error("hashValue2keyValue() found invalid capacity level=" + to_string(level) + " root=" + root + " capacity=" + to_string(fr.toU64(value[8])) + ":" + to_string(fr.toU64(value[9])) + ":" + to_string(fr.toU64(value[10])) + ":" + to_string(fr.toU64(value[11])));
    return ZKR_DB_ERROR;
}

zkresult hashValue2keyValue (const DatabaseMap::MTMap &input, const Goldilocks::Element (&stateRoot)[4], vector<KeyValue> &keyValues)
{
    vector<uint64_t> bits;
    return hashValue2keyValue(input, stateRoot, keyValues, 0, bits);
}

void HashDB::loadDB(const DatabaseMap::MTMap &input, const bool persistent, const Goldilocks::Element (&stateRoot)[4])
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    DatabaseMap::MTMap::const_iterator it;
    if (config.hashDB64)
    {
        vector<KeyValue> keyValues;
        zkresult zkr = hashValue2keyValue(input, stateRoot, keyValues);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("HashDB::loadDB() failed calling hashValue2keyValue() result=" + zkresult2string(zkr));
            exitProcess();
        }
        Goldilocks::Element oldStateRoot[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
        Goldilocks::Element newStateRoot[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};

        zkr = db64.WriteTree(oldStateRoot, keyValues, newStateRoot, persistent);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("HashDB::loadDB() failed calling db64.WriteTree() result=" + zkresult2string(zkr));
            exitProcess();
        }

        if (!feaIsEqual(newStateRoot, stateRoot))
        {
            zklog.error("HashDB::loadDB() failed called db64.WriteTree() but got newStateRoot=" + fea2string(fr, newStateRoot) + " != extected stateRoot=" + fea2string(fr, stateRoot));
            exitProcess();
        }

        stateManager64.setLastConsolidatedStateRoot(stateRoot);
    }
    else
    {
        for (it = input.begin(); it != input.end(); it++)
        {
            db.write(it->first, NULL, it->second, persistent);
        }
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("loadDB", TimeDiff(t));
#endif

}

void HashDB::loadProgramDB(const DatabaseMap::ProgramMap &input, const bool persistent)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    DatabaseMap::ProgramMap::const_iterator it;
    if (config.hashDB64)
    {
        for (it = input.begin(); it != input.end(); it++)
        {
            db64.setProgram(it->first, it->second, persistent);
        }
    }
    else
    {
        for (it = input.begin(); it != input.end(); it++)
        {
            db.setProgram(it->first, it->second, persistent);
        }
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("loadProgramDB", TimeDiff(t));
#endif
}

void HashDB::finishTx (const string &batchUUID, const string &newStateRoot, const Persistence persistence)
{
    if (config.hashDB64)
    {
        if (config.stateManager && (batchUUID.size() != 0))
        {
            stateManager64.finishTx(batchUUID, newStateRoot, persistence);
        }
    }
    else
    {
        if (config.stateManager && (batchUUID.size() != 0))
        {
            stateManager.finishTx(batchUUID, newStateRoot, persistence);
        }
        else
        {
            db.semiFlush();
        }
    }
}

void HashDB::startBlock (const string &batchUUID, const string &oldStateRoot, const Persistence persistence)
{
    if (config.hashDB64)
    {
    }
    else
    {
        if (config.stateManager && (batchUUID.size() != 0))
        {
            stateManager.startBlock(batchUUID, oldStateRoot, persistence);
        }
        else
        {
            db.semiFlush();
        }
    }
}

void HashDB::finishBlock (const string &batchUUID, const string &newStateRoot, const Persistence persistence)
{
    if (config.hashDB64)
    {
        if (config.stateManager && (batchUUID.size() != 0))
        {
            stateManager64.finishBlock(batchUUID, newStateRoot, persistence);
        }
        else
        {
            //db64.semiFlush();
        }
    }
    else
    {
        if (config.stateManager && (batchUUID.size() != 0))
        {
            stateManager.finishBlock(batchUUID, newStateRoot, persistence);
        }
        else
        {
            db.semiFlush();
        }
    }
}

zkresult HashDB::flush (const string &batchUUID, const string &newStateRoot, const Persistence persistence, uint64_t &flushId, uint64_t &storedFlushId)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    zkresult result;
    if (config.hashDB64)
    {
        //result = db64.flush(flushId, storedFlushId);
        //result = stateManager64.flush(batchUUID, newStateRoot, persistence, db64, flushId, storedFlushId);
        zklog.error("HashDB::flush() no longer supported in StateManager64");
        result = ZKR_STATE_MANAGER;
    }
    else
    {
        if (config.stateManager && (batchUUID.size() != 0))
        {
            result = stateManager.flush(batchUUID, newStateRoot, persistence, db, flushId, storedFlushId);
        }
        else
        {
            result = db.flush(flushId, storedFlushId);
        }
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("flush", TimeDiff(t));
    tms.print("HashDB");
    tms.clear();
#endif

    return result;
}

zkresult HashDB::purge (const string &batchUUID, const Goldilocks::Element (&newStateRoot)[4], const Persistence persistence)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    zkresult result;
    if (config.hashDB64 && config.stateManager && (batchUUID.size() != 0))
    {
        //result = stateManager64.purge(batchUUID, fea2string(fr, newStateRoot), persistence, db64);
        zklog.error("HashDB::purge() no longer supported in StateManager64");
        result = ZKR_STATE_MANAGER;
    }
    else
    {
        zklog.error("HashDB::purge() called with invalid configuration");
        result = ZKR_STATE_MANAGER;
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("purge", TimeDiff(t));
#endif

    return result;
}

zkresult HashDB::consolidateState (const Goldilocks::Element (&virtualStateRoot)[4], const Persistence persistence, Goldilocks::Element (&consolidatedStateRoot)[4], uint64_t &flushId, uint64_t &storedFlushId)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    zkresult result;
    if (config.hashDB64)
    {
        if (config.stateManager)
        {
            string consolidatedStateRootString;
            result = stateManager64.consolidateState(fea2string(fr, virtualStateRoot), persistence, consolidatedStateRootString, db64, flushId, storedFlushId);
            if (result == ZKR_SUCCESS)
            {
                string2fea(fr, consolidatedStateRootString, consolidatedStateRoot);
                zklog.info("HashDB::consolidateState() virtualState=" + fea2string(fr, virtualStateRoot) + " consolidatedState=" + consolidatedStateRootString);
            }
        }
        else
        {
            zklog.error("HashDB::consolidateState() called with config.stateManager=false");
            return ZKR_UNSPECIFIED;
        }
    }
    else
    {
        zklog.error("HashDB::consolidateState() called with config.hashDB64=false");
        return ZKR_UNSPECIFIED;
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("consolidateState", TimeDiff(t));
    tms.print("HashDB");
    tms.clear();
#endif

    return result;
}

zkresult HashDB::getFlushStatus(uint64_t &storedFlushId, uint64_t &storingFlushId, uint64_t &lastFlushId, uint64_t &pendingToFlushNodes, uint64_t &pendingToFlushProgram, uint64_t &storingNodes, uint64_t &storingProgram, string &proverId)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    // Get IDs and counters from database
    if (config.hashDB64)
    {
        db64.getFlushStatus(storedFlushId, storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram);
    }
    else
    {
        db.getFlushStatus(storedFlushId, storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram);
    }

    // Get process ID from configuration
    proverId = config.proverID;

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("getFlushStatus", TimeDiff(t));
    tms.print("HashDB");
    tms.clear();
#endif

    return ZKR_SUCCESS;
}

zkresult HashDB::getFlushData(uint64_t flushId, uint64_t &lastSentFlushId, unordered_map<string, string> (&nodes), unordered_map<string, string> (&program), string &nodesStateRoot)
{
    if (!config.dbMultiWrite)
    {
        zklog.error("HashDB::getFlushData() called with config.dbMultiWrite=false");
        return ZKR_DB_ERROR;
    }

    zkresult zkr;

    if (config.hashDB64)
    {
        zklog.error("HashDB::getFlushData() called with config.hashDB64=true");
        return ZKR_DB_ERROR;
    }
    else
    {
        zkr = db.getFlushData(flushId, lastSentFlushId, nodes, program, nodesStateRoot);
    }

    return zkr;
}

void HashDB::clearCache(void)
{
    if (config.hashDB64)
    {
        // We don't use cache in HashDB64
    }
    else
    {
        db.clearCache();
    }
}

zkresult HashDB::readTree (const Goldilocks::Element (&root)[4], vector<KeyValue> &keyValues, vector<HashValueGL> &hashValues)
{
    if (!config.hashDB64)
    {
        zklog.error("HashDB::readTree() called with config.hashDB64=false");
        return ZKR_UNSPECIFIED;
    }

    return db64.ReadTree(root, keyValues, &hashValues);
}

zkresult HashDB::writeTree (const Goldilocks::Element (&oldRoot)[4], const vector<KeyValue> &keyValues, Goldilocks::Element (&newRoot)[4], const bool persistent)
{
    if (config.hashDB64)
    {
        zklog.error("HashDB::writeTree() called with config.hashDB64=false");
        return ZKR_UNSPECIFIED;
    }

    return db64.WriteTree(oldRoot, keyValues, newRoot, persistent);
}

zkresult HashDB::cancelBatch (const string &batchUUID)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    zkresult result;
    if (config.hashDB64 && config.stateManager && (batchUUID.size() != 0))
    {
        result = stateManager64.cancelBatch(batchUUID);
    }
    else if (!config.hashDB64 && config.stateManager && (batchUUID.size() != 0))
    {
        result = stateManager.cancelBatch(batchUUID);
    }
    else
    {
        zklog.error("HashDB::cancelBatch() called with invalid configuration");
        result = ZKR_STATE_MANAGER;
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("cancelBatch", TimeDiff(t));
#endif

    return result;
}

zkresult HashDB::resetDB (void)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    zkresult result;
    if (config.hashDB64 && config.stateManager)
    {
        result = db64.resetDB();
    }
    else
    {
        result = db.resetDB();
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("resetDB", TimeDiff(t));
#endif

    return result;
}

void HashDB::setAutoCommit(const bool autoCommit)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

#ifdef DATABASE_COMMIT
    db.setAutoCommit(autoCommit);
#endif

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("setAutoCommit", TimeDiff(t));
#endif
}

void HashDB::commit()
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

#ifdef DATABASE_COMMIT
    db.commit();
#endif

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("commit", TimeDiff(t));
#endif
}

void HashDB::hashSave(const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], const Persistence persistence, Goldilocks::Element (&hash)[4])
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif
    if (config.hashDB64)
    {
        // TODO: change the way we hashSave() in hashDB64
    }
    else
    {
        SmtContext ctx(db, false, "", 0, 0, persistence);
        smt.hashSave(ctx, a, c, hash);
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("hashSave", TimeDiff(t));
#endif
}