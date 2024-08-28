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
#include "key_utils.hpp"

HashDB::HashDB(Goldilocks &fr, const Config &config) : fr(fr), config(config), db(fr, config), smt(fr)
{
    db.init();
}

HashDB::~HashDB()
{    
#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.print("HashDB");
#endif    
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

    zkresult zkr = smt.set(batchUUID, block, tx, db, oldRoot, key, value, persistence, *r, dbReadLog);

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

    zkresult zkr = smt.get(batchUUID, db, root, key, *r, dbReadLog);

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
    if (config.stateManager)
    {
        zkr = stateManager.writeProgram(batchUUID, block, tx, keyString, data, persistence);
    }
    else
    {
        zkr = db.setProgram(keyString, data, persistence == PERSISTENCE_DATABASE);
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
    if (config.stateManager)
    {
        zkr = stateManager.readProgram(batchUUID, keyString, data, dbReadLog);
        if (zkr != ZKR_SUCCESS)
        {
            zkr = db.getProgram(keyString, data, dbReadLog);
        }
    } else
    {
        zkr = db.getProgram(keyString, data, dbReadLog);
    }
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HashDB::getProgram() failed result=" + zkresult2string(zkr) + " key=" + keyString);
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("getProgram", TimeDiff(t));
#endif

    return zkr;
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
    for (it = input.begin(); it != input.end(); it++)
    {
        db.write(it->first, NULL, it->second, persistent);
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
    for (it = input.begin(); it != input.end(); it++)
    {
        db.setProgram(it->first, it->second, persistent);
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("loadProgramDB", TimeDiff(t));
#endif
}

void HashDB::finishTx (const string &batchUUID, const string &newStateRoot, const Persistence persistence)
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

void HashDB::startBlock (const string &batchUUID, const string &oldStateRoot, const Persistence persistence)
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

void HashDB::finishBlock (const string &batchUUID, const string &newStateRoot, const Persistence persistence)
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

zkresult HashDB::flush (const string &batchUUID, const string &newStateRoot, const Persistence persistence, uint64_t &flushId, uint64_t &storedFlushId)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    zkresult result;
    if (config.stateManager && (batchUUID.size() != 0))
    {
        result = stateManager.flush(batchUUID, newStateRoot, persistence, db, flushId, storedFlushId);
    }
    else
    {
        result = db.flush(flushId, storedFlushId);
    }

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("flush", TimeDiff(t));
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
    db.getFlushStatus(storedFlushId, storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram);

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

    zkresult zkr = db.getFlushData(flushId, lastSentFlushId, nodes, program, nodesStateRoot);

    return zkr;
}

void HashDB::clearCache(void)
{
    db.clearCache();
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
    if (config.stateManager && (batchUUID.size() != 0))
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
    result = db.resetDB();

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
    SmtContext ctx(db, false, "", 0, 0, persistence);
    smt.hashSave(ctx, a, c, hash);

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("hashSave", TimeDiff(t));
#endif
}