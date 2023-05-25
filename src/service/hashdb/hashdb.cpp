#include "hashdb.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sys/time.h>
#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "scalar.hpp"
#include "zkresult.hpp"
#include "database_map.hpp"

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

zkresult HashDB::set(const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result, DatabaseMap *dbReadLog)
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

    zkresult zkr = smt.set(db, oldRoot, key, value, persistent, *r, dbReadLog);
    for (int i = 0; i < 4; i++) newRoot[i] = r->newRoot[i];

    if (result == NULL) delete r;

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("set", TimeDiff(t));
#endif

    return zkr;
}

zkresult HashDB::get(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result, DatabaseMap *dbReadLog)
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

    zkresult zkr = smt.get(db, root, key, *r, dbReadLog);

    value = r->value;

    if (result == NULL) delete r;

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("get", TimeDiff(t));
#endif

    return zkr;
}

zkresult HashDB::setProgram(const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const bool persistent)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    zkresult zkr = db.setProgram(fea2string(fr, key), data, persistent);

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("setProgram", TimeDiff(t));
#endif

    return zkr;
}

zkresult HashDB::getProgram(const Goldilocks::Element (&key)[4], vector<uint8_t> &data, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    data.clear();
    zkresult zkr = db.getProgram(fea2string(fr, key), data, dbReadLog);

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("getProgram", TimeDiff(t));
#endif

    return zkr;
}

void HashDB::loadDB(const DatabaseMap::MTMap &input, const bool persistent)
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
        db.write(it->first, it->second, persistent);
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

zkresult HashDB::flush(uint64_t &flushId, uint64_t &storedFlushId)
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    zkresult result;
    result = db.flush(flushId, storedFlushId);

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

    // Get proces ID from configuration
    proverId = config.proverID;

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("getFlushStatus", TimeDiff(t));
    tms.print("HashDB");
    tms.clear();
#endif

    return ZKR_SUCCESS;
}

zkresult HashDB::getFlushData(uint64_t flushId, uint64_t &lastSentFlushId, vector<FlushData> (&nodes), vector<FlushData> (&nodesUpdate), vector<FlushData> (&program), vector<FlushData> (&programUpdate), string &nodesStateRoot)
{
    if (!config.dbMultiWrite)
    {
        zklog.error("HashDB::getFlushData() called with config.dbMultiWrite=false");
        return ZKR_DB_ERROR;
    }

    return db.getFlushData(flushId, lastSentFlushId, nodes, nodesUpdate, program, programUpdate, nodesStateRoot);
}

void HashDB::clearCache(void)
{
    db.clearCache();
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

void HashDB::hashSave(const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], const bool persistent, Goldilocks::Element (&hash)[4])
{
#ifdef LOG_TIME_STATISTICS_HASHDB
    gettimeofday(&t, NULL);
#endif

#ifdef HASHDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    smt.hashSave(db, a, c, persistent, hash, NULL);

#ifdef LOG_TIME_STATISTICS_HASHDB
    tms.add("hashSave", TimeDiff(t));
#endif
}