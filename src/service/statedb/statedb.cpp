#include "statedb.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sys/time.h>
#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "scalar.hpp"
#include "zkresult.hpp"
#include "database_map.hpp"

StateDB::StateDB(Goldilocks &fr, const Config &config) : fr(fr), config(config), db(fr, config), smt(fr)
{
    db.init();
}

StateDB::~StateDB()
{    
#ifdef LOG_TIME_STATISTICS_STATEDB
    tms.print("StateDB");
#endif    
}

zkresult StateDB::set(const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_STATEDB
    gettimeofday(&t, NULL);
#endif

#ifdef STATEDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    SmtSetResult *r;
    if (result == NULL) r = new SmtSetResult;
    else r = result;

    zkresult zkr = smt.set(db, oldRoot, key, value, persistent, *r, dbReadLog);
    for (int i = 0; i < 4; i++) newRoot[i] = r->newRoot[i];

    if (result == NULL) delete r;

#ifdef LOG_TIME_STATISTICS_STATEDB
    tms.add("set", TimeDiff(t));
#endif

    return zkr;
}

zkresult StateDB::get(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_STATEDB
    gettimeofday(&t, NULL);
#endif

#ifdef STATEDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    SmtGetResult *r;
    if (result == NULL) r = new SmtGetResult;
    else r = result;

    zkresult zkr = smt.get(db, root, key, *r, dbReadLog);

    value = r->value;

    if (result == NULL) delete r;

#ifdef LOG_TIME_STATISTICS_STATEDB
    tms.add("get", TimeDiff(t));
#endif

    return zkr;
}

zkresult StateDB::setProgram(const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const bool persistent)
{
#ifdef LOG_TIME_STATISTICS_STATEDB
    gettimeofday(&t, NULL);
#endif

#ifdef STATEDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    zkresult zkr = db.setProgram(fea2string(fr, key), data, persistent);

#ifdef LOG_TIME_STATISTICS_STATEDB
    tms.add("setProgram", TimeDiff(t));
#endif

    return zkr;
}

zkresult StateDB::getProgram(const Goldilocks::Element (&key)[4], vector<uint8_t> &data, DatabaseMap *dbReadLog)
{
#ifdef LOG_TIME_STATISTICS_STATEDB
    gettimeofday(&t, NULL);
#endif

#ifdef STATEDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    data.clear();
    zkresult zkr = db.getProgram(fea2string(fr, key), data, dbReadLog);

#ifdef LOG_TIME_STATISTICS_STATEDB
    tms.add("getProgram", TimeDiff(t));
#endif

    return zkr;
}

void StateDB::loadDB(const DatabaseMap::MTMap &input, const bool persistent)
{
#ifdef LOG_TIME_STATISTICS_STATEDB
    gettimeofday(&t, NULL);
#endif

#ifdef STATEDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    DatabaseMap::MTMap::const_iterator it;
    for (it = input.begin(); it != input.end(); it++)
    {
        db.write(it->first, it->second, persistent);
    }

#ifdef LOG_TIME_STATISTICS_STATEDB
    tms.add("loadDB", TimeDiff(t));
#endif

}

void StateDB::loadProgramDB(const DatabaseMap::ProgramMap &input, const bool persistent)
{
#ifdef LOG_TIME_STATISTICS_STATEDB
    gettimeofday(&t, NULL);
#endif

#ifdef STATEDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    DatabaseMap::ProgramMap::const_iterator it;
    for (it = input.begin(); it != input.end(); it++)
    {
        db.setProgram(it->first, it->second, persistent);
    }

#ifdef LOG_TIME_STATISTICS_STATEDB
    tms.add("loadProgramDB", TimeDiff(t));
#endif
}

zkresult StateDB::flush()
{
#ifdef LOG_TIME_STATISTICS_STATEDB
    gettimeofday(&t, NULL);
#endif

#ifdef STATEDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    zkresult result;
    result = db.flush();

#ifdef LOG_TIME_STATISTICS_STATEDB
    tms.add("flush", TimeDiff(t));
    tms.print("StateDB");
    tms.clear();
#endif

    return result;
}

void StateDB::setAutoCommit(const bool autoCommit)
{
#ifdef LOG_TIME_STATISTICS_STATEDB
    gettimeofday(&t, NULL);
#endif

#ifdef STATEDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

#ifdef DATABASE_COMMIT
    db.setAutoCommit(autoCommit);
#endif

#ifdef LOG_TIME_STATISTICS_STATEDB
    tms.add("setAutoCommit", TimeDiff(t));
#endif
}

void StateDB::commit()
{
#ifdef LOG_TIME_STATISTICS_STATEDB
    gettimeofday(&t, NULL);
#endif

#ifdef STATEDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

#ifdef DATABASE_COMMIT
    db.commit();
#endif

#ifdef LOG_TIME_STATISTICS_STATEDB
    tms.add("commit", TimeDiff(t));
#endif
}

void StateDB::hashSave(const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], const bool persistent, Goldilocks::Element (&hash)[4])
{
#ifdef LOG_TIME_STATISTICS_STATEDB
    gettimeofday(&t, NULL);
#endif

#ifdef STATEDB_LOCK
    lock_guard<recursive_mutex> guard(mlock);
#endif

    smt.hashSave(db, a, c, persistent, hash);

#ifdef LOG_TIME_STATISTICS_STATEDB
    tms.add("hashSave", TimeDiff(t));
#endif
}