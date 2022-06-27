#include "statedb.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sys/time.h>
#include "goldilocks/goldilocks_base_field.hpp"
#include "config.hpp"

StateDB::StateDB (Goldilocks &fr, const Config &config, bool autoCommit, bool asyncWrite) : fr(fr), config(config), db(fr, autoCommit, asyncWrite), smt(fr)
{
    db.init(config);
}

int StateDB::set(Goldilocks::Element (&oldRoot)[4], Goldilocks::Element (&key)[4], mpz_class &value, const bool persistent, SmtSetResult &result) 
{
    std::lock_guard<std::mutex> lock(mutex);

    smt.set (db, oldRoot, key, value, persistent, result);

    return DB_SUCCESS;
}

int StateDB::get(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], SmtGetResult &result)
{
    std::lock_guard<std::mutex> lock(mutex);
    
    smt.get (db, root, key, result);

    return DB_SUCCESS;
}

int StateDB::setProgram (const string &key, const vector<uint8_t> &value, const bool persistent)
{
    std::lock_guard<std::mutex> lock(mutex);

    return db.setProgram (key, value, persistent);
}

int StateDB::getProgram (const string &key, vector<uint8_t> &value)
{
    std::lock_guard<std::mutex> lock(mutex);
    
    return db.getProgram (key, value);
}

void StateDB::flush()
{
    std::lock_guard<std::mutex> lock(mutex);
    
    db.flush();
}

void StateDB::commit()
{
    std::lock_guard<std::mutex> lock(mutex);
    
    db.commit();
}

void StateDB::hashSave (const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], const bool persistent, Goldilocks::Element (&hash)[4])
{
    std::lock_guard<std::mutex> lock(mutex);

    smt.hashSave(db, a, c, persistent, hash);
}

