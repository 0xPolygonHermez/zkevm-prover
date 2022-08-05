#include "statedb.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sys/time.h>
#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "scalar.hpp"
#include "zkresult.hpp"

StateDB::StateDB (Goldilocks &fr, const Config &config) : fr(fr), config(config), db(fr), smt(fr)
{
    db.init(config);
}

zkresult StateDB::set (const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result) 
{
    unique_lock<mutex> guard(mlock);
    
    SmtSetResult* r;
    if (result==NULL) r = new SmtSetResult;
    else r = result;

    zkresult zkr = smt.set (db, oldRoot, key, value, persistent, *r);
    for (int i=0; i<4; i++) newRoot[i] = r->newRoot[i];

    if (result==NULL) delete r;

    return zkr;
}

zkresult StateDB::get (const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result)
{
    unique_lock<mutex> guard(mlock);
    
    SmtGetResult* r;
    if (result==NULL) r = new SmtGetResult;
    else r = result;

    zkresult zkr = smt.get (db, root, key, *r);
    value = r->value;

    if (result==NULL) delete r;

    return zkr;
}

zkresult StateDB::setProgram (const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const bool persistent)
{
    unique_lock<mutex> guard(mlock);

    return db.setProgram (fea2string(fr, key), data, persistent);
}

zkresult StateDB::getProgram (const Goldilocks::Element (&key)[4], vector<uint8_t> &data)
{
    unique_lock<mutex> guard(mlock);
    
    return db.getProgram (fea2string(fr, key), data);
}

void StateDB::flush()
{
    unique_lock<mutex> guard(mlock);

    db.flush();
}

Database * StateDB::getDatabase (void)
{
    return &db;
}

void StateDB::setAutoCommit (const bool autoCommit)
{
    db.setAutoCommit (autoCommit);
}

void StateDB::commit()
{
    unique_lock<mutex> guard(mlock);
    
    db.commit();
}

void StateDB::hashSave (const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], const bool persistent, Goldilocks::Element (&hash)[4])
{
    unique_lock<mutex> guard(mlock);

    smt.hashSave(db, a, c, persistent, hash);
}