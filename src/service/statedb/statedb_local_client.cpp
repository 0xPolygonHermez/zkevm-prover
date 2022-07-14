#include "statedb_local_client.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sys/time.h>
#include "goldilocks/goldilocks_base_field.hpp"
#include "config.hpp"
#include "scalar.hpp"
#include "result.hpp"

StateDBLocalClient::StateDBLocalClient (Goldilocks &fr, const Config &config) : fr(fr), config(config), db(fr), smt(fr)
{
    db.init(config);
}

result_t StateDBLocalClient::set (const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result) 
{
    std::lock_guard<std::mutex> lock(mutex);
    
    SmtSetResult* r;
    if (result==NULL) r = new SmtSetResult;
    else r= result;

    smt.set (db, oldRoot, key, value, persistent, *r);
    for (int i=0; i<4; i++) newRoot[i] = r->newRoot[i];

    if (result==NULL) delete r;

    return R_SUCCESS;
}

result_t StateDBLocalClient::get (const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result)
{
    std::lock_guard<std::mutex> lock(mutex);
    
    SmtGetResult* r;
    if (result==NULL) r = new SmtGetResult;
    else r= result;

    smt.get (db, root, key, *r);
    value = r->value;

    if (result==NULL) delete r;

    return R_SUCCESS;
}

result_t StateDBLocalClient::setProgram (const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const bool persistent)
{
    std::lock_guard<std::mutex> lock(mutex);

    return db.setProgram (fea2string(fr, key), data, persistent);
}

result_t StateDBLocalClient::getProgram (const Goldilocks::Element (&key)[4], vector<uint8_t> &data)
{
    std::lock_guard<std::mutex> lock(mutex);
    
    return db.getProgram (fea2string(fr, key), data);
}

void StateDBLocalClient::flush()
{
    std::lock_guard<std::mutex> lock(mutex);
    
    db.flush();
}

void StateDBLocalClient::setAutoCommit (const bool autoCommit)
{
    db.setAutoCommit (autoCommit);
}

void StateDBLocalClient::commit()
{
    std::lock_guard<std::mutex> lock(mutex);
    
    db.commit();
}

void StateDBLocalClient::hashSave (const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], const bool persistent, Goldilocks::Element (&hash)[4])
{
    std::lock_guard<std::mutex> lock(mutex);

    smt.hashSave(db, a, c, persistent, hash);
}

