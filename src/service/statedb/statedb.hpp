#ifndef STATEDB_HPP
#define STATEDB_HPP

#include "goldilocks/goldilocks_base_field.hpp"
#include "database.hpp"
#include "config.hpp"
#include "smt.hpp"
#include <mutex>

class StateDB
{
private:
    Goldilocks &fr;
    const Config &config;
    Database db;
    Smt smt;
    std::mutex mutex; // Mutex to protect the requests queues

public:
    StateDB (Goldilocks &fr, const Config &config, bool autoCommit, bool asyncWrite);
    int set(Goldilocks::Element (&oldRoot)[4], Goldilocks::Element (&key)[4], mpz_class &value, const bool persistent, SmtSetResult &result);
    int get(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], SmtGetResult &result);
    int setProgram (const string &key, const vector<uint8_t> &value, const bool persistent);
    int getProgram (const string &key, vector<uint8_t> &value);
    void commit();
    void flush();
    void hashSave (const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], const bool persistent, Goldilocks::Element (&hash)[4]);
    void setDBDebug (bool d) {db.debug = d;}
};

#endif