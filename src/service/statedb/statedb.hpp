#ifndef STATEDB_HPP
#define STATEDB_HPP

#include "goldilocks_base_field.hpp"
#include "database.hpp"
#include "config.hpp"
#include "smt.hpp"
#include "statedb_interface.hpp"
#include "zkresult.hpp"
#include "utils/time_metric.hpp"

class StateDB : public StateDBInterface
{
private:
    Goldilocks &fr;
    const Config &config;
public:
    Database db;
private:
    Smt smt;

#ifdef STATEDB_LOCK
    recursive_mutex mlock;
#endif

#ifdef LOG_TIME_STATISTICS_STATEDB
    TimeMetricStorage tms;
    struct timeval t;
#endif

public:
    StateDB(Goldilocks &fr, const Config &config);
    ~StateDB();
    zkresult set(const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, Goldilocks::Element (&newRoot)[4], SmtSetResult *result, DatabaseMap *dbReadLog);
    zkresult get(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result, DatabaseMap *dbReadLog);
    zkresult setProgram(const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const bool persistent);
    zkresult getProgram(const Goldilocks::Element (&key)[4], vector<uint8_t> &data, DatabaseMap *dbReadLog);
    void loadDB(const DatabaseMap::MTMap &inputDB, const bool persistent);
    void loadProgramDB(const DatabaseMap::ProgramMap &inputProgramDB, const bool persistent);
    zkresult flush();

    // Methods added for testing purposes
    void setAutoCommit(const bool autoCommit);
    void commit();
    void hashSave(const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], const bool persistent, Goldilocks::Element (&hash)[4]);
};

#endif