#ifndef HASHDB_HPP
#define HASHDB_HPP

#include "goldilocks_base_field.hpp"
#include "database.hpp"
#include "database_64.hpp"
#include "config.hpp"
#include "smt.hpp"
#include "hashdb_interface.hpp"
#include "zkresult.hpp"
#include "utils/time_metric.hpp"

class HashDB : public HashDBInterface
{
private:
    Goldilocks &fr;
    const Config &config;
public:
    Database db;
    Database64 db64;
private:
    Smt smt;

#ifdef HASHDB_LOCK
    recursive_mutex mlock;
#endif

#ifdef LOG_TIME_STATISTICS_HASHDB
    TimeMetricStorage tms;
    struct timeval t;
#endif

public:
    HashDB(Goldilocks &fr, const Config &config);
    ~HashDB();

    // HashDBInterface methods
    zkresult getLatestStateRoot (Goldilocks::Element (&stateRoot)[4]);
    zkresult set                (const string &batchUUID, uint64_t block, uint64_t tx, const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const Persistence persistence, Goldilocks::Element (&newRoot)[4], SmtSetResult *result, DatabaseMap *dbReadLog);
    zkresult get                (const string &batchUUID, const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, SmtGetResult *result, DatabaseMap *dbReadLog);
    zkresult setProgram         (const string &batchUUID, uint64_t block, uint64_t tx, const Goldilocks::Element (&key)[4], const vector<uint8_t> &data, const Persistence persistence);
    zkresult getProgram         (const string &batchUUID, const Goldilocks::Element (&key)[4], vector<uint8_t> &data, DatabaseMap *dbReadLog);
    void     loadDB             (const DatabaseMap::MTMap &inputDB, const bool persistent, const Goldilocks::Element (&stateRoot)[4]);
    void     loadProgramDB      (const DatabaseMap::ProgramMap &inputProgramDB, const bool persistent);
    void     finishTx           (const string &batchUUID, const string &newStateRoot, const Persistence persistence);
    void     startBlock         (const string &batchUUID, const string &oldStateRoot, const Persistence persistence);
    void     finishBlock        (const string &batchUUID, const string &newStateRoot, const Persistence persistence);
    zkresult flush              (const string &batchUUID, const string &newStateRoot, const Persistence persistence, uint64_t &flushId, uint64_t &storedFlushId);
    zkresult purge              (const string &batchUUID, const Goldilocks::Element (&newStateRoot)[4], const Persistence persistence);
    zkresult consolidateState   (const Goldilocks::Element (&virtualStateRoot)[4], const Persistence persistence, Goldilocks::Element (&consolidatedStateRoot)[4], uint64_t &flushId, uint64_t &storedFlushId);
    zkresult getFlushStatus     (uint64_t &storedFlushId, uint64_t &storingFlushId, uint64_t &lastFlushId, uint64_t &pendingToFlushNodes, uint64_t &pendingToFlushProgram, uint64_t &storingNodes, uint64_t &storingProgram, string &proverId);
    zkresult getFlushData       (uint64_t flushId, uint64_t &storedFlushId, unordered_map<string, string> (&nodes), unordered_map<string, string> (&program), string &nodesStateRoot);
    void     clearCache         (void);
    zkresult readTree           (const Goldilocks::Element (&root)[4], vector<KeyValue> &keyValues, vector<HashValueGL> &hashValues);
    zkresult writeTree          (const Goldilocks::Element (&oldRoot)[4], const vector<KeyValue> &keyValues, Goldilocks::Element (&newRoot)[4], const bool persistent);
    zkresult cancelBatch        (const string &batchUUID);
    zkresult resetDB            (void);

    // Methods added for testing purposes
    void setAutoCommit(const bool autoCommit);
    void commit();
    void hashSave(const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], const Persistence persistence, Goldilocks::Element (&hash)[4]);
};

#endif