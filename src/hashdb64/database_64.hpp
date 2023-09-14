#ifndef DATABASE_64_HPP
#define DATABASE_64_HPP

#include <vector>
#include <map>
#include <pqxx/pqxx>
#include "goldilocks_base_field.hpp"
#include "compare_fe.hpp"
#include "config.hpp"
#include <semaphore.h>
#include "zkresult.hpp"
#include "database_map.hpp"
#include "database_cache_64.hpp"
#include "database_connection.hpp"
#include "zkassert.hpp"
#include "multi_write_64.hpp"
#include "database_versions_associtive_cache.hpp"
#include "database_kv_associative_cache.hpp"
#include "key_value.hpp"

using namespace std;

class DB64Query
{
public:
    string key;
    Goldilocks::Element keyFea[4];
    string &value; // value can be an input in multiWrite(), or an output in multiRead()
    DB64Query(const string &_key, const Goldilocks::Element (&_keyFea)[4], string &_value) : key(_key), value(_value)
    {
        keyFea[0] = _keyFea[0];
        keyFea[1] = _keyFea[1];
        keyFea[2] = _keyFea[2];
        keyFea[3] = _keyFea[3];
    }
};

class Database64
{
public:
    Goldilocks &fr;
    const Config &config;

#ifdef DATABASE_COMMIT
    bool autoCommit = true;
#endif

    // Basic flags
    bool bInitialized = false;
    bool useRemoteDB = false;

#ifdef DATABASE_COMMIT
    pqxx::work* transaction = NULL;
#endif

    // Connection(s) attributes
    pthread_mutex_t connMutex; // Mutex to protect the connection(s)
    void connLock(void) { pthread_mutex_lock(&connMutex); };
    void connUnlock(void) { pthread_mutex_unlock(&connMutex); };
    DatabaseConnection connection;
    DatabaseConnection * connectionsPool;
    uint64_t nextConnection;
    uint64_t usedConnections;
    DatabaseConnection * getConnection (void);
    void disposeConnection (DatabaseConnection * pConnection);
    void queryFailed (void);

    // Multi write attributes
public:
    MultiWrite64 multiWrite;
    sem_t senderSem; // Semaphore to wakeup database sender thread when flush() is called
    sem_t getFlushDataSem; // Semaphore to unblock getFlushData() callers when new data is available
private:
    pthread_t senderPthread; // Database sender thread
    pthread_t cacheSynchPthread; // Cache synchronization thread
    int maxVersions; // Maximum number of versions to store in the database KV
    int maxVersionsUpload; // Maximum number of versions to upload from the database KV to the cache when there is a cache miss

private:
    // Remote database based on Postgres (PostgreSQL)
    void initRemote(void);
    zkresult readRemote(bool bProgram, const string &key, string &value);
    zkresult writeRemote(bool bProgram, const string &key, const string &value);

    zkresult readRemoteKV(const uint64_t version, const Goldilocks::Element (&key)[4],  mpz_class value, vector<VersionValue> &upstreamVersionValues); 
    zkresult writeRemoteKV(const uint64_t version, const Goldilocks::Element (&key)[4], const mpz_class &value, bool useMultiWrite = true);
    zkresult readRemoteVersion(const Goldilocks::Element (&root)[4], uint64_t version);
    zkresult writeRemoteVersion(const Goldilocks::Element (&root)[4], const uint64_t version); 
    zkresult readRemoteLatestVersion(uint64_t &version);
    zkresult writeRemoteLatestVersion(const uint64_t version);

    zkresult extractVersion(const pqxx::field& fieldData, const uint64_t version, mpz_class &value, vector<VersionValue> &upstreamVersionValues);

public:
#ifdef DATABASE_USE_CACHE
    // Cache static instances
    static DatabaseMTCache64 dbMTCache;
    static DatabaseProgramCache64 dbProgramCache;
    static DatabaseKVAssociativeCache dbKVACache;
    static DatabaseVersionsAssociativeCache dbVersionACache;
    static uint64_t latestVersionCache;

    // This is a fixed key to store the latest state root hash, used to load it to the cache
    // This key is "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    // This key cannot be the result of a hash because it is out of the Goldilocks Element range
    static string dbStateRootKey;
    static Goldilocks::Element dbStateRootvKey[4];

#endif

    // Constructor and destructor
    Database64(Goldilocks &fr, const Config &config);
    ~Database64();

    // Basic methods
    void init(void);
    zkresult read(const string &_key, const Goldilocks::Element (&vkey)[4], string &value, DatabaseMap *dbReadLog, const bool update = false, bool *keys = NULL , uint64_t level=0);
    zkresult read(vector<DB64Query> &dbQueries);
    zkresult write(const string &_key, const Goldilocks::Element* vkey, const string &value, const bool persistent);
    zkresult write(vector<DB64Query> &dbQueries, const bool persistent);
    zkresult getProgram(const string &_key, vector<uint8_t> &value, DatabaseMap *dbReadLog);
    zkresult setProgram(const string &_key, const vector<uint8_t> &value, const bool persistent);
    zkresult readKV(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, DatabaseMap *dbReadLog); 
    zkresult readKV(const Goldilocks::Element (&root)[4], vector<KeyValue> &KVs, DatabaseMap *dbReadLog);
    zkresult writeKV(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, bool persistent);
    zkresult writeKV(const uint64_t& version, const Goldilocks::Element (&key)[4], const mpz_class &value, bool persistent);
    zkresult writeKV(const Goldilocks::Element (&root)[4], const vector<KeyValue> &KVs, bool persistent);
    zkresult writeKV(const uint64_t& version, const vector<KeyValue> &KVs, bool persistent);
    zkresult readVersion(const Goldilocks::Element (&root)[4], uint64_t& version, DatabaseMap *dbReadLog);
    zkresult writeVersion(const Goldilocks::Element (&root)[4], const uint64_t version, bool persistent);
    zkresult readLatestVersion(uint64_t &version);
    zkresult writeLatestVersion(const uint64_t version, const bool persistent);

private:
    zkresult createStateRoot(void);
public:
    zkresult updateStateRoot(const Goldilocks::Element (&stateRoot)[4]);

#ifdef DATABASE_COMMIT
    void setAutoCommit(const bool autoCommit);
    void commit();
#endif

    // Flush multi write pending requests
    zkresult flush(uint64_t &flushId, uint64_t &lastSentFlushId);
    void semiFlush (void);
    zkresult getFlushStatus(uint64_t &storedFlushId, uint64_t &storingFlushId, uint64_t &lastFlushId, uint64_t &pendingToFlushNodes, uint64_t &pendingToFlushProgram, uint64_t &storingNodes, uint64_t &storingProgram);

    // Send multi write data to remote database; called by dbSenderThread
    zkresult sendData(void);

    // Get flush data, written to database by dbSenderThread; it blocks
    zkresult getFlushData(uint64_t flushId, uint64_t &lastSentFlushId, unordered_map<string, string> (&nodes), unordered_map<string, string> (&program), string &nodesStateRoot);

    // Print tree
    void printTree(const string &root, string prefix = "");

    // Clear cache
    void clearCache(void);
};

// Thread to send data to database
void *dbSenderThread64(void *arg);

// Thread to synchronize cache from master hash DB server
void *dbCacheSynchThread64(void *arg);

void loadDb2MemCache64(const Config &config);

#endif