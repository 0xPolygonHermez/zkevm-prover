#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <vector>
#include <map>
#include <pqxx/pqxx>
#include "goldilocks_base_field.hpp"
#include "compare_fe.hpp"
#include "config.hpp"
#include <semaphore.h>
#include "zkresult.hpp"
#include "database_map.hpp"
#include "database_cache.hpp"
#include "database_connection.hpp"
#include "zkassert.hpp"
#include "multi_write.hpp"
#include "database_associative_cache.hpp"

using namespace std;

class Database
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
    MultiWrite multiWrite;
    sem_t senderSem; // Semaphore to wakeup database sender thread when flush() is called
    sem_t getFlushDataSem; // Semaphore to unblock getFlushData() callers when new data is available
private:
    pthread_t senderPthread; // Database sender thread
    pthread_t cacheSynchPthread; // Cache synchronization thread

private:
    // Remote database based on Postgres (PostgreSQL)
    void initRemote(void);
    zkresult readRemote(bool bProgram, const string &key, string &value);
    zkresult readTreeRemote(const string &key, bool *keys, uint64_t level, uint64_t &numberOfFields);
    zkresult writeRemote(bool bProgram, const string &key, const string &value);
    zkresult writeGetTreeFunction(void);

public:
#ifdef DATABASE_USE_CACHE
    // Cache static instances
    static bool useAssociativeCache;
    static DatabaseMTAssociativeCache dbMTACache;
    static DatabaseMTCache dbMTCache;
    static DatabaseProgramCache dbProgramCache;

    // This is a fixed key to store the latest state root hash, used to load it to the cache
    // This key is "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    // This key cannot be the result of a hash because it is out of the Goldilocks Element range
    static string dbStateRootKey;
    static Goldilocks::Element dbStateRootvKey[4];

#endif

    // Constructor and destructor
    Database(Goldilocks &fr, const Config &config);
    ~Database();

    // Basic methods
    void init(void);
    zkresult read(const string &_key, Goldilocks::Element (&vkey)[4], vector<Goldilocks::Element> &value, DatabaseMap *dbReadLog, const bool update = false, bool *keys = NULL , uint64_t level=0);
    zkresult write(const string &_key, const Goldilocks::Element* vkey, const vector<Goldilocks::Element> &value, const bool persistent);
    zkresult getProgram(const string &_key, vector<uint8_t> &value, DatabaseMap *dbReadLog);
    zkresult setProgram(const string &_key, const vector<uint8_t> &value, const bool persistent);
    inline bool usingAssociativeCache(void){ return useAssociativeCache; };

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
void *dbSenderThread(void *arg);

// Thread to synchronize cache from master hash DB server
void *dbCacheSynchThread(void *arg);

void loadDb2MemCache(const Config &config);

#endif