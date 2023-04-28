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
#include "zkassert.hpp"

using namespace std;

class DatabaseConnection
{
public:
    pqxx::connection * pConnection;
    bool bInUse;
    DatabaseConnection() : pConnection(NULL), bInUse(false) {};
};

class MultiWriteData
{
public:
    string program;
    string programUpdate;
    string nodes;
    string nodesUpdate;
    string nodesStateRoot;

    uint64_t programCounter;
    uint64_t programUpdateCounter;
    uint64_t nodesCounter;
    uint64_t nodesUpdateCounter;
    uint64_t nodesStateRootCounter;

    void Reset (void)
    {
        // Reset strings
        program.clear();
        programUpdate.clear();
        nodes.clear();
        nodesUpdate.clear();
        nodesStateRoot.clear();

        // Reset counters
        programCounter = 0;
        programUpdateCounter = 0;
        nodesCounter = 0;
        nodesUpdateCounter = 0;
        nodesStateRootCounter = 0;
    }

    bool IsEmpty (void)
    {
        return (nodes.size() == 0) && (nodesUpdate.size() == 0) && (nodesStateRoot.size() == 0) && (program.size() == 0) && (programUpdate.size() == 0);
    }
};

class MultiWrite
{
public:

    uint64_t lastFlushId;
    uint64_t lastSentFlushId;
    uint64_t sendingFlushId;
    uint64_t processingDataIndex; // Index of data to store data of batches being processed
    uint64_t sendingDataIndex; // Index of data being sent to database

    MultiWriteData data[2];

    pthread_mutex_t mutex; // Mutex to protect the multi write queues
    
    // Constructor
    MultiWrite() :
        lastFlushId(0),
        lastSentFlushId(0),
        sendingFlushId(0),
        processingDataIndex(0),
        sendingDataIndex(1)
    {
        // Init mutex
        pthread_mutex_init(&mutex, NULL);

        // Reset data
        data[0].Reset();
        data[1].Reset();
    };

    // Lock/Unlock
    void Lock(void) { pthread_mutex_lock(&mutex); };
    void Unlock(void) { pthread_mutex_unlock(&mutex); };
    bool IsEmpty(void) { return data[0].IsEmpty() && data[1].IsEmpty(); };
};

class Database
{
private:
    Goldilocks &fr;
    const Config &config;

#ifdef DATABASE_COMMIT
    bool autoCommit = true;
#endif

    // Basic flags
    bool bInitialized = false;
    bool useRemoteDB = false;
    bool useDBMTCache = false;
    bool useDBProgramCache = false;

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

    // Multi write attributes
public:
    MultiWrite multiWrite;
    sem_t senderSem; // Semaphore to wakeup database sender thread when flush() is called
private:
    pthread_t senderPthread; // Database sender thread

private:
    // Remote database based on Postgres (PostgreSQL)
    void initRemote(void);
    zkresult readRemote(bool bProgram, const string &key, string &value);
    zkresult readTreeRemote(const string &key, const vector<uint64_t> *keys, uint64_t level, uint64_t &numberOfFields);
    zkresult writeRemote(bool bProgram, const string &key, const string &value, const bool update);
    zkresult writeGetTreeFunction(void);

public:
#ifdef DATABASE_USE_CACHE
    // Cache static instances
    static DatabaseMTCache dbMTCache;
    static DatabaseProgramCache dbProgramCache;

    // This is a fixed key to store the latest state root hash, used to load it to the cache
    // This key is "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    // This key cannot be the result of a hash because it is out of the Goldilocks Element range
    static string dbStateRootKey;

#endif

    // Constructor and destructor
    Database(Goldilocks &fr, const Config &config);
    ~Database();

    // Basic methods
    void init(void);
    zkresult read(const string &_key, vector<Goldilocks::Element> &value, DatabaseMap *dbReadLog, const bool update = false, const vector<uint64_t> *keys = NULL , uint64_t level=0);
    zkresult write(const string &_key, const vector<Goldilocks::Element> &value, const bool persistent, const bool update = false);
    zkresult getProgram(const string &_key, vector<uint8_t> &value, DatabaseMap *dbReadLog, const bool update = false);
    zkresult setProgram(const string &_key, const vector<uint8_t> &value, const bool persistent, const bool update = false);

#ifdef DATABASE_COMMIT
    void setAutoCommit(const bool autoCommit);
    void commit();
#endif

    // Flush multi write pending requests
    zkresult flush(uint64_t &flushId, uint64_t &lastSentFlushId);
    void getFlushStatus(uint64_t &lastSentFlushId, uint64_t &sendingFlushId, uint64_t &lastFlushId);

    // Send multi write data to remote database; called by dbSenderThread
    zkresult sendData(void);

    // Print tree
    void printTree(const string &root, string prefix = "");

    // Clear cache
    void clearCache(void);
};

void *dbSenderThread(void *arg);

void loadDb2MemCache(const Config config);

#endif