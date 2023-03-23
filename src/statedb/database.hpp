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
    string multiWriteProgram;
    string multiWriteProgramUpdate;
    string multiWriteNodes;
    string multiWriteNodesUpdate;
    string multiWriteNodesStateRoot;
    pthread_mutex_t multiWriteMutex; // Mutex to protect the multi write queues
    void multiWriteLock(void) { pthread_mutex_lock(&multiWriteMutex); };
    void multiWriteUnlock(void) { pthread_mutex_unlock(&multiWriteMutex); };

private:
    // Remote database based on Postgres (PostgreSQL)
    void initRemote(void);
    zkresult readRemote(bool bProgram, const string &key, string &value);
    zkresult writeRemote(bool bProgram, const string &key, const string &value, const bool update);

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
    Database(Goldilocks &fr, const Config &config) :
        fr(fr),
        config(config),
        connectionsPool(NULL)
    {
        // Init mutexes
        pthread_mutex_init(&multiWriteMutex, NULL);
        pthread_mutex_init(&connMutex, NULL);
    };
    ~Database();

    // Basic methods
    void init(void);
    zkresult read(const string &_key, vector<Goldilocks::Element> &value, DatabaseMap *dbReadLog, const bool update = false);
    zkresult write(const string &_key, const vector<Goldilocks::Element> &value, const bool persistent, const bool update = false);
    zkresult getProgram(const string &_key, vector<uint8_t> &value, DatabaseMap *dbReadLog, const bool update = false);
    zkresult setProgram(const string &_key, const vector<uint8_t> &value, const bool persistent, const bool update = false);

#ifdef DATABASE_COMMIT
    void setAutoCommit(const bool autoCommit);
    void commit();
#endif

    // Flush multi write pending requests
    zkresult flush();

    // Print tree
    void printTree(const string &root, string prefix = "");
};

void loadDb2MemCache(const Config config);

#endif