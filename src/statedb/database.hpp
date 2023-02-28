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

using namespace std;

class DatabaseMap;

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

    // Write connection attributes
    pqxx::connection * pConnectionWrite = NULL;
    pthread_mutex_t writeMutex; // Mutex to protect the write connection
    void writeLock(void) { pthread_mutex_lock(&writeMutex); };
    void writeUnlock(void) { pthread_mutex_unlock(&writeMutex); };

    // Read connection attributes
    pqxx::connection * pConnectionRead = NULL;
    pthread_mutex_t readMutex; // Mutex to protect the read connection
    void readLock(void) { pthread_mutex_lock(&readMutex); };
    void readUnlock(void) { pthread_mutex_unlock(&readMutex); };

    // Multi write attributes
    string multiWriteProgram;
    string multiWriteNodes;
    pthread_mutex_t multiWriteMutex; // Mutex to protect the multi write queues
    void multiWriteLock(void) { pthread_mutex_lock(&multiWriteMutex); };
    void multiWriteUnlock(void) { pthread_mutex_unlock(&multiWriteMutex); };

private:
    // Remote database based on Postgres (PostgreSQL)
    void initRemote(void);
    zkresult readRemote(bool bProgram, const string &key, string &value);
    zkresult writeRemote(bool bProgram, const string &key, const string &value);

public:
    // Cache static instances
    static DatabaseMTCache dbMTCache;
    static DatabaseProgramCache dbProgramCache;

    // Constructor and destructor
    Database(Goldilocks &fr, const Config &config) : fr(fr), config(config) {};
    ~Database();

    // Basic methods
    void init(void);
    zkresult read(const string &_key, vector<Goldilocks::Element> &value, DatabaseMap *dbReadLog);
    zkresult write(const string &_key, const vector<Goldilocks::Element> &value, const bool persistent);
    zkresult getProgram(const string &_key, vector<uint8_t> &value, DatabaseMap *dbReadLog);
    zkresult setProgram(const string &_key, const vector<uint8_t> &value, const bool persistent);

#ifdef DATABASE_COMMIT
    void setAutoCommit(const bool autoCommit);
    void commit();
#endif

    // Flush multi write pending requests
    void flush();

    // Print tree
    void printTree(const string &root, string prefix = "");
};

void loadDb2MemCache(const Config config);

#endif