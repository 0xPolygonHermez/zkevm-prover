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

using namespace std;

class DatabaseMap;

class Database
{
private:
    Goldilocks &fr;
    bool autoCommit = true;
    bool asyncWrite = false;
    bool bInitialized = false;
    bool useRemoteDB = false;
    Config config;
    pthread_t writeThread;
    vector<string> writeQueue;
    pthread_mutex_t writeQueueMutex; // Mutex to protect writeQueue list
    pthread_cond_t writeQueueCond; // Cond to signal when queue has new items (no empty)
    pthread_cond_t emptyWriteQueueCond; // Cond to signal when queue is empty
    pqxx::connection * pConnectionWrite = NULL;
    pqxx::connection * pConnectionRead = NULL;
    pqxx::connection * pAsyncWriteConnection = NULL;
    pqxx::work* transaction = NULL;
    string multiWriteProgram;
    string multiWriteNodes;

private:
    // Remote database based on Postgres (PostgreSQL)
    void initRemote(void);
    zkresult readRemote(bool bProgram, const string &key, string &value);
    zkresult writeRemote(bool bProgram, const string &key, const string &value);
    void string2fea(const string os, vector<Goldilocks::Element> &fea);
    void string2ba(const string os, vector<uint8_t> &data);
    string removeBSXIfExists(string s) {return ((s.at(0) == '\\') && (s.at(1) == 'x')) ? s.substr(2) : s;};
    void addWriteQueue(const string sqlWrite);
    void signalEmptyWriteQueue() {};

public:
    static DatabaseMap dbCache; // Local database based on a map attribute
    static bool dbLoaded2Cache; // Indicates if we have already loaded the database into the mem cache for this process

    Database(Goldilocks &fr) : fr(fr) {};
    ~Database();
    void init(const Config &config);
    zkresult read(const string &_key, vector<Goldilocks::Element> &value, DatabaseMap *dbReadLog);
    zkresult write(const string &_key, const vector<Goldilocks::Element> &value, const bool persistent);
    zkresult getProgram(const string &_key, vector<uint8_t> &value, DatabaseMap *dbReadLog);
    zkresult setProgram(const string &_key, const vector<uint8_t> &value, const bool persistent);
    void loadDB2MemCache();
    void processWriteQueue();
    void setAutoCommit(const bool autoCommit);
    void commit();
    void flush();
    void print(void);
    void printTree(const string &root, string prefix = "");
};

void* asyncDatabaseWriteThread(void* arg);

#endif