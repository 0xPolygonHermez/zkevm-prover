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

using namespace std;

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

    // Local database based on a map attribute
    map<string, vector<Goldilocks::Element>> db; // This is in fact a map<fe,fe[16]>

public:
    map<string, vector<Goldilocks::Element>> dbReadLog; // Log data read from the database

private:
    // Remote database based on Postgres (PostgreSQL)
    void initRemote (void);
    zkresult readRemote (const string &key, vector<Goldilocks::Element> &value);
    zkresult writeRemote (const string &key, const vector<Goldilocks::Element> &value);
    void addWriteQueue (const string sqlWrite);
    void signalEmptyWriteQueue () {  };

public:
    Database(Goldilocks &fr) : fr(fr) {};
    ~Database();
    void init (const Config &config);
    zkresult read (const string &key, vector<Goldilocks::Element> &value);
    zkresult write (const string &key, const vector<Goldilocks::Element> &value, const bool persistent);
    zkresult setProgram (const string &key, const vector<uint8_t> &value, const bool persistent);
    zkresult getProgram (const string &key, vector<uint8_t> &value);
    void processWriteQueue ();
    void setAutoCommit (const bool autoCommit);
    void commit ();
    void flush ();    
    void clearDbReadLog ();
    void print (void);
    void printTree (const string &root, string prefix = "");
};

void* asyncDatabaseWriteThread (void* arg);

#endif