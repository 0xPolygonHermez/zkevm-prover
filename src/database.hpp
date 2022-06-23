#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <vector>
#include <map>
#include <pqxx/pqxx>
#include "goldilocks/goldilocks_base_field.hpp"
#include "compare_fe.hpp"
#include "config.hpp"
#include <semaphore.h>

using namespace std;

// Error codes
#define DB_SUCCESS 0
#define DB_KEY_NOT_FOUND 1
#define DB_DB_ERROR 2
#define DB_INTERNAL_ERROR 3

class Database
{
private:
    Goldilocks &fr;
    bool autoCommit;
    bool asyncWrite;
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
    map<string, vector<Goldilocks::Element>> dbNew; // Additions to the original db done through the execution of the prove or execute query
    map<string, vector<Goldilocks::Element>> dbRemote; // Data originally not present in local database that required fetching it from the remote database
    bool debug = false; //·

private:
    // Remote database based on Postgres (PostgreSQL)
    void initRemote (void);
    void readRemote (const string &key, vector<Goldilocks::Element> &value);
    void writeRemote (const string &key, const vector<Goldilocks::Element> &value);
    void addWriteQueue (string sqlWrite);
    void signalEmptyWriteQueue () {  };

public:
    Database(Goldilocks &fr, bool autoCommit, bool asyncWrite) : fr(fr), autoCommit(autoCommit), asyncWrite(asyncWrite) {};
    Database(Goldilocks &fr) : fr(fr), autoCommit(true), asyncWrite(false) {};    
    ~Database();
    void init (const Config &config);
    void read (const string &key, vector<Goldilocks::Element> &value);
    void write (const string &key, const vector<Goldilocks::Element> &value, const bool persistent);
    int setProgram (const string &key, const vector<uint8_t> &value, const bool persistent);
    int getProgram (const string &key, vector<uint8_t> &value);
    void processWriteQueue ();
    void commit();
    void flush ();    
    void print (void);
};

void* asyncDatabaseWriteThread (void* arg);

#endif