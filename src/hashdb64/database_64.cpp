#include <iostream>
#include <thread>
#include "database_64.hpp"
#include "config.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "definitions.hpp"
#include "zkresult.hpp"
#include "utils.hpp"
#include <unistd.h>
#include "timer.hpp"
#include "hashdb_singleton.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkmax.hpp"
#include "hashdb_remote.hpp"
#include "key_value.hpp"
#include "tree_64.hpp"

#ifdef DATABASE_USE_CACHE

// Create static Database64::dbMTCache and DatabaseCacheProgram objects
// This will be used to store DB records in memory and it will be shared for all the instances of Database class
// DatabaseCacheMT and DatabaseCacheProgram classes are thread-safe
DatabaseMTCache64 Database64::dbMTCache;
DatabaseProgramCache64 Database64::dbProgramCache;
DatabaseKVAssociativeCache Database64::dbKVACache;
DatabaseVersionsAssociativeCache Database64::dbVersionACache;
uint64_t Database64::latestVersionCache=0;


string Database64::dbStateRootKey("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"); // 64 f's
Goldilocks::Element Database64::dbStateRootvKey[4] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};


#endif

// Helper functions
string removeBSXIfExists64(string s) {return ((s.at(0) == '\\') && (s.at(1) == 'x')) ? s.substr(2) : s;}

Database64::Database64 (Goldilocks &fr, const Config &config) :
        fr(fr),
        config(config),
        connectionsPool(NULL),
        multiWrite(fr),
        maxVersions(config.kvDBMaxVersions),
        maxVersionsUpload(10) //add into config if needed
{
    // Init mutex
    pthread_mutex_init(&connMutex, NULL);

    // Initialize semaphores
    sem_init(&senderSem, 0, 0);
    sem_init(&getFlushDataSem, 0, 0);
};

Database64::~Database64()
{
    if (config.dbConnectionsPool)
    {
        if (connectionsPool != NULL)
        {
            for (uint64_t i=0; i<config.dbNumberOfPoolConnections; i++)
            {
                if (connectionsPool[i].pConnection != NULL)
                {
                    //zklog.info("Database64::~Database64() deleting writeConnectionsPool[" + to_string(i) + "].pConnection=" + to_string((uint64_t)writeConnectionsPool[i].pConnection));
                    delete[] connectionsPool[i].pConnection;
                }
            }
            delete connectionsPool;
        }
    }
    else
    {
        if (connection.pConnection != NULL)
        {
            delete connection.pConnection;
        }
    }
}

// Database64 class implementation
void Database64::init(void)
{
    // Check that it has not been initialized before
    if (bInitialized)
    {
        zklog.error("Database64::init() called when already initialized");
        exitProcess();
    }

    // Configure the server, if configuration is provided
    if (config.databaseURL != "local")
    {
        // Sender thread creation
        pthread_create(&senderPthread, NULL, dbSenderThread64, this);

        // Cache synchronization thread creation
        if (config.dbCacheSynchURL.size() > 0)
        {
            pthread_create(&cacheSynchPthread, NULL, dbCacheSynchThread64, this);

        }

        initRemote();
        useRemoteDB = true;
    }
    else
    {
        useRemoteDB = false;
    }

    // Mark the database as initialized
    bInitialized = true;
}

zkresult Database64::read (const string &_key, const Goldilocks::Element (&vkey)[4], string &value, DatabaseMap *dbReadLog, const bool update,  const bool *keys, const uint64_t level)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::read() called uninitialized");
        exitProcess();
    }

    struct timeval t;
    if (dbReadLog != NULL) gettimeofday(&t, NULL);

    zkresult r = ZKR_UNSPECIFIED;

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

#ifdef DATABASE_USE_CACHE
    // If the key is found in local database (cached) simply return it
    if( dbMTCache.enabled() && dbMTCache.find(key, value)){
        
        if (dbReadLog != NULL) dbReadLog->add(key, value, true, TimeDiff(t));
        r = ZKR_SUCCESS;
    }
    else
#endif
    // If the key is pending to be stored in database, but already deleted from cache
    if (config.dbMultiWrite && multiWrite.findNode(key, value))
    {
        // Add to the read log
        if (dbReadLog != NULL) dbReadLog->add(key, value, true, TimeDiff(t));

#ifdef DATABASE_USE_CACHE
        // Store it locally to avoid any future remote access for this key
        if(dbMTCache.enabled()){                
            dbMTCache.add(key, value, false);
        }
#endif
        r = ZKR_SUCCESS;
    }
    if (useRemoteDB && (r == ZKR_UNSPECIFIED))
    {
        // If multi write is enabled, flush pending data, since some previously written keys
        // could be in the multi write string but flushed from the cache
        /*if (config.dbMultiWrite)
        {
            flush(); // TODO: manage this situation
        }*/

        // Otherwise, read it remotelly, up to two times
        r = readRemote(false, key, value);
        if ( (r != ZKR_SUCCESS) && (config.dbReadRetryDelay > 0) )
        {
            for (uint64_t i=0; i<config.dbReadRetryCounter; i++)
            {
                zklog.warning("Database64::read() failed calling readRemote() with error=" + zkresult2string(r) + "; will retry after " + to_string(config.dbReadRetryDelay) + "us key=" + key + " i=" + to_string(i));

                // Retry after dbReadRetryDelay us
                usleep(config.dbReadRetryDelay);
                r = readRemote(false, key, value);
                if (r == ZKR_SUCCESS)
                {
                    break;
                }
                zklog.warning("Database64::read() retried readRemote() after dbReadRetryDelay=" + to_string(config.dbReadRetryDelay) + "us and failed with error=" + zkresult2string(r) + " i=" + to_string(i));
            }
        }
        if (r == ZKR_SUCCESS)
        {
#ifdef DATABASE_USE_CACHE
            // Store it locally to avoid any future remote access for this key
            if (dbMTCache.enabled())
            {
                dbMTCache.add(key, value, update);
            }
#endif

            // Add to the read log
            if (dbReadLog != NULL) dbReadLog->add(key, value, false, TimeDiff(t));
        }
    }

    // If we could not find the value, report the error
    if (r == ZKR_UNSPECIFIED)
    {
        zklog.error("Database64::read() requested a key that does not exist (ZKR_DB_KEY_NOT_FOUND): " + key);
        r = ZKR_DB_KEY_NOT_FOUND;
    }

#ifdef LOG_DB_READ
    {
        string s = "Database64::read()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(r);
        s += " key=" + key;
        s += " value=";
        for (uint64_t i = 0; i < value.size(); i++)
            s += fr.toString(value[i], 16) + ":";
        zklog.info(s);
    }
#endif

    return r;
}

zkresult Database64::read (vector<DB64Query> &dbQueries)
{
    zkresult zkr;
    for (uint64_t i=0; i<dbQueries.size(); i++)
    {
        zkr = read(dbQueries[i].key, dbQueries[i].keyFea, dbQueries[i].value, NULL);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::read(DBQueries) failed calling read() result=" + zkresult(zkr));
            return zkr;
        }
    }
    return ZKR_SUCCESS;
}

zkresult Database64::write(const string &_key, const Goldilocks::Element* vkey, const string &value, const bool persistent)
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::write() called uninitialized");
        exitProcess();
    }

    if (config.dbMultiWrite && !dbMTCache.enabled() && !persistent)
    {
        zklog.error("Database64::write() called with multi-write active, cache disabled and no persistance in database, so there is no place to store the date");
        return ZKR_DB_ERROR;
    }

    zkresult r;

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

    if ( useRemoteDB
#ifdef DATABASE_USE_CACHE
         && persistent
#endif
         )
    {
        r = writeRemote(false, key, value);
    }
    else
    {
        r = ZKR_SUCCESS;
    }

#ifdef DATABASE_USE_CACHE
    if ((r == ZKR_SUCCESS) && dbMTCache.enabled() )
    {
        dbMTCache.add(key, value, false);
    }
#endif

#ifdef LOG_DB_WRITE
    {
        string s = "Database64::write()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(r);
        s += " key=" + key;
        s += " value=";
        for (uint64_t i = 0; i < value.size(); i++)
            s += fr.toString(value[i], 16) + ":";
        s += " persistent=" + to_string(persistent);
        zklog.info(s);
    }
#endif

    return r;
}

zkresult Database64::write(const vector<DB64Query> &dbQueries, const bool persistent)
{
    zkresult zkr;
    for (uint64_t i=0; i<dbQueries.size(); i++)
    {
        //zklog.info("Database64::write() writing hash=" + dbQueries[i].key + " size=" + to_string(dbQueries[i].value.size()) + " data=" + ba2string(dbQueries[i].value));
        zkr = write(dbQueries[i].key, dbQueries[i].keyFea, dbQueries[i].value, persistent);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::write(DBQuery) failed calling write() result=" + zkresult(zkr));
            return zkr;
        }
    }
    return ZKR_SUCCESS;
}

zkresult Database64::readKV(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, uint64_t &level ,DatabaseMap *dbReadLog)
{
    level = 128;
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::readKV() called uninitialized");
        exitProcess();
    }

    struct timeval t;
    if (dbReadLog != NULL) gettimeofday(&t, NULL);

    zkresult rkv = ZKR_UNSPECIFIED;
    zkresult rv = ZKR_UNSPECIFIED;
    zkresult rout = ZKR_UNSPECIFIED;

    string keyStr = "";
    if(dbReadLog != NULL && dbReadLog->getSaveKeys()){
        string keyStr_ = fea2string(fr, key[0], key[1], key[2], key[3]); 
        keyStr = NormalizeToNFormat(keyStr_, 64);
    }
    
    uint64_t version;
    rv = readVersion(root, version, dbReadLog);

    if( rv == ZKR_SUCCESS){
       
        // If the key is found in local database (cached) simply return it
        if(dbKVACache.findKey(version, key, value)){

            if (dbReadLog != NULL) dbReadLog->add(keyStr, value, true, TimeDiff(t));
            rkv = ZKR_SUCCESS;

        } 
        // If the key is pending to be stored in database, but already deleted from cache
        else if (config.dbMultiWrite && multiWrite.findKeyValue(version, key, value))
        {
            // Add to the read log
            if (dbReadLog != NULL) dbReadLog->add(keyStr, value, true, TimeDiff(t));
            // We do not store into cache as we do not want to manage the chain of versions
            rkv = ZKR_SUCCESS;
        }
        else if(useRemoteDB)
        {
            vector<VersionValue> upstreamVersionValues;
            rkv = readRemoteKV(version, key, value, upstreamVersionValues);       
            if (rkv == ZKR_SUCCESS)
            {
                dbKVACache.uploadKeyValueVersions(key, upstreamVersionValues);               
                if (dbReadLog != NULL) dbReadLog->add(keyStr, value, false, TimeDiff(t));
            } else {
                
                if( rkv == ZKR_DB_KEY_NOT_FOUND){
                    rout = rkv;
                    // Add a zero into the cache to avoid future remote access for this key (not problematic management of versions as there is only one version)
                    mpz_class   zero(0);
                    dbKVACache.addKeyValueVersion(0, key, zero);
                }else if( rkv == ZKR_DB_VERSION_NOT_FOUND_GLOBAL){
                    rout = rkv;
                    // Add a zero into the cache to avoid future remote access for this key (not problematic management of versions as there is only one version)
                    dbKVACache.uploadKeyValueVersions(key, upstreamVersionValues);
                }else{
                    zkresult rtree = ZKR_UNSPECIFIED;
                    vector<KeyValue> keyValues(1);
                    keyValues[0].key[0] = key[0];
                    keyValues[0].key[1] = key[1];
                    keyValues[0].key[2] = key[2];
                    keyValues[0].key[3] = key[3];
                    rtree = tree64.ReadTree(*this, root, keyValues, NULL);
                    value = keyValues[0].value;
                    rout = rtree;      
                }
            }
        }
    } else {
        zklog.warning("Database64::readKV() requested a root that does not exist in the table statedb.version " + keyStr + " , "
        + zkresult2string(rv) );
        rout = rv;
    }
    
#ifdef LOG_DB_READ
    {
        string s = "Database64::readKV()";
        if (rout != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(rout);
        s += " key=" + keyStr;
        s += " value=";
        s += value.get_str(16) + ";";
        zklog.info(s);
    }
#endif
    return rout;

}

zkresult Database64::readKV(const Goldilocks::Element (&root)[4], vector<KeyValueLevel> &KVLs, DatabaseMap *dbReadLog){
    zkresult zkr;
    for (uint64_t i=0; i<KVLs.size(); i++)
    {
        zkr = readKV(root, KVLs[i].key, KVLs[i].value, KVLs[i].level, dbReadLog);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::readKV(KBs) failed calling read() result=" + zkresult2string(zkr) + " key=" + fea2string(fr, KVLs[i].key) );
            return zkr;
        }
    }
    return ZKR_SUCCESS;
}

zkresult Database64::writeKV(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, bool persistent)
{

    
    zkresult rkv = ZKR_UNSPECIFIED;
    zkresult rv = ZKR_UNSPECIFIED;

    uint64_t version;
    rv = readVersion(root, version, NULL);
    if( rv == ZKR_SUCCESS){
        writeKV(version, key, value, persistent);
    }else{
       rkv=rv;
    }

#ifdef LOG_DB_WRITE
    {
        string rootStr_ = fea2string(fr, root[0], root[1], root[2], root[3]); 
        string rootStr = NormalizeToNFormat(rootStr_, 64);
        string s = "Database64::writeKV()";
        if (rkv != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(rkv);
        s += " key=" + rootStr;
        s += " version=";
        s += to_string(version) + ":";
        s += " persistent=" + to_string(persistent);
        zklog.info(s);
    }
#endif

    return rkv;
}

zkresult Database64::writeKV(const uint64_t& version, const Goldilocks::Element (&key)[4], const mpz_class &value, bool persistent)
{

    // Check that it has  been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::writeKV() called uninitialized");
        exitProcess();
    }
    if (config.dbMultiWrite && !dbKVACache.enabled() && !persistent)
    {
        zklog.error("Database64::writeKV() called with multi-write active, cache disabled and no persistance in database, so there is no place to store the date");
        return ZKR_DB_ERROR;
    }
    zkresult rkv = ZKR_UNSPECIFIED;
    if ( useRemoteDB && persistent)
    {
        
        rkv = writeRemoteKV(version, key, value);
    }
    else
    {
        rkv = ZKR_SUCCESS;
    }

    if (rkv == ZKR_SUCCESS)
    {
        dbKVACache.addKeyValueVersion(version, key, value);
        
    }

#ifdef LOG_DB_WRITE
    {
        string keyStr_ = fea2string(fr, key[0], key[1], key[2], key[3]); 
        string keyStr = NormalizeToNFormat(keyStr_, 64);
        string s = "Database64::writeKV()";
        if (rkv != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(rkv);
        s += " key=" + keyStr;
        s += " value=";
        s += value.get_str(16);
        s += " persistent=" + to_string(persistent);
        zklog.info(s);
    }
#endif

    return rkv;
}

zkresult Database64::writeKV(const Goldilocks::Element (&root)[4], const vector<KeyValue> &KVs, bool persistent){
    zkresult zkr;
    for (uint64_t i=0; i<KVs.size(); i++)
    {
        zkr = writeKV(root, KVs[i].key, KVs[i].value, persistent);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::writeKV(KBs) failed calling write() result=" + zkresult2string(zkr) + " key=" + fea2string(fr, KVs[i].key) );
            return zkr;
        }
    }
    return ZKR_SUCCESS;
}

zkresult Database64::writeKV(const uint64_t& version, const vector<KeyValue> &KVs, bool persistent){
    zkresult zkr;
    for (uint64_t i=0; i<KVs.size(); i++)
    {
        zkr = writeKV(version, KVs[i].key, KVs[i].value, persistent);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::writeKV(KBs) failed calling write() result=" + zkresult2string(zkr) + " key=" + fea2string(fr, KVs[i].key) );
            return zkr;
        }
    }
    return ZKR_SUCCESS;
}

zkresult Database64::readVersion(const Goldilocks::Element (&root)[4], uint64_t& version, DatabaseMap *dbReadLog){

    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::readVersion() called uninitialized");
        exitProcess();
    }

    struct timeval t;
    if (dbReadLog != NULL) gettimeofday(&t, NULL);
   
    zkresult r = ZKR_UNSPECIFIED;

    string rootStr = "";
    if(dbReadLog != NULL &&  dbReadLog->getSaveKeys()){
        string rootStr_ = fea2string(fr, root[0], root[1], root[2], root[3]); 
        rootStr = NormalizeToNFormat(rootStr_, 64);
    }

    // If the key is found in local database (cached) simply return it
    if(dbVersionACache.findKey(root, version)){

        if (dbReadLog != NULL) dbReadLog->add(rootStr, version, true, TimeDiff(t));
        r = ZKR_SUCCESS;

    }else{
        if(dbReadLog==NULL || !dbReadLog->getSaveKeys()){
            string rootStr_ = fea2string(fr, root[0], root[1], root[2], root[3]); 
            rootStr = NormalizeToNFormat(rootStr_, 64);
        } 
        // If the key is pending to be stored in database, but already deleted from cache
        if (config.dbMultiWrite && multiWrite.findVersion(rootStr, version))
        {
            // Add to the read log
            if (dbReadLog != NULL) dbReadLog->add(rootStr, version, true, TimeDiff(t));

            // Store it locally to avoid any future remote access for this key
            
            r = ZKR_SUCCESS;

        }else if(useRemoteDB){
        
            r = readRemoteVersion(root, version);
            if ( (r != ZKR_SUCCESS) && (config.dbReadRetryDelay > 0) )
            {
                for (uint64_t i=0; i<config.dbReadRetryCounter; i++)
                {
                    zklog.warning("Database64::readVersion() failed calling readRemote() with error=" + zkresult2string(r) + "; will retry after " + to_string(config.dbReadRetryDelay) + "us key=" + rootStr + " i=" + to_string(i));

                    // Retry after dbReadRetryDelay us
                    usleep(config.dbReadRetryDelay);
                    r = readRemoteVersion(root, version);
                    if (r == ZKR_SUCCESS)
                    {
                        break;
                    }
                    zklog.warning("Database64::readVersion() retried readRemote() after dbReadRetryDelay=" + to_string(config.dbReadRetryDelay) + "us and failed with error=" + zkresult2string(r) + " i=" + to_string(i));
                }
            }        
            if (r == ZKR_SUCCESS)
            {
                dbVersionACache.addKeyVersion(root, version);
                
                // Add to the read log
                if (dbReadLog != NULL) dbReadLog->add(rootStr, version, false, TimeDiff(t));
            }
        }
    }
    // If we could not find the value, report the error
    if (r == ZKR_UNSPECIFIED)
    {
        zklog.error("Database64::readVersion() requested a key that does not exist (ZKR_DB_KEY_NOT_FOUND): " + rootStr);
        r = ZKR_DB_KEY_NOT_FOUND;
    }

#ifdef LOG_DB_READ 
    {
        if(!dbReadLog->getSaveKeys()){
            string rootStr_ = fea2string(fr, root[0], root[1], root[2], root[3]); 
            rootStr = NormalizeToNFormat(rootStr_, 64);
        }
        string s = "Database64::readKV()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(r);
        s += " key=" + rootStr;
        s += " version=";
        s += to_string(version) + ":";
        zklog.info(s);
    }
#endif
    return r;
}

zkresult Database64::writeVersion(const Goldilocks::Element (&root)[4], const uint64_t version, bool persistent){

    if (!bInitialized)
    {
        zklog.error("Database64::writeVersion() called uninitialized");
        exitProcess();
    }
    if (config.dbMultiWrite && ! dbVersionACache.enabled() && !persistent)
    {
        zklog.error("Database64::writeVersion() called with multi-write active, cache disabled and no persistance in database, so there is no place to store the date");
        return ZKR_DB_ERROR;
    }
    zkresult r;


    if ( useRemoteDB && persistent)
    {
        
        r = writeRemoteVersion(root, version);
    }
    else
    {
        r = ZKR_SUCCESS;
    }

    if (r == ZKR_SUCCESS)
    {
        dbVersionACache.addKeyVersion(root, version);
        
    }

#ifdef LOG_DB_WRITE
    {
        string rootStr_ = fea2string(fr, root[0], root[1], root[2], root[3]); 
        string rootStr = NormalizeToNFormat(rootStr_, 64);
        string s = "Database64::writeKV()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(r);
        s += " root=" + rootStr;
        s += " version=";
        s += to_string(version);
        s += " persistent=" + to_string(persistent);
        zklog.info(s);
    }
#endif

    return r;
}

zkresult Database64::readLatestVersion(uint64_t &version){

    zkresult r=ZKR_UNSPECIFIED;
    if (!bInitialized)
    {
        zklog.error("Database64::readLatestVersion() called uninitialized");
        exitProcess();
    }
    if(latestVersionCache != 0){
        version = latestVersionCache;
        return ZKR_SUCCESS;
    }else{
        r = readRemoteLatestVersion(version);
        if(r == ZKR_SUCCESS){
            latestVersionCache = version;
        }
        return r;
    }
    return ZKR_SUCCESS;
}

zkresult Database64::writeLatestVersion(const uint64_t version, const bool persistent)
{

    if (!bInitialized)
    {
        zklog.error("Database64::writeLatestVersion() called uninitialized");
        exitProcess();
    }
    
    zkresult r;

    if ( useRemoteDB && persistent)
    {
        
        r = writeRemoteLatestVersion(version);
    }
    else
    {
        r = ZKR_SUCCESS;
    }

    if (r == ZKR_SUCCESS)
    {
        latestVersionCache=version;
        
    }
    return r;
}

void Database64::initRemote(void)
{
    TimerStart(DB_INIT_REMOTE);

    try
    {
        // Build the remote database URI
        string uri = config.databaseURL;
        //zklog.info("Database64 URI: " + uri);

        // Create the database connections
        connLock();

        if (config.dbConnectionsPool)
        {
            // Check that we don't support more threads than available connections, including the sender thread
            if (config.dbNumberOfPoolConnections == 0)
            {
                zklog.error("Database64::initRemote() found config.dbNumberOfPoolConnections=" + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }
            if ( config.runHashDBServer && ((config.maxHashDBThreads + 1) > config.dbNumberOfPoolConnections) )
            {
                zklog.error("Database64::initRemote() found config.maxHashDBThreads + 1=" + to_string(config.maxHashDBThreads + 1) + " > config.dbNumberOfPoolConnections=" + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }
            if ( config.runExecutorServer && ((config.maxExecutorThreads + 1) > config.dbNumberOfPoolConnections) )
            {
                zklog.error("Database64::initRemote() found config.maxExecutorThreads + 1=" + to_string(config.maxExecutorThreads + 1) + " > config.dbNumberOfPoolConnections=" + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }
            if ( config.runHashDBServer && config.runExecutorServer && ((config.maxHashDBThreads + config.maxExecutorThreads + 1) > config.dbNumberOfPoolConnections) )
            {
                zklog.error("Database64::initRemote() found config.maxHashDBThreads + config.maxExecutorThreads + 1=" + to_string(config.maxHashDBThreads + config.maxExecutorThreads + 1) + " > config.dbNumberOfPoolConnections=" + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }

            // Allocate write connections pool
            connectionsPool = new DatabaseConnection[config.dbNumberOfPoolConnections];
            if (connectionsPool == NULL)
            {
                zklog.error("Database64::initRemote() failed creating write connection pool of size " + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }

            // Create write connections
            for (uint64_t i=0; i<config.dbNumberOfPoolConnections; i++)
            {
                connectionsPool[i].pConnection = new pqxx::connection{uri};
                if (connectionsPool[i].pConnection == NULL)
                {
                    zklog.error("Database64::initRemote() failed creating write connection " + to_string(i));
                    exitProcess();
                }
                connectionsPool[i].bInUse = false;
                //zklog.info("Database64::initRemote() created write connection i=" + to_string(i) + " connectionsPool[i]=" + to_string((uint64_t)connectionsPool[i].pConnection));
            }

            // Reset counters
            nextConnection = 0;
            usedConnections = 0;
        }
        else
        {
            connection.pConnection = new pqxx::connection{uri};
            if (connection.pConnection == NULL)
            {
                zklog.error("Database64::initRemote() failed creating unique connection");
                exitProcess();
            }
            connection.bInUse = false;
        }
        
        connUnlock();
    }
    catch (const std::exception &e)
    {
        zklog.error("Database64::initRemote() exception: " + string(e.what()));
        exitProcess();
    }

    // Create state root, only useful if database is empty
    if (!config.dbReadOnly)
    {
        createStateRoot();
    }

    TimerStopAndLog(DB_INIT_REMOTE);
}

DatabaseConnection * Database64::getConnection (void)
{
    if (config.dbConnectionsPool)
    {
        connLock();
        DatabaseConnection * pConnection = NULL;
        uint64_t i=0;
        for (i=0; i<config.dbNumberOfPoolConnections; i++)
        {
            if (!connectionsPool[nextConnection].bInUse) break;
            nextConnection++;
            if (nextConnection == config.dbNumberOfPoolConnections)
            {
                nextConnection = 0;
            }
        }
        if (i==config.dbNumberOfPoolConnections)
        {
            zklog.error("Database64::getWriteConnection() run out of free connections");
            exitProcess();
        }

        pConnection = &connectionsPool[nextConnection];
        zkassert(pConnection->bInUse == false);
        pConnection->bInUse = true;
        nextConnection++;
        if (nextConnection == config.dbNumberOfPoolConnections)
        {
            nextConnection = 0;
        }
        usedConnections++;
        if (pConnection->bDisconnect)
        {
            pConnection->pConnection->disconnect();
            pConnection->bDisconnect = false;
        }
        //zklog.info("Database64::getWriteConnection() pConnection=" + to_string((uint64_t)pConnection) + " nextConnection=" + to_string(nextConnection) + " usedConnections=" + to_string(usedConnections));
        connUnlock();
        return pConnection;
    }
    else
    {
        connLock();
        zkassert(connection.bInUse == false);
#ifdef DEBUG
        connection.bInUse = true;
#endif
        return &connection;
    }
}

void Database64::disposeConnection (DatabaseConnection * pConnection)
{
    if (config.dbConnectionsPool)
    {
        connLock();
        zkassert(pConnection->bInUse == true);
        pConnection->bInUse = false;
        zkassert(usedConnections > 0);
        usedConnections--;
        //zklog.info("Database64::disposeWriteConnection() pConnection=" + to_string((uint64_t)pConnection) + " nextConnection=" + to_string(nextConnection) + " usedConnections=" + to_string(usedConnections));
        connUnlock();
    }
    else
    {
        zkassert(pConnection == &connection);
        zkassert(pConnection->bInUse == true);
#ifdef DEBUG
        pConnection->bInUse = false;
#endif
        connUnlock();
    }
}

void Database64::queryFailed (void)
{
    connLock();

    for (uint64_t i=0; i<config.dbNumberOfPoolConnections; i++)
    {
        connectionsPool[i].bDisconnect = true;
    }

    connUnlock();
}

zkresult Database64::readRemote(bool bProgram, const string &key, string &value)
{
    const string &tableName = (bProgram ? config.dbProgramTableName : config.dbNodesTableName);

    if (config.logRemoteDbReads)
    {
        zklog.info("Database64::readRemote() table=" + tableName + " key=" + key);
    }

    // Get a free read db connection
    DatabaseConnection * pDatabaseConnection = getConnection();

    try
    {
        // Prepare the query
        string query = "SELECT * FROM " + tableName + " WHERE hash = E\'\\\\x" + key + "\';";

        pqxx::result rows;

        // Start a transaction.
        pqxx::nontransaction n(*(pDatabaseConnection->pConnection));

        // Execute the query
        rows = n.exec(query);

        // Commit your transaction
        n.commit();

        // Process the result
        if (rows.size() == 0)
        {
            disposeConnection(pDatabaseConnection);
            return ZKR_DB_KEY_NOT_FOUND;
        }
        else if (rows.size() > 1)
        {
            zklog.error("Database64::readRemote() table=" + tableName + " got more than one row for the same key: " + to_string(rows.size()));
            exitProcess();
        }

        pqxx::row const row = rows[0];
        if (row.size() != 2)
        {
            zklog.error("Database64::readRemote() table=" + tableName + " got an invalid number of colums for the row: " + to_string(row.size()));
            exitProcess();
        }
        pqxx::field const fieldData = row[1];
        value = string2ba(removeBSXIfExists64(fieldData.c_str()));
    }
    catch (const std::exception &e)
    {
        zklog.error("Database64::readRemote() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
        queryFailed();
        disposeConnection(pDatabaseConnection);
        return ZKR_DB_ERROR;
    }
    
    // Dispose the read db conneciton
    disposeConnection(pDatabaseConnection);

    return ZKR_SUCCESS;
}

zkresult Database64::writeRemote(bool bProgram, const string &key, const string &value)
{
    zkresult result = ZKR_SUCCESS;
    
    if (config.dbMultiWrite)
    {
        multiWrite.Lock();

        if (bProgram)
        {
            multiWrite.data[multiWrite.pendingToFlushDataIndex].programIntray[key] = value;
#ifdef LOG_DB_WRITE_REMOTE
            zklog.info("Database64::writeRemote() key=" + key + " multiWrite=[" + multiWrite.print() + "]");
#endif
        }
        else
        {
            multiWrite.data[multiWrite.pendingToFlushDataIndex].nodesIntray[key] = value;
        }

        multiWrite.Unlock();
    }
    else
    {
        const string &tableName = (bProgram ? config.dbProgramTableName : config.dbNodesTableName);

        string query = "INSERT INTO " + tableName + " ( hash, data ) VALUES ( E\'\\\\x" + key + "\', E\'\\\\x" + value + "\' ) ON CONFLICT (hash) DO NOTHING;";
            
        DatabaseConnection * pDatabaseConnection = getConnection();

        try
        {        

#ifdef DATABASE_COMMIT
            if (autoCommit)
#endif
            {
                pqxx::work w(*(pDatabaseConnection->pConnection));
                pqxx::result res = w.exec(query);
                w.commit();
            }
#ifdef DATABASE_COMMIT
            else
            {
                if (transaction == NULL)
                    transaction = new pqxx::work{*pConnectionWrite};
                pqxx::result res = transaction->exec(query);
            }
#endif
        }
        catch (const std::exception &e)
        {
            zklog.error("Database64::writeRemote() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
            result = ZKR_DB_ERROR;
            queryFailed();
        }

        disposeConnection(pDatabaseConnection);
    }

    return result;
}

zkresult Database64::readRemoteKV(const uint64_t version, const Goldilocks::Element (&key)[4],  mpz_class& value, vector<VersionValue> &upstreamVersionValues)
{
    const string &tableName = config.dbKeyValueTableName;

    string keyStr_ = fea2string(fr, key[0], key[1], key[2], key[3]); 
    string keyStr = NormalizeToNFormat(keyStr_, 64);

    if (config.logRemoteDbReads)
    {
        zklog.info("Database64::readRemoteKV() table=" + tableName + " key=" + keyStr);
    }

    // Get a free read db connection
    DatabaseConnection * pDatabaseConnection = getConnection();

    try
    {
        // Prepare the query
        string query = "SELECT * FROM " + tableName + " WHERE key = E\'\\\\x" + keyStr + "\';";

        pqxx::result rows;

        // Start a transaction.
        pqxx::nontransaction n(*(pDatabaseConnection->pConnection));

        // Execute the query
        rows = n.exec(query);

        // Commit your transaction
        n.commit();

        // Process the result
        if (rows.size() == 0)
        {
            disposeConnection(pDatabaseConnection);
            return ZKR_DB_KEY_NOT_FOUND;
        }
        else if (rows.size() > 1)
        {
            zklog.error("Database64::readRemoteKV() table=" + tableName + " got more than one row for the same key: " + to_string(rows.size()) + "for key=" + keyStr);
            exitProcess();
        }

        pqxx::row const row = rows[0];
        if (row.size() != 2)
        {
            zklog.error("DatabaseKV::readRemoteKV() table=" + tableName + " got an invalid number of colums for the row: " + to_string(row.size()) + "for key=" + keyStr);
            exitProcess();
        }

        // Dispose the read db conneciton
        disposeConnection(pDatabaseConnection);
        return extractVersion(row[1], version, value, upstreamVersionValues); 
        

    }
    catch (const std::exception &e)
    {
        zklog.error("Database64::readRemoteKV() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
        queryFailed();
        disposeConnection(pDatabaseConnection);
        return ZKR_DB_ERROR;
    }
    
    // Dispose the read db conneciton
    disposeConnection(pDatabaseConnection);

    return ZKR_DB_ERROR;//should never return here
}

zkresult Database64::extractVersion(const pqxx::field& fieldData, const uint64_t version, mpz_class &value, vector<VersionValue> &upstreamVersionValues){

    upstreamVersionValues.clear();
    int data_size = 0;
    if(!fieldData.is_null()){
        string data = removeBSXIfExists64(fieldData.c_str());
        data_size = data.size();
        zkassert(data_size % 80 == 0);
        int nUpstreams = 0;
        for (int i = 0; i < data_size; i += 80)
        {

            string versionStr = data.substr(i, 16);
            mpz_class aux(versionStr, 16);
            uint64_t version_ = aux.get_ui();
            if(nUpstreams < maxVersionsUpload){
                VersionValue vv;
                vv.version = version_;
                vv.value = mpz_class(data.substr(i + 16, 64), 16);
                upstreamVersionValues.push_back(vv);
            }
            if(version >= version_){
                value = mpz_class(data.substr(i + 16, 64), 16);
                return ZKR_SUCCESS;
            }
            ++nUpstreams;
        }
    }
    /*const char * data = fieldData.c_str();
    int data_size = fieldData.size();
    zkassert(data_size % 40 == 0);
    for (int i = 0; i < data_size; i += 40)
    {
        uint64_t version_;
        std::memcpy(&version_, data + i, 8);
        if(version_==version){
            value = mpz_class(data + i + 8, 32);
            return true;
        }
    }*/
    if(data_size < 80*maxVersions){
        return ZKR_DB_VERSION_NOT_FOUND_GLOBAL;
    }else{
        if(data_size != 80*maxVersions){
            zklog.error("Database64::extractVersion() got an invalid data size: " + to_string(data_size));
            exitProcess();
        }
        return ZKR_DB_VERSION_NOT_FOUND_KVDB;
    }
}

zkresult Database64::writeRemoteKV(const uint64_t version, const Goldilocks::Element (&key)[4], const mpz_class &value, bool useMultiWrite)
{
    zkresult result = ZKR_SUCCESS; 
    if (config.dbMultiWrite && useMultiWrite)
    {
        multiWrite.Lock();
        KeyValue auxKV;
        auxKV.key[0] = key[0];
        auxKV.key[1] = key[1];
        auxKV.key[2] = key[2];
        auxKV.key[3] = key[3];
        auxKV.value = value;
        multiWrite.data[multiWrite.pendingToFlushDataIndex].keyValueAIntray[version].push_back(auxKV);
        string keyStr_ = fea2string(fr, key[0], key[1], key[2], key[3]);
        string keyStr = NormalizeToNFormat(keyStr_, 64);
        multiWrite.data[multiWrite.pendingToFlushDataIndex].keyVersionsIntray[keyStr].push_back(version);
#ifdef LOG_DB_WRITE_REMOTE
            zklog.info("Database64::writeRemote() version=" + to_string(version) + " multiWrite=[" + multiWrite.print() + "]");
#endif
        multiWrite.Unlock();
    }
    else
    {
        const string &tableName = config.dbKeyValueTableName;
        string keyStr_ = fea2string(fr, key[0], key[1], key[2], key[3]); 
        string keyStr = NormalizeToNFormat(keyStr_, 64);
        string valueStr = NormalizeToNFormat(value.get_str(16),64);
        string versionStr = NormalizeToNFormat(U64toString(version,16),16); 
        string insertStr = versionStr + valueStr;
        assert(insertStr.size() == 80);

        if (config.logRemoteDbReads)
        {
            zklog.info("Database64::writeRemoteKV() table=" + tableName + " key=" + keyStr);
        }

        // Get a free read db connection
        DatabaseConnection * pDatabaseConnection = getConnection();

        try
        {
            // Prepare the query
            string query = "SELECT * FROM " + tableName + " WHERE key = E\'\\\\x" + keyStr + "\';";

            pqxx::result rows;

            // Start a transaction.
            pqxx::nontransaction n(*(pDatabaseConnection->pConnection));

            // Execute the query
            rows = n.exec(query);

            // Commit your transaction
            n.commit();

            // Process the result
            if(rows.size() > 0){
                if (rows.size() > 1)
                {
                    zklog.error("Database64::writeRemoteKV() table=" + tableName + " got more than one row for the same key: " + to_string(rows.size()) + "for key=" + keyStr);
                    exitProcess();
                }

                const pqxx::row&  row = rows[0];
                if (row.size() != 2)
                {
                    zklog.error("DatabaseKV::writeRemoteKV() table=" + tableName + " got an invalid number of colums for the row: " + to_string(row.size()) + "for key=" + keyStr);
                    exitProcess();
                }
                const pqxx::field& fieldData = row[1];
                //processar insert;
                if(!fieldData.is_null()){
                    string data = removeBSXIfExists64(fieldData.c_str());
                    int data_size = data.size();
                    if(data_size == maxVersions*80){
                        data = data.substr(0, data_size - 80);
                    }
                    insertStr = insertStr + data;
                }
            } else{
                string valueZero = NormalizeToNFormat("0",64);
                string versionZero = NormalizeToNFormat(U64toString(0,16),16); 
                string insertZero = versionStr + valueStr;
                insertStr = insertStr + versionZero + valueZero;
                assert(insertStr.size() == 160);
                mpz_class   zero(0);
                dbKVACache.downstreamAddKeyZeroVersion(version, key);                
                
            }
        }
        catch (const std::exception &e)
        {
            zklog.error("Database64::writeRemoteKV() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
            queryFailed();
            disposeConnection(pDatabaseConnection);
            return ZKR_DB_ERROR;
        }

            
        try
        {        
            string query = "INSERT INTO " + tableName + " ( key, data ) VALUES ( E\'\\\\x" + keyStr + "\', E\'\\\\x" + insertStr + "\' ) ON CONFLICT (key) DO UPDATE SET data = E'\\\\x" + insertStr + "\';";
#ifdef DATABASE_COMMIT
            if (autoCommit)
#endif
            {
                pqxx::work w(*(pDatabaseConnection->pConnection));
                pqxx::result res = w.exec(query);
                w.commit();
            }
#ifdef DATABASE_COMMIT
            else
            {
                if (transaction == NULL)
                    transaction = new pqxx::work{*pConnectionWrite};
                pqxx::result res = transaction->exec(query);
            }
#endif
        }
        catch (const std::exception &e)
        {
            zklog.error("Database::writeRemoteVK() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
            result = ZKR_DB_ERROR;
            queryFailed();
        }

        disposeConnection(pDatabaseConnection);

    }
    return result;
}

zkresult Database64::readRemoteVersion(const Goldilocks::Element (&root)[4], uint64_t& version){
    
    const string &tableName = config.dbVersionTableName;

    string rootStr_ = fea2string(fr, root[0], root[1], root[2], root[3]); 
    string rootStr = NormalizeToNFormat(rootStr_, 64);

    if (config.logRemoteDbReads)
    {
        zklog.info("Database64::readRemoteVersion() table=" + tableName + " key=" + rootStr);
    }

    // Get a free read db connection
    DatabaseConnection * pDatabaseConnection = getConnection();

    try
    {
        // Prepare the query
        string query = "SELECT * FROM " + tableName + " WHERE hash = E\'\\\\x" + rootStr + "\';";

        pqxx::result rows;

        // Start a transaction.
        pqxx::nontransaction n(*(pDatabaseConnection->pConnection));

        // Execute the query
        rows = n.exec(query);

        // Commit your transaction
        n.commit();

        // Process the result
        if (rows.size() == 0)
        {
            disposeConnection(pDatabaseConnection);
            return ZKR_DB_KEY_NOT_FOUND;
        }
        else if (rows.size() > 1)
        {
            zklog.error("Database64::readRemoteVersion() table=" + tableName + " got more than one row for the same key: " + to_string(rows.size()) + "for key: " + rootStr);
            exitProcess();
        }

        pqxx::row const row = rows[0];
        if (row.size() != 2)
        {
            zklog.error("DatabaseKV::readRemoteVersion() table=" + tableName + " got an invalid number of colums for the row: " + to_string(row.size()) + "for key: " + rootStr);
            exitProcess();
        }
        pqxx::field const fieldData = row[1];
        if (!fieldData.is_null()) {
            version = fieldData.as<uint64_t>();
        } else {
            zklog.error("DatabaseKV::readRemoteVersion() table=" + tableName + " got a null version for root: " + rootStr);
        }
    }
    catch (const std::exception &e)
    {
        zklog.error("Database64::readRemoteVersion() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
        queryFailed();
        disposeConnection(pDatabaseConnection);
        return ZKR_DB_ERROR;
    }
    
    // Dispose the read db conneciton
    disposeConnection(pDatabaseConnection);

    return ZKR_SUCCESS;
}

zkresult Database64::writeRemoteVersion(const Goldilocks::Element (&root)[4], const uint64_t version){
    
    zkresult result = ZKR_SUCCESS;
    string rootStr_ = fea2string(fr, root[0], root[1], root[2], root[3]); 
    string rootStr = NormalizeToNFormat(rootStr_, 64);
    
    if (config.dbMultiWrite)
    {
        multiWrite.Lock();
        multiWrite.data[multiWrite.pendingToFlushDataIndex].versionIntray[rootStr] = version;
#ifdef LOG_DB_WRITE_REMOTE
            zklog.info("Database64::writeRemote() root=" + rootStr + " version " + to_string(version) + " multiWrite=[" + multiWrite.print() + "]");
#endif
        multiWrite.Unlock();
    }
    else
    {
        const string &tableName =  config.dbVersionTableName;
        string query = "INSERT INTO " + tableName + " ( hash, version ) VALUES ( E\'\\\\x" + rootStr + "\', " + to_string(version) + " ) ON CONFLICT (hash) DO NOTHING;";
        DatabaseConnection * pDatabaseConnection = getConnection();

        try
        {        

#ifdef DATABASE_COMMIT
            if (autoCommit)
#endif
            {
                pqxx::work w(*(pDatabaseConnection->pConnection));
                pqxx::result res = w.exec(query);
                w.commit();
            }
#ifdef DATABASE_COMMIT
            else
            {
                if (transaction == NULL)
                    transaction = new pqxx::work{*pConnectionWrite};
                pqxx::result res = transaction->exec(query);
            }
#endif
        }
        catch (const std::exception &e)
        {
            zklog.error("Database::writeRemoteVersion() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
            result = ZKR_DB_ERROR;
            queryFailed();
        }

        disposeConnection(pDatabaseConnection);
    }

    return result;
}

zkresult Database64::readRemoteLatestVersion(uint64_t &version){
    
    const string &tableName = config.dbLatestVersionTableName;


    if (config.logRemoteDbReads)
    {
        zklog.info("Database64::readRemoteLatestVersion() table=" + tableName );
    }

    // Get a free read db connection
    DatabaseConnection * pDatabaseConnection = getConnection();

    try
    {
        // Prepare the query
        string query = "SELECT * FROM " + tableName + ";";

        pqxx::result rows;

        // Start a transaction.
        pqxx::nontransaction n(*(pDatabaseConnection->pConnection));

        // Execute the query
        rows = n.exec(query);

        // Commit your transaction
        n.commit();

        // Process the result
        if (rows.size() == 0)
        {
            disposeConnection(pDatabaseConnection);
            return ZKR_DB_KEY_NOT_FOUND;
        }
        else if (rows.size() > 1)
        {
            zklog.error("Database64::readRemoteLatestVersion() table=" + tableName + " got more than one row;" );
            exitProcess();
        }

        pqxx::row const row = rows[0];
        if (row.size() != 1)
        {
            zklog.error("DatabaseKV::readRemoteLatestVersion() table=" + tableName + " got an invalid number of colums;");
            exitProcess();
        }
        pqxx::field const fieldData = row[0];
        if (!fieldData.is_null()) {
            version = fieldData.as<uint64_t>();
        } else {
            zklog.error("DatabaseKV::readRemoteLatestVersion() table=" + tableName + " got a null version;");
        }
    }
    catch (const std::exception &e)
    {
        zklog.error("Database64::readRemoteLatestVersion() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
        queryFailed();
        disposeConnection(pDatabaseConnection);
        return ZKR_DB_ERROR;
    }
    
    // Dispose the read db conneciton
    disposeConnection(pDatabaseConnection);

    return ZKR_SUCCESS;
}
zkresult Database64::writeRemoteLatestVersion(const uint64_t version){
    
    zkresult rw = ZKR_SUCCESS;    

    if (config.dbMultiWrite)
    {
        multiWrite.Lock();
        multiWrite.data[multiWrite.pendingToFlushDataIndex].latestVersion = version;
#ifdef LOG_DB_WRITE_REMOTE
            zklog.info("Database64::writeRemote() root=" + rootStr + " version " + to_string(version) + " multiWrite=[" + multiWrite.print() + "]");
#endif
        multiWrite.Unlock();

    }else{

        const string &tableName =  config.dbLatestVersionTableName;
        string query = "UPDATE " + tableName + " SET version = " + to_string(version) + ";";
        DatabaseConnection * pDatabaseConnection = getConnection();

        try
        {        
#ifdef DATABASE_COMMIT
            if (autoCommit)
#endif
            {
                pqxx::work w(*(pDatabaseConnection->pConnection));
                pqxx::result res = w.exec(query);
                w.commit();
            }
#ifdef DATABASE_COMMIT
            else
            {
                if (transaction == NULL)
                    transaction = new pqxx::work{*pConnectionWrite};
                pqxx::result res = transaction->exec(query);
            }
#endif
        }
        catch (const std::exception &e)
        {
            zklog.error("Database::writeRemoteVersion() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
            rw = ZKR_DB_ERROR;
            queryFailed();
        }
        disposeConnection(pDatabaseConnection);
    
    }
    return rw;
}

zkresult Database64::createStateRoot(void)
{
    // Copy the state root in the first 4 elements of dbValue
    vector<Goldilocks::Element> value;
    for (uint64_t i=0; i<12; i++) value.push_back(fr.zero());
    
    // Prepare the value string
    string valueString = "";
    string aux;
    for (uint64_t i = 0; i < value.size(); i++)
    {
        valueString += PrependZeros(fr.toString(value[i], 16), 16);
    }

    zkresult r = ZKR_SUCCESS;

    if (!config.dbReadOnly)
    {
        // Prepare the query
        string query = "INSERT INTO " + config.dbNodesTableName + " ( hash, data ) VALUES ( E\'\\\\x" + dbStateRootKey + "\', E\'\\\\x" + valueString + "\' ) " +
                    "ON CONFLICT (hash) DO NOTHING;";
            
        DatabaseConnection * pDatabaseConnection = getConnection();

        try
        {        

#ifdef DATABASE_COMMIT
            if (autoCommit)
#endif
            {
                pqxx::work w(*(pDatabaseConnection->pConnection));
                pqxx::result res = w.exec(query);
                w.commit();
            }
#ifdef DATABASE_COMMIT
            else
            {
                if (transaction == NULL)
                    transaction = new pqxx::work{*pConnectionWrite};
                pqxx::result res = transaction->exec(query);
            }
#endif
        }
        catch (const std::exception &e)
        {
            zklog.error("Database64::createStateRoot() table=" + config.dbNodesTableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
            r = ZKR_DB_ERROR;
            queryFailed();
        }

        disposeConnection(pDatabaseConnection);
    }

#ifdef LOG_DB_WRITE
    {
        string s = "Database64::createStateRoot()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(r);
        s += " key=" + dbStateRootKey;
        s += " value=";
        for (uint64_t i = 0; i < value.size(); i++)
            s += fr.toString(value[i], 16) + ":";
        zklog.info(s);
    }
#endif

    return r;
}

zkresult Database64::updateStateRoot(const Goldilocks::Element (&stateRoot)[4])
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::updateStateRoot() called uninitialized");
        exitProcess();
    }
    
    // Prepare the value string
    string valueString = "";
    string aux;
    for (uint64_t i = 0; i < 4; i++)
    {
        valueString += PrependZeros(fr.toString(stateRoot[i], 16), 16);
    }

    zkresult r = ZKR_SUCCESS;

    if ( useRemoteDB )
    {
        if (config.dbMultiWrite)
        {
            multiWrite.Lock();
            multiWrite.data[multiWrite.pendingToFlushDataIndex].nodesStateRoot = valueString;
            multiWrite.Unlock();    
        }
        else
        {
            // Prepare the query
            string query = "UPDATE " + config.dbNodesTableName + " SET data = E\'\\\\x" + valueString + "\' WHERE  hash = E\'\\\\x" + dbStateRootKey + "\';";
                
            DatabaseConnection * pDatabaseConnection = getConnection();

            try
            {        

    #ifdef DATABASE_COMMIT
                if (autoCommit)
    #endif
                {
                    pqxx::work w(*(pDatabaseConnection->pConnection));
                    pqxx::result res = w.exec(query);
                    w.commit();
                }
    #ifdef DATABASE_COMMIT
                else
                {
                    if (transaction == NULL)
                        transaction = new pqxx::work{*pConnectionWrite};
                    pqxx::result res = transaction->exec(query);
                }
    #endif
            }
            catch (const std::exception &e)
            {
                zklog.error("Database64::updateStateRoot() table=" + config.dbNodesTableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
                r = ZKR_DB_ERROR;
                queryFailed();
            }

            disposeConnection(pDatabaseConnection);
        }
    }

#ifdef DATABASE_USE_CACHE
    if ((r == ZKR_SUCCESS) && dbMTCache.enabled())
    {
        // Create in memory cache
        dbMTCache.add(dbStateRootKey, valueString, true);
    }
#endif

#ifdef LOG_DB_WRITE
    {
        string s = "Database64::updateStateRoot()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(r);
        s += " key=" + dbStateRootKey;
        s += " value=";
        for (uint64_t i = 0; i < value.size(); i++)
            s += fr.toString(value[i], 16) + ":";
        zklog.info(s);
    }
#endif

    return r;
}

zkresult Database64::setProgram (const string &_key, const vector<uint8_t> &data, const bool persistent)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::setProgram() called uninitialized");
        exitProcess();
    }

    zkresult r;

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

    if ( useRemoteDB
#ifdef DATABASE_USE_CACHE
         && persistent
#endif
         )
    {
        string sData = "";
        for (uint64_t i=0; i<data.size(); i++)
        {
            sData += byte2string(data[i]);
        }

        r = writeRemote(true, key, sData);
    }
    else
    {
        r = ZKR_SUCCESS;
    }

#ifdef DATABASE_USE_CACHE
    if ((r == ZKR_SUCCESS) && (dbProgramCache.enabled()))
    {
        // Create in memory cache
        dbProgramCache.add(key, data, false);
    }
#endif

#ifdef LOG_DB_WRITE
    {
        string s = "Database64::setProgram()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(r);
        s += " key=" + key;
        s += " data=";
        for (uint64_t i = 0; (i < (data.size()) && (i < 100)); i++)
            s += byte2string(data[i]);
        if (data.size() > 100) s += "...";
        s += " persistent=" + to_string(persistent);
        zklog.info(s);
    }
#endif

    return r;
}

zkresult Database64::getProgram(const string &_key, vector<uint8_t> &data, DatabaseMap *dbReadLog)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::getProgram() called uninitialized");
        exitProcess();
    }

    zkresult r;

    struct timeval t;
    if (dbReadLog != NULL) gettimeofday(&t, NULL);

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

#ifdef DATABASE_USE_CACHE
    // If the key is found in local database (cached) simply return it
    if (dbProgramCache.enabled() && dbProgramCache.find(key, data))
    {
        // Add to the read log
        if (dbReadLog != NULL) dbReadLog->add(key, data, true, TimeDiff(t));

        r = ZKR_SUCCESS;
    }
    // If the key is pending to be stored on database, but already deleted from cache
    else if (config.dbMultiWrite && multiWrite.findProgram(key, data))
    {
        // Add to the read log
        if (dbReadLog != NULL) dbReadLog->add(key, data, true, TimeDiff(t));

        r = ZKR_SUCCESS;
    }
    else
#endif
    if (useRemoteDB)
    {
        // Otherwise, read it remotelly
        string sData;
        r = readRemote(true, key, sData);
        if (r == ZKR_SUCCESS)
        {
            //String to byte/uint8_t vector
            string2ba(sData, data);

#ifdef DATABASE_USE_CACHE
            // Store it locally to avoid any future remote access for this key
            if (dbProgramCache.enabled()) dbProgramCache.add(key, data, false);
#endif

            // Add to the read log
            if (dbReadLog != NULL) dbReadLog->add(key, data, false, TimeDiff(t));
        }
    }
    else
    {
        zklog.error("Database64::getProgram() requested a key that does not exist: " + key);
        r = ZKR_DB_KEY_NOT_FOUND;
    }

#ifdef LOG_DB_READ
    {
        string s = "Database64::getProgram()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(r);
        s += " key=" + key;
        s += " data=";
        for (uint64_t i = 0; (i < (data.size()) && (i < 100)); i++)
            s += byte2string(data[i]);
        if (data.size() > 100) s += "...";
        zklog.info(s);
    }
#endif

    return r;
}
    
zkresult Database64::flush(uint64_t &thisBatch, uint64_t &lastSentBatch)
{
    if (!config.dbMultiWrite)
    {
        return ZKR_SUCCESS;
    }

    // If we are connected to a read-only database, just free memory and pretend to have sent all the data
    if (config.dbReadOnly)
    {
        multiWrite.Lock();
        multiWrite.data[multiWrite.pendingToFlushDataIndex].Reset();
        multiWrite.Unlock();

        return ZKR_SUCCESS;
    }

    //TimerStart(DATABASE_FLUSH);

    multiWrite.Lock();

    // Accept all intray data
    multiWrite.data[multiWrite.pendingToFlushDataIndex].acceptIntray();

    // Increase the last processed batch id and return the last sent batch id
    multiWrite.lastFlushId++;
    thisBatch = multiWrite.lastFlushId;
    lastSentBatch = multiWrite.storedFlushId;

#ifdef LOG_DB_FLUSH
    zklog.info("Database64::flush() thisBatch=" + to_string(thisBatch) + " lastSentBatch=" + to_string(lastSentBatch) + " multiWrite=[" + multiWrite.print() + "]");
#endif

    // Notify the thread
    sem_post(&senderSem);

    multiWrite.Unlock();
    return ZKR_SUCCESS;
}

void Database64::semiFlush (void)
{
    if (!config.dbMultiWrite)
    {
        return;
    }

    multiWrite.Lock();

    multiWrite.data[multiWrite.pendingToFlushDataIndex].acceptIntray();

#ifdef LOG_DB_SEMI_FLUSH
    zklog.info("Database64::semiFlush() called multiWrite=[" + multiWrite.print() + "]");
#endif

    multiWrite.Unlock();
}

zkresult Database64::getFlushStatus(uint64_t &storedFlushId, uint64_t &storingFlushId, uint64_t &lastFlushId, uint64_t &pendingToFlushNodes, uint64_t &pendingToFlushProgram, uint64_t &storingNodes, uint64_t &storingProgram)
{
    multiWrite.Lock();
    storedFlushId = multiWrite.storedFlushId;
    storingFlushId = multiWrite.storingFlushId;
    lastFlushId = multiWrite.lastFlushId;
    pendingToFlushNodes = multiWrite.data[multiWrite.pendingToFlushDataIndex].nodes.size();
    pendingToFlushProgram = multiWrite.data[multiWrite.pendingToFlushDataIndex].program.size();
    storingNodes = multiWrite.data[multiWrite.storingDataIndex].nodes.size();
    storingProgram = multiWrite.data[multiWrite.storingDataIndex].program.size();
    multiWrite.Unlock();
    return ZKR_SUCCESS;
}

zkresult Database64::sendData (void)
{
    zkresult zkr = ZKR_SUCCESS;
    
    // Time calculation variables
    struct timeval t;
    uint64_t timeDiff = 0;
    uint64_t fields = 0;

    // Select proper data instance
    MultiWriteData64 &data = multiWrite.data[multiWrite.storingDataIndex];

    // Check if there is data
    if (data.IsEmpty())
    {
        zklog.warning("Database64::sendData() called with empty data");
        return ZKR_SUCCESS;
    }

    // Check if it has already been stored to database
    if (data.stored)
    {
        zklog.warning("Database64::sendData() called with stored=true");
        return ZKR_SUCCESS;
    }

    // Get a free write db connection
    DatabaseConnection * pDatabaseConnection = getConnection();

    try
    {
        if (config.dbMetrics) gettimeofday(&t, NULL);
        unordered_map<string, string>::const_iterator it;
        if (data.multiQuery.isEmpty())
        {
            // Current query number
            uint64_t currentQuery = 0;
            bool firstValue = false;

            // If there are nodes add the corresponding query
            if (data.nodes.size() > 0)
            {
                it = data.nodes.begin();
                while (it != data.nodes.end())
                {
                    // If queries is empty or last query is full, add a new query
                    if ( (data.multiQuery.queries.size() == 0) || (data.multiQuery.queries[currentQuery].full))
                    {
                        SingleQuery query;
                        data.multiQuery.queries.emplace_back(query);
                        currentQuery = data.multiQuery.queries.size() - 1;
                    }

                    data.multiQuery.queries[currentQuery].query += "INSERT INTO " + config.dbNodesTableName + " ( hash, data ) VALUES ";
                    firstValue = true;
                    for (; it != data.nodes.end(); it++)
                    {
                        if (!firstValue)
                        {
                            data.multiQuery.queries[currentQuery].query += ", ";
                        }
                        firstValue = false;
                        data.multiQuery.queries[currentQuery].query += "( E\'\\\\x" + it->first + "\', E\'\\\\x" + ba2string(it->second) + "\' ) ";
#ifdef LOG_DB_SEND_DATA
                        zklog.info("Database64::sendData() inserting node key=" + it->first + " value=" + it->second);
#endif
                        if (data.multiQuery.queries[currentQuery].query.size() >= config.dbMultiWriteSingleQuerySize)
                        {
                            // Mark query as full
                            data.multiQuery.queries[currentQuery].full = true;
                            break;
                        }
                    }
                    data.multiQuery.queries[currentQuery].query += " ON CONFLICT (hash) DO NOTHING;";
                }
            }

            // If there are keyValues add to db
            if (data.keyValueA.size() > 0)
            {
                map<uint64_t, vector<KeyValue>>::const_iterator it=data.keyValueA.begin();
                while (it != data.keyValueA.end())
                {
                    for(auto it2=it->second.begin(); it2!=it->second.end(); it2++)
                    {   
                        writeRemoteKV(it->first,it2->key,it2->value, false);
                    }
                    ++it;
                }
            }
            // If there are versions add to db
            if (data.version.size() > 0)
            {
                unordered_map<string, uint64_t>::const_iterator it=data.version.begin();
                while (it != data.version.end())
                {
                    // If queries is empty or last query is full, add a new query
                    if ( (data.multiQuery.queries.size() == 0) || (data.multiQuery.queries[currentQuery].full))
                    {
                        SingleQuery query;
                        data.multiQuery.queries.emplace_back(query);
                        currentQuery = data.multiQuery.queries.size() - 1;
                    }

                    data.multiQuery.queries[currentQuery].query += "INSERT INTO " + config.dbVersionTableName + " ( hash, version ) VALUES ";
                    firstValue = true;
                    for (; it != data.version.end(); it++)
                    {
                        if (!firstValue)
                        {
                            data.multiQuery.queries[currentQuery].query += ", ";
                        }
                        firstValue = false;
                        data.multiQuery.queries[currentQuery].query += "( E\'\\\\x" + it->first + "\'," + to_string(it->second) +"  ) ";
#ifdef LOG_DB_SEND_DATA
                        zklog.info("Database64::sendData() inserting version key=" + it->first + " version=" + to_string(it->second));
#endif
                        if (data.multiQuery.queries[currentQuery].query.size() >= config.dbMultiWriteSingleQuerySize)
                        {
                            // Mark query as full
                            data.multiQuery.queries[currentQuery].full = true;
                            break;
                        }
                    }
                    data.multiQuery.queries[currentQuery].query += " ON CONFLICT (hash) DO NOTHING;";
                   
                }
            }

            // If there are program add the corresponding query
            if (data.program.size() > 0)
            {
                it = data.program.begin();
                while (it != data.program.end())
                {
                    // If queries is empty or last query is full, add a new query
                    if ( (data.multiQuery.queries.size() == 0) || (data.multiQuery.queries[currentQuery].full))
                    {
                        SingleQuery query;
                        data.multiQuery.queries.emplace_back(query);
                        currentQuery = data.multiQuery.queries.size() - 1;
                    }

                    data.multiQuery.queries[currentQuery].query += "INSERT INTO " + config.dbProgramTableName + " ( hash, data ) VALUES ";
                    firstValue = true;
                    for (; it != data.program.end(); it++)
                    {
                        if (!firstValue)
                        {
                            data.multiQuery.queries[currentQuery].query += ", ";
                        }
                        firstValue = false;
                        data.multiQuery.queries[currentQuery].query += "( E\'\\\\x" + it->first + "\', E\'\\\\x" + it->second + "\' ) ";
#ifdef LOG_DB_SEND_DATA
                        zklog.info("Database64::sendData() inserting program key=" + it->first + " value=" + it->second);
#endif
                        if (data.multiQuery.queries[currentQuery].query.size() >= config.dbMultiWriteSingleQuerySize)
                        {
                            // Mark query as full
                            data.multiQuery.queries[currentQuery].full = true;
                            break;
                        }
                    }
                    data.multiQuery.queries[currentQuery].query += " ON CONFLICT (hash) DO NOTHING;";
                }
            }

            // If there is a nodes state root query, add it
            if (data.nodesStateRoot.size() > 0)
            {
                // If queries is empty or last query is full, add a new query
                if ( (data.multiQuery.queries.size() == 0) || (data.multiQuery.queries[currentQuery].full))
                {
                    SingleQuery query;
                    data.multiQuery.queries.emplace_back(query);
                    currentQuery = data.multiQuery.queries.size() - 1;
                }

                data.multiQuery.queries[currentQuery].query += "UPDATE " + config.dbNodesTableName + " SET data = E\'\\\\x" + data.nodesStateRoot + "\' WHERE hash = E\'\\\\x" + dbStateRootKey + "\';";

                // Mark query as full
                data.multiQuery.queries[currentQuery].full = true;
#ifdef LOG_DB_SEND_DATA
                zklog.info("Database64::sendData() inserting root=" + data.nodesStateRoot);
#endif
            }
        }

        if (data.multiQuery.isEmpty())
        {
            zklog.warning("Database64::sendData() called without any data to send");
            data.stored = true;
        }
        else
        {
            if (config.dbMetrics)
            {
                fields = data.nodes.size() + data.program.size() + (data.nodesStateRoot.size() > 0 ? 1 : 0);
                zklog.info("Database64::sendData() dbMetrics multiWrite nodes=" + to_string(data.nodes.size()) +
                    " program=" + to_string(data.program.size()) +
                    " nodesStateRootCounter=" + to_string(data.nodesStateRoot.size() > 0 ? 1 : 0) +
                    " query.size=" + to_string(data.multiQuery.size()) + "B=" + to_string(data.multiQuery.size()/zkmax(fields,1)) + "B/field" +
                    " queries.size=" + to_string(data.multiQuery.queries.size()) +
                    " total=" + to_string(fields) + "fields");
            }

            // Send all unsent queries to database
            for (uint64_t i=0; i<data.multiQuery.queries.size(); i++)
            {
                // Skip sent queries
                if (data.multiQuery.queries[i].sent)
                {
                    continue;
                }

                // Start a transaction
                pqxx::work w(*(pDatabaseConnection->pConnection));

                // Execute the query
                pqxx::result res = w.exec(data.multiQuery.queries[i].query);

                // Commit your transaction
                w.commit();

                // Mask as sent
                data.multiQuery.queries[i].sent = true;
            }

            //zklog.info("Database64::flush() sent query=" + query);
            if (config.dbMetrics)
            {
                timeDiff = TimeDiff(t);
                zklog.info("Database64::sendData() dbMetrics multiWrite total=" + to_string(fields) + "fields=" + to_string(timeDiff) + "us=" + to_string(timeDiff/zkmax(fields,1)) + "us/field");
            }

#ifdef LOG_DB_WRITE_QUERY
            {
                string query;
                for (uint64_t i=0; i<data.multiQuery.queries.size(); i++)
                {
                    query += data.multiQuery.queries[i].query;
                }
                zklog.info("Database64::sendData() write query=" + query);
            }
#endif
#ifdef LOG_DB_SEND_DATA
            zklog.info("Database64::sendData() successfully processed query of size= " + to_string(data.multiQuery.size()));
#endif
            // Update status
            data.multiQuery.reset();
            data.stored = true;
        }

        // If we succeeded, update last sent batch
        multiWrite.Lock();
        multiWrite.storedFlushId = multiWrite.storingFlushId;
        multiWrite.Unlock();
    }
    catch (const std::exception &e)
    {
        zklog.error("Database64::sendData() execute query exception: " + string(e.what()));
        zklog.error("Database64::sendData() query.size=" + to_string(data.multiQuery.queries.size()) + (data.multiQuery.isEmpty() ? "" : (" query(<1024)=" + data.multiQuery.queries[0].query.substr(0, 1024))));
        queryFailed();
        zkr = ZKR_DB_ERROR;
    }

    // Dispose the write db connection
    disposeConnection(pDatabaseConnection);

    return zkr;
}

// Get flush data, written to database by dbSenderThread; it blocks
zkresult Database64::getFlushData(uint64_t flushId, uint64_t &storedFlushId, unordered_map<string, string> (&nodes), unordered_map<string, string> (&program), string &nodesStateRoot)
{
    //zklog.info("--> getFlushData()");

    // Set the deadline to now + 60 seconds
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
	deadline.tv_sec += 60;

    // Try to get the semaphore
    int iResult;
    iResult = sem_timedwait(&getFlushDataSem, &deadline);
    if (iResult != 0)
    {
        zklog.info("Database64::getFlushData() timed out");
        return ZKR_SUCCESS;
    }

    multiWrite.Lock();
    MultiWriteData64 &data = multiWrite.data[multiWrite.synchronizingDataIndex];

    zklog.info("Database64::getFlushData woke up: pendingToFlushDataIndex=" + to_string(multiWrite.pendingToFlushDataIndex) +
        " storingDataIndex=" + to_string(multiWrite.storingDataIndex) +
        " synchronizingDataIndex=" + to_string(multiWrite.synchronizingDataIndex) +
        " nodes=" + to_string(data.nodes.size()) +
        " program=" + to_string(data.program.size()) +
        " nodesStateRoot=" + data.nodesStateRoot);

    if (data.nodes.size() > 0)
    {
        nodes = data.nodes;
    }

    if (data.program.size() > 0)
    {
        program = data.program;
    }

    if (data.nodesStateRoot.size() > 0)
    {
        nodesStateRoot = data.nodesStateRoot;
    }

    multiWrite.Unlock();

    //zklog.info("<-- getFlushData()");

    return ZKR_SUCCESS;
}

#ifdef DATABASE_COMMIT

void Database64::setAutoCommit(const bool ac)
{
    if (ac && !autoCommit)
        commit();
    autoCommit = ac;
}

void Database64::commit()
{
    if ((!autoCommit) && (transaction != NULL))
    {
        transaction->commit();
        delete transaction;
        transaction = NULL;
    }
}

#endif

void Database64::printTree(const string &root, string prefix)
{
    /*
    if (prefix == "")
    {
        zklog.info("Printint tree of root=" + root);
    }
    string key = root;
    vector<Goldilocks::Element> value;
    Goldilocks::Element vKey[4];
    string2key(fr, key, vKey);  
    read(key,vKey,value, NULL);

    if (value.size() != 12)
    {
        zklog.error("Database64::printTree() found value.size()=" + to_string(value.size()));
        return;
    }
    if (!fr.equal(value[11], fr.zero()))
    {
        zklog.error("Database64::printTree() found value[11]=" + fr.toString(value[11], 16));
        return;
    }
    if (!fr.equal(value[10], fr.zero()))
    {
        zklog.error("Database64::printTree() found value[10]=" + fr.toString(value[10], 16));
        return;
    }
    if (!fr.equal(value[9], fr.zero()))
    {
        zklog.error("Database64::printTree() found value[9]=" + fr.toString(value[9], 16));
        return;
    }
    if (fr.equal(value[8], fr.zero())) // Intermediate node
    {
        string leftKey = fea2string(fr, value[0], value[1], value[2], value[3]);
        zklog.info(prefix + "Intermediate node - left hash=" + leftKey);
        if (leftKey != "0")
            printTree(leftKey, prefix + "  ");
        string rightKey = fea2string(fr, value[4], value[5], value[6], value[7]);
        zklog.info(prefix + "Intermediate node - right hash=" + rightKey);
        if (rightKey != "0")
            printTree(rightKey, prefix + "  ");
    }
    else if (fr.equal(value[8], fr.one())) // Leaf node
    {
        string rKey = fea2string(fr, value[0], value[1], value[2], value[3]);
        zklog.info(prefix + "rKey=" + rKey);
        string hashValue = fea2string(fr, value[4], value[5], value[6], value[7]);
        zklog.info(prefix + "hashValue=" + hashValue);
        vector<Goldilocks::Element> leafValue;
        Goldilocks::Element vKey[4]={value[4],value[5],value[6],value[7]};
        read(rKey, vKey, leafValue, NULL);
        if (leafValue.size() == 12)
        {
            if (!fr.equal(leafValue[8], fr.zero()))
            {
                zklog.error("Database64::printTree() found leafValue[8]=" + fr.toString(leafValue[8], 16));
                return;
            }
            if (!fr.equal(leafValue[9], fr.zero()))
            {
                zklog.error("Database64::printTree() found leafValue[9]=" + fr.toString(leafValue[9], 16));
                return;
            }
            if (!fr.equal(leafValue[10], fr.zero()))
            {
                zklog.error("Database64::printTree() found leafValue[10]=" + fr.toString(leafValue[10], 16));
                return;
            }
            if (!fr.equal(leafValue[11], fr.zero()))
            {
                zklog.error("Database64::printTree() found leafValue[11]=" + fr.toString(leafValue[11], 16));
                return;
            }
        }
        else if (leafValue.size() == 8)
        {
            zklog.info(prefix + "leafValue.size()=" + to_string(leafValue.size()));
        }
        else
        {
            zklog.error("Database64::printTree() found lleafValue.size()=" + to_string(leafValue.size()));
            return;
        }
        mpz_class scalarValue;
        fea2scalar(fr, scalarValue, leafValue[0], leafValue[1], leafValue[2], leafValue[3], leafValue[4], leafValue[5], leafValue[6], leafValue[7]);
        zklog.info(prefix + "leafValue=" + PrependZeros(scalarValue.get_str(16), 64));
    }
    else
    {
        zklog.error("Database64::printTree() found value[8]=" + fr.toString(value[8], 16));
        return;
    }
    if (prefix == "") zklog.info("");
    */
}

void Database64::clearCache (void)
{
    dbMTCache.clear();
    dbProgramCache.clear();
    dbKVACache.clear();
    dbVersionACache.clear();
    latestVersionCache=0;
}

void *dbSenderThread64 (void *arg)
{
    Database64 *pDatabase = (Database64 *)arg;
    zklog.info("dbSenderThread64() started");
    MultiWrite64 &multiWrite = pDatabase->multiWrite;

    while (true)
    {
        // Wait for the sending semaphore to be released, if there is no more data to send
        struct timespec currentTime;
        int iResult = clock_gettime(CLOCK_REALTIME, &currentTime);
        if (iResult == -1)
        {
            zklog.error("dbSenderThread64() failed calling clock_gettime()");
            exitProcess();
        }

        currentTime.tv_sec += 5;
        sem_timedwait(&pDatabase->senderSem, &currentTime);

        multiWrite.Lock();

        bool bDataEmpty = false;

        // If sending data is not empty (it failed before) then try to send it again
        if (!multiWrite.data[multiWrite.storingDataIndex].multiQuery.isEmpty())
        {
            zklog.warning("dbSenderThread64() found sending data index not empty, probably because of a previous error; resuming...");
        }
        // If processing data is empty, then simply pretend to have sent data
        else if (multiWrite.data[multiWrite.pendingToFlushDataIndex].IsEmpty())
        {
            //zklog.warning("dbSenderThread() found pending to flush data empty");

            // Mark as if we sent all batches
            multiWrite.storedFlushId = multiWrite.lastFlushId;
#ifdef LOG_DB_SENDER_THREAD
            zklog.info("dbSenderThread64() found multi write processing data empty, so ignoring");
#endif
            multiWrite.Unlock();
            continue;
        }
        // Else, switch data indexes
        else
        {
            // Accept all intray data
            multiWrite.data[multiWrite.pendingToFlushDataIndex].acceptIntray(true);

            // Advance processing and sending indexes
            multiWrite.storingDataIndex = (multiWrite.storingDataIndex + 1) % 3;
            multiWrite.pendingToFlushDataIndex = (multiWrite.pendingToFlushDataIndex + 1) % 3;
            multiWrite.data[multiWrite.pendingToFlushDataIndex].Reset();
#ifdef LOG_DB_SENDER_THREAD
            zklog.info("dbSenderThread64() updated: multiWrite=[" + multiWrite.print() + "]");
#endif

            // Record the last processed batch included in this data set
            multiWrite.storingFlushId = multiWrite.lastFlushId;

            // If there is no data to send, just pretend to have sent it
            if (multiWrite.data[multiWrite.storingDataIndex].IsEmpty())
            {
                // Update stored flush ID
                multiWrite.storedFlushId = multiWrite.storingFlushId;

                // Advance synchronizing index
                multiWrite.synchronizingDataIndex = (multiWrite.synchronizingDataIndex + 1) % 3;
#ifdef LOG_DB_SENDER_THREAD
                zklog.info("dbSenderThread64() no data to send: multiWrite=[" + multiWrite.print() + "]");
#endif
                bDataEmpty = true;
            }

        }

        // Unlock to let more processing batch data in
        multiWrite.Unlock();

        if (!bDataEmpty)
        {
#ifdef LOG_DB_SENDER_THREAD
            zklog.info("dbSenderThread64() starting to send data, multiWrite=[" + multiWrite.print() + "]");
#endif
            zkresult zkr;
            zkr = pDatabase->sendData();
            if (zkr == ZKR_SUCCESS)
            {
                multiWrite.Lock();
                multiWrite.storedFlushId = multiWrite.storingFlushId;
#ifdef LOG_DB_SENDER_THREAD
                zklog.info("dbSenderThread64() successfully sent data, multiWrite=[]" + multiWrite.print() + "]");
#endif
                // Advance synchronizing index
                multiWrite.synchronizingDataIndex = (multiWrite.synchronizingDataIndex + 1) % 3;
#ifdef LOG_DB_SENDER_THREAD
                zklog.info("dbSenderThread64() updated: multiWrite=[" + multiWrite.print() + "]");
#endif
                sem_post(&pDatabase->getFlushDataSem);
#ifdef LOG_DB_SENDER_THREAD
                zklog.info("dbSenderThread64() successfully called sem_post(&pDatabase->getFlushDataSem)");
#endif
                multiWrite.Unlock();
            }
            else
            {
                zklog.error("dbSenderThread64() failed calling sendData() error=" + zkresult2string(zkr));
                usleep(1000000);
            }
        }
    }

    zklog.info("dbSenderThread64() done");
    return NULL;
}

void *dbCacheSynchThread64 (void *arg)
{
    Database *pDatabase = (Database *)arg;
    zklog.info("dbCacheSynchThread64() started");

    uint64_t storedFlushId = 0;

    Config config = pDatabase->config;
    config.hashDBURL = config.dbCacheSynchURL;

    while (true)
    {
        HashDBInterface *pHashDBRemote = new HashDBRemote (pDatabase->fr, config);
        if (pHashDBRemote == NULL)
        {
            zklog.error("dbCacheSynchThread64() failed calling new HashDBRemote()");
            sleep(10);
            continue;
        }

        while (true)
        {
            unordered_map<string, string> nodes;
            unordered_map<string, string> program;
            string nodesStateRoot;
            
            // Call getFlushData() remotelly
            zkresult zkr = pHashDBRemote->getFlushData(storedFlushId, storedFlushId, nodes, program, nodesStateRoot);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("dbCacheSynchThread64() failed calling pHashDB->getFlushData() result=" + zkresult2string(zkr));
                sleep(10);
                break;
            }

            if (nodes.size()==0 && program.size()==0 && nodesStateRoot.size()==0)
            {
                zklog.info("dbCacheSynchThread64() called getFlushData() remotely and got no data: storedFlushId=" + to_string(storedFlushId));
                continue;
            }

            TimerStart(DATABASE_CACHE_SYNCH);
            zklog.info("dbCacheSynchThread64() called getFlushData() remotely and got: storedFlushId=" + to_string(storedFlushId) + " nodes=" + to_string(nodes.size()) + " program=" + to_string(program.size()) + " nodesStateRoot=" + nodesStateRoot);

            // Save nodes to cache
            unordered_map<string, string>::const_iterator it;
            if (nodes.size() > 0)
            {
                for (it = nodes.begin(); it != nodes.end(); it++)
                {
                    vector<Goldilocks::Element> value;
                    string2fea(pDatabase->fr, it->second, value);
                    pDatabase->write(it->first, NULL, value, false);
                }
            }

            // Save program to cache
            if (program.size() > 0)
            {
                for (it = program.begin(); it != program.end(); it++)
                {
                    vector<uint8_t> value;
                    string2ba(it->second, value);
                    pDatabase->setProgram(it->first, value, false);
                }
            }

            /* TODO: We cannot overwrite state root to DB.  Do we need to update cache?
            if (nodesStateRoot.size() > 0)
            {
                vector<Goldilocks::Element> value;
                string2fea(pDatabase->fr, nodesStateRoot, value);
                if (value.size() < 4)
                {
                    zklog.error("dbCacheSynchThread() got nodeStateRoot too short=" + nodesStateRoot);
                }
                else
                {
                    Goldilocks::Element stateRoot[4];
                    for (uint64_t i=0; i<4; i++)
                    {
                        stateRoot[i] = value[i];
                    }
                    pDatabase->updateStateRoot(stateRoot);
                }
            }*/

            TimerStopAndLog(DATABASE_CACHE_SYNCH);
        }
        delete pHashDBRemote;
    }

    zklog.info("dbCacheSynchThread64() done");
    return NULL;
}

void loadDb2MemCache64 (const Config &config)
{
    if (config.databaseURL == "local")
    {
        zklog.error("loadDb2MemCache64() called with config.databaseURL==local");
        exitProcess();
    }

#ifdef DATABASE_USE_CACHE

    TimerStart(LOAD_DB_TO_CACHE);

    Goldilocks fr;
    HashDB * pHashDB = (HashDB *)hashDBSingleton.get();

    vector<Goldilocks::Element> dbValue;
    zkresult zkr = pHashDB->db.read(Database64::dbStateRootKey, Database64::dbStateRootvKey, dbValue, NULL, true);

    if (zkr == ZKR_DB_KEY_NOT_FOUND)
    {
        zklog.warning("loadDb2MemCache64() dbStateRootKey=" +  Database64::dbStateRootKey + " not found in database; normal only if database is empty");
        TimerStopAndLog(LOAD_DB_TO_CACHE);
        return;
    }
    else if (zkr != ZKR_SUCCESS)
    {
        zklog.error("loadDb2MemCache64() failed calling db.read result=" + zkresult2string(zkr));
        TimerStopAndLog(LOAD_DB_TO_CACHE);
        return;
    }
    
    string stateRootKey = fea2string(fr, dbValue[0], dbValue[1], dbValue[2], dbValue[3]);
    zklog.info("loadDb2MemCache64() found state root=" + stateRootKey);

    if (stateRootKey == "0")
    {
        zklog.warning("loadDb2MemCache64() found an empty tree");
        TimerStopAndLog(LOAD_DB_TO_CACHE);
        return;
    }

    struct timeval loadCacheStartTime;
    gettimeofday(&loadCacheStartTime, NULL);

    unordered_map<uint64_t, vector<string>> treeMap;
    vector<string> emptyVector;
    string hash, leftHash, rightHash;
    uint64_t counter = 0;

    treeMap[0] = emptyVector;
    treeMap[0].push_back(stateRootKey);
    unordered_map<uint64_t, std::vector<std::string>>::iterator treeMapIterator;
    for (uint64_t level=0; level<256; level++)
    {
        // Spend only 10 seconds
        if (TimeDiff(loadCacheStartTime) > config.loadDBToMemTimeout)
        {
            break;
        }

        treeMapIterator = treeMap.find(level);
        if (treeMapIterator == treeMap.end())
        {
            break;
        }

        if (treeMapIterator->second.size()==0)
        {
            break;
        }

        treeMap[level+1] = emptyVector;

        //zklog.info("loadDb2MemCache() searching at level=" + to_string(level) + " for elements=" + to_string(treeMapIterator->second.size()));
        
        for (uint64_t i=0; i<treeMapIterator->second.size(); i++)
        {
            // Spend only 10 seconds
            if (TimeDiff(loadCacheStartTime) > config.loadDBToMemTimeout)
            {
                break;
            }

            hash = treeMapIterator->second[i];
            dbValue.clear();
            Goldilocks::Element vhash[4];
            if(pHashDB->db.usingAssociativeCache()) string2fea(fr, hash, vhash);
            zkresult zkr = pHashDB->db.read(hash, vhash, dbValue, NULL, true);

            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("loadDb2MemCache64() failed calling db.read(" + hash + ") result=" + zkresult2string(zkr));
                TimerStopAndLog(LOAD_DB_TO_CACHE);
                return;
            }
            if (dbValue.size() != 12)
            {
                zklog.error("loadDb2MemCache64() failed calling db.read(" + hash + ") dbValue.size()=" + to_string(dbValue.size()));
                TimerStopAndLog(LOAD_DB_TO_CACHE);
                return;
            }
            counter++;
            if(Database64::dbMTCache.enabled()){
                double sizePercentage = double(Database64::dbMTCache.getCurrentSize())*100.0/double(Database64::dbMTCache.getMaxSize());
                if ( sizePercentage > 90 )
                {
                    zklog.info("loadDb2MemCache64() stopping since size percentage=" + to_string(sizePercentage));
                    break;
                }
            }
            // If capaxity is X000
            if (fr.isZero(dbValue[9]) && fr.isZero(dbValue[10]) && fr.isZero(dbValue[11]))
            {
                // If capacity is 0000, this is an intermediate node that contains left and right hashes of its children
                if (fr.isZero(dbValue[8]))
                {
                    leftHash = fea2string(fr, dbValue[0], dbValue[1], dbValue[2], dbValue[3]);
                    if (leftHash != "0")
                    {
                        treeMap[level+1].push_back(leftHash);
                        //zklog.info("loadDb2MemCache() level=" + to_string(level) + " found leftHash=" + leftHash);
                    }
                    rightHash = fea2string(fr, dbValue[4], dbValue[5], dbValue[6], dbValue[7]);
                    if (rightHash != "0")
                    {
                        treeMap[level+1].push_back(rightHash);
                        //zklog.info("loadDb2MemCache() level=" + to_string(level) + " found rightHash=" + rightHash);
                    }
                }
                // If capacity is 1000, this is a leaf node that contains right hash of the value node
                else if (fr.isOne(dbValue[8]))
                {
                    rightHash = fea2string(fr, dbValue[4], dbValue[5], dbValue[6], dbValue[7]);
                    if (rightHash != "0")
                    {
                        //zklog.info("loadDb2MemCache() level=" + to_string(level) + " found value rightHash=" + rightHash);
                        dbValue.clear();
                        Goldilocks::Element vRightHash[4]={dbValue[4], dbValue[5], dbValue[6], dbValue[7]};
                        zkresult zkr = pHashDB->db.read(rightHash, vRightHash, dbValue, NULL, true);
                        if (zkr != ZKR_SUCCESS)
                        {
                            zklog.error("loadDb2MemCache64() failed calling db.read(" + rightHash + ") result=" + zkresult2string(zkr));
                            TimerStopAndLog(LOAD_DB_TO_CACHE);
                            return;
                        }
                        counter++;
                    }
                }
            }
        }
    }

    if(Database64::dbMTCache.enabled()){
        zklog.info("loadDb2MemCache64() done counter=" + to_string(counter) + " cache at " + to_string((double(Database64::dbMTCache.getCurrentSize())/double(Database64::dbMTCache.getMaxSize()))*100) + "%");
    }
    TimerStopAndLog(LOAD_DB_TO_CACHE);

#endif
}