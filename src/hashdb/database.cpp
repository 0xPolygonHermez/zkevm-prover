#include <iostream>
#include <thread>
#include "database.hpp"
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

#ifdef DATABASE_USE_CACHE

// Create static Database::dbMTCache and DatabaseCacheProgram objects
// This will be used to store DB records in memory and it will be shared for all the instances of Database class
// DatabaseCacheMT and DatabaseCacheProgram classes are thread-safe
DatabaseMTAssociativeCache Database::dbMTACache;
DatabaseMTCache Database::dbMTCache;
DatabaseProgramCache Database::dbProgramCache;

string Database::dbStateRootKey("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"); // 64 f's
Goldilocks::Element Database::dbStateRootvKey[4] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
bool Database::useAssociativeCache = false;


#endif

// Helper functions
string removeBSXIfExists(string s) {return ((s.at(0) == '\\') && (s.at(1) == 'x')) ? s.substr(2) : s;}

Database::Database (Goldilocks &fr, const Config &config) :
        fr(fr),
        config(config),
        connectionsPool(NULL),
        multiWrite(fr)
{
    // Init mutex
    pthread_mutex_init(&connMutex, NULL);

    // Initialize semaphores
    sem_init(&senderSem, 0, 0);
    sem_init(&getFlushDataSem, 0, 0);
};

Database::~Database()
{
    if (config.dbConnectionsPool)
    {
        if (connectionsPool != NULL)
        {
            for (uint64_t i=0; i<config.dbNumberOfPoolConnections; i++)
            {
                if (connectionsPool[i].pConnection != NULL)
                {
                    //zklog.info("Database::~Database() deleting writeConnectionsPool[" + to_string(i) + "].pConnection=" + to_string((uint64_t)writeConnectionsPool[i].pConnection));
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

// Database class implementation
void Database::init(void)
{
    // Check that it has not been initialized before
    if (bInitialized)
    {
        zklog.error("Database::init() called when already initialized");
        exitProcess();
    }

    // Configure the server, if configuration is provided
    if (config.databaseURL != "local")
    {
        // Sender thread creation
        pthread_create(&senderPthread, NULL, dbSenderThread, this);

        // Cache synchronization thread creation
        if (config.dbCacheSynchURL.size() > 0)
        {
            pthread_create(&cacheSynchPthread, NULL, dbCacheSynchThread, this);

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

zkresult Database::read(const string &_key, Goldilocks::Element (&vkey)[4], vector<Goldilocks::Element> &value, DatabaseMap *dbReadLog, const bool update,  bool *keys, uint64_t level)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database::read() called uninitialized");
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
    if(usingAssociativeCache() && dbMTACache.findKey(vkey,value)){

        if (dbReadLog != NULL) dbReadLog->add(key, value, true, TimeDiff(t));
        r = ZKR_SUCCESS;

    } else if( dbMTCache.enabled() && dbMTCache.find(key, value)){
        
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
        if(usingAssociativeCache()){
            dbMTACache.addKeyValue(vkey, value, false);
        }
        else if(dbMTCache.enabled()){                
            dbMTCache.add(key, value, false);
        }
#endif
        r = ZKR_SUCCESS;
    }
    // If get tree is configured, read the tree from the branch (key hash) to the leaf (keys since level)
    else if (config.dbGetTree && (keys != NULL))
    {
        // Get the tree
        uint64_t numberOfFields;
        r = readTreeRemote(key, keys, level, numberOfFields);

        // Add to the read log, and restart the timer
        if (dbReadLog != NULL)
        {
            dbReadLog->addGetTree(TimeDiff(t), numberOfFields);
            gettimeofday(&t, NULL);
        }

        // Retry if failed, since read-only databases have a synchronization latency
        if ( (r != ZKR_SUCCESS) && (config.dbReadRetryDelay > 0) )
        {
            for (uint64_t i=0; i<config.dbReadRetryCounter; i++)
            {
                zklog.warning("Database::read() failed calling readTreeRemote() with error=" + zkresult2string(r) + "; will retry after " + to_string(config.dbReadRetryDelay) + "us key=" + key);

                // Retry after dbReadRetryDelay us
                usleep(config.dbReadRetryDelay);
                r = readTreeRemote(key, keys, level, numberOfFields);

                // Add to the read log, and restart the timer
                if (dbReadLog != NULL)
                {
                    dbReadLog->addGetTree(TimeDiff(t), numberOfFields);
                    gettimeofday(&t, NULL);
                }

                if (r == ZKR_SUCCESS)
                {
                    break;
                }
                zklog.warning("Database::read() retried readTreeRemote() after dbReadRetryDelay=" + to_string(config.dbReadRetryDelay) + "us and failed with error=" + zkresult2string(r) + " i=" + to_string(i));
            }
        }

        // If succeeded, now the value should be present in the cache
        if ( r == ZKR_SUCCESS)
        {
            if (usingAssociativeCache() && dbMTACache.findKey(vkey,value)){
                if (dbReadLog != NULL) dbReadLog->add(key, value, true, TimeDiff(t));
                r = ZKR_SUCCESS;
            }else if(dbMTCache.enabled() && dbMTCache.find(key, value)){
                if (dbReadLog != NULL) dbReadLog->add(key, value, true, TimeDiff(t));
                r = ZKR_SUCCESS;                
            }
            else
            {
                zklog.warning("Database::read() called readTreeRemote() but key=" + key + " is not present");
                r = ZKR_UNSPECIFIED;
            }
        }
        else r = ZKR_UNSPECIFIED;
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
        string sData;
        r = readRemote(false, key, sData);
        if ( (r != ZKR_SUCCESS) && (config.dbReadRetryDelay > 0) )
        {
            for (uint64_t i=0; i<config.dbReadRetryCounter; i++)
            {
                zklog.warning("Database::read() failed calling readRemote() with error=" + zkresult2string(r) + "; will retry after " + to_string(config.dbReadRetryDelay) + "us key=" + key + " i=" + to_string(i));

                // Retry after dbReadRetryDelay us
                usleep(config.dbReadRetryDelay);
                r = readRemote(false, key, sData);
                if (r == ZKR_SUCCESS)
                {
                    break;
                }
                zklog.warning("Database::read() retried readRemote() after dbReadRetryDelay=" + to_string(config.dbReadRetryDelay) + "us and failed with error=" + zkresult2string(r) + " i=" + to_string(i));
            }
        }
        if (r == ZKR_SUCCESS)
        {
            string2fea(fr, sData, value);

#ifdef DATABASE_USE_CACHE
            // Store it locally to avoid any future remote access for this key
            if(usingAssociativeCache()){
                dbMTACache.addKeyValue(vkey, value, update);
            }else if (dbMTCache.enabled()){
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
        zklog.error("Database::read() requested a key that does not exist (ZKR_DB_KEY_NOT_FOUND): " + key);
        r = ZKR_DB_KEY_NOT_FOUND;
    }

#ifdef LOG_DB_READ
    {
        string s = "Database::read()";
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

zkresult Database::write(const string &_key, const Goldilocks::Element* vkey, const vector<Goldilocks::Element> &value, const bool persistent)
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        zklog.error("Database::write() called uninitialized");
        exitProcess();
    }

    if (config.dbMultiWrite && !(dbMTCache.enabled() || dbMTACache.enabled()) && !persistent)
    {
        zklog.error("Database::write() called with multi-write active, cache disabled and no persistance in database, so there is no place to store the date");
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
        // Prepare the query
        string valueString = "";
        string aux;
        for (uint64_t i = 0; i < value.size(); i++)
        {
            valueString += PrependZeros(fr.toString(value[i], 16), 16);
        }

        r = writeRemote(false, key, valueString);
    }
    else
    {
        r = ZKR_SUCCESS;
    }

#ifdef DATABASE_USE_CACHE
    if ((r == ZKR_SUCCESS) && (dbMTCache.enabled() || dbMTACache.enabled()))
    {
        if(usingAssociativeCache()){
            Goldilocks::Element vkeyf[4];
            if(vkey == NULL){
                string2fea(fr, key, vkeyf);
            }else{
                vkeyf[0] = vkey[0];
                vkeyf[1] = vkey[1];
                vkeyf[2] = vkey[2];
                vkeyf[3] = vkey[3];
            }
            dbMTACache.addKeyValue(vkeyf, value, false);
        }else{
            dbMTCache.add(key, value, false);
        }
    }
#endif

#ifdef LOG_DB_WRITE
    {
        string s = "Database::write()";
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

void Database::initRemote(void)
{
    TimerStart(DB_INIT_REMOTE);

    try
    {
        // Build the remote database URI
        string uri = config.databaseURL;
        //zklog.info("Database URI: " + uri);

        // Create the database connections
        connLock();

        if (config.dbConnectionsPool)
        {
            // Check that we don't support more threads than available connections, including the sender thread
            if (config.dbNumberOfPoolConnections == 0)
            {
                zklog.error("Database::initRemote() found config.dbNumberOfPoolConnections=" + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }
            if ( config.runHashDBServer && ((config.maxHashDBThreads + 1) > config.dbNumberOfPoolConnections) )
            {
                zklog.error("Database::initRemote() found config.maxHashDBThreads + 1=" + to_string(config.maxHashDBThreads + 1) + " > config.dbNumberOfPoolConnections=" + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }
            if ( config.runExecutorServer && ((config.maxExecutorThreads + 1) > config.dbNumberOfPoolConnections) )
            {
                zklog.error("Database::initRemote() found config.maxExecutorThreads + 1=" + to_string(config.maxExecutorThreads + 1) + " > config.dbNumberOfPoolConnections=" + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }
            if ( config.runHashDBServer && config.runExecutorServer && ((config.maxHashDBThreads + config.maxExecutorThreads + 1) > config.dbNumberOfPoolConnections) )
            {
                zklog.error("Database::initRemote() found config.maxHashDBThreads + config.maxExecutorThreads + 1=" + to_string(config.maxHashDBThreads + config.maxExecutorThreads + 1) + " > config.dbNumberOfPoolConnections=" + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }

            // Allocate write connections pool
            connectionsPool = new DatabaseConnection[config.dbNumberOfPoolConnections];
            if (connectionsPool == NULL)
            {
                zklog.error("Database::initRemote() failed creating write connection pool of size " + to_string(config.dbNumberOfPoolConnections));
                exitProcess();
            }

            // Create write connections
            for (uint64_t i=0; i<config.dbNumberOfPoolConnections; i++)
            {
                connectionsPool[i].pConnection = new pqxx::connection{uri};
                if (connectionsPool[i].pConnection == NULL)
                {
                    zklog.error("Database::initRemote() failed creating write connection " + to_string(i));
                    exitProcess();
                }
                connectionsPool[i].bInUse = false;
                //zklog.info("Database::initRemote() created write connection i=" + to_string(i) + " connectionsPool[i]=" + to_string((uint64_t)connectionsPool[i].pConnection));
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
                zklog.error("Database::initRemote() failed creating unique connection");
                exitProcess();
            }
            connection.bInUse = false;
        }
        
        connUnlock();
    }
    catch (const std::exception &e)
    {
        zklog.error("Database::initRemote() exception: " + string(e.what()));
        exitProcess();
    }

    // If configured to use the get tree function, we must install it in the database before using it
    if (config.dbGetTree && !config.dbReadOnly)
    {
        writeGetTreeFunction();
    }

    // Create state root, only useful if database is empty
    if (!config.dbReadOnly)
    {
        createStateRoot();
    }

    TimerStopAndLog(DB_INIT_REMOTE);
}

DatabaseConnection * Database::getConnection (void)
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
            zklog.error("Database::getWriteConnection() run out of free connections");
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
        //zklog.info("Database::getWriteConnection() pConnection=" + to_string((uint64_t)pConnection) + " nextConnection=" + to_string(nextConnection) + " usedConnections=" + to_string(usedConnections));
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

void Database::disposeConnection (DatabaseConnection * pConnection)
{
    if (config.dbConnectionsPool)
    {
        connLock();
        zkassert(pConnection->bInUse == true);
        pConnection->bInUse = false;
        zkassert(usedConnections > 0);
        usedConnections--;
        //zklog.info("Database::disposeWriteConnection() pConnection=" + to_string((uint64_t)pConnection) + " nextConnection=" + to_string(nextConnection) + " usedConnections=" + to_string(usedConnections));
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

void Database::queryFailed (void)
{
    connLock();

    for (uint64_t i=0; i<config.dbNumberOfPoolConnections; i++)
    {
        connectionsPool[i].bDisconnect = true;
    }

    connUnlock();
}

zkresult Database::readRemote(bool bProgram, const string &key, string &value)
{
    const string &tableName = (bProgram ? config.dbProgramTableName : config.dbNodesTableName);

    if (config.logRemoteDbReads)
    {
        zklog.info("Database::readRemote() table=" + tableName + " key=" + key);
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
            zklog.error("Database::readRemote() table=" + tableName + " got more than one row for the same key: " + to_string(rows.size()));
            exitProcess();
        }

        const pqxx::row& row = rows[0];
        if (row.size() != 2)
        {
            zklog.error("Database::readRemote() table=" + tableName + " got an invalid number of colums for the row: " + to_string(row.size()));
            exitProcess();
        }
        const pqxx::field& fieldData = row[1];
        value = removeBSXIfExists(fieldData.c_str());
    }
    catch (const std::exception &e)
    {
        zklog.error("Database::readRemote() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
        queryFailed();
        disposeConnection(pDatabaseConnection);
        return ZKR_DB_ERROR;
    }
    
    // Dispose the read db conneciton
    disposeConnection(pDatabaseConnection);

    return ZKR_SUCCESS;
}

zkresult Database::readTreeRemote(const string &key, bool *keys, uint64_t level, uint64_t &numberOfFields)
{
    zkassert(keys != NULL);

    if (config.logRemoteDbReads)
    {
        zklog.info("Database::readTreeRemote() key=" + key);
    }
    string rkey;
    for (uint64_t i=level; i<256; i++)
    {
        uint8_t auxByte = (uint8_t)(keys[i]);
        if (auxByte > 1)
        {
            zklog.error("Database::readTreeRemote() found invalid keys value=" + to_string(auxByte) + " at position " + to_string(i));
            return ZKR_DB_ERROR;
        }
        rkey.append(1, byte2char(auxByte >> 4));
        rkey.append(1, byte2char(auxByte & 0x0F));
    }

    // Get a free read db connection
    DatabaseConnection * pDatabaseConnection = getConnection();

    numberOfFields = 0;

    try
    {
        // Prepare the query
        string query = "SELECT get_tree (E\'\\\\x" + key + "\', E\'\\\\x" + rkey + "\');";

        pqxx::result rows;

        // Start a transaction.
        pqxx::nontransaction n(*(pDatabaseConnection->pConnection));

        // Execute the query
        rows = n.exec(query);

        // Commit your transaction
        n.commit();

        // Process the result
        numberOfFields = rows.size();
        for (uint64_t i=0; i<numberOfFields; i++)
        {
            pqxx::row const row = rows[i];
            if (row.size() != 1)
            {
                zklog.error("Database::readTreeRemote() got an invalid number of colums for the row: " + to_string(row.size()));
                disposeConnection(pDatabaseConnection);
                return ZKR_UNSPECIFIED;
            }
            pqxx::field const fieldData = row[0];
            string fieldDataString = fieldData.c_str();
            //zklog.info("got value=" + fieldDataString);
            string hash, data;

            string first = "(\"\\\\x";
            string second = "\",\"\\\\x";
            string third = "\")";

            size_t firstPosition = fieldDataString.find(first);
            size_t secondPosition = fieldDataString.find(second);
            size_t thirdPosition = fieldDataString.find(third);

            if ( (firstPosition != 0) ||
                 (firstPosition + first.size() + 32*2 != secondPosition ) ||
                 (secondPosition <= first.size()) ||
                 (thirdPosition == 0) ||
                 ( (secondPosition + second.size() + 12*8*2 != thirdPosition) &&
                   (secondPosition + second.size() + 8*8*2 != thirdPosition) ))
            {
                zklog.error("Database::readTreeRemote() got an invalid field=" + fieldDataString);
                disposeConnection(pDatabaseConnection);
                return ZKR_UNSPECIFIED;
            }

            hash = fieldDataString.substr(firstPosition + first.size(), 32*2);
            data = fieldDataString.substr(secondPosition + second.size(), thirdPosition - secondPosition - second.size());
            vector<Goldilocks::Element> value;
            string2fea(fr, data, value);

#ifdef DATABASE_USE_CACHE
            // Store it locally to avoid any future remote access for this key
            if (dbMTCache.enabled() || dbMTACache.enabled())
            {
                //zklog.info("Database::readTreeRemote() adding hash=" + hash + " to dbMTCache");
                if(usingAssociativeCache()){
                    Goldilocks::Element vhash[4];
                    string2fea(fr, hash, vhash);   
                    dbMTACache.addKeyValue(vhash, value, false);
                }else{
                    dbMTCache.add(hash, value, false);
              }
            }
#endif
        }
    }
    catch (const std::exception &e)
    {
        zklog.warning("Database::readTreeRemote() exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
        queryFailed();
        disposeConnection(pDatabaseConnection);
        return ZKR_DB_ERROR;
    }
    
    // Dispose the read db conneciton
    disposeConnection(pDatabaseConnection);

    if (config.logRemoteDbReads)
    {
        zklog.info("Database::readTreeRemote() key=" + key + " read " + to_string(numberOfFields));
    }

    return ZKR_SUCCESS;
    
}

zkresult Database::writeRemote(bool bProgram, const string &key, const string &value)
{
    zkresult result = ZKR_SUCCESS;
    
    if (config.dbMultiWrite)
    {
        multiWrite.Lock();

        if (bProgram)
        {
            multiWrite.data[multiWrite.pendingToFlushDataIndex].programIntray[key] = value;
#ifdef LOG_DB_WRITE_REMOTE
            zklog.info("Database::writeRemote() key=" + key + " multiWrite=[" + multiWrite.print() + "]");
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
            zklog.error("Database::writeRemote() table=" + tableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
            result = ZKR_DB_ERROR;
            queryFailed();
        }

        disposeConnection(pDatabaseConnection);
    }

    return result;
}

zkresult Database::createStateRoot(void)
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
            zklog.error("Database::createStateRoot() table=" + config.dbNodesTableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
            r = ZKR_DB_ERROR;
            queryFailed();
        }

        disposeConnection(pDatabaseConnection);
    }

#ifdef LOG_DB_WRITE
    {
        string s = "Database::createStateRoot()";
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

zkresult Database::updateStateRoot(const Goldilocks::Element (&stateRoot)[4])
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        zklog.error("Database::updateStateRoot() called uninitialized");
        exitProcess();
    }

    // Copy the state root in the first 4 elements of dbValue
    vector<Goldilocks::Element> value;
    for (uint64_t i=0; i<4; i++) value.push_back(stateRoot[i]);
    for (uint64_t i=0; i<8; i++) value.push_back(fr.zero());
    
    // Prepare the value string
    string valueString = "";
    string aux;
    for (uint64_t i = 0; i < value.size(); i++)
    {
        valueString += PrependZeros(fr.toString(value[i], 16), 16);
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
                zklog.error("Database::updateStateRoot() table=" + config.dbNodesTableName + " exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
                r = ZKR_DB_ERROR;
                queryFailed();
            }

            disposeConnection(pDatabaseConnection);
        }
    }

#ifdef DATABASE_USE_CACHE
    if ((r == ZKR_SUCCESS) && (dbMTCache.enabled() || dbMTACache.enabled()))
    {
        // Create in memory cache
        if(usingAssociativeCache()){
                dbMTACache.addKeyValue(dbStateRootvKey, value, true);
        }else{
                dbMTCache.add(dbStateRootKey, value, true);
        }
    }
#endif

#ifdef LOG_DB_WRITE
    {
        string s = "Database::updateStateRoot()";
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

zkresult Database::writeGetTreeFunction(void)
{
    if (!config.dbGetTree)
    {
        zklog.error("Database::writeGetTreeFunction() dalled with config.dbGetTree=false");
        return ZKR_DB_ERROR;
    }
    
    if (config.databaseURL == "local")
    {
        zklog.error("Database::writeGetTreeFunction() dalled with config.databaseURL=local");
        return ZKR_DB_ERROR;
    }

    zkresult result = ZKR_SUCCESS;

    string query = string("") +
    "create or replace function get_tree (root_hash bytea, remaining_key bytea)\n" +
	"   returns setof state.nodes\n" +
	"   language plpgsql\n" +
    "as $$\n" +
    "declare\n" +
    "	current_hash bytea;\n" +
    "	current_row " + config.dbNodesTableName + "%rowtype;\n" +
    "	remaining_key_length integer;\n" +
    "	remaining_key_bit integer;\n" +
    "	byte_71 integer;\n" +
    "	aux_integer integer;\n" +
    "begin\n" +
    "	remaining_key_length = octet_length(remaining_key);\n" +
    "	current_hash = root_hash;\n" +

    "	-- For every bit (0 or 1) in remaining key\n" +
    "	for counter in 0..(remaining_key_length-1) loop\n" +

    "		-- Get the current_hash row and store it into current_row\n" +
    "		select * into current_row from " + config.dbNodesTableName + " where hash = current_hash;\n" +
    "		if not found then\n" +
    "			raise EXCEPTION 'Hash % not found', current_hash;\n" +
    "		end if;\n" +

    "		-- Return it as a result\n" +
    "		return next current_row;\n" +

    "		-- Data should be a byte array of 12x8 bytes (12 field elements)\n" +
    "		-- Check data length is exactly 12 field elements\n" +
    "		if (octet_length(current_row.data) != 12*8) then\n" +
    "			raise EXCEPTION 'Hash % got invalid data size %', current_hash, octet_length(current_row.data);\n" +
    "		end if;\n" +
	//	-- Check that last 3 field elements are zero
	//	--if (substring(current_row.data from 89 for 8) != E'\\x0000000000000000') then
	//	--	RAISE EXCEPTION 'Hash % got non-null 12th field element data=%', current_hash, current_row.data;
	//	--end if;
	//	--if (substring(current_row.data from 81 for 8) != E'\\x0000000000000000') then
	//	--	RAISE EXCEPTION 'Hash % got non-null 11th field element data=%', current_hash, current_row.data;
	//	--end if;
	//	--if (substring(current_row.data from 73 for 8) != E'\\x0000000000000000') then
	//	--	RAISE EXCEPTION 'Hash % got non-null 10th field element data=%', current_hash, current_row.data;
	//	--end if;
    "		-- If last 4 field elements are 0000, this is an intermediate node\n" +
    "		byte_71 = get_byte(current_row.data, 71);\n" +
    "		case byte_71\n" +
    "		when 0 then\n" +

    "			-- If the next remaining key is a 0, take the left sibling way, if it is a 1, take the right one\n" +
    "			remaining_key_bit = get_byte(remaining_key, counter);\n" +
    "			case remaining_key_bit\n" +
    "			when 0 then\n" +
    "				current_hash =\n" +
    "					substring(current_row.data from 25 for 8) ||\n" +
    "					substring(current_row.data from 17 for 8) ||\n" +
    "					substring(current_row.data from 9 for 8) ||\n" +
    "					substring(current_row.data from 1 for 8);\n" +
    "			when 1 then\n" +
    "				current_hash =\n" +
    "					substring(current_row.data from 57 for 8) ||\n" +
    "					substring(current_row.data from 49 for 8) ||\n" +
    "					substring(current_row.data from 41 for 8) ||\n" +
    "					substring(current_row.data from 33 for 8);\n" +
    "			else\n" +
    "				raise EXCEPTION 'Invalid remaining key bit at position % with value %', counter, remaining_key_bit ;\n" +
    "			end case;\n" +
    
    "			-- If the hash is a 0, we reached the end of the branch\n" +
    "			if (get_byte(current_hash, 0) = 0) and\n" +
    "			   (get_byte(current_hash, 1) = 0) and\n" +
    "			   (get_byte(current_hash, 2) = 0) and\n" +
    "			   (get_byte(current_hash, 3) = 0) and\n" +
    "			   (get_byte(current_hash, 4) = 0) and\n" +
    "			   (get_byte(current_hash, 5) = 0) and\n" +
    "			   (get_byte(current_hash, 6) = 0) and\n" +
    "			   (get_byte(current_hash, 7) = 0) and\n" +
    "			   (get_byte(current_hash, 8) = 0) and\n" +
    "			   (get_byte(current_hash, 9) = 0) and\n" +
    "			   (get_byte(current_hash, 10) = 0) and\n" +
    "			   (get_byte(current_hash, 11) = 0) and\n" +
    "			   (get_byte(current_hash, 12) = 0) and\n" +
    "			   (get_byte(current_hash, 13) = 0) and\n" +
    "			   (get_byte(current_hash, 14) = 0) and\n" +
    "			   (get_byte(current_hash, 15) = 0) and\n" +
    "			   (get_byte(current_hash, 16) = 0) and\n" +
    "			   (get_byte(current_hash, 17) = 0) and\n" +
    "			   (get_byte(current_hash, 18) = 0) and\n" +
    "			   (get_byte(current_hash, 19) = 0) and\n" +
    "			   (get_byte(current_hash, 20) = 0) and\n" +
    "			   (get_byte(current_hash, 21) = 0) and\n" +
    "			   (get_byte(current_hash, 22) = 0) and\n" +
    "			   (get_byte(current_hash, 23) = 0) and\n" +
    "			   (get_byte(current_hash, 24) = 0) and\n" +
    "			   (get_byte(current_hash, 25) = 0) and\n" +
    "			   (get_byte(current_hash, 26) = 0) and\n" +
    "			   (get_byte(current_hash, 27) = 0) and\n" +
    "			   (get_byte(current_hash, 28) = 0) and\n" +
    "			   (get_byte(current_hash, 29) = 0) and\n" +
    "			   (get_byte(current_hash, 30) = 0) and\n" +
    "			   (get_byte(current_hash, 31) = 0) then\n" +
    "			   return;\n" +
    "			end if;\n" +

    "		-- If last 4 field elements are 1000, this is a leaf node\n" +
    "		when 1 then	\n" +

    "			current_hash =\n" +
    "				substring(current_row.data from 57 for 8) ||\n" +
    "				substring(current_row.data from 49 for 8) ||\n" +
    "				substring(current_row.data from 41 for 8) ||\n" +
    "				substring(current_row.data from 33 for 8);\n" +
    "			select * into current_row from " + config.dbNodesTableName + " where hash = current_hash;\n" +
    "			if not found then\n" +
    "				raise EXCEPTION 'Hash % not found', current_hash;\n" +
    "			end if;\n" +
    "			return next current_row;\n" +
    "			return;\n" +

    "		else\n" +
    "			raise EXCEPTION 'Hash % got invalid 9th field element data=%', current_hash, current_row.data;\n" +
    "		end case;\n" +
			
    "	end loop;\n" +

    "	return;\n" +
    "end;$$\n";
        
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
        zklog.error("Database::writeGetTreeFunction() exception: " + string(e.what()) + " connection=" + to_string((uint64_t)pDatabaseConnection));
        result = ZKR_DB_ERROR;
        queryFailed();
    }
    
    disposeConnection(pDatabaseConnection);

    zklog.info("Database::writeGetTreeFunction() returns " + zkresult2string(result));
        
    return result;
}

zkresult Database::setProgram (const string &_key, const vector<uint8_t> &data, const bool persistent)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database::setProgram() called uninitialized");
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
        string s = "Database::setProgram()";
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

zkresult Database::getProgram(const string &_key, vector<uint8_t> &data, DatabaseMap *dbReadLog)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database::getProgram() called uninitialized");
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
        zklog.error("Database::getProgram() requested a key that does not exist: " + key);
        r = ZKR_DB_KEY_NOT_FOUND;
    }

#ifdef LOG_DB_READ
    {
        string s = "Database::getProgram()";
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
    
zkresult Database::flush(uint64_t &thisBatch, uint64_t &lastSentBatch)
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
    zklog.info("Database::flush() thisBatch=" + to_string(thisBatch) + " lastSentBatch=" + to_string(lastSentBatch) + " multiWrite=[" + multiWrite.print() + "]");
#endif

    // Notify the thread
    sem_post(&senderSem);

    multiWrite.Unlock();
    return ZKR_SUCCESS;
}

void Database::semiFlush (void)
{
    if (!config.dbMultiWrite)
    {
        return;
    }

    multiWrite.Lock();

    multiWrite.data[multiWrite.pendingToFlushDataIndex].acceptIntray();

#ifdef LOG_DB_SEMI_FLUSH
    zklog.info("Database::semiFlush() called multiWrite=[" + multiWrite.print() + "]");
#endif

    multiWrite.Unlock();
}

zkresult Database::getFlushStatus(uint64_t &storedFlushId, uint64_t &storingFlushId, uint64_t &lastFlushId, uint64_t &pendingToFlushNodes, uint64_t &pendingToFlushProgram, uint64_t &storingNodes, uint64_t &storingProgram)
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

zkresult Database::sendData (void)
{
    // If we have read-only access to database, just pretend to have sent all data
    if (config.dbReadOnly)
    {
        // If we succeeded, update last sent batch
        multiWrite.Lock();
        multiWrite.storedFlushId = multiWrite.storingFlushId;
        multiWrite.Unlock();

        return ZKR_SUCCESS;
    }

    zkresult zkr = ZKR_SUCCESS;
    
    // Time calculation variables
    struct timeval t;
    uint64_t timeDiff = 0;
    uint64_t fields = 0;

    // Select proper data instance
    MultiWriteData &data = multiWrite.data[multiWrite.storingDataIndex];

    // Check if there is data
    if (data.IsEmpty())
    {
        zklog.warning("Database::sendData() called with empty data");
        return ZKR_SUCCESS;
    }

    // Check if it has already been stored to database
    if (data.stored)
    {
        zklog.warning("Database::sendData() called with stored=true");
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
                        data.multiQuery.queries[currentQuery].query += "( E\'\\\\x" + it->first + "\', E\'\\\\x" + it->second + "\' ) ";
#ifdef LOG_DB_SEND_DATA
                        zklog.info("Database::sendData() inserting node key=" + it->first + " value=" + it->second);
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
                        zklog.info("Database::sendData() inserting program key=" + it->first + " value=" + it->second);
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
                zklog.info("Database::sendData() inserting root=" + data.nodesStateRoot);
#endif
            }
        }

        if (data.multiQuery.isEmpty())
        {
            zklog.warning("Database::sendData() called without any data to send");
            data.stored = true;
        }
        else
        {
            if (config.dbMetrics)
            {
                fields = data.nodes.size() + data.program.size() + (data.nodesStateRoot.size() > 0 ? 1 : 0);
                zklog.info("Database::sendData() dbMetrics multiWrite nodes=" + to_string(data.nodes.size()) +
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

            //zklog.info("Database::flush() sent query=" + query);
            if (config.dbMetrics)
            {
                timeDiff = TimeDiff(t);
                zklog.info("Database::sendData() dbMetrics multiWrite total=" + to_string(fields) + "fields=" + to_string(timeDiff) + "us=" + to_string(timeDiff/zkmax(fields,1)) + "us/field");
            }

#ifdef LOG_DB_WRITE_QUERY
            {
                string query;
                for (uint64_t i=0; i<data.multiQuery.queries.size(); i++)
                {
                    query += data.multiQuery.queries[i].query;
                }
                zklog.info("Database::sendData() write query=" + query);
            }
#endif
#ifdef LOG_DB_SEND_DATA
            zklog.info("Database::sendData() successfully processed query of size= " + to_string(data.multiQuery.size()));
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
        zklog.error("Database::sendData() execute query exception: " + string(e.what()));
        zklog.error("Database::sendData() query.size=" + to_string(data.multiQuery.queries.size()) + (data.multiQuery.isEmpty() ? "" : (" query(<1024)=" + data.multiQuery.queries[0].query.substr(0, 1024))));
        queryFailed();
        zkr = ZKR_DB_ERROR;
    }

    // Dispose the write db connection
    disposeConnection(pDatabaseConnection);

    return zkr;
}

// Get flush data, written to database by dbSenderThread; it blocks
zkresult Database::getFlushData(uint64_t flushId, uint64_t &storedFlushId, unordered_map<string, string> (&nodes), unordered_map<string, string> (&program), string &nodesStateRoot)
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
        zklog.info("Database::getFlushData() timed out");
        return ZKR_SUCCESS;
    }

    multiWrite.Lock();
    MultiWriteData &data = multiWrite.data[multiWrite.synchronizingDataIndex];

    zklog.info("Database::getFlushData woke up: pendingToFlushDataIndex=" + to_string(multiWrite.pendingToFlushDataIndex) +
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

void Database::setAutoCommit(const bool ac)
{
    if (ac && !autoCommit)
        commit();
    autoCommit = ac;
}

void Database::commit()
{
    if ((!autoCommit) && (transaction != NULL))
    {
        transaction->commit();
        delete transaction;
        transaction = NULL;
    }
}

#endif

void Database::printTree(const string &root, string prefix)
{
    if (prefix == "")
    {
        zklog.info("Printint tree of root=" + root);
    }
    string key = root;
    vector<Goldilocks::Element> value;
    Goldilocks::Element vKey[4];
    if(Database::useAssociativeCache) string2fea(fr, key, vKey);  
    read(key,vKey,value, NULL);

    if (value.size() != 12)
    {
        zklog.error("Database::printTree() found value.size()=" + to_string(value.size()));
        return;
    }
    if (!fr.equal(value[11], fr.zero()))
    {
        zklog.error("Database::printTree() found value[11]=" + fr.toString(value[11], 16));
        return;
    }
    if (!fr.equal(value[10], fr.zero()))
    {
        zklog.error("Database::printTree() found value[10]=" + fr.toString(value[10], 16));
        return;
    }
    if (!fr.equal(value[9], fr.zero()))
    {
        zklog.error("Database::printTree() found value[9]=" + fr.toString(value[9], 16));
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
                zklog.error("Database::printTree() found leafValue[8]=" + fr.toString(leafValue[8], 16));
                return;
            }
            if (!fr.equal(leafValue[9], fr.zero()))
            {
                zklog.error("Database::printTree() found leafValue[9]=" + fr.toString(leafValue[9], 16));
                return;
            }
            if (!fr.equal(leafValue[10], fr.zero()))
            {
                zklog.error("Database::printTree() found leafValue[10]=" + fr.toString(leafValue[10], 16));
                return;
            }
            if (!fr.equal(leafValue[11], fr.zero()))
            {
                zklog.error("Database::printTree() found leafValue[11]=" + fr.toString(leafValue[11], 16));
                return;
            }
        }
        else if (leafValue.size() == 8)
        {
            zklog.info(prefix + "leafValue.size()=" + to_string(leafValue.size()));
        }
        else
        {
            zklog.error("Database::printTree() found lleafValue.size()=" + to_string(leafValue.size()));
            return;
        }
        mpz_class scalarValue;
        fea2scalar(fr, scalarValue, leafValue[0], leafValue[1], leafValue[2], leafValue[3], leafValue[4], leafValue[5], leafValue[6], leafValue[7]);
        zklog.info(prefix + "leafValue=" + PrependZeros(scalarValue.get_str(16), 64));
    }
    else
    {
        zklog.error("Database::printTree() found value[8]=" + fr.toString(value[8], 16));
        return;
    }
    if (prefix == "") zklog.info("");
}

void Database::clearCache (void)
{
    dbMTCache.clear();
    dbProgramCache.clear();
    dbMTACache.clear();
}

void *dbSenderThread (void *arg)
{
    Database *pDatabase = (Database *)arg;
    zklog.info("dbSenderThread() started");
    MultiWrite &multiWrite = pDatabase->multiWrite;

    while (true)
    {
        // Wait for the sending semaphore to be released, if there is no more data to send
        struct timespec currentTime;
        int iResult = clock_gettime(CLOCK_REALTIME, &currentTime);
        if (iResult == -1)
        {
            zklog.error("dbSenderThread() failed calling clock_gettime()");
            exitProcess();
        }

        currentTime.tv_sec += 5;
        sem_timedwait(&pDatabase->senderSem, &currentTime);

        multiWrite.Lock();

        bool bDataEmpty = false;

        // If sending data is not empty (it failed before) then try to send it again
        if (!multiWrite.data[multiWrite.storingDataIndex].multiQuery.isEmpty())
        {
            zklog.warning("dbSenderThread() found sending data index not empty, probably because of a previous error; resuming...");
        }
        // If processing data is empty, then simply pretend to have sent data
        else if (multiWrite.data[multiWrite.pendingToFlushDataIndex].IsEmpty())
        {
            //zklog.warning("dbSenderThread() found pending to flush data empty");

            // Mark as if we sent all batches
            multiWrite.storedFlushId = multiWrite.lastFlushId;
#ifdef LOG_DB_SENDER_THREAD
            zklog.info("dbSenderThread() found multi write processing data empty, so ignoring");
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
            zklog.info("dbSenderThread() updated: multiWrite=[" + multiWrite.print() + "]");
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
                zklog.info("dbSenderThread() no data to send: multiWrite=[" + multiWrite.print() + "]");
#endif
                bDataEmpty = true;
            }

        }

        // Unlock to let more processing batch data in
        multiWrite.Unlock();

        if (!bDataEmpty)
        {
#ifdef LOG_DB_SENDER_THREAD
            zklog.info("dbSenderThread() starting to send data, multiWrite=[" + multiWrite.print() + "]");
#endif
            zkresult zkr;
            zkr = pDatabase->sendData();
            if (zkr == ZKR_SUCCESS)
            {
                multiWrite.Lock();
                multiWrite.storedFlushId = multiWrite.storingFlushId;
#ifdef LOG_DB_SENDER_THREAD
                zklog.info("dbSenderThread() successfully sent data, multiWrite=[]" + multiWrite.print() + "]");
#endif
                // Advance synchronizing index
                multiWrite.synchronizingDataIndex = (multiWrite.synchronizingDataIndex + 1) % 3;
#ifdef LOG_DB_SENDER_THREAD
                zklog.info("dbSenderThread() updated: multiWrite=[" + multiWrite.print() + "]");
#endif
                sem_post(&pDatabase->getFlushDataSem);
#ifdef LOG_DB_SENDER_THREAD
                zklog.info("dbSenderThread() successfully called sem_post(&pDatabase->getFlushDataSem)");
#endif
                multiWrite.Unlock();
            }
            else
            {
                zklog.error("dbSenderThread() failed calling sendData() error=" + zkresult2string(zkr));
                usleep(1000000);
            }
        }
    }

    zklog.info("dbSenderThread() done");
    return NULL;
}

void *dbCacheSynchThread (void *arg)
{
    Database *pDatabase = (Database *)arg;
    zklog.info("dbCacheSynchThread() started");

    uint64_t storedFlushId = 0;

    Config config = pDatabase->config;
    config.hashDBURL = config.dbCacheSynchURL;

    while (true)
    {
        HashDBInterface *pHashDBRemote = new HashDBRemote (pDatabase->fr, config);
        if (pHashDBRemote == NULL)
        {
            zklog.error("dbCacheSynchThread() failed calling new HashDBRemote()");
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
                zklog.error("dbCacheSynchThread() failed calling pHashDB->getFlushData() result=" + zkresult2string(zkr));
                sleep(10);
                break;
            }

            if (nodes.size()==0 && program.size()==0 && nodesStateRoot.size()==0)
            {
                zklog.info("dbCacheSynchThread() called getFlushData() remotely and got no data: storedFlushId=" + to_string(storedFlushId));
                continue;
            }

            TimerStart(DATABASE_CACHE_SYNCH);
            zklog.info("dbCacheSynchThread() called getFlushData() remotely and got: storedFlushId=" + to_string(storedFlushId) + " nodes=" + to_string(nodes.size()) + " program=" + to_string(program.size()) + " nodesStateRoot=" + nodesStateRoot);

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

    zklog.info("dbCacheSynchThread() done");
    return NULL;
}

void loadDb2MemCache(const Config &config)
{
    if (config.databaseURL == "local")
    {
        zklog.error("loadDb2MemCache() called with config.databaseURL==local");
        exitProcess();
    }

#ifdef DATABASE_USE_CACHE

    TimerStart(LOAD_DB_TO_CACHE);

    Goldilocks fr;
    HashDB * pHashDB = (HashDB *)hashDBSingleton.get();

    vector<Goldilocks::Element> dbValue;
    zkresult zkr = pHashDB->db.read(Database::dbStateRootKey, Database::dbStateRootvKey, dbValue, NULL, true);

    if (zkr == ZKR_DB_KEY_NOT_FOUND)
    {
        zklog.warning("loadDb2MemCache() dbStateRootKey=" +  Database::dbStateRootKey + " not found in database; normal only if database is empty");
        TimerStopAndLog(LOAD_DB_TO_CACHE);
        return;
    }
    else if (zkr != ZKR_SUCCESS)
    {
        zklog.error("loadDb2MemCache() failed calling db.read result=" + zkresult2string(zkr));
        TimerStopAndLog(LOAD_DB_TO_CACHE);
        return;
    }
    
    string stateRootKey = fea2string(fr, dbValue[0], dbValue[1], dbValue[2], dbValue[3]);
    zklog.info("loadDb2MemCache() found state root=" + stateRootKey);

    if (stateRootKey == "0")
    {
        zklog.warning("loadDb2MemCache() found an empty tree");
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
            string hashNorm = NormalizeToNFormat(hash, 64);
            if(pHashDB->db.usingAssociativeCache()) string2fea(fr, hashNorm, vhash);
            zkresult zkr = pHashDB->db.read(hash, vhash, dbValue, NULL, true);

            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("loadDb2MemCache() failed calling db.read(" + hash + ") result=" + zkresult2string(zkr));
                TimerStopAndLog(LOAD_DB_TO_CACHE);
                return;
            }
            if (dbValue.size() != 12)
            {
                zklog.error("loadDb2MemCache() failed calling db.read(" + hash + ") dbValue.size()=" + to_string(dbValue.size()));
                TimerStopAndLog(LOAD_DB_TO_CACHE);
                return;
            }
            counter++;
            if(Database::dbMTCache.enabled()){
                double sizePercentage = double(Database::dbMTCache.getCurrentSize())*100.0/double(Database::dbMTCache.getMaxSize());
                if ( sizePercentage > 90 )
                {
                    zklog.info("loadDb2MemCache() stopping since size percentage=" + to_string(sizePercentage));
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
                            zklog.error("loadDb2MemCache() failed calling db.read(" + rightHash + ") result=" + zkresult2string(zkr));
                            TimerStopAndLog(LOAD_DB_TO_CACHE);
                            return;
                        }
                        counter++;
                    }
                }
            }
        }
    }

    if(Database::dbMTCache.enabled()){
        zklog.info("loadDb2MemCache() done counter=" + to_string(counter) + " cache at " + to_string((double(Database::dbMTCache.getCurrentSize())/double(Database::dbMTCache.getMaxSize()))*100) + "%");
    }
    TimerStopAndLog(LOAD_DB_TO_CACHE);

#endif
}