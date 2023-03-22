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
#include "statedb_singleton.hpp"

#ifdef DATABASE_USE_CACHE

// Create static Database::dbMTCache and DatabaseCacheProgram objects
// This will be used to store DB records in memory and it will be shared for all the instances of Database class
// DatabaseCacheMT and DatabaseCacheProgram classes are thread-safe
DatabaseMTCache Database::dbMTCache;
DatabaseProgramCache Database::dbProgramCache;

string Database::dbStateRootKey("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"); // 64 f's

#endif

// Helper functions
string removeBSXIfExists(string s) {return ((s.at(0) == '\\') && (s.at(1) == 'x')) ? s.substr(2) : s;}

// Database class implementation
void Database::init(void)
{
    // Check that it has not been initialized before
    if (bInitialized)
    {
        cerr << "Error: Database::init() called when already initialized" << endl;
        exitProcess();
    }

#ifdef DATABASE_USE_CACHE
    useDBMTCache = dbMTCache.enabled();
    useDBProgramCache = dbProgramCache.enabled();
#endif

    // Configure the server, if configuration is provided
    if (config.databaseURL != "local")
    {
        initRemote();
        useRemoteDB = true;
    } else useRemoteDB = false;

    // Mark the database as initialized
    bInitialized = true;
}

zkresult Database::read(const string &_key, vector<Goldilocks::Element> &value, DatabaseMap *dbReadLog, const bool update)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::read() called uninitialized" << endl;
        exitProcess();
    }

    zkresult r;

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

#ifdef DATABASE_USE_CACHE
    // If the key is found in local database (cached) simply return it
    if ((useDBMTCache) && (Database::dbMTCache.find(key, value)))
    {
        // Add to the read log
        if (dbReadLog != NULL) dbReadLog->add(key, value);

        r = ZKR_SUCCESS;
    }
    else
#endif
    if (useRemoteDB)
    {
        // If multi write is enabled, flush pending data, since some previously written keys
        // could be in the multi write string but flushed from the cache
        if (config.dbMultiWrite)
        {
            flush();
        }

        // Otherwise, read it remotelly
        string sData;
        r = readRemote(false, key, sData);
        if (r == ZKR_SUCCESS)
        {
            string2fea(fr, sData, value);

#ifdef DATABASE_USE_CACHE
            // Store it locally to avoid any future remote access for this key
            if (useDBMTCache) Database::dbMTCache.add(key, value, update);
#endif

            // Add to the read log
            if (dbReadLog != NULL) dbReadLog->add(key, value);
        }
    }
    else
    {
        cerr << "Error: Database::read() requested a key that does not exist: " << key << endl;
        r = ZKR_DB_KEY_NOT_FOUND;
    }

#ifdef LOG_DB_READ
    cout << "Database::read()";
    if (r != ZKR_SUCCESS)
        cout << " ERROR=" << r << " (" << zkresult2string(r) << ")";
    cout << " key=" << key;
    cout << " value=";
    for (uint64_t i = 0; i < value.size(); i++)
        cout << fr.toString(value[i], 16) << ":";
    cout << endl;
#endif

    return r;
}

zkresult Database::write(const string &_key, const vector<Goldilocks::Element> &value, const bool persistent, const bool update)
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::write() called uninitialized" << endl;
        exitProcess();
    }

    if (config.dbMultiWrite && !useDBMTCache && !persistent)
    {
        cerr << "Error: Database::write() called with multi-write active, cache disabled and no persistance in database, so there is no place to store the date" << endl;
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

        r = writeRemote(false, key, valueString, update);
    }
    else
    {
        r = ZKR_SUCCESS;
    }

#ifdef DATABASE_USE_CACHE
    if ((r == ZKR_SUCCESS) && (useDBMTCache))
    {
        // Create in memory cache
        Database::dbMTCache.add(key, value, update);
    }
#endif

#ifdef LOG_DB_WRITE
    cout << "Database::write()";
    if (r != ZKR_SUCCESS)
        cout << " ERROR=" << r << " (" << zkresult2string(r) << ")";
    cout << " key=" << key;
    cout << " value=";
    for (uint64_t i = 0; i < value.size(); i++)
        cout << fr.toString(value[i], 16) << ":";
    cout << " persistent=" << persistent << " update=" << update << endl;
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
        //cout << "Database URI: " << uri << endl;

        // Create the database connections
        connLock();

        if (config.dbConnectionsPool)
        {
            // Check that we don't support more threads than available connections
            if ( config.runStateDBServer && (config.maxStateDBThreads > config.dbNumberOfPoolConnections) )
            {
                cerr << "Error: Database::initRemote() found config.maxStateDBThreads=" << config.maxStateDBThreads << " > config.dbNumberOfPoolConnections=" << config.dbNumberOfPoolConnections << endl;
                exitProcess();
            }
            if ( config.runExecutorServer && (config.maxExecutorThreads > config.dbNumberOfPoolConnections) )
            {
                cerr << "Error: Database::initRemote() found config.maxExecutorThreads=" << config.maxExecutorThreads << " > config.dbNumberOfPoolConnections=" << config.dbNumberOfPoolConnections << endl;
                exitProcess();
            }
            if ( config.runStateDBServer && config.runExecutorServer && ((config.maxStateDBThreads+config.maxExecutorThreads) > config.dbNumberOfPoolConnections) )
            {
                cerr << "Error: Database::initRemote() found config.maxStateDBThreads+config.maxExecutorThreads=" << config.maxStateDBThreads+config.maxExecutorThreads << " > config.dbNumberOfPoolConnections=" << config.dbNumberOfPoolConnections << endl;
                exitProcess();
            }

            // Allocate write connections pool
            connectionsPool = new DatabaseConnection[config.dbNumberOfPoolConnections];
            if (connectionsPool == NULL)
            {
                cerr << "Error: Database::initRemote() failed creating write connection pool of size " << config.dbNumberOfPoolConnections << endl;
                exitProcess();
            }

            // Create write connections
            for (uint64_t i=0; i<config.dbNumberOfPoolConnections; i++)
            {
                connectionsPool[i].pConnection = new pqxx::connection{uri};
                if (connectionsPool[i].pConnection == NULL)
                {
                    cerr << "Error: Database::initRemote() failed creating write connection " << i << endl;
                    exitProcess();
                }
                connectionsPool[i].bInUse = false;
                //cout << "Database::initRemote() created write connection i=" << i << " connectionsPool[i]=" <<connectionsPool[i].pConnection << endl;
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
                cerr << "Error: Database::initRemote() failed creating unique connection" << endl;
                exitProcess();
            }
            connection.bInUse = false;
        }
        
        connUnlock();
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::initRemote() exception: " << e.what() << endl;
        exitProcess();
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
            cerr << "Error: Database::getWriteConnection() run out of free connections" << endl;
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
        //cout << "Database::getWriteConnection() pConnection=" << pConnection << " nextConnection=" << to_string(nextConnection) << " usedConnections=" << to_string(usedConnections) << endl;
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
        //cout << "Database::disposeWriteConnection() pConnection=" << pConnection << " nextConnection=" << to_string(nextConnection) << " usedConnections=" << to_string(usedConnections) << endl;
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

zkresult Database::readRemote(bool bProgram, const string &key, string &value)
{
    const string &tableName = (bProgram ? config.dbProgramTableName : config.dbNodesTableName);

    if (config.logRemoteDbReads)
    {
        cout << "   Database::readRemote() table=" << tableName << " key=" << key << endl;
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
            cerr << "Error: Database::readRemote() table="<< tableName << " got more than one row for the same key: " << rows.size() << endl;
            exitProcess();
        }

        pqxx::row const row = rows[0];
        if (row.size() != 2)
        {
            cerr << "Error: Database::readRemote() table="<< tableName << " got an invalid number of colums for the row: " << row.size() << endl;
            exitProcess();
        }
        pqxx::field const fieldData = row[1];
        value = removeBSXIfExists(fieldData.c_str());
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::readRemote() table="<< tableName << " exception: " << e.what() << " connection=" << pDatabaseConnection << endl;
        exitProcess();
    }
    
    // Dispose the read db conneciton
    disposeConnection(pDatabaseConnection);

    return ZKR_SUCCESS;
}

zkresult Database::writeRemote(bool bProgram, const string &key, const string &value, const bool update)
{
    zkresult result = ZKR_SUCCESS;

    const string &tableName = (bProgram ? config.dbProgramTableName : config.dbNodesTableName);
    
    if (config.dbMultiWrite)
    {
        if (update && (key==dbStateRootKey))
        {
            multiWriteLock();
            multiWriteNodesStateRoot = "INSERT INTO " + tableName + " ( hash, data ) VALUES ( E\'\\\\x" + key + "\', E\'\\\\x" + value + "\' ) ";
            multiWriteUnlock();       
        }
        else
        {
            string &multiWrite = bProgram ? (update ? multiWriteProgramUpdate : multiWriteProgram) : (update ? multiWriteNodesUpdate : multiWriteNodes);
            multiWriteLock();
            if (multiWrite.size() == 0)
            {
                multiWrite = "INSERT INTO " + tableName + " ( hash, data ) VALUES ( E\'\\\\x" + key + "\', E\'\\\\x" + value + "\' ) ";
            }
            else
            {
                multiWrite += ", ( E\'\\\\x" + key + "\', E\'\\\\x" + value + "\' )";
            }
            multiWriteUnlock();       
        }
    }
    else
    {
        string query = "INSERT INTO " + tableName + " ( hash, data ) VALUES ( E\'\\\\x" + key + "\', E\'\\\\x" + value + "\' ) " +
                    (update ? "ON CONFLICT (hash) DO UPDATE SET data = EXCLUDED.data;" : "ON CONFLICT (hash) DO NOTHING;");
            
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
                disposeConnection(pDatabaseConnection);
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
            cerr << "Error: Database::writeRemote() table="<< tableName << " exception: " << e.what() << " connection=" << pDatabaseConnection << endl;
            result = ZKR_DB_ERROR;
        }
    }

    return result;
}

zkresult Database::setProgram(const string &_key, const vector<uint8_t> &data, const bool persistent, const bool update)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::setProgram() called uninitialized" << endl;
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

        r = writeRemote(true, key, sData, update);
    }
    else
    {
        r = ZKR_SUCCESS;
    }

#ifdef DATABASE_USE_CACHE
    if ((r == ZKR_SUCCESS) && (useDBProgramCache))
    {
        // Create in memory cache
        Database::dbProgramCache.add(key, data, update);
    }
#endif

#ifdef LOG_DB_WRITE
    cout << "Database::setProgram()";
    if (r != ZKR_SUCCESS)
        cout << " ERROR=" << r << " (" << zkresult2string(r) << ")";
    cout << " key=" << key;
    cout << " data=";
    for (uint64_t i = 0; (i < (data.size()) && (i < 100)); i++)
        cout << byte2string(data[i]);
    if (data.size() > 100) cout << "...";
    cout << " persistent=" << persistent << " update=" << update << endl;
#endif

    return r;
}

zkresult Database::getProgram(const string &_key, vector<uint8_t> &data, DatabaseMap *dbReadLog, const bool update)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::getProgram() called uninitialized" << endl;
        exitProcess();
    }

    zkresult r;

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

#ifdef DATABASE_USE_CACHE
    // If the key is found in local database (cached) simply return it
    if (useDBProgramCache && !update && Database::dbProgramCache.find(key, data))
    {
        // Add to the read log
        if (dbReadLog != NULL) dbReadLog->add(key, data);

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
            if (useDBProgramCache) Database::dbProgramCache.add(key, data, update);
#endif

            // Add to the read log
            if (dbReadLog != NULL) dbReadLog->add(key, data);
        }
    }
    else
    {
        cerr << "Error: Database::getProgram() requested a key that does not exist: " << key << endl;
        r = ZKR_DB_KEY_NOT_FOUND;
    }

#ifdef LOG_DB_READ
    cout << "Database::getProgram()";
    if (r != ZKR_SUCCESS)
        cout << " ERROR=" << r << " (" << zkresult2string(r) << ")";
    cout << " key=" << key;
    cout << " data=";
    for (uint64_t i = 0; (i < (data.size()) && (i < 100)); i++)
        cout << byte2string(data[i]);
    if (data.size() > 100) cout << "...";
    cout << " update=" << update << endl;
#endif

    return r;
}

zkresult Database::flush()
{
    if (!config.dbMultiWrite)
    {
        return ZKR_SUCCESS;
    }

    //TimerStart(DATABASE_FLUSH);

    zkresult zkr = ZKR_SUCCESS;

    multiWriteLock();

    if ( (multiWriteNodes.size() > 0) || (multiWriteNodesStateRoot.size() > 0) || (multiWriteNodesUpdate.size() > 0) || (multiWriteProgram.size() > 0) || (multiWriteProgramUpdate.size() > 0) )
    {

        // Get a free write db connection
        DatabaseConnection * pDatabaseConnection = getConnection();

        try
        {
            string query;
            if (multiWriteProgram.size() > 0)
            {
                query = multiWriteProgram + " ON CONFLICT (hash) DO NOTHING;";
                
                // Start a transaction
                pqxx::work w(*(pDatabaseConnection->pConnection));

                // Execute the query
                pqxx::result res = w.exec(query);

                // Commit your transaction
                w.commit();

                //cout << "Database::flush() sent " << query << endl;

                // Delete the accumulated query data only if the query succeeded
                multiWriteProgram.clear();
            }
            if (multiWriteProgramUpdate.size() > 0)
            {
                query = multiWriteProgramUpdate + " ON CONFLICT (hash) DO UPDATE SET data = EXCLUDED.data;";
                
                // Start a transaction
                pqxx::work w(*(pDatabaseConnection->pConnection));

                // Execute the query
                pqxx::result res = w.exec(query);

                // Commit your transaction
                w.commit();

                //cout << "Database::flush() sent " << query << endl;

                // Delete the accumulated query data only if the query succeeded
                multiWriteProgramUpdate.clear();
            }
            if (multiWriteNodes.size() > 0)
            {
                query = multiWriteNodes + " ON CONFLICT (hash) DO NOTHING;";
                
                // Start a transaction
                pqxx::work w(*(pDatabaseConnection->pConnection));

                // Execute the query
                pqxx::result res = w.exec(query);

                // Commit your transaction
                w.commit();

                //cout << "Database::flush() sent " << query << endl;

                // Delete the accumulated query data only if the query succeeded
                multiWriteNodes.clear();
            }
            if (multiWriteNodesUpdate.size() > 0)
            {
                query = multiWriteNodesUpdate + " ON CONFLICT (hash) DO UPDATE SET data = EXCLUDED.data;";
                
                // Start a transaction
                pqxx::work w(*(pDatabaseConnection->pConnection));

                // Execute the query
                pqxx::result res = w.exec(query);

                // Commit your transaction
                w.commit();

                //cout << "Database::flush() sent " << query << endl;

                // Delete the accumulated query data only if the query succeeded
                multiWriteNodesUpdate.clear();
            }
            if (multiWriteNodesStateRoot.size() > 0)
            {
                query = multiWriteNodesStateRoot + " ON CONFLICT (hash) DO UPDATE SET data = EXCLUDED.data;";
                
                // Start a transaction
                pqxx::work w(*(pDatabaseConnection->pConnection));

                // Execute the query
                pqxx::result res = w.exec(query);

                // Commit your transaction
                w.commit();

                //cout << "Database::flush() sent " << query << endl;

                // Delete the accumulated query data only if the query succeeded
                multiWriteNodesStateRoot.clear();
            }
        }
        catch (const std::exception &e)
        {
            cerr << "Error: Database::flush() execute query exception: " << e.what() << endl;
            zkr = ZKR_DB_ERROR;
        }

        // Dispose the write db connection
        disposeConnection(pDatabaseConnection);
    }
    multiWriteUnlock();

    //TimerStopAndLog(DATABASE_FLUSH);
    
    return zkr;
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
        cout << "Printint tree of root=" << root << endl;
    string key = root;
    vector<Goldilocks::Element> value;
    read(key, value, NULL);
    if (value.size() != 12)
    {
        cerr << "Error: Database::printTree() found value.size()=" << value.size() << endl;
        return;
    }
    if (!fr.equal(value[11], fr.zero()))
    {
        cerr << "Error: Database::printTree() found value[11]=" << fr.toString(value[11], 16) << endl;
        return;
    }
    if (!fr.equal(value[10], fr.zero()))
    {
        cerr << "Error: Database::printTree() found value[10]=" << fr.toString(value[10], 16) << endl;
        return;
    }
    if (!fr.equal(value[9], fr.zero()))
    {
        cerr << "Error: Database::printTree() found value[9]=" << fr.toString(value[9], 16) << endl;
        return;
    }
    if (fr.equal(value[8], fr.zero())) // Intermediate node
    {
        string leftKey = fea2string(fr, value[0], value[1], value[2], value[3]);
        cout << prefix << "Intermediate node - left hash=" << leftKey << endl;
        if (leftKey != "0")
            printTree(leftKey, prefix + "  ");
        string rightKey = fea2string(fr, value[4], value[5], value[6], value[7]);
        cout << prefix << "Intermediate node - right hash=" << rightKey << endl;
        if (rightKey != "0")
            printTree(rightKey, prefix + "  ");
    }
    else if (fr.equal(value[8], fr.one())) // Leaf node
    {
        string rKey = fea2string(fr, value[0], value[1], value[2], value[3]);
        cout << prefix << "rKey=" << rKey << endl;
        string hashValue = fea2string(fr, value[4], value[5], value[6], value[7]);
        cout << prefix << "hashValue=" << hashValue << endl;
        vector<Goldilocks::Element> leafValue;
        read(hashValue, leafValue, NULL);
        if (leafValue.size() == 12)
        {
            if (!fr.equal(leafValue[8], fr.zero()))
            {
                cerr << "Error: Database::printTree() found leafValue[8]=" << fr.toString(leafValue[8], 16) << endl;
                return;
            }
            if (!fr.equal(leafValue[9], fr.zero()))
            {
                cerr << "Error: Database::printTree() found leafValue[9]=" << fr.toString(leafValue[9], 16) << endl;
                return;
            }
            if (!fr.equal(leafValue[10], fr.zero()))
            {
                cerr << "Error: Database::printTree() found leafValue[10]=" << fr.toString(leafValue[10], 16) << endl;
                return;
            }
            if (!fr.equal(leafValue[11], fr.zero()))
            {
                cerr << "Error: Database::printTree() found leafValue[11]=" << fr.toString(leafValue[11], 16) << endl;
                return;
            }
        }
        else if (leafValue.size() == 8)
        {
            cout << prefix << "leafValue.size()=" << leafValue.size() << endl;
        }
        else
        {
            cerr << "Error: Database::printTree() found lleafValue.size()=" << leafValue.size() << endl;
            return;
        }
        mpz_class scalarValue;
        fea2scalar(fr, scalarValue, leafValue[0], leafValue[1], leafValue[2], leafValue[3], leafValue[4], leafValue[5], leafValue[6], leafValue[7]);
        cout << prefix << "leafValue=" << PrependZeros(scalarValue.get_str(16), 64) << endl;
    }
    else
    {
        cerr << "Error: Database::printTree() found value[8]=" << fr.toString(value[8], 16) << endl;
        return;
    }
    if (prefix == "") cout << endl;
}

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
                    //cout << "Database::~Database() deleting writeConnectionsPool[" << i << "].pConnection=" << writeConnectionsPool[i].pConnection << endl;
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

void loadDb2MemCache(const Config config)
{
    if (config.databaseURL == "local")
    {
        cerr << "Error: loadDb2MemCache() called with config.stateDBURL==local" << endl;
        exitProcess();
    }

#ifdef DATABASE_USE_CACHE

    TimerStart(LOAD_DB_TO_CACHE);

    Goldilocks fr;
    StateDB * pStateDB = (StateDB *)stateDBSingleton.get(fr, config);

    vector<Goldilocks::Element> dbValue;

    zkresult zkr = pStateDB->db.read(Database::dbStateRootKey, dbValue, NULL, true);
    if (zkr == ZKR_DB_KEY_NOT_FOUND)
    {
        cout << "Warning: loadDb2MemCache() dbStateRootKey=" <<  Database::dbStateRootKey << " not found in database; normal only if database is empty" << endl;
        TimerStopAndLog(LOAD_DB_TO_CACHE);
        return;
    }
    else if (zkr != ZKR_SUCCESS)
    {
        cerr << "Error: loadDb2MemCache() failed calling db.read result=" << zkr << "=" << zkresult2string(zkr) << endl;
        TimerStopAndLog(LOAD_DB_TO_CACHE);
        return;
    }
    
    string stateRootKey = fea2string(fr, dbValue[0], dbValue[1], dbValue[2], dbValue[3]);
    cout << "loadDb2MemCache() found state root=" << stateRootKey << endl;

    if (stateRootKey == "0")
    {
        cout << "loadDb2MemCache() found an empty tree" << endl;
        TimerStopAndLog(LOAD_DB_TO_CACHE);
        return;
    }

    unordered_map<uint64_t, vector<string>> treeMap;
    vector<string> emptyVector;
    string hash, leftHash, rightHash;
    uint64_t counter = 0;

    treeMap[0] = emptyVector;
    treeMap[0].push_back(stateRootKey);
    unordered_map<uint64_t, std::vector<std::string>>::iterator treeMapIterator;
    for (uint64_t level=0; level<256; level++)
    {
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

        //cout << "loadDb2MemCache() searching at level=" << level << " for elements=" << treeMapIterator->second.size() << endl;
        
        for (uint64_t i=0; i<treeMapIterator->second.size(); i++)
        {
            hash = treeMapIterator->second[i];
            dbValue.clear();
            zkresult zkr = pStateDB->db.read(hash, dbValue, NULL, true);
            if (zkr != ZKR_SUCCESS)
            {
                cerr << "Error: loadDb2MemCache() failed calling db.read(" << hash << ") result=" << zkr << "=" << zkresult2string(zkr) << endl;
                TimerStopAndLog(LOAD_DB_TO_CACHE);
                return;
            }
            if (dbValue.size() != 12)
            {
                cerr << "Error: loadDb2MemCache() failed calling db.read(" << hash << ") dbValue.size()=" << dbValue.size() << endl;
                TimerStopAndLog(LOAD_DB_TO_CACHE);
                return;
            }
            counter++;
            double sizePercentage = double(Database::dbMTCache.getCurrentSize())*100.0/double(Database::dbMTCache.getMaxSize());
            if ( sizePercentage > 90 )
            {
                cout << "loadDb2MemCache() stopping since size percentage=" << sizePercentage << endl;
                break;
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
                        //cout << "loadDb2MemCache() level=" << level << " found leftHash=" << leftHash << endl;
                    }
                    rightHash = fea2string(fr, dbValue[4], dbValue[5], dbValue[6], dbValue[7]);
                    if (rightHash != "0")
                    {
                        treeMap[level+1].push_back(rightHash);
                        //cout << "loadDb2MemCache() level=" << level << " found rightHash=" << rightHash << endl;
                    }
                }
                // If capacity is 1000, this is a leaf node that contains right hash of the value node
                else if (fr.isOne(dbValue[8]))
                {
                    rightHash = fea2string(fr, dbValue[4], dbValue[5], dbValue[6], dbValue[7]);
                    if (rightHash != "0")
                    {
                        //cout << "loadDb2MemCache() level=" << level << " found value rightHash=" << rightHash << endl;
                        dbValue.clear();
                        zkresult zkr = pStateDB->db.read(rightHash, dbValue, NULL, true);
                        if (zkr != ZKR_SUCCESS)
                        {
                            cerr << "Error: loadDb2MemCache() failed calling db.read(" << rightHash << ") result=" << zkr << "=" << zkresult2string(zkr) << endl;
                            TimerStopAndLog(LOAD_DB_TO_CACHE);
                            return;
                        }
                        counter++;
                    }
                }
            }
        }
    }

    cout << "loadDb2MemCache() done counter=" << counter << " cache at " << (double(Database::dbMTCache.getCurrentSize())/double(Database::dbMTCache.getMaxSize()))*100 << "%" << endl;

    TimerStopAndLog(LOAD_DB_TO_CACHE);

#endif
}