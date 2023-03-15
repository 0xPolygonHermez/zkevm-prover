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

#ifdef DATABASE_USE_CACHE

// Create static Database::dbMTCache and DatabaseCacheProgram objects
// This will be used to store DB records in memory and it will be shared for all the instances of Database class
// DatabaseCacheMT and DatabaseCacheProgram classes are thread-safe
DatabaseMTCache Database::dbMTCache;
DatabaseProgramCache Database::dbProgramCache;

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

zkresult Database::read(const string &_key, vector<Goldilocks::Element> &value, DatabaseMap *dbReadLog)
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
            if (useDBMTCache) Database::dbMTCache.add(key, value);
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

zkresult Database::write(const string &_key, const vector<Goldilocks::Element> &value, const bool persistent)
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::write() called uninitialized" << endl;
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
    if ((r == ZKR_SUCCESS) && (useDBMTCache))
    {
        // Create in memory cache
        Database::dbMTCache.add(key, value);
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
    cout << " persistent=" << persistent << endl;
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
        writeLock();

        if (config.dbConnectionsPool)
        {
            // Check that we don't support more threads than available connections
            if ( config.runStateDBServer && (config.maxStateDBThreads > config.dbNumberOfWritePoolConnections) )
            {
                cerr << "Error: Database::initRemote() found config.maxStateDBThreads=" << config.maxStateDBThreads << " > config.dbNumberOfWritePoolConnections=" << config.dbNumberOfWritePoolConnections << endl;
                exitProcess();
            }
            if ( config.runExecutorServer && (config.maxExecutorThreads > config.dbNumberOfWritePoolConnections) )
            {
                cerr << "Error: Database::initRemote() found config.maxExecutorThreads=" << config.maxExecutorThreads << " > config.dbNumberOfWritePoolConnections=" << config.dbNumberOfWritePoolConnections << endl;
                exitProcess();
            }

            // Allocate write connections pool
            writeConnectionsPool = new DatabaseConnection[config.dbNumberOfWritePoolConnections];
            if (writeConnectionsPool == NULL)
            {
                cerr << "Error: Database::initRemote() failed creating write connection pool of size " << config.dbNumberOfWritePoolConnections << endl;
                exitProcess();
            }

            // Create write connections
            for (uint64_t i=0; i<config.dbNumberOfWritePoolConnections; i++)
            {
                writeConnectionsPool[i].pConnection = new pqxx::connection{uri};
                if (writeConnectionsPool[i].pConnection == NULL)
                {
                    cerr << "Error: Database::initRemote() failed creating write connection " << i << endl;
                    exitProcess();
                }
                writeConnectionsPool[i].bInUse = false;
                //cout << "Database::initRemote() created write connection i=" << i << " writeConnectionsPool[i]=" << writeConnectionsPool[i].pConnection << endl;
            }

            // Reset counters
            nextWriteConnection = 0;
            usedWriteConnections = 0;
        }
        else
        {
            writeConnection.pConnection = new pqxx::connection{uri};
            if (writeConnection.pConnection == NULL)
            {
                cerr << "Error: Database::initRemote() failed creating unique write connection" << endl;
                exitProcess();
            }
        }
        
        writeUnlock();
        
        readLock();
        
        if (config.dbConnectionsPool)
        {
            // Check that we don't support more threads than available connections
            if ( config.runStateDBServer && (config.maxStateDBThreads > config.dbNumberOfReadPoolConnections) )
            {
                cerr << "Error: Database::initRemote() found config.maxStateDBThreads=" << config.maxStateDBThreads << " > config.dbNumberOfReadPoolConnections=" << config.dbNumberOfReadPoolConnections << endl;
                exitProcess();
            }
            if ( config.runExecutorServer && (config.maxExecutorThreads > config.dbNumberOfReadPoolConnections) )
            {
                cerr << "Error: Database::initRemote() found config.maxExecutorThreads=" << config.maxExecutorThreads << " > config.dbNumberOfReadPoolConnections=" << config.dbNumberOfReadPoolConnections << endl;
                exitProcess();
            }

            // Allocate read connections pool
            readConnectionsPool = new DatabaseConnection[config.dbNumberOfReadPoolConnections];
            if (readConnectionsPool == NULL)
            {
                cerr << "Error: Database::initRemote() failed creating read connection pool of size " << config.dbNumberOfReadPoolConnections << endl;
                exitProcess();
            }

            // Create read connections
            for (uint64_t i=0; i<config.dbNumberOfReadPoolConnections; i++)
            {
                readConnectionsPool[i].pConnection = new pqxx::connection{uri};
                if (readConnectionsPool[i].pConnection == NULL)
                {
                    cerr << "Error: Database::initRemote() failed creating read connection " << i << endl;
                    exitProcess();
                }
                readConnectionsPool[i].bInUse = false;
                //cout << "Database::initRemote() created read connection i=" << i << " readConnectionsPool[i]=" << readConnectionsPool[i].pConnection << endl;
            }

            // Reset counters
            nextReadConnection = 0;
            usedReadConnections = 0;
        }
        else
        {
            readConnection.pConnection = new pqxx::connection{uri};
            if (readConnection.pConnection == NULL)
            {
                cerr << "Error: Database::initRemote() failed creating unique read connection" << endl;
                exitProcess();
            }
        }

        readUnlock();
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::initRemote() exception: " << e.what() << endl;
        exitProcess();
    }

    TimerStopAndLog(DB_INIT_REMOTE);
}

DatabaseConnection * Database::getWriteConnection (void)
{
    if (config.dbConnectionsPool)
    {
        writeLock();
        DatabaseConnection * pConnection = NULL;
        uint64_t i=0;
        for (i=0; i<config.dbNumberOfWritePoolConnections; i++)
        {
            if (!writeConnectionsPool[nextWriteConnection].bInUse) break;
            nextWriteConnection++;
            if (nextWriteConnection == config.dbNumberOfWritePoolConnections)
            {
                nextWriteConnection = 0;
            }
        }
        if (i==config.dbNumberOfWritePoolConnections)
        {
            cerr << "Error: Database::getWriteConnection() run out of free connections" << endl;
            exitProcess();
        }

        pConnection = &writeConnectionsPool[nextWriteConnection];
        zkassert(pConnection->bInUse == false);
        pConnection->bInUse = true;
        nextWriteConnection++;
        if (nextWriteConnection == config.dbNumberOfWritePoolConnections)
        {
            nextWriteConnection = 0;
        }
        usedWriteConnections++;
        //cout << "Database::getWriteConnection() pConnection=" << pConnection << " nextWriteConnection=" << to_string(nextWriteConnection) << " usedWriteConnections=" << to_string(usedWriteConnections) << endl;
        writeUnlock();
        return pConnection;
    }
    else
    {
        writeLock();
        zkassert(writeConnection.bInUse == false);
#ifdef DEBUG
        writeConnection.bInUse = true;
#endif
        return &writeConnection;
    }
}

void Database::disposeWriteConnection (DatabaseConnection * pConnection)
{
    if (config.dbConnectionsPool)
    {
        writeLock();
        zkassert(pConnection->bInUse == true);
        pConnection->bInUse = false;
        zkassert(usedWriteConnections > 0);
        usedWriteConnections--;
        //cout << "Database::disposeWriteConnection() pConnection=" << pConnection << " nextWriteConnection=" << to_string(nextWriteConnection) << " usedWriteConnections=" << to_string(usedWriteConnections) << endl;
        writeUnlock();
    }
    else
    {
        zkassert(pConnection == &writeConnection);
        zkassert(pConnection->bInUse == true);
#ifdef DEBUG
        pConnection->bInUse = false;
#endif
        writeUnlock();
    }
}

DatabaseConnection * Database::getReadConnection (void)
{
    if (config.dbConnectionsPool)
    {
        readLock();
        DatabaseConnection * pConnection = NULL;
        uint64_t i=0;
        for (i=0; i<config.dbNumberOfReadPoolConnections; i++)
        {
            if (!readConnectionsPool[nextReadConnection].bInUse) break;
            nextReadConnection++;
            if (nextReadConnection == config.dbNumberOfReadPoolConnections)
            {
                nextReadConnection = 0;
            }
        }
        if (i==config.dbNumberOfReadPoolConnections)
        {
            cerr << "Error: Database::getReadConnection() run out of free connections" << endl;
            exitProcess();
        }

        pConnection = &readConnectionsPool[nextReadConnection];
        zkassert(pConnection->bInUse == false);
        pConnection->bInUse = true;
        nextReadConnection++;
        if (nextReadConnection == config.dbNumberOfReadPoolConnections)
        {
            nextReadConnection = 0;
        }
        usedReadConnections++;
        //cout << "Database::getReadConnection() pConnection=" << pConnection << " nextReadConnection=" << to_string(nextReadConnection) << " usedReadConnections=" << to_string(usedReadConnections) << endl;
        readUnlock();
        return pConnection;
    }
    else
    {
        readLock();
        zkassert(readConnection.bInUse == false);
#ifdef DEBUG
        readConnection.bInUse = true;
#endif
        return &readConnection;
    }
}

void Database::disposeReadConnection (DatabaseConnection * pConnection)
{
    if (config.dbConnectionsPool)
    {
        readLock();
        zkassert(pConnection->bInUse == true);
        pConnection->bInUse = false;
        zkassert(usedReadConnections > 0);
        usedReadConnections--;
        //cout << "Database::disposeReadConnection() pConnection=" << pConnection << " nextReadConnection=" << to_string(nextReadConnection) << " usedReadConnections=" << to_string(usedReadConnections) << endl;
        readUnlock();
    }
    else
    {
        zkassert(pConnection == &readConnection);
        zkassert(pConnection->bInUse == true);
#ifdef DEBUG
        pConnection->bInUse = false;
#endif
        readUnlock();
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
    DatabaseConnection * pDatabaseConnection = getReadConnection();

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
            disposeReadConnection(pDatabaseConnection);
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
    disposeReadConnection(pDatabaseConnection);

    return ZKR_SUCCESS;
}

zkresult Database::writeRemote(bool bProgram, const string &key, const string &value)
{
    zkresult result = ZKR_SUCCESS;

    const string &tableName = (bProgram ? config.dbProgramTableName : config.dbNodesTableName);
    
    if (config.dbMultiWrite)
    {
        string &multiWrite = bProgram ? multiWriteProgram : multiWriteNodes;
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
    else
    {
        string query = "INSERT INTO " + tableName + " ( hash, data ) VALUES ( E\'\\\\x" + key + "\', E\'\\\\x" + value + "\' ) " +
                    "ON CONFLICT (hash) DO NOTHING;";
            
        DatabaseConnection * pDatabaseConnection = getWriteConnection();

        try
        {        

#ifdef DATABASE_COMMIT
            if (autoCommit)
#endif
            {
                pqxx::work w(*(pDatabaseConnection->pConnection));
                pqxx::result res = w.exec(query);
                w.commit();
                disposeWriteConnection(pDatabaseConnection);
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

zkresult Database::setProgram(const string &_key, const vector<uint8_t> &data, const bool persistent)
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

        r = writeRemote(true, key, sData);
    }
    else
    {
        r = ZKR_SUCCESS;
    }

#ifdef DATABASE_USE_CACHE
    if ((r == ZKR_SUCCESS) && (useDBProgramCache))
    {
        // Create in memory cache
        Database::dbProgramCache.add(key, data);
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
    cout << " persistent=" << persistent << endl;
#endif

    return r;
}

zkresult Database::getProgram(const string &_key, vector<uint8_t> &data, DatabaseMap *dbReadLog)
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
    if ((useDBProgramCache) && (Database::dbProgramCache.find(key, data)))
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
            if (useDBProgramCache) Database::dbProgramCache.add(key, data);
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
        cout << byte2string(data[i]) << endl;
    if (data.size() > 100) cout << "...";
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

    if ( (multiWriteProgram.size() > 0) || (multiWriteNodes.size() > 0) )
    {

        // Get a free write db connection
        DatabaseConnection * pDatabaseConnection = getWriteConnection();

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

                //cout << "Database::flush() sent " << multiWriteProgram << endl;

                // Delete the accumulated query data only if the query succeeded
                multiWriteProgram.clear();
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

                //cout << "Database::flush() sent " << multiWriteNodes << endl;

                // Delete the accumulated query data only if the query succeeded
                multiWriteNodes.clear();
            }
        }
        catch (const std::exception &e)
        {
            cerr << "Error: Database::flush() execute query exception: " << e.what() << endl;
            zkr = ZKR_DB_ERROR;
        }

        // Dispose the write db connection
        disposeWriteConnection(pDatabaseConnection);
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
        if (writeConnectionsPool != NULL)
        {
            for (uint64_t i=0; i<config.dbNumberOfWritePoolConnections; i++)
            {
                if (writeConnectionsPool[i].pConnection != NULL)
                {
                    //cout << "Database::~Database() deleting writeConnectionsPool[" << i << "].pConnection=" << writeConnectionsPool[i].pConnection << endl;
                    delete[] writeConnectionsPool[i].pConnection;
                }
            }
            delete writeConnectionsPool;
        }

        if (readConnectionsPool != NULL)
        {
            for (uint64_t i=0; i<config.dbNumberOfReadPoolConnections; i++)
            {
                if (readConnectionsPool[i].pConnection != NULL)
                {
                    //cout << "Database::~Database() deleting readConnectionsPool[" << i << "].pConnection=" << readConnectionsPool[i].pConnection << endl;
                    delete[] readConnectionsPool[i].pConnection;
                }
            }
            delete readConnectionsPool;
        }
    }
    else
    {
        if (writeConnection.pConnection != NULL)
        {
            delete writeConnection.pConnection;
        }

        if (readConnection.pConnection != NULL)
        {
            delete readConnection.pConnection;
        }
    }
}

void loadDb2MemCache(const Config config)
{
#ifdef DATABASE_USE_CACHE

    TimerStart(LOAD_DB_TO_CACHE);
    
    Goldilocks fr;
    pqxx::connection *pConnection = NULL;

    try
    {
        // Create the database connection
        pConnection = new pqxx::connection{config.databaseURL};
    }
    catch(const std::exception& e)
    {
        cerr << "Error: Database::loadDb2MemCache() exception: " << e.what() << endl;
        exitProcess();
    }

    try
    {
        if (config.dbMTCacheSize > 0) 
        {
            // Start a transaction.
            pqxx::nontransaction n(*pConnection);

            // Prepare the query
            string query = "SELECT * FROM " + config.dbNodesTableName +";";

            // Execute the query
            pqxx::result rows = n.exec(query);
            pqxx::result::size_type i;
            uint64_t count = 0;
            for (i=0; i < rows.size(); i++)
            {
                count++;
                vector<Goldilocks::Element> value;
                string2fea(fr, removeBSXIfExists(rows[i][1].c_str()), value);

                if (Database::dbMTCache.add(removeBSXIfExists(rows[i][0].c_str()), value))
                {
                    // Cache is full stop loading records
                    cout << "MT cache full, stop loading records" << endl;
                    break;
                }
            }

            cout << "MT cache loaded. Count=" << count << endl;

            // Commit your transaction
            n.commit();
        }
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::loadDb2MemCache() table=" << config.dbNodesTableName << " exception: " << e.what() << endl;
        exitProcess();
    }

    try
    {
        if (config.dbProgramCacheSize)
        {
            // Start a transaction.
            pqxx::nontransaction n(*pConnection);

            // Prepare the query
            string query = "SELECT * FROM " + config.dbProgramTableName +";";

            // Execute the query
            pqxx::result rows = n.exec(query);
            pqxx::result::size_type i = 0;
            uint64_t count = 0;
            for (i=0; i < rows.size(); i++)
            {
                count++;
                vector<uint8_t> value;
                string2ba(removeBSXIfExists(rows[i][1].c_str()), value);

                if (Database::dbProgramCache.add(removeBSXIfExists(rows[i][0].c_str()), value))
                {
                    // Cache is full stop loading records
                    cout << "Program cache full, stop loading records" << endl;
                    break;
                }
            }

            cout << "Program cache loaded. Count=" << count << endl;

            // Commit your transaction
            n.commit();
        }
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::loadDb2MemCache() table=" << config.dbProgramTableName << " exception: " << e.what() << endl;
        exitProcess();
    }

    if (pConnection != NULL)
        delete pConnection;

    TimerStopAndLog(LOAD_DB_TO_CACHE);

#endif
}