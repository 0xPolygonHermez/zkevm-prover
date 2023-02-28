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
void* asyncDatabaseWriteThread(void* arg);

// Database class implementation
void Database::init(void)
{
    // Check that it has not been initialized before
    if (bInitialized)
    {
        cerr << "Error: Database::init() called when already initialized" << endl;
        exitProcess();
    }

    // Init mutexes
    pthread_mutex_init(&multiWriteMutex, NULL);
    pthread_mutex_init(&writeMutex, NULL);
    pthread_mutex_init(&readMutex, NULL);

    useDBMTCache = dbMTCache.enabled();
    useDBProgramCache = dbProgramCache.enabled();

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
    else r = ZKR_SUCCESS;

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

    return ZKR_SUCCESS;
}

void Database::initRemote(void)
{
    try
    {
        // Build the remote database URI
        string uri = config.databaseURL;
        cout << "Database URI: " << uri << endl;

        // Create the connection
        pConnectionWrite = new pqxx::connection{uri};
        pConnectionRead = new pqxx::connection{uri};
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::initRemote() exception: " << e.what() << endl;
        exitProcess();
    }
}

zkresult Database::readRemote(bool bProgram, const string &key, string &value)
{
    const string &tableName = (bProgram ? config.dbProgramTableName : config.dbNodesTableName);

    if (config.logRemoteDbReads)
    {
        cout << "   Database::readRemote() table=" << tableName << " key=" << key << endl;
    }

    try
    {
        // Prepare the query
        string query = "SELECT * FROM " + tableName + " WHERE hash = E\'\\\\x" + key + "\';";

        readLock();

        // Start a transaction.
        pqxx::nontransaction n(*pConnectionRead);

        // Execute the query
        pqxx::result rows = n.exec(query);

        // Commit your transaction
        n.commit();

        readUnlock();

        // Process the result
        if (rows.size() == 0)
        {
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
        cerr << "Error: Database::readRemote() table="<< tableName << " exception: " << e.what() << endl;
        exitProcess();
    }

    return ZKR_SUCCESS;
}

zkresult Database::writeRemote(bool bProgram, const string &key, const string &value)
{
    const string &tableName = (bProgram ? config.dbProgramTableName : config.dbNodesTableName);
    try
    {
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

#ifdef DATABASE_COMMIT
            if (autoCommit)
#endif
            {
                writeLock();
                pqxx::work w(*pConnectionWrite);
                pqxx::result res = w.exec(query);
                w.commit();
                writeUnlock();
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
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::writeRemote() table="<< tableName << " exception: " << e.what() << endl;
        exitProcess();
    }

    return ZKR_SUCCESS;
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
    else r = ZKR_SUCCESS;

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

    return ZKR_SUCCESS;
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

void Database::flush()
{
    if (config.dbMultiWrite)
    {
        multiWriteLock();
        if ( (multiWriteProgram.size() == 0) && (multiWriteNodes.size() == 0) )
        {
            multiWriteUnlock();
            return;
        }

        try
        {
            if (multiWriteProgram.size() > 0)
            {
                multiWriteProgram += " ON CONFLICT (hash) DO NOTHING;";

                writeLock();
                
                // Start a transaction
                pqxx::work w(*pConnectionWrite);

                // Execute the query
                pqxx::result res = w.exec(multiWriteProgram);

                // Commit your transaction
                w.commit();

                writeUnlock();

                //cout << "Database::flush() sent " << multiWriteProgram << endl;

                multiWriteProgram.clear();
            }
            if (multiWriteNodes.size() > 0)
            {
                multiWriteNodes += " ON CONFLICT (hash) DO NOTHING;";

                writeLock();
                
                // Start a transaction
                pqxx::work w(*pConnectionWrite);

                // Execute the query
                pqxx::result res = w.exec(multiWriteNodes);

                // Commit your transaction
                w.commit();

                writeUnlock();

                //cout << "Database::flush() sent " << multiWriteNodes << endl;

                multiWriteNodes.clear();
            }
        }
        catch (const std::exception &e)
        {
            cerr << "Error: Database::flush() execute query exception: " << e.what() << endl;
        }
        multiWriteUnlock();
    }
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
    if (pConnectionWrite != NULL)
    {
        delete pConnectionWrite;
    }

    if (pConnectionRead != NULL)
    {
        delete pConnectionRead;
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