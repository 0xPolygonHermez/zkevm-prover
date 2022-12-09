#include <iostream>
#include "database.hpp"
#include "config.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "definitions.hpp"
#include "zkresult.hpp"
#include "utils.hpp"

// Create static Database::dbCache object. This will be used to store DB records in memory
// and it will be shared for all the instances of Database class. DatabaseMap class is thread-safe
DatabaseMap Database::dbCache;
bool Database::dbLoaded2Cache = false;

void Database::init(const Config &_config)
{
    // Check that it has not been initialized before
    if (bInitialized)
    {
        cerr << "Error: Database::init() called when already initialized" << endl;
        exitProcess();
    }

    config = _config;

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

    // If the key is found in local database (cached) simply return it
    if (Database::dbCache.findMT(key, value))
    {
        // Add to the read log
        if (dbReadLog != NULL) dbReadLog->add(key, value);

        r = ZKR_SUCCESS;
    }
    else if (useRemoteDB)
    {
        // Otherwise, read it remotelly
        string sData;
        r = readRemote(config.dbNodesTableName, key, sData);
        if (r == ZKR_SUCCESS)
        {
            string2fea(sData, value);

            // Store it locally to avoid any future remote access for this key
            Database::dbCache.add(key, value);

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

    if (useRemoteDB && persistent)
    {
        // Prepare the query
        string valueString = "";
        string aux;
        for (uint64_t i = 0; i < value.size(); i++)
        {
            valueString += PrependZeros(fr.toString(value[i], 16), 16);
        }

        r = writeRemote(config.dbNodesTableName, key, valueString);
    }
    else r = ZKR_SUCCESS;

    if (r == ZKR_SUCCESS)
    {
        // Create in memory cache
        Database::dbCache.add(key, value);
    }

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

        // Create the thread to process asynchronous writes to de DB
        if (config.dbAsyncWrite)
        {
            pthread_cond_init(&writeQueueCond, 0);
            pthread_cond_init(&emptyWriteQueueCond, 0);
            pthread_mutex_init(&writeQueueMutex, NULL);
            pthread_create(&writeThread, NULL, asyncDatabaseWriteThread, this);
        }
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::initRemote() exception: " << e.what() << endl;
        exitProcess();
    }
}

zkresult Database::readRemote(const string tableName, const string &key, string &value)
{
    if (config.logRemoteDbReads)
    {
        cout << "   Database::readRemote() table=" << tableName << " key=" << key << endl;
    }

    try
    {
        // Start a transaction.
        pqxx::nontransaction n(*pConnectionRead);

        // Prepare the query
        string query = "SELECT * FROM " + tableName + " WHERE hash = E\'\\\\x" + key + "\';";

        // Execute the query
        pqxx::result rows = n.exec(query);

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

        // Commit your transaction
        n.commit();
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::readRemote() table="<< tableName << " exception: " << e.what() << endl;
        exitProcess();
    }

    return ZKR_SUCCESS;
}

zkresult Database::writeRemote(const string tableName, const string &key, const string &value)
{
    try
    {
        string query = "INSERT INTO " + tableName + " ( hash, data ) VALUES ( E\'\\\\x" + key + "\', E\'\\\\x" + value + "\' ) " +
                       "ON CONFLICT (hash) DO NOTHING;";

        if (config.dbAsyncWrite)
        {
            addWriteQueue(query);
        }
        else
        {
            if (autoCommit)
            {
                pqxx::work w(*pConnectionWrite);
                pqxx::result res = w.exec(query);
                w.commit();
            }
            else
            {
                if (transaction == NULL)
                    transaction = new pqxx::work{*pConnectionWrite};
                pqxx::result res = transaction->exec(query);
            }
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

    if (useRemoteDB && persistent)
    {
        string sData = "";
        for (uint64_t i=0; i<data.size(); i++)
        {
            sData += byte2string(data[i]);
        }

        r = writeRemote(config.dbProgramTableName, key, sData);
    }
    else r = ZKR_SUCCESS;

    if (r == ZKR_SUCCESS)
    {
        // Create in memory cache
        Database::dbCache.add(key, data);
    }

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

    // If the key is found in local database (cached) simply return it
    if (Database::dbCache.findProgram(key, data))
    {
        // Add to the read log
        if (dbReadLog != NULL) dbReadLog->add(key, data);

        r = ZKR_SUCCESS;
    }
    else if (useRemoteDB)
    {
        // Otherwise, read it remotelly
        string sData;
        r = readRemote(config.dbProgramTableName, key, sData);
        if (r == ZKR_SUCCESS)
        {
            //String to byte/uint8_t vector
            string2ba(sData, data);

            // Store it locally to avoid any future remote access for this key
            Database::dbCache.add(key, data);

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

void Database::string2fea(const string os, vector<Goldilocks::Element> &fea)
{
    Goldilocks::Element fe;
    for (uint64_t i = 0; i < os.size(); i += 16)
    {
        if (i + 16 > os.size())
        {
            cerr << "Error: Database::string2fea() found incorrect DATA column size: " << os.size() << endl;
            exitProcess();
        }
        string2fe(fr, os.substr(i, 16), fe);
        fea.push_back(fe);
    }
}

void Database::string2ba(const string os, vector<uint8_t> &data)
{
    string s = Remove0xIfPresent(os);

    if (s.size()%2 != 0)
    {
        s = "0" + s;
    }

    uint64_t dsize = s.size()/2;
    const char *p = s.c_str();
    for (uint64_t i=0; i<dsize; i++)
    {
        data.push_back(char2byte(p[2*i])*16 + char2byte(p[2*i + 1]));
    }
}

void Database::loadDB2MemCache()
{
    if (!useRemoteDB) return;

    if (Database::dbLoaded2Cache) return;

    cout << "Loading SQL Database to memory cache..." << endl;

    try
    {
        // Start a transaction.
        pqxx::nontransaction n(*pConnectionRead);

        // Prepare the query
        string query = "SELECT * FROM " + config.dbNodesTableName +";";

        // Execute the query
        pqxx::result rows = n.exec(query);

        for (pqxx::result::size_type i=0; i < rows.size(); i++)
        {
            vector<Goldilocks::Element> value;

            string2fea(removeBSXIfExists(rows[i][1].c_str()), value);

            Database::dbCache.add(removeBSXIfExists(rows[i][0].c_str()), value);
        }

        // Commit your transaction
        n.commit();
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::loadDB2MemCache() table=" << config.dbNodesTableName << " exception: " << e.what() << endl;
        exitProcess();
    }

    try
    {
        // Start a transaction.
        pqxx::nontransaction n(*pConnectionRead);

        // Prepare the query
        string query = "SELECT * FROM " + config.dbProgramTableName +";";

        // Execute the query
        pqxx::result rows = n.exec(query);

        for (pqxx::result::size_type i=0; i < rows.size(); i++)
        {
            vector<uint8_t> value;

            string2ba(removeBSXIfExists(rows[i][1].c_str()), value);

            Database::dbCache.add(removeBSXIfExists(rows[i][0].c_str()), value);
        }

        // Commit your transaction
        n.commit();
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::loadDB2MemCache() table=" << config.dbProgramTableName << " exception: " << e.what() << endl;
        exitProcess();
    }

    Database::dbLoaded2Cache = true;
    
    cout << "Load done" << endl;        
}

void Database::addWriteQueue(const string sqlWrite)
{
    pthread_mutex_lock(&writeQueueMutex);
    writeQueue.push_back(sqlWrite);
    pthread_cond_signal(&writeQueueCond);
    pthread_mutex_unlock(&writeQueueMutex);
}

void Database::flush()
{
    if (config.dbAsyncWrite)
    {
        pthread_mutex_lock(&writeQueueMutex);
        while (writeQueue.size() > 0)
            pthread_cond_wait(&emptyWriteQueueCond, &writeQueueMutex);
        pthread_mutex_unlock(&writeQueueMutex);
    }
}

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

void Database::processWriteQueue()
{
    string writeQuery;

    cout << "Database::processWriteQueue() started" << endl;

    try
    {
        // Build the remote database URI
        string uri = config.databaseURL;
        cout << "Database::processWriteQueue URI: " << uri << endl;

        // Create the connection
        pAsyncWriteConnection = new pqxx::connection{uri};
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::processWriteQueue() connection to the DB exception: " << e.what() << endl;
        return;
    }

    while (true)
    {
        pthread_mutex_lock(&writeQueueMutex);

        // Wait for the pending writes in the queue, if there are no more pending writes
        if (writeQueue.size() == 0)
        {
            pthread_cond_signal(&emptyWriteQueueCond);
            pthread_cond_wait(&writeQueueCond, &writeQueueMutex);
        }

        // Check that the pending writes queue is not empty
        if (writeQueue.size() > 0)
        {
            try
            {
                // Get the query for the pending write
                writeQuery = writeQueue[0];
                writeQueue.erase(writeQueue.begin());
                pthread_mutex_unlock(&writeQueueMutex);

                // Start a transaction
                pqxx::work w(*pConnectionWrite);

                // Execute the query
                pqxx::result res = w.exec(writeQuery);

                // Commit your transaction
                w.commit();
            }
            catch (const std::exception &e)
            {
                cerr << "Error: Database::processWriteQueue() execute query exception: " << e.what() << endl;
            }
        }
        else
        {
            cout << "Database::processWriteQueue() found pending writes queue empty, so ignoring" << endl;
            pthread_mutex_unlock(&writeQueueMutex);
        }
    }

    if (pAsyncWriteConnection != NULL)
    {
        delete pAsyncWriteConnection;
    }
}

void Database::print(void)
{
    DatabaseMap::MTMap mtDB = Database::dbCache.getMTDB();
    cout << "Database of " << mtDB.size() << " elements:" << endl;
    for (DatabaseMap::MTMap::iterator it = mtDB.begin(); it != mtDB.end(); it++)
    {
        vector<Goldilocks::Element> vect = it->second;
        cout << "key:" << it->first << " ";
        for (uint64_t i = 0; i < vect.size(); i++)
            cout << fr.toString(vect[i], 16) << ":";
        cout << endl;
    }
}

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
        delete pConnectionWrite;
    if (pConnectionRead != NULL)
        delete pConnectionRead;

    if (config.dbAsyncWrite)
    {
        pthread_mutex_destroy(&writeQueueMutex);
        pthread_cond_destroy(&writeQueueCond);
        pthread_cond_destroy(&emptyWriteQueueCond);
    }
}

void *asyncDatabaseWriteThread(void *arg)
{
    Database *db = (Database *)arg;
    db->processWriteQueue();
    return NULL;
}