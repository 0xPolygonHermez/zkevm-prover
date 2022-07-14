#include <iostream>
#include "config.hpp"
#include "database.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "definitions.hpp"
#include "result.hpp"

void Database::init(const Config &_config)
{
    // Check that it has not been initialized before
    if (bInitialized)
    {
        cerr << "Error: Database::init() called when already initialized" << endl;
        exit(-1);
    }

    config = _config;

    // Configure the server, if configuration is provided
    if (config.databaseURL!="local")
    {
        initRemote();
        useRemoteDB = true;
    } else useRemoteDB = false;

    // Mark the database as initialized
    bInitialized = true;
}

result_t Database::read (const string &_key, vector<Goldilocks::Element> &value)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::read() called uninitialized" << endl;
        exit(-1);
    }
    
    result_t r;

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

    // If the value is found in local database (cached) simply return it
    if (db.find(key) != db.end())
    {
        value = db[key];
        r = R_SUCCESS;
    } 
    else if (useRemoteDB)
    {
        // Otherwise, read it remotelly
        r = readRemote(key, value);
        if (r == R_SUCCESS) {
            // Store it locally to avoid any future remote access for this key
            db[key] = value;
            dbRemote[key] = value;
        }
    }
    else
    {
        cerr << "Error: Database::read() requested a key that does not exist: " << key << endl;
        r = R_DB_KEY_NOT_FOUND;
    }

#ifdef LOG_DB
    cout << "Database::read()" << endl;
    if (r != R_SUCCESS) cout << "  ERROR=" << r << " (" << result2string(r) << ")" << endl;
    cout << "  key=" << key << endl;
    cout << "  value=";
    for (uint64_t i = 0; i < value.size(); i++)
        cout << fr.toString(value[i], 16) << ":";
    cout << endl;    
#endif    

    return r;
}

result_t Database::write (const string &_key, const vector<Goldilocks::Element> &value, const bool persistent)
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::write() called uninitialized" << endl;
        exit(-1);
    }

    result_t r;
    
    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

    if (useRemoteDB && persistent)
    {
        r = writeRemote(key, value);
    } else r = R_SUCCESS;

    if (r == R_SUCCESS) {
        // Create in memory cache
        db[key] = value;
        dbNew[key] = value;
    }

#ifdef LOG_DB
    cout << "Database::write()" << endl;
    if (r != R_SUCCESS) cout << "  ERROR=" << r << " (" << result2string(r) << ")" << endl;
    cout << "  key=" << key << endl;
    cout << "  value=";
    for (uint64_t i = 0; i < value.size(); i++)
        cout << fr.toString(value[i], 16) << ":";
    cout << endl;    
    cout << "  persistent=" << persistent << endl;    
#endif      

    return R_SUCCESS;
}

void Database::initRemote (void)
{
    try
    {
        // Build the remote database URI
        string uri = config.databaseURL;
        cout << "Database URI: " << uri << endl;

        // Create the connection
        pConnectionWrite = new pqxx::connection{uri};
        pConnectionRead = new pqxx::connection{uri};

        //Create the thread to process asynchronous writes to de DB
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
        exit(-1);
    }
}

result_t Database::readRemote (const string &key, vector<Goldilocks::Element> &value)
{
    value.clear();
    try
    {
        // Start a transaction.
        pqxx::nontransaction n(*pConnectionRead);

        // Prepare the query
        string query = "SELECT * FROM " + config.dbTableName + " WHERE hash = E\'\\\\x" + key + "\';";

        // Execute the query
        pqxx::result rows = n.exec(query);

        // Process the result
        if (rows.size() == 0)
        {
            return R_DB_KEY_NOT_FOUND;
        } 
        else if (rows.size() > 1)
        {
            cerr << "Error: Database::readRemote() got more than one row for the same key: " << rows.size() << endl;
            exit(-1);
        }
        
        pqxx::row const row = rows[0];
        if (row.size() != 2)
        {
            cerr << "Error: Database::readRemote() got an invalid number of colums for the row: " << row.size() << endl;
            exit(-1);
        }
        pqxx::field const fieldData = row[1];
        string sData = fieldData.c_str();

        Goldilocks::Element fe;
        string aux;
        for (uint64_t i=2; i<sData.size(); i+=64)
        {
            if (i+64 > sData.size())
            {
                cerr << "Error: Database::readRemote() found incorrect DATA column size: " << sData.size() << endl;
                exit(-1);
            }
            aux = sData.substr(i, 64);
            string2fe(fr, aux, fe);
            value.push_back(fe);
        }

        // Commit your transaction
        n.commit();
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::readRemote() exception: " << e.what() << endl;
        exit(-1);
    }

    return R_SUCCESS;
}

result_t Database::writeRemote (const string &key, const vector<Goldilocks::Element> &value)
{
    try
    {
        // Prepare the query
        string valueString;
        string aux;
        for (uint64_t i = 0; i < value.size(); i++)
        {
            aux = fr.toString(value[i], 16);
            valueString += NormalizeToNFormat(aux, 64); //· porque formateamos con 64 chars? no sería 16?
        }
        string query = "INSERT INTO " + config.dbTableName + " ( hash, data ) VALUES ( E\'\\\\x" + key + "\', E\'\\\\x" + valueString + "\' ) "+
                       "ON CONFLICT (hash) DO NOTHING;";

        if (config.dbAsyncWrite) {
            addWriteQueue(query);
        } else {
            if (autoCommit) {
                pqxx::work w(*pConnectionWrite);
                pqxx::result res = w.exec(query);
                w.commit();
            } else {
                if (transaction==NULL) transaction = new pqxx::work{*pConnectionWrite};
                pqxx::result res = transaction->exec(query);
            }
        }
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::writeRemote() exception: " << e.what() << endl;
        exit(-1);
    }

    return R_SUCCESS;
}

void Database::print(void)
{
    cout << "Database of " << db.size() << " elements:" << endl;
    for (map<string, vector<Goldilocks::Element>>::iterator it = db.begin(); it != db.end(); it++)
    {
        vector<Goldilocks::Element> vect = it->second;
        cout << "key:" << it->first << " ";
        for (uint64_t i = 0; i < vect.size(); i++)
            cout << fr.toString(vect[i], 16) << ":";
        cout << endl;
    }
}

result_t Database::setProgram (const string &key, const vector<uint8_t> &data, const bool persistent)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::setProgram() called uninitialized" << endl;
        exit(-1);
    }

#ifdef LOG_DB
    cout << "Database::setProgram()" << endl;
#endif  

    vector<Goldilocks::Element> feValue;
    Goldilocks::Element fe;
    for (uint64_t i=0; i<data.size(); i++)
    {
        fe = fr.fromU64(data[i]);
        feValue.push_back(fe);
    }

    return write(key, feValue, persistent);
}

result_t Database::getProgram (const string &key, vector<uint8_t> &data)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::getProgram() called uninitialized" << endl;
        exit(-1);
    }
    
#ifdef LOG_DB
    cout << "Database::getProgram()" << endl;
#endif  

    result_t r;

    vector<Goldilocks::Element> feValue;

    r = read(key, feValue);

    if (r == R_SUCCESS) 
    {
        for (uint64_t i=0; i<feValue.size(); i++)
        {
            uint64_t uValue;
            uValue = fr.toU64(feValue[i]);
            zkassert(uValue < (1<<8));
            data.push_back((uint8_t)uValue);
        }
    }

    return r;
}

void Database::addWriteQueue (const string sqlWrite)
{
    pthread_mutex_lock(&writeQueueMutex);
    writeQueue.push_back(sqlWrite);
    pthread_cond_signal(&writeQueueCond);
    pthread_mutex_unlock(&writeQueueMutex);
}

void Database::flush ()
{
    if (config.dbAsyncWrite) {
        pthread_mutex_lock(&writeQueueMutex);
        while (writeQueue.size()>0) pthread_cond_wait(&emptyWriteQueueCond, &writeQueueMutex);
        pthread_mutex_unlock(&writeQueueMutex);
    }
}

void Database::setAutoCommit (const bool ac)
{
    if (ac && !autoCommit) commit ();
    autoCommit = ac;
}

void Database::commit ()
{
    if ((!autoCommit)&&(transaction!=NULL)) {      
        transaction->commit();
        delete transaction;
        transaction = NULL;
    }
}

void Database::processWriteQueue () 
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
        if (writeQueue.size() == 0) {
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
            } catch (const std::exception &e) {
                cerr << "Error: Database::processWriteQueue() execute query exception: " << e.what() << endl;
            }        
        } else {
            cout << "Database::processWriteQueue() found pending writes queue empty, so ignoring" << endl;
            pthread_mutex_unlock(&writeQueueMutex);
        }
    }

    if (pAsyncWriteConnection != NULL)
    {
        delete pAsyncWriteConnection;
    }    
}

Database::~Database()
{
    if (pConnectionWrite != NULL) delete pConnectionWrite;
    if (pConnectionRead != NULL) delete pConnectionRead;
    
    if (config.dbAsyncWrite)
    {
        pthread_mutex_destroy(&writeQueueMutex);
        pthread_cond_destroy(&writeQueueCond);
        pthread_cond_destroy(&emptyWriteQueueCond);
    }    
}

void* asyncDatabaseWriteThread (void* arg) {
    Database* db = (Database*)arg;   
    db->processWriteQueue();
    return NULL;
}