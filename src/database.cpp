#include <iostream>
#include "config.hpp"
#include "database.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"

void Database::init(const Config &_config)
{
    // Check that it has not been initialized before
    if (bInitialized)
    {
        cerr << "Error: Database::init() called when already initialized" << endl;
        exit(-1);
    }

    // Copy the database configuration
    config = _config;

    // Configure the server, if configuration is provided
    if (config.runStateDBServer)
    {
        initRemote();
    }

    // Mark the database as initialized
    bInitialized = true;
}

void Database::read (const string &key, vector<Goldilocks::Element> &value)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::read() called uninitialized" << endl;
        exit(-1);
    }

    // If the value is found in local database (cached) simply return it
    if (db.find(key) != db.end())
    {
        value = db[key];
        return;
    }
    
    if (config.runStateDBServer)
    {
        // Otherwise, read it remotelly
        //· cout << "Database::read() trying to read key remotely, key: " << key << endl;
        readRemote(key, value);

        // Store it locally to avoid any future remote access for this key
        db[key] = value;
        dbRemote[key] = value;
        //· cout << "Database::read() read key remotely, key: " << key << " length: " << to_string(value.size()) << endl;
    }
    else
    {
        cerr << "Error: Database::read() requested a key that is not present in database: " << key << endl;
        exit(-1);
    }
}

/*void Database::write (const string &key, const vector<Goldilocks::Element> &value)
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::write() called uninitialized" << endl;
        exit(-1);
    }

    // Store in local database; no need to update remote database
    db[key] = value;
    dbNew[key] = value;
}*/

void Database::create (const string &key, const vector<Goldilocks::Element> &value, const bool persistent)
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::create() called uninitialized" << endl;
        exit(-1);
    }

    // Create in local database; no need to update remote database
    db[key] = value;
    dbNew[key] = value;

    if (config.runStateDBServer && persistent)
    {
        createRemote(key, value);
    }
}

void Database::initRemote (void)
{
    try
    {
        /* Start in localhost with::
        /etc/postgresql/14/main$ sudo -u postgres psql postgres
        postgres-# \du
        /usr/lib/postgresql/14/bin$ $ sudo -u postgres psql postgres
        /usr/lib/postgresql/14/bin$ sudo -u postgres createuser -P hermez
        /usr/lib/postgresql/14/bin$ dropuser hermez // to undo previous command
        /usr/lib/postgresql/14/bin$ sudo -u postgres createdb polygon-hermez
        /usr/lib/postgresql/14/bin$ sudo -u postgres dropdb polygon-hermez // to undo previous command
        */

        // Build the remote database URI
        string uri = "postgresql://" + config.dbUser + ":" + config.dbPassword + "@" + config.dbHost + ":" + to_string(config.dbPort) + "/" + config.dbDatabaseName;
        cout << "Database URI: " << uri << endl;

        // Create the connection
        pConnectionWrite = new pqxx::connection{uri};
        pConnectionRead = new pqxx::connection{uri};

        /*pqxx::work w3(*pConnectionWrite);
        string createSchemaQuery = "CREATE SCHEMA state;";
        pqxx::result res3 = w3.exec(createSchemaQuery);
        w3.commit();*/

#ifdef DATABASE_INIT_WITH_INPUT_DB
        pqxx::work w(*pConnectionWrite);
        //string createQuery = "CREATE TABLE state_merkletree ( hash varchar(255), value0 varchar(255), value1 varchar(255), value2 varchar(255), value3 varchar(255), value4 varchar(255), value5 varchar(255), value6 varchar(255), value7 varchar(255), value8 varchar(255), value9 varchar(255), value10 varchar(255), value11 varchar(255), value12 varchar(255), value13 varchar(255), value14 varchar(255), value15 varchar(255) );";
        string createQuery = "CREATE TABLE " + config.tableName + " ( hash BYTEA PRIMARY KEY, data BYTEA NOT NULL );";
        pqxx::result res = w.exec(createQuery);
        w.commit();
#endif
        //Create the thread to process asynchronous writes to de DB
        if (asyncWrite)
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

void Database::readRemote (const string &key, vector<Goldilocks::Element> &value)
{
    value.clear();
    try
    {
        // Start a transaction.
        pqxx::nontransaction n(*pConnectionRead);

        // Prepare the query
        string keyString = NormalizeToNFormat(key, 64);
        string query = "SELECT * FROM " + config.dbTableName + " WHERE hash = E\'\\\\x" + keyString + "\';";
        //·cout << "Database::readRemote() query: " << query << endl;

        // Execute the query
        pqxx::result res = n.exec(query);

        // Process the result
        if (res.size() != 1)
        {
            cerr << "Error: Database::readRemote() got an invalid result size: " << res.size() << endl;
            exit(-1);
        }
        pqxx::row const row = res[0];
        if (row.size() != 2)
        {
            cerr << "Error: Database::readRemote() got an invalid row size: " << row.size() << endl;
            exit(-1);
        }
        pqxx::field const field = row[1];
        string stringResult = field.c_str();

        //cout << "stringResult size=" << to_string(stringResult.size()) << " value=" << stringResult << endl;

        Goldilocks::Element fe;
        string aux;
        for (uint64_t i=2; i<stringResult.size(); i+=64)
        {
            if (i+64 > stringResult.size())
            {
                cerr << "Error: Database::readRemote() found incorrect result size: " << stringResult.size() << endl;
                exit(-1);
            }
            aux = stringResult.substr(i, 64);
            string2fe(fr, aux, fe);
            value.push_back(fe);
        }

        // Commit your transaction
        n.commit();
        //w.commit();
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::readRemote() exception: " << e.what() << endl;
        exit(-1);
    }
}

/*void Database::writeRemote (const string &key, const vector<Goldilocks::Element> &value)
{
    try
    {
        // Start a transaction.
        pqxx::work w(*pConnectionWrite);

        // Prepare the query
        string keyString = NormalizeToNFormat(key, 64);
        string valueString;
        string aux;
        for (uint64_t i = 0; i < value.size(); i++)
        {
            aux = fr.toString(value[i], 16);
            valueString += NormalizeToNFormat(aux, 64);
        }
        string query = "UPDATE " + config.dbTableName + " SET data = E\'\\\\x" + valueString + "\' WHERE key = E\'\\\\x" + keyString + "\';";

        cout << "Database::writeRemote() query: " << query << endl;

        // Execute the query
        pqxx::result res = w.exec(query);

        // Commit your transaction
        w.commit();
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::writeRemote() exception: " << e.what() << endl;
        exit(-1);
    }
}*/

void Database::createRemote (const string &key, const vector<Goldilocks::Element> &value)
{
    try
    {
        // Prepare the query
        string keyString = NormalizeToNFormat(key, 64);
        string valueString;
        string aux;
        for (uint64_t i = 0; i < value.size(); i++)
        {
            aux = fr.toString(value[i], 16);
            valueString += NormalizeToNFormat(aux, 64);
        }
        string query = "INSERT INTO " + config.dbTableName + " ( hash, data ) VALUES ( E\'\\\\x" + keyString + "\', E\'\\\\x" + valueString + "\' ) "+
                       "ON CONFLICT (hash) DO NOTHING;";

        //cout << "Database::createRemote() query: " << query << endl;

        if (asyncWrite) {
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
        cerr << "Error: Database::createRemote() exception: " << e.what() << endl;
        exit(-1);
    }
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

Database::~Database()
{
    if (pConnectionWrite != NULL) delete pConnectionWrite;
    if (pConnectionRead != NULL) delete pConnectionRead;
    
    if (asyncWrite)
    {
        pthread_mutex_destroy(&writeQueueMutex);
        pthread_cond_destroy(&writeQueueCond);
        pthread_cond_destroy(&emptyWriteQueueCond);
    }    
}

int Database::setProgram (const string &key, const vector<uint8_t> &value, const bool persistent)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::read() called uninitialized" << endl;
        return DB_INTERNAL_ERROR;
    }

    vector<Goldilocks::Element> feValue;
    Goldilocks::Element fe;
    for (uint64_t i=0; i<value.size(); i++)
    {
        fe = fr.fromU64(value[i]);
        feValue.push_back(fe);
    }
    create(key, feValue);

    if (config.runStateDBServer && persistent)
    {
        createRemote(key, feValue);
    }

    return DB_SUCCESS;
}

int Database::getProgram (const string &key, vector<uint8_t> &value)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::read() called uninitialized" << endl;
        return DB_INTERNAL_ERROR;
    }

    vector<Goldilocks::Element> feValue;

    // If the value is found in local database (cached) simply return it
    if (db.find(key) != db.end())
    {
        read(key, feValue);
    } else if (config.runStateDBServer)
    {
        // Otherwise, read it remotelly
        readRemote(key, feValue);
    } else {
        return DB_KEY_NOT_FOUND;
    }

    for (uint64_t i=0; i<feValue.size(); i++)
    {
        uint64_t uValue;
        uValue = fr.toU64(feValue[i]);
        zkassert(uValue < (1<<8));
        value.push_back((uint8_t)uValue);
    }
    return DB_SUCCESS;
}

void Database::addWriteQueue (string sqlWrite)
{
    pthread_mutex_lock(&writeQueueMutex);
    writeQueue.push_back(sqlWrite);
    pthread_cond_signal(&writeQueueCond);
    pthread_mutex_unlock(&writeQueueMutex);
}

void Database::flush ()
{
    if (asyncWrite) {
        pthread_mutex_lock(&writeQueueMutex);
        while (writeQueue.size()>0) pthread_cond_wait(&emptyWriteQueueCond, &writeQueueMutex);
        pthread_mutex_unlock(&writeQueueMutex);
    }
}

void Database::commit()
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
        string uri = "postgresql://" + config.dbUser + ":" + config.dbPassword + "@" + config.dbHost + ":" + to_string(config.dbPort) + "/" + config.dbDatabaseName;
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

void* asyncDatabaseWriteThread (void* arg) {
    Database* db = (Database*)arg;   
    db->processWriteQueue();
    return NULL;
}