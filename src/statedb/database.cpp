#include <iostream>
#include "config.hpp"
#include "database.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "definitions.hpp"

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

void Database::read (const string &_key, vector<Goldilocks::Element> &value)
{
    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

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
    } 
    else if (useRemoteDB)
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
        cerr << "Error: Database::read() requested a key that does not exist: " << key << endl;
        exit(-1);
    }

#ifdef LOG_DB_READ
        cout << "Database::read() key=" << key << " value=";
        for (uint64_t i = 0; i < value.size(); i++)
            cout << fr.toString(value[i], 16) << ":";
        cout << endl;        
#endif    
}

void Database::write (const string &_key, const vector<Goldilocks::Element> &value, const bool persistent)
{
    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

    // Check that it has  been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::write() called uninitialized" << endl;
        exit(-1);
    }

    // Create in local database; no need to update remote database
    db[key] = value;
    dbNew[key] = value;

    if (useRemoteDB && persistent)
    {
        writeRemote(key, value);
    }

#ifdef LOG_DB_WRITE
        cout << "Database::write() key=" << key << " value=";
        for (uint64_t i = 0; i < value.size(); i++)
            cout << fr.toString(value[i], 16) << ":";
        cout << endl;        
#endif      
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
        string uri = config.databaseURL;
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
        string createQuery = "CREATE TABLE " + config.tableName + " ( hash BYTEA PRIMARY KEY, data BYTEA NOT NULL );";
        pqxx::result res = w.exec(createQuery);
        w.commit();
#endif
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
                cerr << "Error: Database::readRemote() found incorrect value size: " << stringResult.size() << endl;
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

void Database::writeRemote (const string &key, const vector<Goldilocks::Element> &value)
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

        //cout << "Database::writeRemote() query: " << query << endl;

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
void Database::printTree (const string &root, string prefix)
{
    if (prefix=="") cout << "Printint tree of root=" << root << endl;
    string key = root;
    vector<Goldilocks::Element> value;
    read(key, value);
    if (value.size() != 12)
    {
        cerr << "Error: Database::printTree() found value.size()=" << value.size() << endl;
        return;
    }
    if (!fr.equal(value[11], fr.zero()))
    {
        cerr << "Error: Database::printTree() found value[11]=" << fr.toString(value[11],16) << endl;
        return;
    }
    if (!fr.equal(value[10], fr.zero()))
    {
        cerr << "Error: Database::printTree() found value[10]=" << fr.toString(value[10],16) << endl;
        return;
    }
    if (!fr.equal(value[9], fr.zero()))
    {
        cerr << "Error: Database::printTree() found value[9]=" << fr.toString(value[9],16) << endl;
        return;
    }
    if (fr.equal(value[8], fr.zero())) // Intermediate node
    {
        string leftKey = fea2string(fr, value[0], value[1], value[2], value[3]);
        cout << prefix << "Intermediate node - left hash=" << leftKey << endl;
        printTree(leftKey, prefix+"  ");
        string rightKey = fea2string(fr, value[4], value[5], value[6], value[7]);
        cout << prefix << "Intermediate node - right hash=" << rightKey << endl;
        printTree(rightKey, prefix+"  ");
    }
    else if (fr.equal(value[8], fr.one())) // Leaf node
    {
        string rKey = fea2string(fr, value[0], value[1], value[2], value[3]);
        cout << prefix << "rKey=" << rKey << endl;
        string hashValue = fea2string(fr, value[4], value[5], value[6], value[7]);
        cout << prefix << "hashValue=" << hashValue << endl;
        vector<Goldilocks::Element> leafValue;
        read(hashValue, leafValue);
        if (leafValue.size() == 12)
        {
            if (!fr.equal(leafValue[8], fr.zero()))
            {
                cerr << "Error: Database::printTree() found leafValue[8]=" << fr.toString(leafValue[8],16) << endl;
                return;
            }
            if (!fr.equal(leafValue[9], fr.zero()))
            {
                cerr << "Error: Database::printTree() found leafValue[9]=" << fr.toString(leafValue[9],16) << endl;
                return;
            }
            if (!fr.equal(leafValue[10], fr.zero()))
            {
                cerr << "Error: Database::printTree() found leafValue[10]=" << fr.toString(leafValue[10],16) << endl;
                return;
            }
            if (!fr.equal(leafValue[11], fr.zero()))
            {
                cerr << "Error: Database::printTree() found leafValue[11]=" << fr.toString(leafValue[11],16) << endl;
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
        cout << prefix << "leafValue=" << NormalizeToNFormat(scalarValue.get_str(16), 64) << endl;
    }
    else
    {
        cerr << "Error: Database::printTree() found value[8]=" << fr.toString(value[8],16) << endl;
        return;
    }
    if (prefix=="") cout << endl;
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

int Database::setProgram (const string &_key, const vector<uint8_t> &data, const bool persistent)
{
    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::setProgram() called uninitialized" << endl;
        exit(-1);
    }

    vector<Goldilocks::Element> feValue;
    Goldilocks::Element fe;
    for (uint64_t i=0; i<data.size(); i++)
    {
        fe = fr.fromU64(data[i]);
        feValue.push_back(fe);
    }
    write(key, feValue, persistent);

    if (useRemoteDB && persistent)
    {
        writeRemote(key, feValue);
    }

    return DB_SUCCESS;
}

int Database::getProgram (const string &_key, vector<uint8_t> &data)
{
    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::getProgram() called uninitialized" << endl;
        exit(-1);
    }

    vector<Goldilocks::Element> feValue;

    // If the value is found in local database (cached) simply return it
    if (db.find(key) != db.end())
    {
        read(key, feValue);
    } else if (useRemoteDB)
    {
        readRemote(key, feValue);
    } else {
        cerr << "Error: Database::getProgram() requested a hash that does not exist: " << key << endl;
        exit(-1);
    }

    for (uint64_t i=0; i<feValue.size(); i++)
    {
        uint64_t uValue;
        uValue = fr.toU64(feValue[i]);
        zkassert(uValue < (1<<8));
        data.push_back((uint8_t)uValue);
    }

    return DB_SUCCESS;
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

void* asyncDatabaseWriteThread (void* arg) {
    Database* db = (Database*)arg;   
    db->processWriteQueue();
    return NULL;
}