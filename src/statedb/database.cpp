#include <iostream>
#include "config.hpp"
#include "database.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "definitions.hpp"
#include "zkresult.hpp"
#include "utils.hpp"

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
    if (config.databaseURL!="local")
    {
        initRemote();
        useRemoteDB = true;
    } else useRemoteDB = false;

    // Mark the database as initialized
    bInitialized = true;
}

zkresult Database::read (const string &_key, vector<Goldilocks::Element> &value)
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

    // If the value is found in local database (cached) simply return it
    if (db.find(key) != db.end())
    {
        value = db[key];
        r = ZKR_SUCCESS;
    } 
    else if (useRemoteDB)
    {
        // Otherwise, read it remotelly
        r = readRemote(key, value);
        if (r == ZKR_SUCCESS) {
            // Store it locally to avoid any future remote access for this key
            db[key] = value;
        }
    }
    else
    {
        cerr << "Error: Database::read() requested a key that does not exist: " << key << endl;
        r = ZKR_DB_KEY_NOT_FOUND;
    }

#ifdef LOG_DB_READ
    cout << "Database::read()";
    if (r != ZKR_SUCCESS) cout << " ERROR=" << r << " (" << zkresult2string(r) << ")";
    cout << " key=" << key;
    cout << " value=";
    for (uint64_t i = 0; i < value.size(); i++)
        cout << fr.toString(value[i], 16) << ":";
    cout << endl;    
#endif    

    return r;
}

zkresult Database::write (const string &_key, const vector<Goldilocks::Element> &value, const bool persistent)
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
        r = writeRemote(key, value);
    } else r = ZKR_SUCCESS;

    if (r == ZKR_SUCCESS) {
        // Create in memory cache
        db[key] = value;
    }

#ifdef LOG_DB_WRITE
    cout << "Database::write()";
    if (r != ZKR_SUCCESS) cout << " ERROR=" << r << " (" << zkresult2string(r) << ")";
    cout << " key=" << key;
    cout << " value=";
    for (uint64_t i = 0; i < value.size(); i++)
        cout << fr.toString(value[i], 16) << ":";
    cout << " persistent=" << persistent << endl;    
#endif      

    return ZKR_SUCCESS;
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
        exitProcess();
    }
}

zkresult Database::readRemote (const string &key, vector<Goldilocks::Element> &value)
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
            return ZKR_DB_KEY_NOT_FOUND;
        } 
        else if (rows.size() > 1)
        {
            cerr << "Error: Database::readRemote() got more than one row for the same key: " << rows.size() << endl;
            exitProcess();
        }
        
        pqxx::row const row = rows[0];
        if (row.size() != 2)
        {
            cerr << "Error: Database::readRemote() got an invalid number of colums for the row: " << row.size() << endl;
            exitProcess();
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
                exitProcess();
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
        exitProcess();
    }

    return ZKR_SUCCESS;
}

zkresult Database::writeRemote (const string &key, const vector<Goldilocks::Element> &value)
{
    try
    {
        // Prepare the query
        string valueString;
        string aux;
        for (uint64_t i = 0; i < value.size(); i++)
        {
            aux = fr.toString(value[i], 16);
            valueString += NormalizeToNFormat(aux, 64);
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
        exitProcess();
    }

    return ZKR_SUCCESS;
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
        if (leftKey != "0") printTree(leftKey, prefix+"  ");
        string rightKey = fea2string(fr, value[4], value[5], value[6], value[7]);
        cout << prefix << "Intermediate node - right hash=" << rightKey << endl;
        if (rightKey != "0") printTree(rightKey, prefix+"  ");
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

zkresult Database::setProgram (const string &key, const vector<uint8_t> &data, const bool persistent)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::setProgram() called uninitialized" << endl;
        exitProcess();
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

zkresult Database::getProgram (const string &key, vector<uint8_t> &data)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::getProgram() called uninitialized" << endl;
        exitProcess();
    }
    
#ifdef LOG_DB
    cout << "Database::getProgram()" << endl;
#endif  

    zkresult r;

    vector<Goldilocks::Element> feValue;

    r = read(key, feValue);

    if (r == ZKR_SUCCESS) 
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