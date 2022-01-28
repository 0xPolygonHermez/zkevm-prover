#include <iostream>
#include "config.hpp"
#include "database.hpp"
#include "scalar.hpp"

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
    if (config.bServer)
    {
        initRemote();
    }

    // Mark the database as initialized
    bInitialized = true;
}

void Database::read (const RawFr::Element &key, vector<RawFr::Element> &value)
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::read() called uninitialized" << endl;
        exit(-1);
    }

    if (config.bServer)
    {
        // If the value is found in local database (cached) simply return it
        if (db.find(key) != db.end())
        {
            value = db[key];
            return;
        }

        // Otherwise, read it remotelly
        cout << "Database::read() trying to read key remotely, key: " << fr.toString(key,16) << endl;
        readRemote(key, value);

        // Store it locally to avoid any future remote access for this key
        db[key] = value;
        cout << "Database::read() read key remotely, key: " << fr.toString(key,16) << " length: " << to_string(value.size()) << endl;
    }
    else
    {
        value = db[key];
    }
}

void Database::write (const RawFr::Element &key, const vector<RawFr::Element> &value)
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::write() called uninitialized" << endl;
        exit(-1);
    }

    // Store in local database; no need to update remote database
    db[key] = value;
}

void Database::create (const RawFr::Element &key, const vector<RawFr::Element> &value)
{
    // Check that it has  been initialized before
    if (!bInitialized)
    {
        cerr << "Error: Database::create() called uninitialized" << endl;
        exit(-1);
    }

    // Create in local database; no need to update remote database
    db[key] = value;
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
        pConnection = new pqxx::connection{uri};

        /*pqxx::work w3(*pConnection);
        string createSchemaQuery = "CREATE SCHEMA state;";
        pqxx::result res3 = w3.exec(createSchemaQuery);
        w3.commit();*/

#ifdef DATABASE_INIT_WITH_INPUT_DB
        pqxx::work w(*pConnection);
        //string createQuery = "CREATE TABLE state_merkletree ( hash varchar(255), value0 varchar(255), value1 varchar(255), value2 varchar(255), value3 varchar(255), value4 varchar(255), value5 varchar(255), value6 varchar(255), value7 varchar(255), value8 varchar(255), value9 varchar(255), value10 varchar(255), value11 varchar(255), value12 varchar(255), value13 varchar(255), value14 varchar(255), value15 varchar(255) );";
        string createQuery = "CREATE TABLE " + config.tableName + " ( hash BYTEA PRIMARY KEY, data BYTEA NOT NULL );";
        pqxx::result res = w.exec(createQuery);
        w.commit();
#endif
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::initRemote() exception: " << e.what() << endl;
        exit(-1);
    }
}

void Database::readRemote (const RawFr::Element &key, vector<RawFr::Element> &value)
{
    value.clear();
    try
    {
        // Start a transaction.
        pqxx::work w(*pConnection);

        // Prepare the query
        string aux = fr.toString(key, 16);
        string keyString = NormalizeToNFormat(aux, 64);
        string query = "SELECT * FROM " + config.dbTableName + " WHERE hash = E\'\\\\x" + keyString + "\';";
        cout << "Database::readRemote() query: " << query << endl;

        // Execute the query
        pqxx::result res = w.exec(query);

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

        RawFr::Element fe;
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
        w.commit();
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::readRemote() exception: " << e.what() << endl;
        exit(-1);
    }
}

void Database::writeRemote (const RawFr::Element &key, const vector<RawFr::Element> &value)
{
    try
    {
        // Start a transaction.
        pqxx::work w(*pConnection);

        // Prepare the query
        string aux = fr.toString(key, 16);
        string keyString = NormalizeToNFormat(aux, 64);
        string valueString;
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
}

void Database::createRemote (const RawFr::Element &key, const vector<RawFr::Element> &value)
{
    try
    {
        // Start a transaction.
        pqxx::work w(*pConnection);

        // Prepare the query
        string aux = fr.toString(key, 16);
        string keyString = NormalizeToNFormat(aux, 64);
        string valueString;
        for (uint64_t i = 0; i < value.size(); i++)
        {
            aux = fr.toString(value[i], 16);
            valueString += NormalizeToNFormat(aux, 64);
        }
        string query = "INSERT INTO " + config.dbTableName + " ( hash, data ) VALUES ( E\'\\\\x" + keyString + "\', E\'\\\\x" + valueString + "\' );";

        //cout << "Database::createRemote() query: " << query << endl;

        // Execute the query
        pqxx::result res = w.exec(query);

        // Commit your transaction
        w.commit();
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
    for (map<RawFr::Element, vector<RawFr::Element>, CompareFe>::iterator it = db.begin(); it != db.end(); it++)
    {
        RawFr::Element fe = it->first;
        vector<RawFr::Element> vect = it->second;
        cout << "key:" << fr.toString(fe, 16);
        for (uint64_t i = 0; i < vect.size(); i++)
            cout << " " << i << ":" << fr.toString(vect[i], 16);
        cout << endl;
    }
}

Database::~Database()
{
    if (pConnection != NULL)
    {
        delete pConnection;
    }
}