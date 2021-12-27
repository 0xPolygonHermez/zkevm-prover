#include <iostream>
#include "config.hpp"
#include "database.hpp"
#include "scalar.hpp"

/************/
/* Wrappers */
/************/

eDbResult Database::init(void)
{
#ifdef USE_LOCAL_DATABASE
    return initLocal();
#else
    return initRemote();
#endif
}

eDbResult Database::read (RawFr::Element &key, vector<RawFr::Element> &value)
{
    switch (state)
    {
    case dbs_initialized_local:
    {
        return readLocal(key, value);
    }
    case dbs_initialized_remote:
    {
        return readRemote(key, value);
    }
    default:
    {
        break;
    }
    }
    cerr << "Error: Database::read() called uninitialized" << endl;
    exit(-1);
}

eDbResult Database::write (RawFr::Element &key, vector<RawFr::Element> &value)
{
    switch (state)
    {
    case dbs_initialized_local:
    {
        return writeLocal(key, value);
    }
    case dbs_initialized_remote:
    {
        return writeRemote(key, value);
    }
    default:
    {
        break;
    }
    }
    cerr << "Error: Database::write() called uninitialized" << endl;
    exit(-1);
}

eDbResult Database::create (RawFr::Element &key, vector<RawFr::Element> &value)
{
    switch (state)
    {
    case dbs_initialized_local:
    {
        return createLocal(key, value);
    }
    case dbs_initialized_remote:
    {
        return createRemote(key, value);
    }
    default:
    {
        break;
    }
    }
    cerr << "Error: Database::create() called uninitialized" << endl;
    exit(-1);
}

/*********/
/* Local */
/*********/

eDbResult Database::initLocal (void)
{
    if (state != dbs_uninitialized)
    {
        cerr << "Error: Database::initLocal() called in already initialized" << endl;
        exit(-1);
    }
    state = dbs_initialized_local;
    return dbr_ok;
}

eDbResult Database::readLocal (RawFr::Element &key, vector<RawFr::Element> &value)
{
    value = db[key];
    return dbr_ok;
}

eDbResult Database::writeLocal (RawFr::Element &key, vector<RawFr::Element> &value)
{
    db[key] = value;
    return dbr_ok;
}

eDbResult Database::createLocal (RawFr::Element &key, vector<RawFr::Element> &value)
{
    db[key] = value;
    return dbr_ok;
}

/**********/
/* Remote */
/**********/

eDbResult Database::initRemote (void)
{
    if (state != dbs_uninitialized)
    {
        cerr << "Error: Database::initRemote() called in already initialized" << endl;
        exit(-1);
    }

    try
    {
        /* Start in localhost with::
        $ sudo -u postgres psql postgres
        sudo -u postgres createuser -P hermez
        //dropuser hermez
        sudo -u postgres createdb polygon-hermez
        sudo -u postgres dropdb polygon-hermez

        */

        // Build the remote database URI
        string user = "hermez";
        string pwd = "polygon";
        string host = "localhost";
        string dbname = "polygon-hermez";
        uint64_t port = 5432;
        string uri = "postgresql://" + user + ":" + pwd + "@" + host + ":" + to_string(port) + "/" + dbname;
        cout << "Database URI: " << uri << endl;

        // Create the connection
        pConnection = new pqxx::connection{uri};

#ifdef INIT_DATABASE_WITH_INPUT_JSON_DATA
         tableName = "state_merkletree";
#else
         tableName = "state.merkletree";
#endif

        /*pqxx::work w3(*pConnection);
        string createSchemaQuery = "CREATE SCHEMA state;";
        pqxx::result res3 = w3.exec(createSchemaQuery);
        w3.commit();*/

#ifdef USE_LOCALHOST_DATABASE
        pqxx::work w(*pConnection);
        string createQuery = "CREATE TABLE state_merkletree ( hash varchar(255), value0 varchar(255), value1 varchar(255), value2 varchar(255), value3 varchar(255), value4 varchar(255), value5 varchar(255), value6 varchar(255), value7 varchar(255), value8 varchar(255), value9 varchar(255), value10 varchar(255), value11 varchar(255), value12 varchar(255), value13 varchar(255), value14 varchar(255), value15 varchar(255) );";
        pqxx::result res2 = w.exec(createQuery);
        w.commit();
#endif
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::initRemote() exception: " << e.what() << endl;
        exit(-1);
        return dbr_error;
    }

    state = dbs_initialized_remote;
    return dbr_ok;
}

eDbResult Database::readRemote (RawFr::Element &key, vector<RawFr::Element> &value)
{
    value.clear();
    try
    {
        // Start a transaction.
        pqxx::work w(*pConnection);

        // Prepare the query
        string keyString;
        keyString = fr.toString(key, 16);
        string query = "SELECT * FROM " + tableName + " WHERE hash = \'" + keyString + "\';"; // TODO: replace by state.merkletree
        cout << "Database::readRemote() query: " << query << endl;

        // Execute the query
        pqxx::result res = w.exec(query);

        /*for (pqxx::row::size_type r=0; r<res.size(); r++)
        {
            pqxx::row const row = res[r];
            for (pqxx::row::size_type c = 0; c < row.size(); c++)
            {
                pqxx::field const field = row[c];
                cout << "Database row: " << r << " column: " << c << " field: " << field.c_str() << endl;
            }
        }*/

        // Process the result
        if (res.size() != 1)
        {
            cerr << "Error: Database::readRemote() got an invalid result size: " << res.size() << endl;
            exit(-1);
        }
        pqxx::row const row = res[0];
        for (pqxx::row::size_type c = 1; c < row.size(); c++)
        {
            pqxx::field const field = row[c];
            cout << "Database row: " << 0 << " column: " << c << " field: " << field.c_str() << endl;
            RawFr::Element fe;
            string2fe(fr, field.c_str(), fe);
            value.push_back(fe);
        }

        // Commit your transaction
        w.commit();
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::readRemote() exception: " << e.what() << endl;
        exit(-1);
        return dbr_error;
    }
    return dbr_ok;
}

eDbResult Database::writeRemote (RawFr::Element &key, vector<RawFr::Element> &value)
{
    try
    {
        // Start a transaction.
        pqxx::work w(*pConnection);

        // Prepare the query
        string keyString;
        keyString = fr.toString(key, 16);
        string valuesString;
        for (uint64_t i = 0; i < value.size(); i++)
        {
            if (i > 0)
            {
                valuesString += ", ";
            }
            string aux;
            aux = fr.toString(value[i], 16);
            valuesString += "value" + to_string(i) + " = \'" + aux + "\'";
        }
        string query = "UPDATE " + tableName + " SET " + valuesString + " WHERE key = \'" + keyString + "\';";
        cout << "Database::writeRemote() query: " << query << endl;
/*
UPDATE Customers
SET ContactName = 'Alfred Schmidt', City= 'Frankfurt'
WHERE CustomerID = 1;
*/
        // Execute the query
        pqxx::result res = w.exec(query);

        // Commit your transaction
        w.commit();
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::writeRemote() exception: " << e.what() << endl;
        exit(-1);
        return dbr_error;
    }

    return dbr_ok;
}

eDbResult Database::createRemote (RawFr::Element &key, vector<RawFr::Element> &value)
{
    try
    {
        // Start a transaction.
        pqxx::work w(*pConnection);

        // Prepare the query
        string keyString;
        keyString = fr.toString(key, 16);
        string columnsString = "( hash, ";
        string valuesString = "(\'" + keyString + "\', ";
        for (uint64_t i = 0; i < value.size(); i++)
        {
            if (i > 0)
            {
                columnsString += ", ";
                valuesString += ", ";
            }
            string aux;
            aux = fr.toString(value[i], 16);
            valuesString += "\'" + aux + "\'";
            columnsString += "value" + to_string(i);
        }
        valuesString += ")";
        columnsString += ")";
        string query = "INSERT INTO " + tableName + " " + columnsString + " VALUES " + valuesString + ";"; // TODO: replace by state.merkletree
        cout << "Database::createRemote() query: " << query << endl;

        // Execute the query
        pqxx::result res = w.exec(query);

        // Commit your transaction
        w.commit();
    }
    catch (const std::exception &e)
    {
        cerr << "Error: Database::createRemote() exception: " << e.what() << endl;
        exit(-1);
        return dbr_error;
    }

    return dbr_ok;
}

/*********/
/* Other */
/*********/

void Database::print(void)
{
    switch (state)
    {
    case dbs_initialized_local:
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
        return;
    }
    case dbs_initialized_remote:
    default:
        return;
    }
}

Database::~Database()
{
    switch (state)
    {
    case dbs_initialized_local:
        break;
    case dbs_initialized_remote:
        if (pConnection != NULL)
            delete pConnection;
        break;
    default:
        break;
    }
}