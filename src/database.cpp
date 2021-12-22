#include <iostream>
#include "database.hpp"


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

eDbResult Database::read (const RawFr::Element &key, vector<RawFr::Element> &value)
{
    switch (state)
    {
        case dbs_initialized_local:
        {
            return readLocal(key, value);
        }
        case dbs_initialized_postgres:
        {
            return readPostgres(key, value);
        }
        default:
        {
            break;
        }
    }
    cerr << "Error: Database::read() called uninitialized" << endl;
    exit(-1);
}

eDbResult Database::write (const RawFr::Element &key, const vector<RawFr::Element> &value)
{
    switch (state)
    {
        case dbs_initialized_local:
        {
            return writeLocal(key, value);
        }
        case dbs_initialized_postgres:
        {
            return writePostgres(key, value);
        }
        default:
        {
            break;
        }
    }
    cerr << "Error: Database::write() called uninitialized" << endl;
    exit(-1);
}

eDbResult Database::readLocal (const RawFr::Element &key, vector<RawFr::Element> &value)
{
    value = db[key];
    return dbr_ok;
}

eDbResult Database::writeLocal (const RawFr::Element &key, const vector<RawFr::Element> &value)
{
    db[key] = value;
    return dbr_ok;
}

eDbResult Database::readPostgres (const RawFr::Element &key, vector<RawFr::Element> &value)
{
    cerr << "Error: Database::readPostgres() not implemented" << endl;
    exit(-1);
}

eDbResult Database::writePostgres (const RawFr::Element &key, const vector<RawFr::Element> &value)
{
    cerr << "Error: Database::writePostgres() not implemented" << endl;
    exit(-1);
}


void Database::print (void)
{
    switch (state)
    {
        case dbs_initialized_local:
        {
            cout << "Database of " << db.size() << " elements:" << endl;
            for ( map< RawFr::Element, vector<RawFr::Element>, CompareFe >::iterator it = db.begin(); it!=db.end(); it++)
            {
                RawFr::Element fe = it->first;
                vector<RawFr::Element> vect = it->second;
                cout << "key:" << fr.toString(fe, 16);
                for (uint64_t i=0; i<vect.size(); i++)
                    cout << " " << i << ":" << fr.toString(vect[i], 16);
                cout << endl;
            }
            return;
        }
        case dbs_initialized_postgres:
        default:
            return;
    }
}