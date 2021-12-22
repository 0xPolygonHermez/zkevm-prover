#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <vector>
#include <map>
#include "ffiasm/fr.hpp"
#include "compare_fe.hpp"

using namespace std;

enum eDbState
{
    dbs_uninitialized = 0,
    dbs_initialized_local = 1,
    dbs_initialized_postgres = 2
};

enum eDbResult
{
    dbr_ok = 0,
    dbr_error = 1
};

class Database
{
private:
    RawFr &fr;
    eDbState state;

    // Database based on local map attribute
    map< RawFr::Element, vector<RawFr::Element>, CompareFe > db; // This is in fact a map<fe,fe[16]>
    eDbResult readLocal (const RawFr::Element &key, vector<RawFr::Element> &value);
    eDbResult writeLocal (const RawFr::Element &key, const vector<RawFr::Element> &value);

    // Database based on Postgres (PostgreSQL)
    eDbResult readPostgres (const RawFr::Element &key, vector<RawFr::Element> &value);
    eDbResult writePostgres (const RawFr::Element &key, const vector<RawFr::Element> &value);

public:
    Database(RawFr &fr) : fr(fr) { state = dbs_uninitialized; };
    eDbResult initLocal (void);
    eDbResult read (const RawFr::Element &key, vector<RawFr::Element> &value);
    eDbResult write (const RawFr::Element &key, const vector<RawFr::Element> &value);
    void print (void);
};

#endif