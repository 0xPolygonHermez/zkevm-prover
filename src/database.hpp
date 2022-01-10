#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <vector>
#include <map>
#include <pqxx/pqxx>
#include "ffiasm/fr.hpp"
#include "compare_fe.hpp"
#include "config.hpp"

using namespace std;


// TODO: Document installation: sudo apt install libpqxx-dev


enum eDbState
{
    dbs_uninitialized = 0,
    dbs_initialized_local = 1,
    dbs_initialized_remote = 2
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

    // Local database based on a map attribute
    map< RawFr::Element, vector<RawFr::Element>, CompareFe > db; // This is in fact a map<fe,fe[16]>
    eDbResult initLocal (void);
    eDbResult initRemote (void);
    eDbResult readLocal (RawFr::Element &key, vector<RawFr::Element> &value);
    eDbResult writeLocal (RawFr::Element &key, const vector<RawFr::Element> &value);
    eDbResult createLocal (RawFr::Element &key, const vector<RawFr::Element> &value);

    // Remote database based on Postgres (PostgreSQL)
    pqxx::connection * pConnection;
    string tableName;
    eDbResult readRemote (RawFr::Element &key, vector<RawFr::Element> &value);
    eDbResult writeRemote (RawFr::Element &key, const vector<RawFr::Element> &value);
    eDbResult createRemote (RawFr::Element &key, const vector<RawFr::Element> &value);

public:
    Database(RawFr &fr) : fr(fr) { state = dbs_uninitialized; pConnection = NULL; };
    ~Database();
    eDbResult init (void);
    eDbResult read (RawFr::Element &key, vector<RawFr::Element> &value); // TODO: key to be const when ffi library allows
    eDbResult write (RawFr::Element &key, const vector<RawFr::Element> &value);
    eDbResult create (RawFr::Element &key, const vector<RawFr::Element> &value);
    void print (void);
};

#endif