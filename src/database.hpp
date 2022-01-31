#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <vector>
#include <map>
#include <pqxx/pqxx>
#include "ffiasm/fr.hpp"
#include "compare_fe.hpp"
#include "config.hpp"

using namespace std;

class Database
{
private:
    RawFr &fr;
    bool bInitialized;
    Config config;

    // Local database based on a map attribute
    map< RawFr::Element, vector<RawFr::Element>, CompareFe > db; // This is in fact a map<fe,fe[16]>
public:
    map< RawFr::Element, vector<RawFr::Element>, CompareFe > dbNew; // Additions to the original db done through the execution of the prove or execute query
    map< RawFr::Element, vector<RawFr::Element>, CompareFe > dbRemote; // Data originally not present in local database that required fetching it from the remote database

private:
    // Remote database based on Postgres (PostgreSQL)
    pqxx::connection * pConnection;
    void initRemote (void);
    void readRemote (const RawFr::Element &key, vector<RawFr::Element> &value);
    void writeRemote (const RawFr::Element &key, const vector<RawFr::Element> &value);
    void createRemote (const RawFr::Element &key, const vector<RawFr::Element> &value);

public:
    Database(RawFr &fr) : fr(fr)
    { 
        bInitialized = false;
        pConnection = NULL;
    };
    ~Database();
    void init (const Config &config);
    void read (const RawFr::Element &key, vector<RawFr::Element> &value);
    void write (const RawFr::Element &key, const vector<RawFr::Element> &value);
    void create (const RawFr::Element &key, const vector<RawFr::Element> &value);
    void print (void);
};

#endif