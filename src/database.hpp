#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <vector>
#include <map>
#include <pqxx/pqxx>
#include "ff/ff.hpp"
#include "compare_fe.hpp"
#include "config.hpp"

using namespace std;

class Database
{
private:
    FiniteField &fr;
    bool bInitialized;
    Config config;

    // Local database based on a map attribute
    map<string, vector<FieldElement>> db; // This is in fact a map<fe,fe[16]>
public:
    map<string, vector<FieldElement>> dbNew; // Additions to the original db done through the execution of the prove or execute query
    map<string, vector<FieldElement>> dbRemote; // Data originally not present in local database that required fetching it from the remote database

private:
    // Remote database based on Postgres (PostgreSQL)
    pqxx::connection * pConnection;
    void initRemote (void);
    void readRemote (const string &key, vector<FieldElement> &value);
    void writeRemote (const string &key, const vector<FieldElement> &value);
    void createRemote (const string &key, const vector<FieldElement> &value);

public:
    Database(FiniteField &fr) : fr(fr)
    { 
        bInitialized = false;
        pConnection = NULL;
    };
    ~Database();
    void init (const Config &config);
    void read (const string &key, vector<FieldElement> &value);
    void write (const string &key, const vector<FieldElement> &value);
    void create (const string &key, const vector<FieldElement> &value);
    void setProgram (const string &key, const vector<uint8_t> &value);
    void getProgram (const string &key, vector<uint8_t> &value);
    void print (void);
};

#endif