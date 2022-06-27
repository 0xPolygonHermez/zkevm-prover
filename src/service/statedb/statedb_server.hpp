#ifndef STATEDB_SERVER_HPP
#define STATEDB_SERVER_HPP

#include "goldilocks/goldilocks_base_field.hpp"
#include "database.hpp"
#include "statedb.hpp"

class StateDBServer
{
private:    
    Goldilocks &fr;
    Config &config;
    StateDB &stateDB;
    pthread_t t;

public:
    StateDBServer (Goldilocks &fr, Config &config, StateDB &stateDB) : fr(fr), config(config), stateDB(stateDB) {}; 
    void run (void);
    void runThread (void);
    void waitForThread (void);
};

void* stateDBServerThread(void* arg);

#endif